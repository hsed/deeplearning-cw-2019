import keras
import pandas as pd
from datetime import datetime
import numpy as np
import os
import tensorflow as tf
from pprint import pformat
import yaml
import subprocess

from utils import plot_denoise_v2

from read_data import DenoiseHPatches
from utils import plot_denoise

from read_data import MultiDataLoader

import re

def add_text_tb(summary_writer, text, tag='default_tag', index=0):
    text_tensor = tf.make_tensor_proto(text, dtype=tf.string)
    meta = tf.SummaryMetadata()
    meta.plugin_data.plugin_name = "text"
    summary = tf.Summary()
    summary.value.add(tag=tag, metadata=meta, tensor=text_tensor)
    summary_writer.add_summary(summary, index) #tf.summary.text('info', tf.constant("This is test data"))

def add_img_tb(summary_writer, img_buf, img_shape, tag='default_tag', index=0):
    #tf.image.decode_jpeg(image_data, channels = 3, dct_method='INTEGER_ACCURATE')
    #tf.image.decode_png('plots/denoise_plot.png', channels=3)
    #https://stackoverflow.com/questions/38543850/tensorflow-how-to-display-custom-images-in-tensorboard-e-g-matplotlib-plots

    # Convert PNG buffer to TF image
    image_string = img_buf.getvalue()
    
    h,w,c = img_shape[0],img_shape[1],img_shape[2]
    #print("H, W, C: ", h, w, c)
    image = tf.Summary.Image(height=h, width=w, colorspace=c, encoded_image_string=image_string)
    # image = tf.image.decode_png(img_buf.getvalue(), channels=4)

    # # Add the batch dimension
    # image = tf.expand_dims(image, 0)

    # Add image summary
    #summary_op = tf.summary.image(tag, image)

    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=image)])

    #summary = tf.summary.image(tag, imgs_tensor, max_outputs=5, family=tag)
    summary_writer.add_summary(summary, index)##summary.eval(session=tf.keras.backend.get_session()) #tf.keras.backend.get_session().run(summary_op)
    summary_writer.flush()
    img_buf.close()

# histogram_freq ==> for activation and weight histograms, set to 1 for every epoch
# https://keras.io/callbacks/#tensorboard
def get_callbacks(callback_type='denoiser', tag = datetime.now().strftime("%Y%m%d_%H%M"), yaml_config=None,
                  data_obj=None):
    ### makedir in logs folder for each experiment!
    ### call it tag string or maybe add some name too?


    tb_log_dir = './logs/%s_%s' % (callback_type, tag)
    csv_log_dir = './logs/%s_hist_csv' % callback_type
    cache_dir = './cache/training/%s' % (tag)

    if not os.path.isdir(tb_log_dir):
        os.makedirs(tb_log_dir, exist_ok=True)
    if not os.path.isdir(csv_log_dir):
        os.makedirs(csv_log_dir, exist_ok=True)
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    tensorboard_callback = \
        keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=0,
                                    batch_size=32, write_graph=True,
                                    write_grads=False, write_images=False,
                                    embeddings_freq=0, embeddings_layer_names=None,
                                    embeddings_metadata=None, embeddings_data=None,
                                    update_freq='epoch')

    regular_checkpoint_callback = \
        keras.callbacks.ModelCheckpoint('cache/training/%s/%s_{epoch:02d}.hdf5' % (tag, callback_type),
                                        monitor='val_loss', verbose=1,
                                        save_best_only=False, save_weights_only=False,
                                        period=1)#period=2

    best_checkpoint_callback = \
        keras.callbacks.ModelCheckpoint('cache/training/%s/%s_best.hdf5' % (tag, callback_type),
                                        monitor='val_loss', verbose=1,
                                        save_best_only=True, save_weights_only=False,
                                        period=1)#period=2

    loss_callback = LossHistory(callback_type, tag=tag, log_dir=csv_log_dir)

    extra_info_callback = ExtraInfoCallback(tensorboard_callback, yaml_config, name=callback_type, tag=tag, log_dir=tb_log_dir,
                                            data_obj=data_obj)

    # any MAE > 6 is automatically bad for validation

    early_stop_callback_baseline = \
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=False) #baseline=(6 if callback_type == 'denoiser' else 0.99)
    ## handles image printing and text information printig
    ## including upload urls for final best weights
    #denoiser_extra_info_callback

    return [tensorboard_callback, regular_checkpoint_callback,
            best_checkpoint_callback, loss_callback, extra_info_callback, early_stop_callback_baseline]



# https://keras.io/callbacks/
class LossHistory(keras.callbacks.Callback):
    def __init__(self, name="model", tag = datetime.now().strftime("%Y%m%d_%H%M"),
                 log_dir="logs"):
        self.name = name
        self.tag = tag
        self.log_dir = log_dir

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.epochs = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.epochs.append(epoch)


    def on_train_end(self, logs={}):
        ## need to transpose cause as it stands otherwise
        ## each column will be row by default!
        file_path = os.path.join(self.log_dir, 'loss_%s_%s.csv' % (self.name, self.tag))
        data_table = np.array([self.epochs, self.losses, self.val_losses]).T
        df = pd.DataFrame(data = data_table,
                           columns=["epoch", "loss", "val_loss"])
        df.to_csv(file_path, index=False)


# class UploadCheckpoints(keras.callbacks.Callback):
#     def __init__(self, frequency=5):
#         self.save_freq = 5

#     def on_train_begin(self, logs={}):
#         self.urls = []


#     def on_epoch_end(self, epoch, logs={}):
#         if epoch % self.save_freq == 0:
#             print("Uploading ")



class ExtraInfoCallback(keras.callbacks.Callback):
    def __init__(self, tb_callback, yaml_config, name="model", tag = datetime.now().strftime("%Y%m%d_%H%M"),
                 log_dir="logs",  save_root_dir='cache/training', data_obj: MultiDataLoader =None):
        self.log_dir = log_dir
        self.name = name
        self.tag = tag
        self.tb_callback = tb_callback
        ## tensorboard requires markdown style new line which is "<spc><spc>\n"
        self.config_info = yaml.dump(yaml_config)#re.sub(r"[\n]*", "  \n", )

        if 'epoch_reshuffle' in yaml_config['training']:
            #print("EPOCH RESHUFFLE = ON")
            self.do_epoch_reshuffle = yaml_config['training']['epoch_reshuffle']
        else:
            #print("EPOCH RESHUFFLE = OFF")
            self.do_epoch_reshuffle = False


        self.data_obj = data_obj


        self.final_best_file_path = os.path.normpath(os.path.join(save_root_dir, tag, name + "_best.hdf5"))
        print("Info: Final best weights will be %s" % self.final_best_file_path)

    def on_epoch_end(self, epoch, logs={}):
        #self.model
        # self.losses.append(logs.get('loss'))
        # self.val_losses.append(logs.get('val_loss'))
        # self.epochs.append(epoch)
        #generator = DenoiseHPatches(['./hpatches/v_talent'], cache_overwrite=True, toy_data=True, batch_size=16)## add option to skip cache loading and saving
        #imgs, imgs_clean = generator[999]# get batch of 32 images
        #index = np.random.randint(0, imgs.shape[0])
        #imgs_den = self.model.predict(imgs) # tensor

        if self.name == 'denoiser':
            buf, img_shape = plot_denoise_v2(self.model, show_plt=False, save_fig=False, return_buf=True)
            add_img_tb(self.tb_callback.writer, buf, img_shape, 'denoised', epoch)
        
        if self.name == 'descriptor' and self.data_obj is not None and self.do_epoch_reshuffle:
            print("Data_Obj found, recreating training descriptor dataset...")
            self.data_obj.training_generator.on_epoch_end()
            self.data_obj.val_generator.on_epoch_end()
            

    def on_train_begin(self, logs={}):
        # note need to do this here cause otherwise
        # writer object is not initialised by keras
        add_text_tb(self.tb_callback.writer, self.config_info, 'info/config')
        add_text_tb(self.tb_callback.writer, str(self.model.count_params()), 'info/trainable_params')

    def on_train_end(self, logs={}):
        ## upload best model
        ## need to get accurate string
        self.tb_callback.writer = tf.summary.FileWriter(self.log_dir)
        self.upload_best_weights()


    def upload_best_weights(self):
        print("=> Uploading Best Weights...")

        output = b""

        # list_of_files = glob.iglob(weight_dir + '/denoiser*.hdf5')

        best_file = 'cache/training/%s/%s_best.hdf5' % (self.tag, self.name)

        if best_file is not None:
            print('Best Weight File: %s' % best_file)
            #print("Warning weight uploading disabled!")
            output += subprocess.check_output('curl -F "file=@%s" https://file.io' %\
                                                best_file, shell=True)
            print("Output:\n", output)

            add_text_tb(self.tb_callback.writer, output, 'info/url')
        