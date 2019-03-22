import os
import glob
import subprocess
from datetime import datetime
import keras
import tensorflow as tf

from modules.utils import generate_desc_csv, plot_denoise, plot_triplet

from keras.models import Model


def plots(denoise_model, training_generator):
    pass
    ### plot_denoise(denoise_model) ## need to make this savefig
    ### plot_triplet(training_generator)

def perform_tests(descriptor_model, seqs_test, denoise_model, use_clean=False,
                  denoiser_tag='DEFAULT', descr_logdir='logs', write_event=False):
    import subprocess
    #### final testing stage ###
    ### generate csv file
    print("\nGenerating descriptor database as CSV...")
    generate_desc_csv(descriptor_model, seqs_test, denoise_model=denoise_model, use_clean=use_clean)

    print("Patch verif....")
    output = ''
    ## patch verif
    import sys
    output += subprocess.check_output('python ./hpatches-benchmark/hpatches_eval.py \
    --descr-name=custom --descr-dir=out/ --task=verification --delimiter=";"', shell=True).decode(sys.stdout.encoding)
    output += subprocess.check_output('python ./hpatches-benchmark/hpatches_results.py \
    --descr=custom --results-dir=./hpatches-benchmark/results/ --task=verification', shell=True).decode(sys.stdout.encoding)

    ## matching
    print("Matching....")
    output += subprocess.check_output('python ./hpatches-benchmark/hpatches_eval.py \
    --descr-name=custom --descr-dir=out/ --task=matching --delimiter=";"', shell=True).decode(sys.stdout.encoding)
    output += subprocess.check_output('python ./hpatches-benchmark/hpatches_results.py \
    --descr=custom --results-dir=./hpatches-benchmark/results/ --task=matching', shell=True).decode(sys.stdout.encoding)

    ## patch retrieval
    print("Retrieval....")
    output += subprocess.check_output('python ./hpatches-benchmark/hpatches_eval.py \
    --descr-name=custom --descr-dir=out/ --task=retrieval --delimiter=";"', shell=True).decode(sys.stdout.encoding)
    output += subprocess.check_output('python ./hpatches-benchmark/hpatches_results.py \
    --descr=custom --results-dir=./hpatches-benchmark/results/ --task=retrieval', shell=True).decode(sys.stdout.encoding)

    print(output)

    #os.system('zip -rq descriptors.zip ./out/custom')

    if write_event and os.path.isdir(descr_logdir):
        from callbacks import add_text_tb
        writer = tf.summary.FileWriter(descr_logdir)
        add_text_tb(writer, output, 'results')



def upload_recent_weights(weight_dir='weights/', tag=datetime.now().strftime("%Y%m%d_%H%M")):
    print("=> Uploading Most Recent Weights...")

    output = b""
    
    list_of_files = glob.iglob(weight_dir + '/denoiser*.hdf5')

    latest_file = max(list_of_files, key=os.path.getctime, default=None)

    if latest_file is not None:
        print('Denoiser File: %s' % latest_file)
        output += subprocess.check_output('curl -F "file=@%s" https://file.io' %\
                                            latest_file, shell=True)
        print("Output:\n", output)


    list_of_files = glob.iglob(weight_dir + '/descriptor*.hdf5')
    latest_file = max(list_of_files, key=os.path.getctime, default=None)

    if latest_file is not None:
        print('Descriptor File: %s' % latest_file)
        output += subprocess.check_output('curl -F "file=@%s" https://file.io' %\
                                            latest_file, shell=True)
        print("Output:\n", output)
    
    if not os.path.isdir("logs/upload_urls"):
        os.mkdir("logs/upload_urls")

    with open("logs/upload_urls/upload_%s.txt" % tag, "wb") as text_file:
        text_file.write(output)




if __name__ == '__main__':
    '''

    '''
    from read_data import MultiDataLoader
    import argparse

    parser = argparse.ArgumentParser(description='Evaluation using HPatches Benchmark.')
    parser.add_argument('-dnf', '--denoiser_file', help="Denoiser filenames", type=str,
                        metavar='FNAME', default=None)
    parser.add_argument('-dsf', '--descriptor_file', help="Descriptor filenames", type=str,
                        metavar='FNAME', default=None)
    
    args = parser.parse_args()

    USE_CLEAN = False

    weight_dir='weights/'

    if not USE_CLEAN:
        if args.denoiser_file is not None and os.path.isfile(args.denoiser_file):
            print("=> Loading given denoising model for evalauation...")
            latest_file = args.denoiser_file
        else: 
            list_of_files = glob.iglob(weight_dir + '/denoiser*.hdf5')
            latest_file = max(list_of_files, key=os.path.getctime, default=None)

        if latest_file is not None:
            print("=> Loading latest denoising model for evaluation...")
            denoise_model = keras.models.load_model(latest_file,
                                                    custom_objects={
                                                        '<lambda>': 'mean_absolute_error',
                                                        'DSSIMObjective': 'mean_absolute_error'
                                                        })
        else:
            raise RuntimeError("No denoise model found in dir: %s" % weight_dir)
        print("Loaded %s successfully..." % latest_file)
    else:
        denoise_model = None



    if args.descriptor_file is not None and os.path.isfile(args.descriptor_file):
        print("=> Loading given descriptor model for evaluation...")
        latest_file = args.descriptor_file

    else:
        list_of_files = glob.iglob(weight_dir + '/descriptor*.hdf5')
        latest_file = max(list_of_files, key=os.path.getctime, default=None)

    if latest_file is not None:
        print("=> Loading latest descriptor model for evaluation...")
        descriptor_model = keras.models.load_model(latest_file,
                                                   custom_objects={
                                                                    '<lambda>': 'mean_absolute_error',
                                                                    'loss': 'mean_absolute_error'
                                                   })
    else:
        raise RuntimeError("No descriptor model found in dir: %s" % weight_dir)
    print("Loaded %s successfully..." % latest_file)


    ## a check to see if the network descriptor is of triplet type i.e. 3 networks with shared
    ## weights or if it is a standard model, for triplet first 3 layers must be input type
    is_triplet = True
    for i in range(3):
        if not isinstance(descriptor_model.layers[i], keras.engine.input_layer.InputLayer):
            is_triplet = False

    if is_triplet:
        print("Info: descriptor detected as triplet model, will use extracted sub_model for eval")
        ## extract relevant nested model
        inner_model = Model(descriptor_model.get_layer('sequential_1').get_input_at(0), output=\
                    descriptor_model.get_layer('sequential_1').get_output_at(0))
    else:
        print("Info: descriptor detected as standard model (non-triplet) for eval")
        inner_model = descriptor_model
    #print(model2.summary())
    #exit()
    perform_tests(inner_model, MultiDataLoader().seqs_test, denoise_model, use_clean=USE_CLEAN)

    # zip -rq descriptors.zip ./out/custom