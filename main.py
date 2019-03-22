import sys
import json
import os
import glob
import time
import tensorflow as tf
import numpy as np
import cv2
import random

# "host": "ssh4.vast.ai",
# "port": 34429,


### some things may be extra
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Input, UpSampling2D, concatenate

import argparse

from pprint import pprint

from modules.utils import generate_desc_csv, plot_denoise, plot_triplet
from modules.arch import DenoiserModel, DescriptorModel
from modules.read_data import MultiDataLoader
from evaluate import perform_tests, upload_recent_weights
from modules.callbacks import get_callbacks

from datetime import datetime

import yaml

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('-c', '--config', default='config/baseline.yaml', type=str,
                    help='config file path (default: config/baseline.yaml)')
parser.add_argument('-ep', '--epochs', type=int, default=None, help='overwrite num epochs from config')
parser.add_argument('-red', '--reduced', action='store_true', help='use reduced dataset for training')
parser.add_argument('-dc', '--disable_callbacks', action='store_true', help='disable all kinds of keras callbacks')

args = parser.parse_args()
if args.config:
        # load config file
        config = yaml.load(open(args.config))
elif args.resume:
      # TODO
      # load config file from checkpoint, in case new config file is not given.
      # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
      config = None  # torch.load(args.resume)['config']

if args.reduced:
	print("***WARNING: USING REDUCED DATASET***")

print("--- CONFIG ---")
pprint(config)
#printm()

#### fix seeds ####
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

epochs = config['training']['epochs'] if args.epochs is None else args.epochs


if config['train_mode'] == 'denoiser':
      print("Descriptor training will be skipped!")
      train_denoise = True
      train_descript = False
elif config['train_mode'] == 'descriptor':
      print("Denoiser training will be skipped!")
      train_denoise = False
      train_descript = True
elif config['train_mode'] == 'both' or config['train_mode'] == 'all':
      print("Both models will be trained!")
      train_denoise = True
      train_descript = True
else:
      raise ValueError("Config Train_Mode = %s is unrecognized!" %
                       config['train_mode'])



### data loader init, no actual loading ###
data_obj = MultiDataLoader()


### experiment datetime for logs and weights ###
exp_datetime_str = datetime.now().strftime("%Y%m%d_%H%M")
print("\n\n--- STATS ---")
print("EXPERIMENT: %s" % exp_datetime_str,
      "EP: %d" % epochs,)  
      # "TRAIN_BS: %d" % train_batch_sz, "VAL_BS: %d" % val_batch_sz

#### denoiser training ####

# load denoiser weights if requested
if len(config['denoiser_cache_tags']) > 0:
 print("Loading last denoiser cache tag: ", config['denoiser_cache_tags'][-1])
 denoise_model = \
                  keras.models.load_model('cache/training/%s/denoiser_best.hdf5' % config['denoiser_cache_tags'][-1],
                                          custom_objects={'<lambda>': 'mean_absolute_error',
                                                      'DSSIMObjective': 'mean_absolute_error'})



if train_denoise:
      den_ep = epochs
      if 'denoiser_epochs' in config['training'] and args.epochs is None:
            den_ep = config['training']['denoiser_epochs']
            print("Setting denoiser epochs from config: %d" % den_ep)
      denoise_model = DenoiserModel(**config['denoiser_arch']).model
      data_obj.load_denoiser_dataset(reduced_dataset=args.reduced, **config['dataloader']['denoiser'])
      print("\n=> Training denoiser...")
      denoise_history = denoise_model.fit_generator(
							generator=data_obj.denoise_generator,
							epochs=den_ep, verbose=1,
							validation_data=data_obj.denoise_generator_val,
							workers=4,
							callbacks=(get_callbacks('denoiser', tag=exp_datetime_str,
													yaml_config=config) \
                                                                              if not args.disable_callbacks else None)
							)


if train_descript:
      desc_ep = epochs
      if 'descriptor_epochs' in config['training'] and args.epochs is None:
            desc_ep = config['training']['descriptor_epochs']
            print("Setting descriptor epochs from config: %d" % desc_ep)
      descriptor_model_trip, descriptor_model = DescriptorModel(**config['descriptor_arch']).get_both_models()
      ### must load desc dataset AFTER training denoiser ###
      data_obj.load_descr_dataset(denoise_model=denoise_model, **config['dataloader']['descriptor'])
      print("\n=> Training descriptor...")
      descriptor_history = descriptor_model_trip.fit_generator(
								generator=data_obj.training_generator,
								epochs=desc_ep, verbose=1, workers=4,
								validation_data=data_obj.val_generator,
								callbacks=(get_callbacks('descriptor',
                                                                        tag=exp_datetime_str, #pass data_obj for shuffling!
                                                                        yaml_config=config, data_obj=data_obj)
                                                                        if not args.disable_callbacks else None)
      						)

      print("Evaluating descriptor model...")
      ### final testing mAP ####
      perform_tests(descriptor_model, data_obj.seqs_test, denoise_model)
