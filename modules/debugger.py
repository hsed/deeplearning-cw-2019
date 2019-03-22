from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras.optimizers import Adam
#import keras.backend as K

import numpy as np
import tensorflow as tf
import keras

from numpy.random import randn as randn


def get_batch_size(inputs):
        ### special case for list type for multi inputs
        if isinstance(inputs, list):
            batch_size = inputs[0].shape[0] # assume homogenous
        else:
            batch_size = inputs.shape[0]
        return batch_size


class KD(object):
    '''
        KD == KerasDebugger
    '''

    def __init__(self, keras_model=None, pre_compiled=False, loss='mean_absolute_error', optimizer=keras.optimizers.Adam()):

        if keras_model is None:
            print("Debug using default model...")
            self.keras_model = Sequential([Conv2D(6, 3, padding='same', input_shape=(32,32,1), use_bias = True)])
        else:
            self.keras_model = keras_model
        
        if not pre_compiled:
            self.keras_model.compile(loss=loss, optimizer=optimizer)

    def __call__(self, inputs: np.ndarray):
        
        batch_size = get_batch_size(inputs)
        result = self.keras_model.predict(inputs, batch_size=batch_size)
    
        return result
    

    def overfit(self, inputs: np.ndarray, targets: np.ndarray = None, epochs=10, verbose=0):
        '''
            overfit to see loss reduction
            returns history dictionary
        '''
        if targets is None:
            print("WARNING: No targets supplied for fitting, will produce random data...")
            outputs = self(inputs)

            ## use '*' to unpack as arguments
            targets = np.random.randn(*outputs.shape)

            print("Outputs.shape: ", outputs.shape, "Targets.shape: ", targets.shape)
        
        else:
            assert inputs.shape[0] == targets.shape[0], "Last dimension must match!"

        callback_obj = self.keras_model.fit(x=inputs, y=targets, epochs=epochs, verbose=verbose, batch_size=get_batch_size(inputs))

        return callback_obj.history
    

    def get_model(self):
        return self.keras_model





if __name__ == "__main__":
    ### debugging the debugger!!

    '''
        To use this tool simply place a break point at a point
        where you are adding layers to your model and then 
        pass inputs as expected and you will get output shape

        you can also call overfit to see if backprop is working

        loss: default 'mean_absolute_error'
        optimizer: default 'adam'
    '''

    ### example
    from arch import *

    INPUT_SHAPE = (32,32,1) # H, W, C

    init_weights = keras.initializers.he_normal()
  
    descriptor_model = Sequential()
    descriptor_model.add(Conv2D(32, 3, padding='same', input_shape=INPUT_SHAPE, use_bias = True, kernel_initializer=init_weights))
    descriptor_model.add(BatchNormalization(axis = -1))
    descriptor_model.add(Activation('relu'))

    ### kets debug at this point using batch size of 10
    ### batch size inferred by last dim
    inputs = np.random.randn(10,32,32,1)
    outputs = KD(descriptor_model)(inputs)

    print("Outputs Shape: ", outputs.shape)


    ### add some more stuff
    descriptor_model.add(Conv2D(32, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
    descriptor_model.add(BatchNormalization(axis = -1))
    descriptor_model.add(Activation('relu'))

    descriptor_model.add(Conv2D(64, 3, padding='same', strides=2, use_bias = True, kernel_initializer=init_weights))
    descriptor_model.add(BatchNormalization(axis = -1))
    descriptor_model.add(Activation('relu'))

    ## lets say we just want to know losses, dunno output shape but thats fine..
    losses = KD(descriptor_model).overfit(inputs)

    print("Losses:\n", losses)

    ## ofc can also supply targets
    #targets = np.random.randn(10, ...)
    #losses = KD(descriptor_model).overfit(inputs, targets)