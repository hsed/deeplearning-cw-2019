import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization 
from keras.layers import Input, UpSampling2D, concatenate 
from keras.layers import Lambda

from triplet_loss import batch_hard_triplet_loss, batch_all_triplet_loss


from keras_contrib.losses.dssim import DSSIMObjective


#from keras.con

from rdn import get_denoise_model_rdn

'''
    Models and loss
We now define three functions that define the main modules of our baseline.

get_denoise_model(..) returns the denoising model.
    The input for the function is the size of the patch, which will be 1x32x32, 
    and it outputs a keras denoising model.
get_descriptor_model(..) builts the descriptor model.
    The input for the function is the size of the patch, which will be 1x32x32,
    and it outputs a keras descriptor model. The model we use as baseline returns 
    a descriptor of dimension 128x1.
triplet_loss(..) defines the loss function which is used to train the descriptor model.
    You can modify the models in these functions and run the training code again.
    For example, the given denoising model is quite shallow, maybe using a deeper
    network can improve results. Or testing new initializations for the weights.
    Or maybe adding dropout. Or modifying the loss function somehow...

'''



'''
    improve denoise model:
    - more layers see unet model
    - more filters
    - use of residual dense network
    - attention??
    - pxiel shuffle
'''

def none2emptydict(obj_list):
    '''
        simple generator mapper

        map: [None, x, None, ..., None] -> [{}, x, {}, ..., {}]
        where x is a non-null object of anytype

    '''
    return ({} if item is None else item for item in obj_list)

def get_denoise_model(shape):

    
  inputs = Input(shape)
  #32x32x3

  ## Encoder starts
  
  conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  #16x16x16

  ## Bottleneck
  conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  #16x16x32

  ## Now the decoder starts
  up1 = UpSampling2D(size = (2,2))(conv2)
  # 32x32x32

  up3 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up1)
  # 32x32x64

  merge3 = concatenate([conv1,up3], axis = -1)
  # 32x32x80

  conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
  # 32 x 32 x 64

  conv4 = Conv2D(1, 3,  padding = 'same')(conv3)
  # 32 x 32 x 1

  shallow_net = Model(inputs = inputs, outputs = conv4)
  
  return shallow_net


def get_denoise_model_1a(shape):
  '''  
        standard model
  '''

  # 32 x 32 x 1
  inputs = Input(shape)
  #32x32x1

  ## Encoder starts
  
  # 32x32x1 -> 32x32x16
  conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
  
  # 32x32x16 -> 16x16x16
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  ## Bottleneck

  ## 16x16x16 -> 16x16x32
  conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)


  ## Now the decoder starts
  # 16x16x32 -> 32x32x32
  up1 = UpSampling2D(size = (2,2))(conv2)

  # 32x32x32 -> 32x32x64 -- note a concatenation was skipped here! TODO
  # this is conv wrongly named upsampling!
  up3 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up1)

  # [32x32x16, 32x32x64] -> 32x32x80  # Note this assymetrical concatenation is bad!! fix this TODO
  merge3 = concatenate([conv1,up3], axis = -1)

  # 32x32x80 -> 32x32x64
  conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
  
  # 32x32x64 -> 32x32x1
  conv4 = Conv2D(1, 3,  padding = 'same')(conv3)

  shallow_net = Model(inputs = inputs, outputs = conv4)
  
  return shallow_net


def make_conv(inputs, kernel, out_channels, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'):
	# e.g. out_channel = 16, kernel = 3
	return Conv2D(out_channels, kernel, activation = activation, padding = padding, kernel_initializer = kernel_initializer)(inputs)


def make_pool(inputs, pool_size=(2,2)):
	# 2,2 => stride 2, kernel 2, produces downsample of spatial dim by 2 so WxHxC -> W/2xH/2xC
	return MaxPooling2D(pool_size=pool_size)(inputs)

def make_upsample(inputs, size = (2,2)):
	# extrapolation
	# HxWxC -> H*2 x W*2 x C
	return UpSampling2D(size = (2,2))(inputs)



def make_convpooldown(inputs, conv_steps, kernel, out_channels):
	'''
		inputs -> conv_steps*make_conv -> make_pool

		Note: out channels is fixed for all conv.
	'''

	# HxWxC_in -> HxWxC_out -> ... -> HxWxC_out
	convs = inputs
	for i in range(conv_steps):
		convs = make_conv(convs, kernel, out_channels)
	
	## downsample
	# HxWxC_out -> H/2 x W/2 x C_out
	# also return convs for future concat
	return make_pool(convs, pool_size=(2,2)), convs


def make_upconvconcatconv(inputs, concat, conv_steps, kernel, out_channels):
	'''
		in -> up -> conv -> concat -> conv_steps*conv -> out
	'''
	### for symmetry, this is good to have
	assert concat.shape[-1] == out_channels

	# HxWxC -> H*2 x W*2 x C
	outputs = make_upsample(inputs, size=(2,2))

	# H*2 x W*2 x C -> H*2 x W*2 x C_out
	outputs = make_conv(outputs, kernel, out_channels)

	### concat
	## C_out -> C_out+C_concat
	outputs = concatenate([concat,outputs], axis = -1)

	## C_out+C_concat -> C_out
	## all extra conv, need atleast 1 to maintain C
	for i in range(conv_steps):
		# default kernel=3
		outputs = make_conv(outputs, kernel, out_channels)
	
	# H*2 x W*2 x C_out
	return outputs


	## note this conv needs to be same size as out_channels

def get_denoise_model_1b(shape, top_lvl_channels = 16, kernel = 3, conv_repeats = 1):
  '''  
        top_lvl_channels -> starting channels, max_channels = top_lvl_channels*8
		conv_repeats -> repeated convs after filter fixing and after concat+filter_fix

        for orig model:
        get_denoise_model_1b((32,32,1), 64, conv_repeats=2).summary()
  '''

  '''
	1 -> 16 ->																	            32 -> 16+16 -> 16 -> 1
		 16 ->	32 ->													     64 -> 32+32 -> 32
				32 -> 64 ->							         128 -> 64+64 -> 64			
					  64 -> 128 ->		   256 -> 128+128 -> 128
							128 -> 256 --> 256
  '''
  
  


  ### input: 32x32x1
  inputs = Input(shape)

  ### 32x32x1 -> 
  convpool1, conv1 = make_convpooldown(inputs, conv_repeats, kernel, top_lvl_channels) # 16x16x16
  convpool2, conv2 = make_convpooldown(convpool1, conv_repeats, kernel, top_lvl_channels*2) # 8x8x32
  convpool3, conv3 = make_convpooldown(convpool2, conv_repeats, kernel, top_lvl_channels*4) # 4x4x64
  convpool4, conv4 = make_convpooldown(convpool3, conv_repeats, kernel, top_lvl_channels*8) # 2x2x128
  
  ### bottelneck 2x2x128 -> 2x2x256
  bottleneck_conv = make_conv(convpool4, kernel, top_lvl_channels*16)
  ###

  ### 2x2x256 ->
  upsampconv1 = make_upconvconcatconv(bottleneck_conv, conv4, conv_repeats, kernel, top_lvl_channels*8) # 4x4x128
  upsampconv2 = make_upconvconcatconv(upsampconv1, conv3, conv_repeats, kernel, top_lvl_channels*4) # 8x8x64
  upsampconv3 = make_upconvconcatconv(upsampconv2, conv2, conv_repeats, kernel, top_lvl_channels*2) # 16x16x32
  upsampconv4 = make_upconvconcatconv(upsampconv3, conv1, conv_repeats, kernel, top_lvl_channels) # 32x32x16

  ### final output 32x32x16 -> 32x32x1
  output = make_conv(upsampconv4, kernel, 1)

  return Model(inputs = inputs, outputs = output)

def get_descriptor_model_orig(shape):
  
  '''Architecture copies HardNet architecture'''
  
  init_weights = keras.initializers.he_normal()
  
  descriptor_model = Sequential()
  descriptor_model.add(Conv2D(32, 3, padding='same', input_shape=shape, use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(32, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(64, 3, padding='same', strides=2, use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(64, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(128, 3, padding='same', strides=2,  use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(128, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))
  descriptor_model.add(Dropout(0.3))

  descriptor_model.add(Conv2D(128, 8, padding='valid', use_bias = True, kernel_initializer=init_weights))
  
  # Final descriptor reshape
  descriptor_model.add(Reshape((128,)))

  
  return descriptor_model

def get_descriptor_model_1c(shape):
  cardinality = 1#32
  from keras import layers
  
  '''Architecture copies HardNet architecture'''
  def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y

  def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y


  def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y
  
  def add_common_layer_orig(x):
      x = BatchNormalization(axis = -1)(x)
      return Activation('relu')(x)
  

  init_weights = keras.initializers.he_normal()
  inputs = Input(shape)
  x = Conv2D(32, 3, padding='same', input_shape=shape, use_bias = True, kernel_initializer=init_weights)(inputs)
  x = add_common_layer_orig(x)
  #descriptor_model = Sequential()
#   descriptor_model.add()
#   descriptor_model.add()
#   descriptor_model.add()
  x = Conv2D(32, 3, padding='same', use_bias = True, kernel_initializer=init_weights)(x)
  x = add_common_layer_orig(x)

  for i in range(2):
    # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
    strides = (2, 2) if i == 0 else (1, 1)
    x = residual_block(x, 32, 64, _strides=strides)

#   descriptor_model.add(Conv2D(64, 3, padding='same', strides=2, use_bias = True, kernel_initializer=init_weights))
#   x = add_common_layer_orig(x)
# 

  x = Conv2D(64, 3, padding='same', use_bias = True, kernel_initializer=init_weights)(x)
  x = add_common_layer_orig(x)

  x = Conv2D(128, 3, padding='same', strides=2,  use_bias = True, kernel_initializer=init_weights)(x)
  x = add_common_layer_orig(x)

  x = Conv2D(128, 3, padding='same', use_bias = True, kernel_initializer=init_weights)(x)
  x = add_common_layer_orig(x)
  x = Dropout(0.3)(x)
  x = Conv2D(128, 8, padding='valid', use_bias = True, kernel_initializer=init_weights)(x)
  
  x = Reshape((128,))(x)

  # l2_norm
  # descriptor_model.add(Lambda(lambda  x: K.l2_normalize(x,axis=1)))

  
  return Model(inputs=inputs, outputs=x)



def get_descriptor_model_1d(shape):
  
  '''Architecture copies HardNet architecture'''
  
  init_weights = keras.initializers.he_normal()
  conv_multiplier = 2
  
  descriptor_model = Sequential()
  descriptor_model.add(Conv2D(32*conv_multiplier, 3, padding='same', input_shape=shape, use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(32*conv_multiplier, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(64*conv_multiplier, 3, padding='same', strides=2, use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(64*conv_multiplier, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(128*conv_multiplier, 3, padding='same', strides=2,  use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(128*conv_multiplier, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))
  descriptor_model.add(Dropout(0.3))

  descriptor_model.add(Conv2D(128, 8, padding='valid', use_bias = True, kernel_initializer=init_weights))
  
  # Final descriptor reshape
  descriptor_model.add(Reshape((128,)))

  # l2_norm
  # descriptor_model.add(Lambda(lambda  x: K.l2_normalize(x,axis=1)))

  
  return descriptor_model


def get_descriptor_model(shape):
  
  '''Architecture copies HardNet architecture'''
  
  init_weights = keras.initializers.he_normal()
  
  descriptor_model = Sequential()
  descriptor_model.add(Conv2D(32, 3, padding='same', input_shape=shape, use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(32, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(64, 3, padding='same', strides=2, use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(64, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(128, 3, padding='same', strides=2,  use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))

  descriptor_model.add(Conv2D(128, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
  descriptor_model.add(BatchNormalization(axis = -1))
  descriptor_model.add(Activation('relu'))
  descriptor_model.add(Dropout(0.3))

  descriptor_model.add(Conv2D(128, 8, padding='valid', use_bias = True, kernel_initializer=init_weights))
  
  # Final descriptor reshape
  descriptor_model.add(Reshape((128,)))

  # l2_norm
  # descriptor_model.add(Lambda(lambda  x: K.l2_normalize(x,axis=1)))

  
  return descriptor_model


def make_offline_triplet_model(shape):
    # get simple descriptor model
    descriptor_model = get_descriptor_model(shape)

    xa = Input(shape=shape, name='a')
    xp = Input(shape=shape, name='p')
    xn = Input(shape=shape, name='n')

    ## replicate the network with shared weights
    ea = descriptor_model(xa)
    ep = descriptor_model(xp)
    en = descriptor_model(xn)
    
    ## apply loss as actual output
    loss = Lambda(triplet_loss)([ea, ep, en])

    return Model(inputs=[xa, xp, xn], outputs=loss), descriptor_model

def make_offline_triplet_model_resnet(shape):
    # get simple descriptor model
    descriptor_model = get_descriptor_model_1c(shape)

    xa = Input(shape=shape, name='a')
    xp = Input(shape=shape, name='p')
    xn = Input(shape=shape, name='n')

    ## replicate the network with shared weights
    ea = descriptor_model(xa)
    ep = descriptor_model(xp)
    en = descriptor_model(xn)
    
    ## apply loss as actual output
    loss = Lambda(triplet_loss)([ea, ep, en])

    return Model(inputs=[xa, xp, xn], outputs=loss), descriptor_model

def make_offline_triplet_model_more_filters(shape):
    # get simple descriptor model
    descriptor_model = get_descriptor_model_1d(shape)

    xa = Input(shape=shape, name='a')
    xp = Input(shape=shape, name='p')
    xn = Input(shape=shape, name='n')

    ## replicate the network with shared weights
    ea = descriptor_model(xa)
    ep = descriptor_model(xp)
    en = descriptor_model(xn)
    
    ## apply loss as actual output
    loss = Lambda(triplet_loss)([ea, ep, en])

    return Model(inputs=[xa, xp, xn], outputs=loss), descriptor_model



def make_online_hardnet_triplet_model(shape):
    # get simple descriptor model
    descriptor_model = get_descriptor_model(shape)

    xa = Input(shape=shape, name='a')
    xp = Input(shape=shape, name='p')
    #xn = Input(shape=shape, name='n')

    ## replicate the network with shared weights
    ea = descriptor_model(xa)
    ep = descriptor_model(xp)
    #en = descriptor_model(xn)
    
    ## apply loss as actual output
    loss = Lambda(loss_hardnet)([ea, ep])

    '''
        Try newer easier to understand method...
        Basically:
        descriptor_model_trip = Model(inputs=[xa, xp, xn], outputs=[ea, ep, en])
        descriptor_model_trip.compile(loss=triplet_loss, optimizer=sgd)

        also try incorporating multiinput multi out within the model
    '''

    return Model(inputs=[xa, xp], outputs=loss), descriptor_model

import numpy as np
import tensorflow as tf
def distance_matrix_vector(A, B):
    #https://gist.github.com/mbsariyildiz/34cdc26afb630e8cae079048eef91865
    # # A = anchor
    # # B = positive
    # r = tf.reduce_sum(A*B, 1)
    # eps = tf.cast(tf.convert_to_tensor([1e-6]), A.dtype)
    # # turn r into column vector
    # r = tf.reshape(r, [-1, 1])
    # return tf.sqrt((r - 2*tf.matmul(A, tf.transpose(B)) + tf.transpose(r))+eps)
        # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)
    
    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0)) + 1e-5
    return D


def loss_hardnet(x, anchor_swap = False, anchor_ave = False,\
                margin = 100.0, batch_reduce = 'min', loss_type = "triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """
    anchor, positive = x
    #print("anchor_shape, pos shape")
    #assert anchor.shape == positive.shape, "Input sizes between positive and negative must be equal."
    assert len(anchor.shape) == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive)+eps
    eye = tf.cast(tf.diag(tf.fill(tf.shape(dist_matrix[0]), 1)), dist_matrix.dtype)

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = tf.linalg.tensor_diag_part(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    #con =  tf.constant([0.008], dtype=dist_without_min_on_diag.dtype)
    #mask = (tf.cast(tf.math.greater_equal(dist_without_min_on_diag, con), dist_without_min_on_diag.dtype)-1)*-1
    #mask = (dist_without_min_on_diag.ge(0.008).float()-1)*-1
    
    #mask = tf.cast(mask, dist_without_min_on_diag.dtype)*10
    
    #dist_without_min_on_diag = dist_without_min_on_diag+mask
    if batch_reduce == 'min':
        min_neg = K.min(dist_without_min_on_diag,1)[0]
        if anchor_swap:
            min_neg2 = K.min(dist_without_min_on_diag,0)[0]
            min_neg = K.min(min_neg,min_neg2)
        # if False:
        #     dist_matrix_a = distance_matrix_vector(anchor, anchor)+ eps
        #     dist_matrix_p = distance_matrix_vector(positive,positive)+eps
        #     dist_without_min_on_diag_a = dist_matrix_a+eye*10
        #     dist_without_min_on_diag_p = dist_matrix_p+eye*10
        #     min_neg_a = torch.min(dist_without_min_on_diag_a,1)[0]
        #     min_neg_p = torch.t(torch.min(dist_without_min_on_diag_p,0)[0])
        #     min_neg_3 = torch.min(min_neg_p,min_neg_a)
        #     min_neg = torch.min(min_neg,min_neg_3)
        #     print (min_neg_a)
        #     print (min_neg_p)
        #     print (min_neg_3)
        #     print (min_neg)
        min_neg = min_neg
        pos = pos1
        #print("min_neg, pos", K.eval(min_neg), K.eval(pos))
    # elif batch_reduce == 'average':
    #     pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
    #     min_neg = dist_without_min_on_diag.view(-1,1)
    #     if anchor_swap:
    #         min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
    #         min_neg = torch.min(min_neg,min_neg2)
    #     min_neg = min_neg.squeeze(0)
    # elif batch_reduce == 'random':
    #     idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
    #     min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
    #     if anchor_swap:
    #         min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1)) 
    #         min_neg = torch.min(min_neg,min_neg2)
    #     min_neg = torch.t(min_neg).squeeze(0)
    #     pos = pos1
    # else: 
    #     print ('Unknown batch reduce mode. Try min, average or random')
    #     sys.exit(1)
    if loss_type == "triplet_margin":
        loss = K.clip(margin + pos - min_neg, 0.0, 1e10)
    # elif loss_type == 'softmax':
    #     exp_pos = torch.exp(2.0 - pos);
    #     exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
    #     loss = - torch.log( exp_pos / exp_den )
    # elif loss_type == 'contrastive':
    #     loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    # else: 
    #     print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
    #     sys.exit(1)
    loss = K.expand_dims(loss, axis=1) #K.mean(loss)
    return loss



  
def triplet_loss(x):
  
  output_dim = 128
  a, p, n = x
  _alpha = 1.0
  positive_distance = K.mean(K.square(a - p), axis=-1)
  negative_distance = K.mean(K.square(a - n), axis=-1)
  
  return K.expand_dims(K.maximum(0.0, positive_distance - negative_distance + _alpha), axis = 1)



#def dssim

def init_loss_fn(loss_str, loss_params):
    if loss_params == {}:
            print("Empty Params! Using Default Params for DSSIM: k1=0.01, k2=0.03, kernel_size=3, max_value=1.0")

    if loss_str == 'dssim':
        ## check params
        ## empty dict automatically uses default params
        return DSSIMObjective(**loss_params)
    elif loss_str == 'dssim_and_mae':
        if 'alpha_dssim' not in loss_params:
            print("alpha_dssim not found using 0.01")
            alpha = 0.01
        else:
            alpha = loss_params['alpha_dssim']
            del loss_params['alpha_dssim']
        dssim = DSSIMObjective(**loss_params)
        mae = keras.losses.mae
        return lambda y_true, y_pred: alpha*dssim(y_true, y_pred)+(1-alpha)*mae(y_true, y_pred)
    else:
        raise NotImplementedError("Error: Custom Loss Fn %s Not Implemented" % loss_str)



def get_semihard_loss(margin=1.0):
    def loss(y_true, y_pred):
        loss, pos_trip = batch_all_triplet_loss(y_true, y_pred, margin)
        return loss
    return loss



class DenoiserModel:
    '''
        Returns fully compiled model ready for action by `DenoiserModel().model`

    '''
    models = {
        '1': get_denoise_model,
        '1a': get_denoise_model_1a,
        '1b': get_denoise_model_1b,
        '2a': get_denoise_model_rdn,
    }

    optims = {
        'adam': keras.optimizers.Adam,
        'sgd': keras.optimizers.SGD
    }

    # metrics = {
    #     'mae',
    #     'mse'
    # }
    def_sgd_params = {
        'lr': 0.00001,
        'momentum': 0.9,
        'nesterov': True,
    }

    losses = {
        'mean_absolute_error': 'mean_absolute_error',
        'mean_square_error': 'mean_square_error',
        'dssim': None, # placeholder
        'dssim_and_mae': None, #[losses['mean_absolute_error'], losses['dssim']]
    }


    def __init__(self, model_revision='1', optim_name='adam', metric_name='mae', loss_name='mean_absolute_error',
                 model_params = None, optim_params = None, future_params = None):
        

        model_params, optim_params, future_params = \
            none2emptydict([model_params, optim_params, future_params])
        
        ### denoiser_model ###
        optim = DenoiserModel.optims[optim_name](**optim_params)

        #adam = keras.optimizers.Adam()
        #sgd = keras.optimizers.SGD(lr=0.00001, momentum=0.9, nesterov=True)
        
        loss = DenoiserModel.losses[loss_name]
        
        ## special case for custom losses
        if loss is None:
            ## special case need to init loss as y_pred, y_true fn
            if 'loss_params' not in future_params:
                # if no params then add empty dict value
                loss_params = {}
            else:
                loss_params = future_params['loss_params'].copy() # copy the dict so del can be performed safely
            loss = init_loss_fn(loss_name, loss_params)
        
        input_shape = (32, 32, 1)
        
        denoise_model = DenoiserModel.models[model_revision](input_shape, **model_params)
        
        denoise_model.compile(loss=loss,
                              optimizer=optim,#adam,#sgd
                              metrics=[metric_name])   # sgd defined in arch.py
        
        self.model = denoise_model
        




class DescriptorModel:
    '''
        Returns fully compiled model ready for action by `DescriptorModel().model`

        Use `.get_both_models()` to get descriptor_model_trip, descriptor_model

    '''

    models = {
        '1': make_offline_triplet_model,
        '1b': make_online_hardnet_triplet_model,
        '1c': make_offline_triplet_model_resnet,
        '1d': make_offline_triplet_model_more_filters,
        '1e': lambda shape: (get_descriptor_model(shape), None),
        '1f': lambda shape: (get_descriptor_model_1c(shape), None),
    }

    optims = {
        'adagrad': keras.optimizers.Adagrad,
        'adam': keras.optimizers.Adam,
        'sgd': keras.optimizers.SGD
    }

    losses = {
        'mean_absolute_error': 'mean_absolute_error',
        'semihard': get_semihard_loss(margin=0.1)
        #'triplet_loss': triplet_loss, #lambda ea, ep, en: Lambda(triplet_loss)([ea, ep, en])
    }

    def __init__(self, model_revision='1', optim_name='sgd', loss_name='mean_absolute_error',
                 model_params = None, optim_params = None, future_params = None):
        
        model_params, optim_params, future_params = \
            none2emptydict([model_params, optim_params, future_params])

        ### descriptor_model ###
        shape = (32, 32, 1)
        
        # provide two models one of which simply dupliates the model
        descriptor_model_trip, descriptor_model = DescriptorModel.models[model_revision](shape, **model_params)

        optim = DescriptorModel.optims[optim_name](**optim_params)
        loss = DescriptorModel.losses[loss_name]
        
        # adagrad = keras.optimizers.adagrad(lr=0.1)
        # sgd = keras.optimizers.SGD(lr=0.1)
        # adam = keras.optimizers.Adam()

        descriptor_model_trip.compile(loss=loss, optimizer=optim)#adagrad)#adam)#sgd

        self.model = descriptor_model_trip
        self._descriptor_model = descriptor_model ## needed for final testing
    
    def get_both_models(self):
        return self.model, self._descriptor_model



if __name__ == "__main__":
	from debugger import *
	# inputs = Input((32,32,1)); KD(Model(inputs, make_convpool(inputs, 1, 3, 16))).overfit(np.random.randn(10,32,32,1))


	# inputs = Input((16,16,32)); concat = Input((32,32,16)); outs = make_upsampleconcatconv(inputs, concat, 1, 3, 16); KD(Model([inputs,concat], outs)).overfit([randn(10,16,16,32), randn(10,32,32,16)])


	# KD(get_denoise_model_1b((32,32,1), 32, 3, 2)).get_model().summary()
	# KD(get_denoise_model_1b((32,32,1), 32, 3, 2))(randn(10,32,32,1))
	# KD(get_denoise_model_1b((32,32,1), 32, 3, 2)).overfit(randn(10,32,32,1))