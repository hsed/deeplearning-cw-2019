import tensorflow as tf
from keras.layers import concatenate, Input, Activation, Add, Conv2D, Lambda, UpSampling2D
from keras.models import Model

'''
    https://github.com/idealo/image-super-resolution
'''


def get_denoise_model_rdn(input_shape, C=3, D=6, G=64, G0=64, x=1, c_dim=1, kernel_size=3):
    """ Returns the model.
    Used to select the model.

    See RDN class for details on param
    """
    arch_params = {
        'C': C,#3,#6 -- conv layers inside RDB
        'D': D,#6,#20 -- rdb layers
        'G': G, #64,#64 -- num_filters in each conv layer in each RDB
        'G0': G0, #64,#64
        'x': x, #1#2
    }

    #c_dim = input_shape[2]
    #patch_size = input_shape[0]

    return RDN(arch_params, patch_size=32, c_dim=c_dim, kernel_size=kernel_size).model


class RDN:
    """Implementation of the Residual Dense Network for image super-scaling.
    The network is the one described in https://arxiv.org/abs/1802.08797 (Zhang et al. 2018).
    Args:
        arch_params: dictionary, contains the network parameters C, D, G, G0, x.
        patch_size: integer or None, determines the input size. Only needed at
            training time, for prediction is set to None.
        c_dim: integer, number of channels of the input image.
        kernel_size: integer, common kernel size for convolutions.
        upscaling: string, 'ups' or 'shuffle', determines which implementation
            of the upscaling layer to use.
    Attributes:
        C: integer, number of conv layer inside each residual dense blocks (RDB).
        D: integer, number of RDBs.
        G: integer, number of convolution output filters inside the RDBs.
        G0: integer, number of output filters of each RDB.
        x: integer, the scaling factor.
        model: Keras model of the RDN.
        name: name used to identify what upscaling network is used during training.
        model.name: identifies this network as the generator network
            in the compound model built by the trainer class.
    """

    ## they use mse loss for this!!
    ## also use lr decay every 100 ep i think with lr *= 0.5 uptil lr == 1e-6
    ## batch size is 16
    adam_params = {
        'epsilon': 0.0000001,
        'learning_rate': 0.0004,
    }

    def __init__(self, arch_params={}, patch_size=None, c_dim=3, kernel_size=3, upscaling='ups'):
        self.params = arch_params
        self.C = self.params['C']
        self.D = self.params['D']
        self.G = self.params['G']
        self.G0 = self.params['G0']
        self.scale = self.params['x']
        self.patch_size = patch_size
        self.c_dim = c_dim
        self.kernel_size = kernel_size
        self.upscaling = upscaling
        self.model = self._build_rdn()
        self.model.name = 'generator'
        self.name = 'rdn'

    def _upsampling_block(self, input_layer):
        """ Upsampling block for old weights. """

        x = Conv2D(self.c_dim * self.scale ** 2, kernel_size=3, padding='same', name='UPN3')(
            input_layer
        )
        return UpSampling2D(size=self.scale, name='UPsample')(x)

    def _pixel_shuffle(self, input_layer):
        """ PixelShuffle implementation of the upscaling layer. """

        x = Conv2D(self.c_dim * self.scale ** 2, kernel_size=3, padding='same', name='UPN3')(
            input_layer
        )
        return Lambda(
            lambda x: tf.depth_to_space(x, block_size=self.scale, data_format='NHWC'),
            name='PixelShuffle',
        )(x)

    def _UPN(self, input_layer):
        """ Upscaling layers. With old weights use _upsampling_block instead of _pixel_shuffle. """

        x = Conv2D(64, kernel_size=5, strides=1, padding='same', name='UPN1')(input_layer)
        x = Activation('relu', name='UPN1_Relu')(x)
        x = Conv2D(32, kernel_size=3, padding='same', name='UPN2')(x)
        x = Activation('relu', name='UPN2_Relu')(x)
        if self.upscaling == 'shuffle':
            return self._pixel_shuffle(x)
        elif self.upscaling == 'ups':
            return self._upsampling_block(x)
        else:
            raise ValueError('Invalid choice of upscaling layer.')

    def _RDBs(self, input_layer):
        """RDBs blocks.
        Args:
            input_layer: input layer to the RDB blocks (e.g. the second convolutional layer F_0).
        Returns:
            concatenation of RDBs output feature maps with G0 feature maps.
        """
        rdb_concat = list()
        rdb_in = input_layer
        for d in range(1, self.D + 1):
            x = rdb_in
            for c in range(1, self.C + 1):
                F_dc = Conv2D(
                    self.G, kernel_size=self.kernel_size, padding='same', name='F_%d_%d' % (d, c)
                )(x)
                F_dc = Activation('relu', name='F_%d_%d_Relu' % (d, c))(F_dc)
                # concatenate input and output of ConvRelu block
                # x = [input_layer,F_11(input_layer),F_12([input_layer,F_11(input_layer)]), F_13..]
                x = concatenate([x, F_dc], axis=3, name='RDB_Concat_%d_%d' % (d, c))
            # 1x1 convolution (Local Feature Fusion)
            x = Conv2D(self.G0, kernel_size=1, name='LFF_%d' % (d))(x)
            # Local Residual Learning F_{i,LF} + F_{i-1}
            rdb_in = Add(name='LRL_%d' % (d))([x, rdb_in])
            rdb_concat.append(rdb_in)

        assert len(rdb_concat) == self.D

        return concatenate(rdb_concat, axis=3, name='LRLs_Concat')

    def _build_rdn(self):
        LR_input = Input(shape=(self.patch_size, self.patch_size, self.c_dim), name='LR')
        F_m1 = Conv2D(self.G0, kernel_size=self.kernel_size, padding='same', name='F_m1')(LR_input)
        F_0 = Conv2D(self.G0, kernel_size=self.kernel_size, padding='same', name='F_0')(F_m1)
        FD = self._RDBs(F_0)
        # Global Feature Fusion
        # 1x1 Conv of concat RDB layers -> G0 feature maps
        GFF1 = Conv2D(self.G0, kernel_size=1, padding='same', name='GFF_1')(FD)
        GFF2 = Conv2D(self.G0, kernel_size=self.kernel_size, padding='same', name='GFF_2')(GFF1)
        # Global Residual Learning for Dense Features
        FDF = Add(name='FDF')([GFF2, F_m1])
        # Upscaling, bypass if no scaling required
        FU = self._UPN(FDF) if self.scale == 1 else FDF
        # Compose SR image
        SR = Conv2D(self.c_dim, kernel_size=self.kernel_size, padding='same', name='SR')(FU)

        return Model(inputs=LR_input, outputs=SR)