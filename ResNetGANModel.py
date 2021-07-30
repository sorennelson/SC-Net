import tensorflow as tf
import tensorflow_addons as tfa


# Generator
# TODO: Change to params dict
def resnet_generator(depth=6, input_shape=(128,128), output_shape=(128,128), norm='instance'): 
    if norm == 'instance':
      norm_layer = tfa.layers.InstanceNormalization
    elif norm == 'batch':
      norm_layer = tf.keras.layers.BatchNormalization

    if input_shape == (320,240) and output_shape == (32,32):
        return resnet_generator_320x240_32x32(depth=depth, norm_layer=norm_layer)
    elif input_shape == (32,32) and output_shape == (320,240):
        return resnet_generator_32x32_320x240(depth=depth, norm_layer=norm_layer)
    else:
        return resnet_generator_1x1(depth=depth)



def resnet_generator_320x240_32x32(depth=6, norm_layer=tf.keras.layers.BatchNormalization, activation=tf.keras.layers.ReLU):
    input = tf.keras.layers.Input((240, 320, 3))
    # c7 -> (320, 240), (164, 120, 64), (86, 60, 128), (47, 30, 128), (28, 16, 128), (16, 16, 256) 
    # -> resnet layers 
    # -> (32, 32, 128), (32, 32, 64) -> c7

    c7s1_64 = tf.pad(input, [[0,0], [3,3], [3,3], [0,0]], 'REFLECT')
    c7s1_64 = conv_norm_act(c7s1_64, 64, 7, padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=activation())

    # Down
    d64 = tf.pad(c7s1_64, [[0,0], [0,0], [4,4], [0,0]], 'REFLECT')
    d64 = conv_norm_act(d64, 64, 3, (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d128 = tf.pad(d64, [[0,0], [0,0], [4,4], [0,0]], 'REFLECT')
    d128 = conv_norm_act(d128, 128, 3, (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d128 = tf.pad(d128, [[0,0], [0,0], [4,4], [0,0]], 'REFLECT')
    d128 = conv_norm_act(d128, 128, 3, (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d128 = tf.pad(d128, [[0,0], [1,1], [4,4], [0,0]], 'REFLECT')
    d128 = conv_norm_act(d128, 128, 3, (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d256 = tf.pad(d128, [[0,0], [0,0], [2,2], [0,0]], 'REFLECT')
    d256 = conv_norm_act(d256, 256, 3, (1,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())

    # Residual layers
    r = d256
    for i in range(depth):
      r = residual_block(r, 256, 3, dilation_rate=1, norm_layer=norm_layer, activation=activation())
    
    # Up
    u256 = conv_norm_act(r, 256, 3, dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u128 = convtranspose_norm_act(u256, 128, 3, dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u64 = conv_norm_act(u128, 64, 3, dilation_rate=1, norm_layer=norm_layer, activation=activation())

    c7s1_3 = tf.pad(u64, [[0,0], [3,3], [3,3], [0,0]], 'REFLECT')
    c7s1_3 = conv_norm_act(c7s1_3, 3, 7, padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=tf.keras.activations.sigmoid)
    # c7s1_3 = tf.keras.layers.Activation('tanh')(c7s1_3)

    model = tf.keras.models.Model(input, c7s1_3, name='resnet_generator_320x240_32x32')
    return model


def resnet_generator_32x32_320x240(depth=6, norm_layer=tf.keras.layers.BatchNormalization, activation=tf.keras.layers.ReLU):
    input = tf.keras.layers.Input((32, 32, 3))
    # c7 -> (32, 30), (16, 15, 128), (16, 15, 256)
    # -> resnet layers
    # -> (32, 30, 256), (80, 60, 128), (160, 120, 128), -(320, 240, 128)-, (320, 240, 64) -> c7

    c7s1_64 = tf.pad(input, [[0,0], [2,2], [3,3], [0,0]], 'REFLECT')
    c7s1_64 = conv_norm_act(c7s1_64, 64, 7, padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=activation())

    # Down
    d128 = conv_norm_act(c7s1_64, 128, 3, (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d256 = conv_norm_act(d128, 256, 3, (1,1), dilation_rate=1, norm_layer=norm_layer, activation=activation())

    # Residual layers
    r = d256
    for i in range(depth):
      r = residual_block(r, 256, 3, dilation_rate=1, norm_layer=norm_layer, activation=activation())

    # Up
    u256 = convtranspose_norm_act(r, 256, 3, dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u256 = tf.pad(u256, [[0,0], [0,0], [4,4], [0,0]], 'REFLECT')
    u128 = convtranspose_norm_act(u256, 128, 3, dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u128 = convtranspose_norm_act(u128, 128, 3, dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u64 = convtranspose_norm_act(u128, 64, 3, dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u64 = conv_norm_act(u64, 64, 3, (1,1), dilation_rate=1, norm_layer=norm_layer, activation=activation())

    c7s1_3 = tf.pad(u64, [[0,0], [3,3], [3,3], [0,0]], 'REFLECT')
    c7s1_3 = conv_norm_act(c7s1_3, 3, 7, padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=tf.keras.activations.sigmoid)
    # c7s1_3 = tf.keras.layers.Activation('tanh')(c7s1_3)

    model = tf.keras.models.Model(input, c7s1_3, name='resnet_generator_32x32_320x240')
    return model


def conv_norm_act(input, filters, kernel_size=3, strides=(1,1), padding='same', dilation_rate=1,
                  norm_layer=tf.keras.layers.BatchNormalization, activation=tf.keras.layers.ReLU()):
    dk = tf.keras.layers.Convolution2D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate)(input)
    dk = norm_layer(axis=1)(dk)
    dk = activation(dk)
    return dk

def residual_block(input, filters, kernel_size=3, padding=[[1,1], [1,1]], dilation_rate=1,
                   norm_layer=tf.keras.layers.BatchNormalization, activation=tf.keras.layers.ReLU()):
    r_1 = tf.pad(input, [[0,0], padding[0], padding[1], [0,0]], 'REFLECT')
    r_1 = conv_norm_act(r_1, filters, kernel_size, (1,1), 'valid', dilation_rate, norm_layer, activation)

    r_2 = tf.pad(r_1, [[0,0], padding[0], padding[1], [0,0]], 'REFLECT')
    r_2 = tf.keras.layers.Convolution2D(filters, kernel_size, padding='valid', dilation_rate=dilation_rate)(r_2)
    r_2 = norm_layer(axis=1)(r_2)

    x = tf.keras.layers.Add()([input, r_2])
    x = activation(x) # TODO: Test with removed
    return x

def convtranspose_norm_act(input, filters, kernel_size=3, dilation_rate=1,
                           norm_layer=tf.keras.layers.BatchNormalization, activation=tf.keras.layers.ReLU()):
    uk = tf.keras.layers.Convolution2DTranspose(filters, kernel_size, strides=(2,2), padding='same', dilation_rate=dilation_rate)(input)
    uk = norm_layer(axis=1)(uk)
    uk = activation(uk)
    return uk
