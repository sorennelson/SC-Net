import tensorflow as tf
import tensorflow_addons as tfa

from CycleReconstructorImageNet import *
from CycleReconstructorBlocks import *

def cycle_reconstructor(params):
    filters = params['filters']
    kernels = params['kernels']
    dilation_rate = params['dilation_rate']
    dropout = params['dropout']
    res_depth = params['res_depth']
    input_shape = params['input_shape']
    output_shape = params['output_shape']
    num_channels = params['num_channels']
    channel_format = params['channel_format']
    compression = params['compression']

    if params['norm'] == 'instance': norm_layer = tfa.layers.InstanceNormalization
    elif params['norm'] == 'batch':  norm_layer = tf.keras.layers.BatchNormalization
    elif params['norm'] == 'spectral': norm_layer = tfa.layers.SpectralNormalization

    if params['activation'] == 'relu': activation = tf.keras.layers.ReLU

    if input_shape == (240,320) and output_shape == (32,32):
        return cycle_reconstructor_320x240_32x32(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels)
    elif input_shape == (32,32) and output_shape == (240,320):
        return cycle_reconstructor_32x32_320x240(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels)
    elif input_shape == (180,240) and output_shape == (32,32):
        if compression:
            return cycle_reconstructor_180x240_32x32(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels)
        else:
            return cycle_reconstructor_180x240_32x32_no_compression(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels)
    elif input_shape == (32,32) and output_shape == (180,240):
        if compression:
            return cycle_reconstructor_32x32_180x240(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels)
        else:
            return cycle_reconstructor_32x32_180x240_no_compression(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels)
    elif input_shape == (60,80) and output_shape == (32,32):
        return cycle_reconstructor_60x80_32x32(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels)
    elif input_shape == (32,32) and output_shape == (60,80):
        return cycle_reconstructor_32x32_60x80(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels)
    elif input_shape == (120,160) and output_shape == (32,32):
        return cycle_reconstructor_120x160_32x32(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels)
    elif input_shape == (32,32) and output_shape == (120,160):
        return cycle_reconstructor_32x32_120x160(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels)
    elif input_shape == (224,224) and output_shape == (240,320):
        return cycle_reconstructor_224x224_240x320(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels)
    elif input_shape == (240,320) and output_shape == (224,224):
        return cycle_reconstructor_240x320_224x224(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels)
    elif input_shape == (180,240) and output_shape == (224,224):
        return cycle_reconstructor_180x240_224x224(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels)
    elif input_shape == (224,224) and output_shape == (180,240):
        return cycle_reconstructor_224x224_180x240(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels)
    else:
        return cycle_reconstructor_even(filters, kernels, dilation_rate, dropout, res_depth, norm_layer, activation, channel_format, num_channels, input_shape)


## 32x32 -> 180x240
def cycle_reconstructor_32x32_180x240(filters={'down':[64], 'up':[64,64,64,64,64]}, 
                                      kernels=[3,3], 
                                      dilation_rate=1, 
                                      dropout={'down':[0.2], 'up':[0.2,0.2,0.2,0.2,0.2]},
                                      res_depth={'down':1, 'bottom':0, 'up':1},
                                      norm_layer=tf.keras.layers.BatchNormalization, 
                                      activation=tf.keras.layers.ReLU,
                                      channel_format='NCHW', num_channels=3):
    """
    filters['down'] can have any number of layers >= 1 where the last layer is downsampling and the rest are normal 
    filters['up'] must have 4 layers where the first 3 are upsampling and the last layer is normal
    """
    latents = []
    skips = []
    concat_axis = 1 if channel_format == 'NCHW' else -1
    down_filters, up_filters = filters['down'], filters['up']
    down_dropout, up_dropout = dropout['down'], dropout['up']
    
    if kernels[0] == 9: 
        in_layer_padding = [[7,7], [8,8]]
        out_layer_padding = [[4,4], [4,4]]
    elif kernels[0] == 7: 
        in_layer_padding = [[2,2], [3,3]]
        out_layer_padding = [[3,3], [3,3]]
    elif kernels[0] == 5: 
        if dilation_rate == 1: in_layer_padding = [[1,1], [2,2]]
        elif dilation_rate == 2: in_layer_padding = [[4,4], [4,4]]
        # elif dilation_rate == 2: in_layer_padding = [[3,3], [4,4]]
        out_layer_padding = [[2,2], [2,2]]
    elif kernels[0] == 3: 
        if dilation_rate == 1: in_layer_padding = [[0,0], [1,1]]
        elif dilation_rate == 2: in_layer_padding = [[1,1], [2,2]]
        out_layer_padding = [[1,1], [1,1]]

    residual_padding = dilation_rate
    if kernels[1] == 5: residual_padding += dilation_rate
    if kernels[1] == 5 and dilation_rate == 1:
        bottom_residual_padding = [[residual_padding-1,residual_padding-1], [residual_padding-1,residual_padding-1]]
    else:
        bottom_residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]
    residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

    if channel_format == 'NCHW':
        input = tf.keras.layers.Input((num_channels, 32, 32))
    else:
        input = tf.keras.layers.Input((32, 32, num_channels))
    top = reflect_padding(input, in_layer_padding, channel_format=channel_format)
    top_norm = None if norm_layer == tfa.layers.InstanceNormalization or norm_layer == tfa.layers.SpectralNormalization else norm_layer
    top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=top_norm, activation=activation())
    # for _ in range(res_depth['down']):
    #     top = residual_block(top, 64, kernels[0], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    skips.append(top)

    # Down
    down = top
    for i in range(len(down_filters)-1):
        down = conv_norm_act(down, down_filters[i], kernels[1], norm_layer=norm_layer, activation=activation())
        if i == 1: latents.append(down) #
        down = tf.keras.layers.Dropout(down_dropout[i])(down)
        for _ in range(res_depth['down']):
            down = residual_block(down, down_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
        skips.append(down)
    
    # TODO:
    down = conv_norm_act(down, down_filters[-1], kernels[1], (2,2), norm_layer=norm_layer, activation=activation())
    # latents.append(down) #
    down = tf.keras.layers.Dropout(down_dropout[-1])(down)
    for _ in range(res_depth['down']):
        down = residual_block(down, down_filters[-1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # TODO:
    # down = conv_norm_act(down, down_filters[-1], kernels[1], norm_layer=norm_layer, activation=activation())
    # # latents.append(down) #
    # down = tf.keras.layers.Dropout(down_dropout[-1])(down)

    # Bottom
    r = down
    # latents.append(r)
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[-1], kernels[1], padding=bottom_residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    # if res_depth['bottom'] > 0: latents.append(r) #

    # r = non_local_gaussian(r, down_filters[-1] // 2, channel_format=channel_format)

    # Up
    if kernels[1] == 3: u1 = reflect_padding(r, [[3,3], [4,4]], channel_format=channel_format)
    else: u1 = reflect_padding(r, [[2,3], [4,4]], channel_format=channel_format)
    u1 = convtranspose_norm_act(u1, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    # u1 = skip(True, u1, skips[-1], concat_axis, up_filters[0], kernels[1], dilation_rate, norm_layer=norm_layer, activation=activation())
    # latents.append(u1) #
    # u1 = non_local_gaussian(u1, up_filters[0] // 2, channel_format=channel_format, mode='emb')
    u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
    for _ in range(res_depth['up']):
        u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u2 = reflect_padding(u1, [[1,1], [4,4]], channel_format=channel_format)
    u2 = convtranspose_norm_act(u2, up_filters[1], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    # u2 = skip(True, u2, skips[-2], concat_axis, up_filters[1], kernels[1], dilation_rate, norm_layer=norm_layer, activation=activation())
    u2 = tf.keras.layers.Dropout(up_dropout[1])(u2)
    for _ in range(res_depth['up']):
        u2 = residual_block(u2, up_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u3 = reflect_padding(u2, [[1,1], [4,4]], channel_format=channel_format)
    u3 = convtranspose_norm_act(u3, up_filters[2], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    # u3 = skip(True, u3, skips[-3], concat_axis, up_filters[2], kernels[1], dilation_rate, norm_layer=norm_layer, activation=activation())
    # latents.append(u3) #
    u3 = tf.keras.layers.Dropout(up_dropout[2])(u3)
    for _ in range(res_depth['up']):
        u3 = residual_block(u3, up_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # u4 = conv_norm_act(u3, up_filters[3], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    # u4 = tf.keras.layers.Dropout(up_dropout[3])(u4)
    # for _ in range(res_depth['up']):
    #     u4 = residual_block(u4, up_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Top
    top = reflect_padding(u3, out_layer_padding, channel_format=channel_format)
    # top = skip(True, top, skips[0], concat_axis, 64, kernels[0], dilation_rate, norm_layer=norm_layer, activation=activation())
    ending_act = tf.keras.activations.sigmoid
    # ending_act = tf.keras.layers.Activation('tanh')
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)
    # top = conv_norm(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer)

    # model = tf.keras.models.Model(input, top, name='resnet_generator_32x32_180x240')
    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='resnet_generator_32x32_180x240')
    return model

def cycle_reconstructor_180x240_32x32(filters={'down':[64,64,64,64], 'up':[64]}, 
                                      kernels=[3,3], 
                                      dilation_rate=1, 
                                      dropout={'down':[0.2,0.2,0.2,0.2], 'up':[0.2]},
                                      res_depth={'down':1, 'bottom':0, 'up':1},
                                      norm_layer=tf.keras.layers.BatchNormalization, 
                                      activation=tf.keras.layers.ReLU, 
                                      channel_format='NCHW', num_channels=3):
    """
    filters['down'] must have 4 layers
    filters['up'] can have any number of layers >= 1 where the first layer is upsampling and the rest are normal
    dropout['up'] and dropout['down'] must be the same length as filters['up'] and filters['down']
    """
    latents = []
    skips = []
    concat_axis = 1 if channel_format == 'NCHW' else -1
    down_filters, up_filters = filters['down'], filters['up']
    down_dropout, up_dropout = dropout['down'], dropout['up']
    
    if kernels[0] == 9: top_layer_padding = 4
    elif kernels[0] == 7: top_layer_padding = 3
    elif kernels[0] == 5: top_layer_padding = 2
    elif kernels[0] == 3: top_layer_padding = 1
    top_layer_padding = [[top_layer_padding,top_layer_padding], [top_layer_padding,top_layer_padding]]

    residual_padding = dilation_rate
    if kernels[1] == 5: residual_padding += dilation_rate
    residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

    if channel_format == 'NCHW':
      input = tf.keras.layers.Input((num_channels, 180, 240))
    else:
      input = tf.keras.layers.Input((180, 240, num_channels))
    
    top = reflect_padding(input, top_layer_padding, channel_format=channel_format)
    top_norm = None if norm_layer == tfa.layers.InstanceNormalization or norm_layer == tfa.layers.SpectralNormalization else norm_layer
    top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=top_norm, activation=activation())
    # for _ in range(res_depth['down']):
    #     top = residual_block(top, 64, kernels[0], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    skips.append(top)

    # top = attention(top, 64, channel_format=channel_format) ##

    # Down
    d1 = reflect_padding(top, [[4,4], [4,4]], channel_format=channel_format)
    d1 = conv_norm_act(d1, down_filters[0], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    # d1 = non_local_gaussian(d1, down_filters[0] // 2, channel_format=channel_format, mode='emb')
    d1 = tf.keras.layers.Dropout(down_dropout[0])(d1)
    for _ in range(res_depth['down']):
        d1 = residual_block(d1, down_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    skips.append(d1)

    d2 = reflect_padding(d1, [[4,4], [2,2]], channel_format=channel_format)
    d2 = conv_norm_act(d2, down_filters[1], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    # d2 = non_local_gaussian(d2, down_filters[1] // 2, channel_format=channel_format, mode='emb')
    latents.append(d2)
    d2 = tf.keras.layers.Dropout(down_dropout[1])(d2)
    for _ in range(res_depth['down']):
        d2 = residual_block(d2, down_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    skips.append(d2)

    d3 = reflect_padding(d2, [[4,4], [0,0]], channel_format=channel_format)
    d3 = conv_norm_act(d3, down_filters[2], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    # d3 = non_local_gaussian(d3, down_filters[2] // 2, channel_format=channel_format, mode='emb')
    d3 = tf.keras.layers.Dropout(down_dropout[2])(d3)
    for _ in range(res_depth['down']):
        d3 = residual_block(d3, down_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d4 = reflect_padding(d3, [[1,1], [0,0]], channel_format=channel_format)
    # d4 = reflect_padding(d3, [[1,2], [0,0]], channel_format=channel_format)
    skips.append(d4)
    d4 = conv_norm_act(d4, down_filters[3], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    # latents.append(d4) #
    # d4 = non_local_gaussian(d4, down_filters[3] // 2, channel_format=channel_format, mode='emb')
    d4 = tf.keras.layers.Dropout(down_dropout[3])(d4)
    for _ in range(res_depth['down']):
        d4 = residual_block(d4, down_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # d4 = attention(d4, down_filters[3], channel_format=channel_format) ##
    # d4 = non_local_gaussian(d4, down_filters[3], channel_format=channel_format)

    # TODO:
    # d4 = conv_norm_act(d4, down_filters[-1], kernels[1], norm_layer=norm_layer, activation=activation())
    # # latents.append(d4) #
    # d4 = tf.keras.layers.Dropout(down_dropout[-1])(d4)

    # Bottom
    r = d4
    # latents.append(r)
    for _ in range(res_depth['bottom']):
        # TODO:
        r = residual_block(r, down_filters[-1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    # if res_depth['bottom'] > 0: latents.append(r) #

    # r = non_local_gaussian(r, down_filters[-1] // 2, channel_format=channel_format)
    # r = non_local_gaussian(r, 512, channel_format=channel_format)

    # Up
    u1 = convtranspose_norm_act(r, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    # u1 = resize_conv_norm_act(r, up_filters[0], kernels[1], dilation_rate=1, strides=(2,2), norm_layer=norm_layer, activation=activation())
    # u1 = skip(True, u1, skips[-1], concat_axis, up_filters[0], kernels[1], dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())
    # latents.append(u1) #
    u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
    for _ in range(res_depth['up']):
        # u1 = non_local_gaussian(u1, up_filters[0] // 2, channel_format=channel_format, mode='emb')
        u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # attn = attention(u1, up_filters[0], channel_format=channel_format) ##

    up = u1 # attn #
    for i in range(1, len(up_filters)):
        up = conv_norm_act(up, up_filters[i], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
        # up = skip(True, up, skips[-(i+1)], concat_axis, up_filters[i], kernels[1], dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())
        if i == 1: 
            latents.append(up) #
            # up = non_local_gaussian(up, up_filters[i] // 2, channel_format=channel_format)
        up = tf.keras.layers.Dropout(up_dropout[i])(up)
        for _ in range(res_depth['up']):
            up = residual_block(up, up_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

        # up = attention(up, up_filters[i], channel_format=channel_format) ##


    top = reflect_padding(up, top_layer_padding, channel_format=channel_format)
    # top = skip(True, top, skips[0], concat_axis, 64, kernels[0], dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())
    ending_act = tf.keras.activations.sigmoid
    # ending_act = tf.keras.layers.Activation('tanh')
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)
    # top = conv_norm(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer)

    # model = tf.keras.models.Model(input, top, name='resnet_generator_180x240_32x32')
    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='resnet_generator_180x240_32x32')
    return model


## 320->32
def cycle_reconstructor_32x32_240x320(filters={'down':[64], 'up':[64,64,64,64,64]}, 
                                      kernels=[3,3], 
                                      dilation_rate=1, 
                                      dropout={'down':[0.2], 'up':[0.2,0.2,0.2,0.2,0.2]},
                                      res_depth={'down':1, 'bottom':0, 'up':1},
                                      norm_layer=tf.keras.layers.BatchNormalization, 
                                      activation=tf.keras.layers.ReLU,
                                      channel_format='NCHW', num_channels=3):
    """
    filters['down'] can have any number of layers >= 1 where the last layer is downsampling and the rest are normal 
    filters['up'] must have 4 layers where the first 3 are upsampling and the last layer is normal
    """
    latents = []
    concat_axis = 1 if channel_format == 'NCHW' else -1
    down_filters, up_filters = filters['down'], filters['up']
    down_dropout, up_dropout = dropout['down'], dropout['up']
    
    if kernels[0] == 5: 
        if dilation_rate == 2: in_layer_padding = [[3,3], [6,6]]
        else: raise ValueError('G is only set up for dilation rate 2')
        out_layer_padding = [[2,2], [6,6]]
    else:
        raise ValueError('G is only set up for kernel size [5,5]')

    residual_padding = dilation_rate
    if kernels[1] == 5: residual_padding += dilation_rate
    if kernels[1] == 5 and dilation_rate == 1:
        bottom_residual_padding = [[residual_padding-1,residual_padding-1], [residual_padding-1,residual_padding-1]]
    else:
        bottom_residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]
    residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

    if channel_format == 'NCHW':
        input = tf.keras.layers.Input((num_channels, 32, 32))
    else:
        input = tf.keras.layers.Input((32, 32, num_channels))
    top = reflect_padding(input, in_layer_padding, channel_format=channel_format)
    top_norm = None if norm_layer == tfa.layers.InstanceNormalization or norm_layer == tfa.layers.SpectralNormalization else norm_layer
    top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=top_norm, activation=activation())

    # Down
    down = top
    for i in range(len(down_filters)-1):
        down = conv_norm_act(down, down_filters[i], kernels[1], norm_layer=norm_layer, activation=activation())
        # if i == 1: latents.append(down) #
        down = tf.keras.layers.Dropout(down_dropout[i])(down)
        for _ in range(res_depth['down']):
            down = residual_block(down, down_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    
    down = conv_norm_act(down, down_filters[-1], kernels[1], (2,2), norm_layer=norm_layer, activation=activation())
    down = tf.keras.layers.Dropout(down_dropout[-1])(down)
    for _ in range(res_depth['down']):
        down = residual_block(down, down_filters[-1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)


    # Bottom
    r = down
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[-1], kernels[1], padding=bottom_residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    latents.append(r)

    # Up
    u1 = reflect_padding(r, [[4,4], [6,6]], channel_format=channel_format)
    u1 = convtranspose_norm_act(u1, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
    for _ in range(res_depth['up']):
        u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u2 = reflect_padding(u1, [[5,5], [6,6]], channel_format=channel_format)
    u2 = convtranspose_norm_act(u2, up_filters[1], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u2 = tf.keras.layers.Dropout(up_dropout[1])(u2)
    for _ in range(res_depth['up']):
        u2 = residual_block(u2, up_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u3 = reflect_padding(u2, [[4,4], [6,6]], channel_format=channel_format)
    u3 = convtranspose_norm_act(u3, up_filters[2], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u3 = tf.keras.layers.Dropout(up_dropout[2])(u3)
    for _ in range(res_depth['up']):
        u3 = residual_block(u3, up_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Top
    top = reflect_padding(u3, out_layer_padding, channel_format=channel_format)
    ending_act = tf.keras.activations.sigmoid
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)

    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='cycle_reconstructor_32x32_240x320')
    return model

def cycle_reconstructor_240x320_32x32(filters={'down':[64,64,64,64], 'up':[64]}, 
                                      kernels=[3,3], 
                                      dilation_rate=1, 
                                      dropout={'down':[0.2,0.2,0.2,0.2], 'up':[0.2]},
                                      res_depth={'down':1, 'bottom':0, 'up':1},
                                      norm_layer=tf.keras.layers.BatchNormalization, 
                                      activation=tf.keras.layers.ReLU, 
                                      channel_format='NCHW', num_channels=3):
    """
    filters['down'] must have 4 layers
    filters['up'] can have any number of layers >= 1 where the first layer is upsampling and the rest are normal
    dropout['up'] and dropout['down'] must be the same length as filters['up'] and filters['down']
    """
    latents = []
    concat_axis = 1 if channel_format == 'NCHW' else -1
    down_filters, up_filters = filters['down'], filters['up']
    down_dropout, up_dropout = dropout['down'], dropout['up']
    
    if kernels[0] == 5: top_layer_padding = 0
    else: raise ValueError('G is only set up for kernel size [5,5]')
    top_layer_padding = [[top_layer_padding,top_layer_padding], [top_layer_padding,top_layer_padding]]

    residual_padding = dilation_rate
    if kernels[1] == 5: residual_padding += dilation_rate
    else: raise ValueError('G is only set up for kernel size [5,5]')
    residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

    if channel_format == 'NCHW':
      input = tf.keras.layers.Input((num_channels, 240, 320))
    else:
      input = tf.keras.layers.Input((240, 320, num_channels))
    
    top_norm = None if norm_layer == tfa.layers.InstanceNormalization or norm_layer == tfa.layers.SpectralNormalization else norm_layer
    top = conv_norm_act(input, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=top_norm, activation=activation())
    
    # Down
    d1 = conv_norm_act(top, down_filters[0], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d1 = tf.keras.layers.Dropout(down_dropout[0])(d1)
    for _ in range(res_depth['down']):
        d1 = residual_block(d1, down_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d2 = conv_norm_act(d1, down_filters[1], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d2 = tf.keras.layers.Dropout(down_dropout[1])(d2)
    for _ in range(res_depth['down']):
        d2 = residual_block(d2, down_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d3 = conv_norm_act(d2, down_filters[2], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d3 = tf.keras.layers.Dropout(down_dropout[2])(d3)
    for _ in range(res_depth['down']):
        d3 = residual_block(d3, down_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d4 = reflect_padding(d3, [[3,3], [0,0]], channel_format=channel_format)
    d4 = conv_norm_act(d4, down_filters[3], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d4 = tf.keras.layers.Dropout(down_dropout[3])(d4)
    for _ in range(res_depth['down']):
        d4 = residual_block(d4, down_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Bottom
    r = tf.keras.layers.Cropping2D(((0,0),(1,1)))(d4)
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[-1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    latents.append(r)

    # Up
    u1 = convtranspose_norm_act(r, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
    for _ in range(res_depth['up']):
        u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    up = u1
    for i in range(1, len(up_filters)):
        up = conv_norm_act(up, up_filters[i], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
        up = tf.keras.layers.Dropout(up_dropout[i])(up)
        for _ in range(res_depth['up']):
            up = residual_block(up, up_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    ending_act = tf.keras.activations.sigmoid
    top = conv_norm_act(up, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)

    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='cycle_reconstructor_240x320_32x32')
    return model

# def cycle_reconstructor_320x240_32x32(filters={'down':[64,64,64,64], 'up':[64]}, 
#                                       kernels=[3,3], 
#                                       dilation_rate=1, 
#                                       dropout={'down':[0.2,0.2,0.2,0.2], 'up':[0.2]},
#                                       res_depth={'down':1, 'bottom':0, 'up':1},
#                                       norm_layer=tf.keras.layers.BatchNormalization, 
#                                       activation=tf.keras.layers.ReLU, 
#                                       channel_format='NCHW', num_channels=3):
  
#     down_filters, up_filters = filters['down'], filters['up']
#     down_dropout, up_dropout = dropout['down'], dropout['up']
    
#     if kernels[0] == 7: top_layer_padding = 3
#     elif kernels[0] == 5: top_layer_padding = 2
#     elif kernels[0] == 3: top_layer_padding = 1
#     top_layer_padding = [[top_layer_padding,top_layer_padding], [top_layer_padding,top_layer_padding]]

#     residual_padding = dilation_rate
#     if kernels[1] == 5: residual_padding += dilation_rate
#     residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

#     if channel_format == 'NCHW':
#       input = tf.keras.layers.Input((num_channels, 240, 320))
#     else:
#       input = tf.keras.layers.Input((240, 320, num_channels))
    
#     top = reflect_padding(input, top_layer_padding, channel_format=channel_format)
#     top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())

#     # Down
#     d1 = reflect_padding(top, [[0,0], [4,4]], channel_format=channel_format)
#     d1 = conv_norm_act(d1, down_filters[0], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
#     d1 = tf.keras.layers.Dropout(down_dropout[0])(d1)
#     for _ in range(res_depth['down']):
#         d1 = residual_block(d1, down_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

#     d2 = reflect_padding(d1, [[0,0], [4,4]], channel_format=channel_format)
#     d2 = conv_norm_act(d2, down_filters[1], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
#     d2 = tf.keras.layers.Dropout(down_dropout[1])(d2)
#     for _ in range(res_depth['down']):
#         d2 = residual_block(d2, down_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

#     d3 = reflect_padding(d2, [[0,0], [4,4]], channel_format=channel_format)
#     d3 = conv_norm_act(d3, down_filters[2], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
#     d3 = tf.keras.layers.Dropout(down_dropout[2])(d3)
#     for _ in range(res_depth['down']):
#         d3 = residual_block(d3, down_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

#     d4 = reflect_padding(d3, [[1,1], [4,4]], channel_format=channel_format)
#     d4 = conv_norm_act(d4, down_filters[3], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
#     d4 = tf.keras.layers.Dropout(down_dropout[3])(d4)
#     for _ in range(res_depth['down']):
#         d4 = residual_block(d4, down_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

#     d5 = conv_norm_act(d4, down_filters[4], kernels[1], (1,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
#     d5 = tf.keras.layers.Dropout(down_dropout[4])(d5)
#     for _ in range(res_depth['down']):
#         d5 = residual_block(d5, down_filters[4], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)


#     # Bottom
#     r = d5
#     for _ in range(res_depth['bottom']):
#         r = residual_block(r, down_filters[-1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)    


#     # Up
#     u1 = reflect_padding(r, [[0,0], [1,1]], channel_format=channel_format)
#     u1 = convtranspose_norm_act(u1, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
#     u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
#     for _ in range(res_depth['up']):
#         u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

#     up = u1
#     for i in range(1, len(up_filters)):
#         up = conv_norm_act(up, up_filters[i], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
#         up = tf.keras.layers.Dropout(up_dropout[i])(up)
#         for _ in range(res_depth['up']):
#             up = residual_block(up, up_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)


#     top = reflect_padding(up, top_layer_padding, channel_format=channel_format)
#     ending_act = tf.keras.activations.sigmoid
#     # ending_act = tf.keras.layers.Activation('tanh')
#     top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)
#     # top = conv_norm(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer)

#     model = tf.keras.models.Model(input, top, name='cycle_reconstructor_320x240_32x32')
#     return model

# def cycle_reconstructor_32x32_320x240(filters={'down':[64], 'up':[64,64,64,64,64]}, 
#                                       kernels=[3,3], 
#                                       dilation_rate=1, 
#                                       dropout={'down':[0.2], 'up':[0.2,0.2,0.2,0.2,0.2]},
#                                       res_depth={'down':1, 'bottom':0, 'up':1},
#                                       norm_layer=tf.keras.layers.BatchNormalization, 
#                                       activation=tf.keras.layers.ReLU,
#                                       channel_format='NCHW', num_channels=3):
#     """
#     filters['down'] can have any number of layers >= 1 where the last layer is downsampling and the rest are normal 
#     filters['up'] must have 4 layers where the first 3 are upsampling and the last layer is normal
#     """
#     down_filters, up_filters = filters['down'], filters['up']
#     down_dropout, up_dropout = dropout['down'], dropout['up']
    
#     if kernels[0] == 7: 
#         in_layer_padding = [[2,2], [3,3]]
#         out_layer_padding = [[3,3], [3,3]]
#     elif kernels[0] == 5: 
#         if dilation_rate == 1: in_layer_padding = [[1,1], [2,2]]
#         elif dilation_rate == 2: in_layer_padding = [[3,3], [4,4]]
#         out_layer_padding = [[2,2], [2,2]]
#     elif kernels[0] == 3: 
#         if dilation_rate == 1: in_layer_padding = [[0,0], [1,1]]
#         elif dilation_rate == 2: in_layer_padding = [[1,1], [2,2]]
#         out_layer_padding = [[1,1], [1,1]]

#     residual_padding = dilation_rate
#     if kernels[1] == 5: residual_padding += dilation_rate
#     if kernels[1] == 5 and dilation_rate == 1:
#         bottom_residual_padding = [[residual_padding-1,residual_padding-1], [residual_padding-1,residual_padding-1]]
#     else:
#         bottom_residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]
#     residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

#     if channel_format == 'NCHW':
#       input = tf.keras.layers.Input((num_channels, 32, 32))
#     else:
#       input = tf.keras.layers.Input((32, 32, num_channels))
#     top = reflect_padding(input, in_layer_padding, channel_format=channel_format)
#     top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())

#     # Down
#     down = top
#     for i in range(len(down_filters)-1):
#       down = conv_norm_act(down, down_filters[i], kernels[1], norm_layer=norm_layer, activation=activation())
#       down = tf.keras.layers.Dropout(down_dropout[i])(down)
#       for _ in range(res_depth['down']):
#           down = residual_block(down, down_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    
#     down = conv_norm_act(down, down_filters[-1], kernels[1], (2,2), norm_layer=norm_layer, activation=activation())
#     down = tf.keras.layers.Dropout(down_dropout[-1])(down)
#     for _ in range(res_depth['down']):
#         down = residual_block(down, down_filters[-1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)


#     # Bottom
#     r = down
#     for _ in range(res_depth['bottom']):
#         r = residual_block(r, down_filters[-1], kernels[1], padding=bottom_residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

#     # Up
#     u1 = convtranspose_norm_act(r, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
#     u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
#     for _ in range(res_depth['up']):
#         u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

#     u2 = reflect_padding(u1, [[0,0], [4,4]], channel_format=channel_format)
#     u2 = convtranspose_norm_act(u2, up_filters[1], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
#     u2 = tf.keras.layers.Dropout(up_dropout[1])(u2)
#     for _ in range(res_depth['up']):
#         u2 = residual_block(u2, up_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

#     u3 = convtranspose_norm_act(u2, up_filters[2], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
#     u3 = tf.keras.layers.Dropout(up_dropout[2])(u3)
#     for _ in range(res_depth['up']):
#         u3 = residual_block(u3, up_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

#     u4 = convtranspose_norm_act(u3, up_filters[3], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
#     u4 = tf.keras.layers.Dropout(up_dropout[3])(u4)
#     for _ in range(res_depth['up']):
#         u4 = residual_block(u4, up_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

#     u5 = conv_norm_act(u4, up_filters[4], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
#     u5 = tf.keras.layers.Dropout(up_dropout[4])(u5)
#     for _ in range(res_depth['up']):
#         u5 = residual_block(u5, up_filters[4], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

#     # Top
#     top = reflect_padding(u4, out_layer_padding, channel_format=channel_format)
#     ending_act = tf.keras.activations.sigmoid
#     # ending_act = tf.keras.layers.Activation('tanh')
#     top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)
#     # top = conv_norm(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer)

#     model = tf.keras.models.Model(input, top, name='cycle_reconstructor_32x32_320x240')
#     return model


## 160->32
def cycle_reconstructor_120x160_32x32(filters={'down':[64,64,64,64], 'up':[64]}, 
                                      kernels=[3,3], 
                                      dilation_rate=1, 
                                      dropout={'down':[0.2,0.2,0.2,0.2], 'up':[0.2]},
                                      res_depth={'down':1, 'bottom':0, 'up':1},
                                      norm_layer=tf.keras.layers.BatchNormalization, 
                                      activation=tf.keras.layers.ReLU, 
                                      channel_format='NCHW', num_channels=3):
    """
    filters['down'] must have 4 layers
    filters['up'] can have any number of layers >= 1 where the first layer is upsampling and the rest are normal
    dropout['up'] and dropout['down'] must be the same length as filters['up'] and filters['down']
    """
    latents = []
    concat_axis = 1 if channel_format == 'NCHW' else -1
    down_filters, up_filters = filters['down'], filters['up']
    down_dropout, up_dropout = dropout['down'], dropout['up']
    
    if kernels[0] == 5: top_layer_padding = 2
    else: raise ValueError('G is only set up for kernel size [5,5]')
    top_layer_padding = [[top_layer_padding,top_layer_padding], [top_layer_padding,top_layer_padding]]

    residual_padding = dilation_rate
    if kernels[1] == 5: residual_padding += dilation_rate
    else: raise ValueError('G is only set up for kernel size [5,5]')
    residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

    if channel_format == 'NCHW':
      input = tf.keras.layers.Input((num_channels, 120, 160))
    else:
      input = tf.keras.layers.Input((160, 120, num_channels))
    
    top = reflect_padding(input, top_layer_padding, channel_format=channel_format)
    top_norm = None if norm_layer == tfa.layers.InstanceNormalization or norm_layer == tfa.layers.SpectralNormalization else norm_layer
    top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=top_norm, activation=activation())
    
    # Down
    d1 = reflect_padding(top, [[4,4], [4,4]], channel_format=channel_format)
    d1 = conv_norm_act(d1, down_filters[0], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d1 = tf.keras.layers.Dropout(down_dropout[0])(d1)
    for _ in range(res_depth['down']):
        d1 = residual_block(d1, down_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d2 = reflect_padding(d1, [[4,4], [2,2]], channel_format=channel_format)
    d2 = conv_norm_act(d2, down_filters[1], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(d2)
    d2 = tf.keras.layers.Dropout(down_dropout[1])(d2)
    for _ in range(res_depth['down']):
        d2 = residual_block(d2, down_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d3 = reflect_padding(d2, [[4,4], [0,0]], channel_format=channel_format)
    d3 = conv_norm_act(d3, down_filters[2], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d3 = tf.keras.layers.Dropout(down_dropout[2])(d3)
    for _ in range(res_depth['down']):
        d3 = residual_block(d3, down_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d4 = reflect_padding(d3, [[5,5], [5,5]], channel_format=channel_format)
    d4 = conv_norm_act(d4, down_filters[3], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d4 = tf.keras.layers.Dropout(down_dropout[3])(d4)
    for _ in range(res_depth['down']):
        d4 = residual_block(d4, down_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Bottom
    r = d4
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[-1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Up
    u1 = convtranspose_norm_act(r, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
    for _ in range(res_depth['up']):
        u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    up = u1
    for i in range(1, len(up_filters)):
        up = conv_norm_act(up, up_filters[i], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
        up = tf.keras.layers.Dropout(up_dropout[i])(up)
        for _ in range(res_depth['up']):
            up = residual_block(up, up_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)


    top = reflect_padding(up, top_layer_padding, channel_format=channel_format)
    ending_act = tf.keras.activations.sigmoid
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)

    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='cycle_reconstructor_120x160_32x32')
    return model

def cycle_reconstructor_32x32_120x160(filters={'down':[64], 'up':[64,64,64,64,64]}, 
                                      kernels=[3,3], 
                                      dilation_rate=1, 
                                      dropout={'down':[0.2], 'up':[0.2,0.2,0.2,0.2,0.2]},
                                      res_depth={'down':1, 'bottom':0, 'up':1},
                                      norm_layer=tf.keras.layers.BatchNormalization, 
                                      activation=tf.keras.layers.ReLU,
                                      channel_format='NCHW', num_channels=3):
    """
    filters['down'] can have any number of layers >= 1 where the last layer is downsampling and the rest are normal 
    filters['up'] must have 4 layers where the first 3 are upsampling and the last layer is normal
    """
    latents = []
    concat_axis = 1 if channel_format == 'NCHW' else -1
    down_filters, up_filters = filters['down'], filters['up']
    down_dropout, up_dropout = dropout['down'], dropout['up']
    
    if kernels[0] == 5: 
        if dilation_rate == 2: in_layer_padding = [[3,3], [4,4]]
        else: raise ValueError('G is only set up for dilation rate 2')
        out_layer_padding = [[0,0], [2,2]]
    else:
        raise ValueError('G is only set up for kernel size [5,5]')

    residual_padding = dilation_rate
    if kernels[1] == 5: residual_padding += dilation_rate
    if kernels[1] == 5 and dilation_rate == 1:
        bottom_residual_padding = [[residual_padding-1,residual_padding-1], [residual_padding-1,residual_padding-1]]
    else:
        bottom_residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]
    residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

    if channel_format == 'NCHW':
        input = tf.keras.layers.Input((num_channels, 32, 32))
    else:
        input = tf.keras.layers.Input((32, 32, num_channels))
    top = reflect_padding(input, in_layer_padding, channel_format=channel_format)
    top_norm = None if norm_layer == tfa.layers.InstanceNormalization or norm_layer == tfa.layers.SpectralNormalization else norm_layer
    top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=top_norm, activation=activation())

    # Down
    down = top
    for i in range(len(down_filters)-1):
        down = conv_norm_act(down, down_filters[i], kernels[1], norm_layer=norm_layer, activation=activation())
        if i == 1: latents.append(down) #
        down = tf.keras.layers.Dropout(down_dropout[i])(down)
        for _ in range(res_depth['down']):
            down = residual_block(down, down_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    
    down = conv_norm_act(down, down_filters[-1], kernels[1], (2,2), norm_layer=norm_layer, activation=activation())
    down = tf.keras.layers.Dropout(down_dropout[-1])(down)
    for _ in range(res_depth['down']):
        down = residual_block(down, down_filters[-1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)


    # Bottom
    r = down
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[-1], kernels[1], padding=bottom_residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)


    # Up
    u1 = convtranspose_norm_act(r, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
    for _ in range(res_depth['up']):
        u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u2 = reflect_padding(u1, [[0,0], [2,2]], channel_format=channel_format)
    u2 = convtranspose_norm_act(u2, up_filters[1], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u2 = tf.keras.layers.Dropout(up_dropout[1])(u2)
    for _ in range(res_depth['up']):
        u2 = residual_block(u2, up_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u3 = reflect_padding(u2, [[1,1], [4,4]], channel_format=channel_format)
    u3 = convtranspose_norm_act(u3, up_filters[2], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u3 = tf.keras.layers.Dropout(up_dropout[2])(u3)
    for _ in range(res_depth['up']):
        u3 = residual_block(u3, up_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Top
    top = reflect_padding(u3, out_layer_padding, channel_format=channel_format)
    ending_act = tf.keras.activations.sigmoid
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)

    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='cycle_reconstructor_32x32_120x160')
    return model


## 80->32
def cycle_reconstructor_60x80_32x32(filters={'down':[64,64,64,64], 'up':[64]}, 
                                      kernels=[3,3], 
                                      dilation_rate=1, 
                                      dropout={'down':[0.2,0.2,0.2,0.2], 'up':[0.2]},
                                      res_depth={'down':1, 'bottom':0, 'up':1},
                                      norm_layer=tf.keras.layers.BatchNormalization, 
                                      activation=tf.keras.layers.ReLU, 
                                      channel_format='NCHW', num_channels=3):
    """
    filters['down'] must have 4 layers
    filters['up'] can have any number of layers >= 1 where the first layer is upsampling and the rest are normal
    dropout['up'] and dropout['down'] must be the same length as filters['up'] and filters['down']
    """
    latents = []
    down_filters, up_filters = filters['down'], filters['up']
    down_dropout, up_dropout = dropout['down'], dropout['up']
    
    if kernels[0] == 5: top_layer_padding = 5
    else: raise ValueError('G is only set up for kernel size [5,5]')
    top_layer_padding = [[top_layer_padding,top_layer_padding], [top_layer_padding,top_layer_padding]]

    residual_padding = dilation_rate
    if kernels[1] == 5: residual_padding += dilation_rate
    else: raise ValueError('G is only set up for kernel size [5,5]')
    residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

    if channel_format == 'NCHW':
      input = tf.keras.layers.Input((num_channels, 60, 80))
    else:
      input = tf.keras.layers.Input((60, 80, num_channels))
    
    top = reflect_padding(input, top_layer_padding, channel_format=channel_format)
    top_norm = None if norm_layer == tfa.layers.InstanceNormalization or norm_layer == tfa.layers.SpectralNormalization else norm_layer
    top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=top_norm, activation=activation())
    
    # Down
    d1 = reflect_padding(top, [[4,4], [4,4]], channel_format=channel_format)
    d1 = conv_norm_act(d1, down_filters[0], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d1 = tf.keras.layers.Dropout(down_dropout[0])(d1)
    for _ in range(res_depth['down']):
        d1 = residual_block(d1, down_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d2 = reflect_padding(d1, [[4,4], [2,2]], channel_format=channel_format)
    d2 = conv_norm_act(d2, down_filters[1], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(d2)
    d2 = tf.keras.layers.Dropout(down_dropout[1])(d2)
    for _ in range(res_depth['down']):
        d2 = residual_block(d2, down_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d3 = reflect_padding(d2, [[4,4], [2,2]], channel_format=channel_format)
    d3 = conv_norm_act(d3, down_filters[2], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d3 = tf.keras.layers.Dropout(down_dropout[2])(d3)
    for _ in range(res_depth['down']):
        d3 = residual_block(d3, down_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d4 = reflect_padding(d3, [[5,5], [5,5]], channel_format=channel_format)
    d4 = conv_norm_act(d4, down_filters[3], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d4 = tf.keras.layers.Dropout(down_dropout[3])(d4)
    for _ in range(res_depth['down']):
        d4 = residual_block(d4, down_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Bottom
    r = d4
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[-1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Up
    u1 = convtranspose_norm_act(r, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
    for _ in range(res_depth['up']):
        u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    up = u1
    for i in range(1, len(up_filters)):
        up = conv_norm_act(up, up_filters[i], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
        up = tf.keras.layers.Dropout(up_dropout[i])(up)
        for _ in range(res_depth['up']):
            up = residual_block(up, up_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)


    top = reflect_padding(up, top_layer_padding, channel_format=channel_format)
    ending_act = tf.keras.activations.sigmoid
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)

    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='cycle_reconstructor_60x80_32x32')
    return model

def cycle_reconstructor_32x32_60x80(filters={'down':[64], 'up':[64,64,64,64,64]}, 
                                      kernels=[3,3], 
                                      dilation_rate=1, 
                                      dropout={'down':[0.2], 'up':[0.2,0.2,0.2,0.2,0.2]},
                                      res_depth={'down':1, 'bottom':0, 'up':1},
                                      norm_layer=tf.keras.layers.BatchNormalization, 
                                      activation=tf.keras.layers.ReLU,
                                      channel_format='NCHW', num_channels=3):
    """
    filters['down'] can have any number of layers >= 1 where the last layer is downsampling and the rest are normal 
    filters['up'] must have 4 layers where the first 3 are upsampling and the last layer is normal
    """
    latents = []
    concat_axis = 1 if channel_format == 'NCHW' else -1
    down_filters, up_filters = filters['down'], filters['up']
    down_dropout, up_dropout = dropout['down'], dropout['up']
    
    if kernels[0] == 5: 
        if dilation_rate == 2: in_layer_padding = [[3,3], [4,4]]
        else: raise ValueError('G is only set up for dilation rate 2')
        out_layer_padding = [[0,0], [2,2]]
    else:
        raise ValueError('G is only set up for kernel size [5,5]')

    residual_padding = dilation_rate
    if kernels[1] == 5: residual_padding += dilation_rate
    if kernels[1] == 5 and dilation_rate == 1:
        bottom_residual_padding = [[residual_padding-1,residual_padding-1], [residual_padding-1,residual_padding-1]]
    else:
        bottom_residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]
    residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

    if channel_format == 'NCHW':
        input = tf.keras.layers.Input((num_channels, 32, 32))
    else:
        input = tf.keras.layers.Input((32, 32, num_channels))
    top = reflect_padding(input, in_layer_padding, channel_format=channel_format)
    top_norm = None if norm_layer == tfa.layers.InstanceNormalization or norm_layer == tfa.layers.SpectralNormalization else norm_layer
    top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=top_norm, activation=activation())

    # Down
    down = top
    for i in range(len(down_filters)-1):
        down = conv_norm_act(down, down_filters[i], kernels[1], norm_layer=norm_layer, activation=activation())
        if i == 1: latents.append(down) #
        down = tf.keras.layers.Dropout(down_dropout[i])(down)
        for _ in range(res_depth['down']):
            down = residual_block(down, down_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    
    down = conv_norm_act(down, down_filters[-1], kernels[1], (2,2), norm_layer=norm_layer, activation=activation())
    down = tf.keras.layers.Dropout(down_dropout[-1])(down)
    for _ in range(res_depth['down']):
        down = residual_block(down, down_filters[-1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)


    # Bottom
    r = down
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[-1], kernels[1], padding=bottom_residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)


    # Up
    u1 = convtranspose_norm_act(r, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
    for _ in range(res_depth['up']):
        u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u2 = reflect_padding(u1, [[1,1], [4,4]], channel_format=channel_format)
    u2 = convtranspose_norm_act(u2, up_filters[1], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u2 = tf.keras.layers.Dropout(up_dropout[1])(u2)
    for _ in range(res_depth['up']):
        u2 = residual_block(u2, up_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u3 = conv_norm_act(u2, up_filters[2], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u3 = tf.keras.layers.Dropout(up_dropout[2])(u3)
    for _ in range(res_depth['up']):
        u3 = residual_block(u3, up_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Top
    top = reflect_padding(u3, out_layer_padding, channel_format=channel_format)
    ending_act = tf.keras.activations.sigmoid
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)

    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='cycle_reconstructor_32x32_60x80')
    return model


## No compression
def cycle_reconstructor_32x32_180x240_no_compression(filters={'down':[64], 'up':[64,64,64,64,64]}, 
                                                    kernels=[3,3], 
                                                    dilation_rate=1, 
                                                    dropout={'down':[0.2], 'up':[0.2,0.2,0.2,0.2,0.2]},
                                                    res_depth={'down':1, 'bottom':0, 'up':1},
                                                    norm_layer=tf.keras.layers.BatchNormalization, 
                                                    activation=tf.keras.layers.ReLU,
                                                    channel_format='NCHW', num_channels=3):
    """
    filters['down'] can have any number of layers >= 1 where the last layer is downsampling and the rest are normal 
    filters['up'] must have 4 layers where the first 3 are upsampling and the last layer is normal
    """
    latents = []
    down_filters, up_filters = filters['down'], filters['up']
    down_dropout, up_dropout = dropout['down'], dropout['up']
    
    if kernels[0] == 7: 
        in_layer_padding = [[2,2], [3,3]]
        out_layer_padding = [[3,3], [3,3]]
    elif kernels[0] == 5: 
        if dilation_rate == 1: in_layer_padding = [[1,1], [2,2]]
        elif dilation_rate == 2: in_layer_padding = [[4,4], [4,4]]
        # elif dilation_rate == 2: in_layer_padding = [[3,3], [4,4]]
        out_layer_padding = [[2,2], [2,2]]
    elif kernels[0] == 3: 
        if dilation_rate == 1: in_layer_padding = [[0,0], [1,1]]
        elif dilation_rate == 2: in_layer_padding = [[1,1], [2,2]]
        out_layer_padding = [[1,1], [1,1]]

    residual_padding = dilation_rate
    if kernels[1] == 5: residual_padding += dilation_rate
    if kernels[1] == 5 and dilation_rate == 1:
        bottom_residual_padding = [[residual_padding-1,residual_padding-1], [residual_padding-1,residual_padding-1]]
    else:
        bottom_residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]
    residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

    if channel_format == 'NCHW':
        input = tf.keras.layers.Input((num_channels, 32, 32))
    else:
        input = tf.keras.layers.Input((32, 32, num_channels))
    top = reflect_padding(input, in_layer_padding, channel_format=channel_format)
    top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())

    # Down
    down = top
    for i in range(len(down_filters)):
        down = conv_norm_act(down, down_filters[i], kernels[1], norm_layer=norm_layer, activation=activation())
        if i == len(down_filters) - 1: latents.append(down) #
        down = tf.keras.layers.Dropout(down_dropout[i])(down)
        for _ in range(res_depth['down']):
            down = residual_block(down, down_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Bottom
    r = down
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[-1], kernels[1], padding=bottom_residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    if res_depth['bottom'] > 0: latents.append(r)

    # Up
    u1 = reflect_padding(r, [[4,4], [6,6]], channel_format=channel_format)
    # u1 = reflect_padding(r, [[3,3], [4,4]], channel_format=channel_format)
    u1 = conv_norm_act(u1, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
    for _ in range(res_depth['up']):
        u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u2 = reflect_padding(u1, [[2,2], [6,6]], channel_format=channel_format)
    u2 = convtranspose_norm_act(u2, up_filters[1], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u2 = tf.keras.layers.Dropout(up_dropout[1])(u2)
    for _ in range(res_depth['up']):
        u2 = residual_block(u2, up_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u3 = reflect_padding(u2, [[1,1], [4,4]], channel_format=channel_format)
    u3 = convtranspose_norm_act(u3, up_filters[2], kernels[1], dilation_rate=1, strides=(2,2), norm_layer=norm_layer, activation=activation())
    u3 = tf.keras.layers.Dropout(up_dropout[2])(u3)
    for _ in range(res_depth['up']):
        u3 = residual_block(u3, up_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Top
    top = reflect_padding(u3, out_layer_padding, channel_format=channel_format)
    ending_act = tf.keras.activations.sigmoid
    # ending_act = tf.keras.layers.Activation('tanh')
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)
    # top = conv_norm(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer)

    # model = tf.keras.models.Model(input, top, name='resnet_generator_32x32_180x240')
    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='cycle_reconstructor_32x32_180x240_no_compression')
    return model

def cycle_reconstructor_180x240_32x32_no_compression(filters={'down':[64,64,64,64], 'up':[64]}, 
                                                      kernels=[3,3], 
                                                      dilation_rate=1, 
                                                      dropout={'down':[0.2,0.2,0.2,0.2], 'up':[0.2]},
                                                      res_depth={'down':1, 'bottom':0, 'up':1},
                                                      norm_layer=tf.keras.layers.BatchNormalization, 
                                                      activation=tf.keras.layers.ReLU, 
                                                      channel_format='NCHW', num_channels=3):
    """
    filters['down'] must have 4 layers
    filters['up'] can have any number of layers >= 1 where the first layer is upsampling and the rest are normal
    dropout['up'] and dropout['down'] must be the same length as filters['up'] and filters['down']
    """
    latents = []
    down_filters, up_filters = filters['down'], filters['up']
    down_dropout, up_dropout = dropout['down'], dropout['up']
    
    if kernels[0] == 7: top_layer_padding = 3
    elif kernels[0] == 5: top_layer_padding = 2
    elif kernels[0] == 3: top_layer_padding = 1
    top_layer_padding = [[top_layer_padding,top_layer_padding], [top_layer_padding,top_layer_padding]]

    residual_padding = dilation_rate
    if kernels[1] == 5: residual_padding += dilation_rate
    residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

    if channel_format == 'NCHW':
      input = tf.keras.layers.Input((num_channels, 180, 240))
    else:
      input = tf.keras.layers.Input((180, 240, num_channels))
    
    top = reflect_padding(input, top_layer_padding, channel_format=channel_format)
    top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())

    # Down
    d1 = reflect_padding(top, [[4,4], [4,4]], channel_format=channel_format)
    d1 = conv_norm_act(d1, down_filters[0], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d1 = tf.keras.layers.Dropout(down_dropout[0])(d1)
    for _ in range(res_depth['down']):
        d1 = residual_block(d1, down_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d2 = reflect_padding(d1, [[4,4], [2,2]], channel_format=channel_format)
    d2 = conv_norm_act(d2, down_filters[1], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d2 = tf.keras.layers.Dropout(down_dropout[1])(d2)
    for _ in range(res_depth['down']):
        d2 = residual_block(d2, down_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d3 = reflect_padding(d2, [[4,4], [0,0]], channel_format=channel_format)
    d3 = conv_norm_act(d3, down_filters[2], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    d3 = tf.keras.layers.Dropout(down_dropout[2])(d3)
    for _ in range(res_depth['down']):
        d3 = residual_block(d3, down_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d4 = reflect_padding(d3, [[1,2], [0,0]], channel_format=channel_format)
    d4 = conv_norm_act(d4, down_filters[3], kernels[1], (1,1), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(d4) #
    d4 = tf.keras.layers.Dropout(down_dropout[3])(d4)
    for _ in range(res_depth['down']):
        d4 = residual_block(d4, down_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Bottom
    r = d4
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    if res_depth['bottom'] > 0: latents.append(r) #

    # Up
    up = r # attn #
    for i in range(len(up_filters)):
        up = conv_norm_act(up, up_filters[i], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
        # if i == 1: latents.append(up) #
        up = tf.keras.layers.Dropout(up_dropout[i])(up)
        for _ in range(res_depth['up']):
            up = residual_block(up, up_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
        # latents.append(up) #


    top = reflect_padding(up, top_layer_padding, channel_format=channel_format)
    ending_act = tf.keras.activations.sigmoid
    # ending_act = tf.keras.layers.Activation('tanh')
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)
    # top = conv_norm(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer)

    # model = tf.keras.models.Model(input, top, name='resnet_generator_180x240_32x32')
    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='cycle_reconstructor_180x240_32x32_no_compression')
    return model


## 128->128
def cycle_reconstructor_even(filters={'down':[64,64,64,64], 'up':[64]}, 
                                          kernels=[3,3], 
                                          dilation_rate=1, 
                                          dropout={'down':[0.2,0.2,0.2,0.2], 'up':[0.2]},
                                          res_depth={'down':1, 'bottom':0, 'up':1},
                                          norm_layer=tf.keras.layers.BatchNormalization, 
                                          activation=tf.keras.layers.ReLU, 
                                          channel_format='NCHW', num_channels=3, input_shape=(128,128)):
    """
    filters['down'] must have 4 layers
    filters['up'] can have any number of layers >= 1 where the first layer is upsampling and the rest are normal
    dropout['up'] and dropout['down'] must be the same length as filters['up'] and filters['down']
    """
    latents = []
    skips = []
    concat_axis = 1 if channel_format == 'NCHW' else -1
    down_filters, up_filters = filters['down'], filters['up']
    down_dropout, up_dropout = dropout['down'], dropout['up']
    
    if kernels[0] == 7: top_layer_padding = 3
    elif kernels[0] == 5: top_layer_padding = 2
    elif kernels[0] == 3: top_layer_padding = 1
    top_layer_padding = [[top_layer_padding,top_layer_padding], [top_layer_padding,top_layer_padding]]

    residual_padding = dilation_rate
    if kernels[1] == 5: residual_padding += dilation_rate
    residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

    if channel_format == 'NCHW':
      input = tf.keras.layers.Input((num_channels, input_shape[0], input_shape[1]))
    else:
      input = tf.keras.layers.Input((input_shape[0], input_shape[1], num_channels))
    
    top = reflect_padding(input, top_layer_padding, channel_format=channel_format)
    # top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())
    top_norm = None if norm_layer == tfa.layers.InstanceNormalization else norm_layer
    top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=top_norm, activation=activation())

    # Down
    d1 = conv_norm_act(top, down_filters[0], kernels[1], (1,1), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    # d1 = tf.keras.layers.Dropout(down_dropout[0])(d1)
    for _ in range(res_depth['down']):
        d1 = residual_block(d1, down_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    latents.append(d1)
    skips.append(d1)

    down = d1
    for i in range(1, len(down_filters)):
        down = conv_norm_act(down, down_filters[i], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
        # down = tf.keras.layers.Dropout(down_dropout[i])(down)
        for _ in range(res_depth['down']):
            down = residual_block(down, down_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
        if i != len(down_filters) - 1:
            skips.append(down)

    # Bottom
    r = down
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    # latents.append(r)

    # Up
    up = r
    for i in range(len(up_filters)):
        if i == len(up_filters) - 1: 
            if input_shape == (338,338):
                up = reflect_padding(up, [[1,0], [1,0]], channel_format=channel_format)
            elif input_shape == (468,468):
                up = reflect_padding(up, [[1,1], [1,1]], channel_format=channel_format)
        up = convtranspose_norm_act(up, up_filters[i], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
        up = skip(False, up, skips[-(i+1)], concat_axis, up_filters[i], kernels[1], dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())
        # up = tf.keras.layers.Dropout(up_dropout[i])(up)
        for _ in range(res_depth['up']):
            up = residual_block(up, up_filters[i], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    ## 32 ADDED
    up = conv_norm_act(up, 64, kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())

    top = reflect_padding(up, top_layer_padding, channel_format=channel_format)
    ending_act = tf.keras.activations.sigmoid
    # ending_act = tf.keras.layers.Activation('tanh')
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)
    # top = conv_norm(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer)

    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='cycle_reconstructor_even')
    return model


## Attention / Non-local
class GammaLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GammaLayer, self).__init__() #name='Gamma')

    def build(self, input_shape):
        self.gamma = self.add_weight("gamma", [1], initializer='zeros', trainable=True)
        super(GammaLayer, self).build(input_shape)
                
    def call(self, o):
        return self.gamma * o

    def compute_output_shape(self, input_shape):
        return input_shape

def attention(input, filters, channel_format):
  f = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')(input)
  f = tf.keras.layers.MaxPooling2D()(f)  # GB
  g = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')(input)
  h = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')(input)
  # h = tf.keras.layers.Conv2D(filters // 2, kernel_size=1, padding='same')(input)  # GB
  h = tf.keras.layers.MaxPooling2D()(h)  # GB

  # N = h * w
  channel = 1 if channel_format == 'NCHW' else -1
  flatten_g = tf.keras.layers.Reshape([-1, g.shape[channel]])(g)
  flatten_f = tf.keras.layers.Reshape([-1, f.shape[channel]])(f)

  s = tf.matmul(flatten_g, flatten_f, transpose_b=True) # # [bs, N, N]
  beta = tf.nn.softmax(s)  # attention map

  flatten_h = tf.keras.layers.Reshape([-1, h.shape[channel]])(h)
  o = tf.matmul(beta, flatten_h) # [bs, N, C]

  if channel == 1:
      # reshape = [input.shape[1] // 2, input.shape[2], input.shape[3]] # GB
      reshape = [input.shape[1], input.shape[2], input.shape[3]]
  else:
      reshape = [input.shape[1], input.shape[2], input.shape[3] // 2]  # GB
  o = tf.keras.layers.Reshape(reshape)(o)
  o = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')(o)  # GB
  
  gamma = GammaLayer()(o)
  x = gamma + input
  print(gamma.name)
  
  return x

def non_local_gaussian(input, filters, channel_format, mode='emb'):
    channel = 1 if channel_format == 'NCHW' else -1
    num_input_channels = input.shape[channel]

    # Gaussian
    if mode == 'gaussian':
        theta = input
        phi = input
        if channel == 1:
            theta = tf.keras.layers.Permute([2, 3, 1])(theta)
            phi = tf.keras.layers.Permute([2, 3, 1])(phi)

        theta = tf.keras.layers.Reshape([-1, filters])(theta) # [BS, HW, C]
        phi = tf.keras.layers.Reshape([-1, filters])(phi) # [BS, HW, C]

        # Attention map
        s = tf.matmul(theta, phi, transpose_b=True) # [BS, HW, HW]
        s = tf.nn.softmax(s)

    # Embedded
    elif mode == 'emb':
        theta = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')(input)
        phi = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')(input)
        phi = tf.keras.layers.MaxPooling2D()(phi)  # GB

        if channel == 1:
            theta = tf.keras.layers.Permute([2, 3, 1])(theta)
            phi = tf.keras.layers.Permute([2, 3, 1])(phi)

        theta = tf.keras.layers.Reshape([-1, filters])(theta) # [BS, HW, C]
        phi = tf.keras.layers.Reshape([-1, filters])(phi) # [BS, HW, C]

        # Attention map
        s = tf.matmul(theta, phi, transpose_b=True) # [BS, HW, HW]
        s = tf.nn.softmax(s)


    g = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')(input)
    if mode == 'emb': g = tf.keras.layers.MaxPooling2D()(g)  # GB
    if channel == 1: g = tf.keras.layers.Permute([2, 3, 1])(g)
    g = tf.keras.layers.Reshape([-1, filters])(g) # [BS, HW, C]

    o = tf.matmul(s, g) # [BS, HW, C]
    if channel == 1:
        o = tf.keras.layers.Reshape([input.shape[2], input.shape[3], filters])(o)
        o = tf.keras.layers.Permute([3, 1, 2])(o) # [BS, H, W, C]
    else:
        o = tf.keras.layers.Reshape([input.shape[1], input.shape[2], filters])(o) # [BS, H, W, C]
    o = tf.keras.layers.Conv2D(num_input_channels, kernel_size=1, padding='same', kernel_initializer='zeros')(o)
    o = tf.keras.layers.BatchNormalization(axis=channel)(o) # Section 4.1

    x = o + input
    return x

