import tensorflow as tf
import tensorflow_addons as tfa

from CycleReconstructorBlocks import *


# 180->224
def cycle_reconstructor_180x240_224x224(filters={'down':[64,64,64,64], 'up':[64]}, 
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
    # for _ in range(res_depth['down']):
    #     top = residual_block(top, 64, kernels[0], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    # skips.append(top)

    # Down
    d1 = reflect_padding(top, [[5,5], [5,5]], channel_format=channel_format)
    d1 = conv_norm_act(d1, down_filters[0], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(d1)
    d1 = tf.keras.layers.Dropout(down_dropout[0])(d1)
    for _ in range(res_depth['down']):
        d1 = residual_block(d1, down_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    # skips.append(d1)

    d2 = reflect_padding(d1, [[5,5], [3,3]], channel_format=channel_format)
    d2 = conv_norm_act(d2, down_filters[1], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(d2)
    d2 = tf.keras.layers.Dropout(down_dropout[1])(d2)
    for _ in range(res_depth['down']):
        d2 = residual_block(d2, down_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    # skips.append(d2)

    d3 = reflect_padding(d2, [[5,5], [1,1]], channel_format=channel_format)
    d3 = conv_norm_act(d3, down_filters[2], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(d3)
    d3 = tf.keras.layers.Dropout(down_dropout[2])(d3)
    for _ in range(res_depth['down']):
        d3 = residual_block(d3, down_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d4 = reflect_padding(d3, [[4,4], [3,3]], channel_format=channel_format)
    # skips.append(d4)

    d4 = conv_norm_act(d4, down_filters[3], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(d4) #
    d4 = tf.keras.layers.Dropout(down_dropout[3])(d4)
    for _ in range(res_depth['down']):
        d4 = residual_block(d4, down_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Bottom
    r = d4
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
        latents.append(r)
    # if res_depth['bottom'] > 0: latents.append(r) #

    # Up
    u1 = reflect_padding(r, [[4,4], [4,4]], channel_format=channel_format)
    u1 = convtranspose_norm_act(u1, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    # u1 = skip(False, u1, skips[-1], concat_axis, up_filters[0], kernels[1], dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())
    latents.append(u1) #
    u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
    for _ in range(res_depth['up']):
        u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u2 = reflect_padding(u1, [[4,4], [4,4]], channel_format=channel_format)
    u2 = convtranspose_norm_act(u1, up_filters[1], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u2 = tf.keras.layers.Dropout(up_dropout[1])(u2)
    for _ in range(res_depth['up']):
        u2 = residual_block(u2, up_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u3 = convtranspose_norm_act(u2, up_filters[2], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u3 = tf.keras.layers.Dropout(up_dropout[2])(u3)
    for _ in range(res_depth['up']):
        u3 = residual_block(u3, up_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)


    top = reflect_padding(u3, top_layer_padding, channel_format=channel_format)
    # top = skip(True, top, skips[0], concat_axis, 64, kernels[0], dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())
    ending_act = tf.keras.activations.sigmoid
    # ending_act = tf.keras.layers.Activation('tanh')
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)
    # top = conv_norm(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer)

    # model = tf.keras.models.Model(input, top, name='resnet_generator_180x240_32x32')
    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='cycle_reconstructor_180x240_224x224')
    return model


def cycle_reconstructor_224x224_180x240(filters={'down':[64], 'up':[64,64,64,64,64]}, 
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
        input = tf.keras.layers.Input((num_channels, 224, 224))
    else:
        input = tf.keras.layers.Input((224, 224, num_channels))
    top = reflect_padding(input, in_layer_padding, channel_format=channel_format)
    top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())
    # skips.append(top)

    # Down
    down = conv_norm_act(top, down_filters[0], kernels[1], (2,2), norm_layer=norm_layer, activation=activation())
    # latents.append(down)
    down = tf.keras.layers.Dropout(down_dropout[0])(down)
    for _ in range(res_depth['down']):
        down = residual_block(down, down_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    
    down = reflect_padding(down, [[0,0], [2,2]], channel_format=channel_format)
    down = conv_norm_act(down, down_filters[1], kernels[1], (2,2), norm_layer=norm_layer, activation=activation())
    # latents.append(down)
    down = tf.keras.layers.Dropout(down_dropout[1])(down)
    for _ in range(res_depth['down']):
        down = residual_block(down, down_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    down = reflect_padding(down, [[0,0], [2,2]], channel_format=channel_format)
    down = conv_norm_act(down, down_filters[2], kernels[1], (2,2), norm_layer=norm_layer, activation=activation())
    # latents.append(down)
    down = tf.keras.layers.Dropout(down_dropout[2])(down)
    for _ in range(res_depth['down']):
        down = residual_block(down, down_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    
    down = reflect_padding(down, [[2,2], [0,0]], channel_format=channel_format)
    down = conv_norm_act(down, down_filters[-1], kernels[1], (2,2), norm_layer=norm_layer, activation=activation())
    # latents.append(down)
    down = tf.keras.layers.Dropout(down_dropout[-1])(down)
    for _ in range(res_depth['down']):
        down = residual_block(down, down_filters[-1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # latents.append(down) #

    # Bottom
    r = down
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[-1], kernels[1], padding=bottom_residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
        latents.append(r)
    # if res_depth['bottom'] > 0: latents.append(r)

    # Up
    if kernels[1] == 3:
        u1 = reflect_padding(r, [[3,3], [4,4]], channel_format=channel_format)
    else:
        u1 = reflect_padding(r, [[2,3], [4,4]], channel_format=channel_format)
    u1 = convtranspose_norm_act(u1, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(u1)
    # u1 = skip(True, u1, skips[-1], concat_axis, up_filters[0], kernels[1], dilation_rate, norm_layer=norm_layer, activation=activation())
    u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
    for _ in range(res_depth['up']):
        u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u2 = reflect_padding(u1, [[1,1], [4,4]], channel_format=channel_format)
    u2 = convtranspose_norm_act(u2, up_filters[1], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(u2)
    # u2 = skip(True, u2, skips[-2], concat_axis, up_filters[1], kernels[1], dilation_rate, norm_layer=norm_layer, activation=activation())
    u2 = tf.keras.layers.Dropout(up_dropout[1])(u2)
    for _ in range(res_depth['up']):
        u2 = residual_block(u2, up_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u3 = reflect_padding(u2, [[1,1], [4,4]], channel_format=channel_format)
    u3 = convtranspose_norm_act(u3, up_filters[2], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(u3)
    # u3 = skip(True, u3, skips[-3], concat_axis, up_filters[2], kernels[1], dilation_rate, norm_layer=norm_layer, activation=activation())
    u3 = tf.keras.layers.Dropout(up_dropout[2])(u3)
    for _ in range(res_depth['up']):
        u3 = residual_block(u3, up_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Top
    top = reflect_padding(u3, out_layer_padding, channel_format=channel_format)
    # top = skip(True, top, skips[0], concat_axis, 64, kernels[0], dilation_rate, norm_layer=norm_layer, activation=activation())
    ending_act = tf.keras.activations.sigmoid
    # ending_act = tf.keras.layers.Activation('tanh')
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)
    # top = conv_norm(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer)

    # model = tf.keras.models.Model(input, top, name='resnet_generator_32x32_180x240')
    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='cycle_reconstructor_224x224_180x240')
    return model



## 320->224
def cycle_reconstructor_240x320_224x224(filters={'down':[64,64,64,64], 'up':[64]}, 
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
    
    if kernels[0] == 7: top_layer_padding = 3
    elif kernels[0] == 5: top_layer_padding = 2
    elif kernels[0] == 3: top_layer_padding = 1
    top_layer_padding = [[top_layer_padding,top_layer_padding], [top_layer_padding,top_layer_padding]]

    residual_padding = dilation_rate
    if kernels[1] == 5: residual_padding += dilation_rate
    residual_padding = [[residual_padding,residual_padding], [residual_padding,residual_padding]]

    if channel_format == 'NCHW':
      input = tf.keras.layers.Input((num_channels, 240, 320))
    else:
      input = tf.keras.layers.Input((240, 320, num_channels))
    
    top = reflect_padding(input, top_layer_padding, channel_format=channel_format)
    top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())
    # for _ in range(res_depth['down']):
    #     top = residual_block(top, 64, kernels[0], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    # skips.append(top)


    # Down
    d1 = reflect_padding(top, [[4,4], [4,4]], channel_format=channel_format)
    d1 = conv_norm_act(d1, down_filters[0], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(d1)
    d1 = tf.keras.layers.Dropout(down_dropout[0])(d1)
    for _ in range(res_depth['down']):
        d1 = residual_block(d1, down_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    # skips.append(d1)


    d2 = reflect_padding(d1, [[4,4], [0,0]], channel_format=channel_format)
    d2 = conv_norm_act(d2, down_filters[1], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(d2)
    d2 = tf.keras.layers.Dropout(down_dropout[1])(d2)
    for _ in range(res_depth['down']):
        d2 = residual_block(d2, down_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    # skips.append(d2)


    d3 = reflect_padding(d2, [[4,4], [0,0]], channel_format=channel_format)
    d3 = conv_norm_act(d3, down_filters[2], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(d3)
    d3 = tf.keras.layers.Dropout(down_dropout[2])(d3)
    for _ in range(res_depth['down']):
        d3 = residual_block(d3, down_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d4 = reflect_padding(d3, [[2,2], [0,0]], channel_format=channel_format)
    # skips.append(d4)

    d4 = conv_norm_act(d4, down_filters[3], kernels[1], (2,2), dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(d4) #
    d4 = tf.keras.layers.Dropout(down_dropout[3])(d4)
    for _ in range(res_depth['down']):
        d4 = residual_block(d4, down_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    d4 = reflect_padding(d4, [[3,4], [3,4]], channel_format=channel_format)

    # Bottom
    r = d4
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[3], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
        latents.append(r)
    # if res_depth['bottom'] > 0: latents.append(r) #


    # Up
    u1 = convtranspose_norm_act(r, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    # u1 = skip(False, u1, skips[-1], concat_axis, up_filters[0], kernels[1], dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())
    latents.append(u1) #
    u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
    for _ in range(res_depth['up']):
        u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)


    u2 = convtranspose_norm_act(u1, up_filters[1], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u2 = tf.keras.layers.Dropout(up_dropout[1])(u2)
    for _ in range(res_depth['up']):
        u2 = residual_block(u2, up_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u3 = convtranspose_norm_act(u2, up_filters[2], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    u3 = tf.keras.layers.Dropout(up_dropout[2])(u3)
    for _ in range(res_depth['up']):
        u3 = residual_block(u3, up_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)


    top = reflect_padding(u3, top_layer_padding, channel_format=channel_format)
    # top = skip(True, top, skips[0], concat_axis, 64, kernels[0], dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())
    ending_act = tf.keras.activations.sigmoid
    # ending_act = tf.keras.layers.Activation('tanh')
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)

    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='cycle_reconstructor_240x320_224x224')
    return model

def cycle_reconstructor_224x224_240x320(filters={'down':[64], 'up':[64,64,64,64,64]}, 
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
        input = tf.keras.layers.Input((num_channels, 224, 224))
    else:
        input = tf.keras.layers.Input((224, 224, num_channels))
    top = reflect_padding(input, in_layer_padding, channel_format=channel_format)
    top = conv_norm_act(top, 64, kernels[0], padding='valid', dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation())
    # skips.append(top)

    # Down
    down = conv_norm_act(top, down_filters[0], kernels[1], (2,2), norm_layer=norm_layer, activation=activation())
    # latents.append(down)
    down = tf.keras.layers.Dropout(down_dropout[0])(down)
    for _ in range(res_depth['down']):
        down = residual_block(down, down_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    
    down = reflect_padding(down, [[4,4], [4,4]], channel_format=channel_format)
    down = conv_norm_act(down, down_filters[1], kernels[1], (2,2), norm_layer=norm_layer, activation=activation())
    # latents.append(down)
    down = tf.keras.layers.Dropout(down_dropout[1])(down)
    for _ in range(res_depth['down']):
        down = residual_block(down, down_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    down = reflect_padding(down, [[4,4], [4,4]], channel_format=channel_format)
    down = conv_norm_act(down, down_filters[2], kernels[1], (2,2), norm_layer=norm_layer, activation=activation())
    # latents.append(down)
    down = tf.keras.layers.Dropout(down_dropout[2])(down)
    for _ in range(res_depth['down']):
        down = residual_block(down, down_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
    
    down = reflect_padding(down, [[4,4], [4,4]], channel_format=channel_format)
    down = conv_norm_act(down, down_filters[-1], kernels[1], (2,2), norm_layer=norm_layer, activation=activation())
    # latents.append(down)
    down = tf.keras.layers.Dropout(down_dropout[-1])(down)
    for _ in range(res_depth['down']):
        down = residual_block(down, down_filters[-1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # latents.append(down) #

    # Bottom
    r = down
    for _ in range(res_depth['bottom']):
        r = residual_block(r, down_filters[-1], kernels[1], padding=bottom_residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)
        latents.append(r)
    # if res_depth['bottom'] > 0: latents.append(r)

    # Up
    u1 = reflect_padding(r, [[2,3], [6,6]], channel_format=channel_format)
    u1 = convtranspose_norm_act(u1, up_filters[0], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(u1)
    # u1 = skip(True, u1, skips[-1], concat_axis, up_filters[0], kernels[1], dilation_rate, norm_layer=norm_layer, activation=activation())
    u1 = tf.keras.layers.Dropout(up_dropout[0])(u1)
    for _ in range(res_depth['up']):
        u1 = residual_block(u1, up_filters[0], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u2 = reflect_padding(u1, [[4,4], [5,5]], channel_format=channel_format)
    u2 = convtranspose_norm_act(u2, up_filters[1], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(u2)
    # u2 = skip(True, u2, skips[-2], concat_axis, up_filters[1], kernels[1], dilation_rate, norm_layer=norm_layer, activation=activation())
    u2 = tf.keras.layers.Dropout(up_dropout[1])(u2)
    for _ in range(res_depth['up']):
        u2 = residual_block(u2, up_filters[1], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    u3 = reflect_padding(u2, [[0,0], [4,4]], channel_format=channel_format)
    u3 = convtranspose_norm_act(u3, up_filters[2], kernels[1], dilation_rate=1, norm_layer=norm_layer, activation=activation())
    latents.append(u3)
    # u3 = skip(True, u3, skips[-3], concat_axis, up_filters[2], kernels[1], dilation_rate, norm_layer=norm_layer, activation=activation())
    u3 = tf.keras.layers.Dropout(up_dropout[2])(u3)
    for _ in range(res_depth['up']):
        u3 = residual_block(u3, up_filters[2], kernels[1], padding=residual_padding, dilation_rate=dilation_rate, norm_layer=norm_layer, activation=activation(), channel_format=channel_format)

    # Top
    top = reflect_padding(u3, out_layer_padding, channel_format=channel_format)
    # top = skip(True, top, skips[0], concat_axis, 64, kernels[0], dilation_rate, norm_layer=norm_layer, activation=activation())
    ending_act = tf.keras.activations.sigmoid
    # ending_act = tf.keras.layers.Activation('tanh')
    top = conv_norm_act(top, num_channels, kernels[0], padding='valid', dilation_rate=1, norm_layer=norm_layer, activation=ending_act)

    out = [top] + latents
    model = tf.keras.models.Model(input, out, name='cycle_reconstructor_224x224_240x320')
    return model
