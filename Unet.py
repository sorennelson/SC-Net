import tensorflow as tf

def full_resize_Unet(starting_filters=32, input_shape=(240,320), num_channels=3):
    # input = tf.keras.layers.Input((240, 320, 3))
    input = tf.keras.layers.Input((num_channels, input_shape[0], input_shape[1]))

    l2 = encoder_block(input, starting_filters)

    l3 = tf.keras.layers.MaxPooling2D()(l2)
    l3 = encoder_block(l3, starting_filters*2)

    l4 = tf.keras.layers.MaxPooling2D()(l3)
    l4 = encoder_block(l4, starting_filters*4)

    l5 = tf.keras.layers.MaxPooling2D()(l4)
    l5 = encoder_block(l5, starting_filters*8)

    l6 = tf.keras.layers.MaxPooling2D()(l5)
    l6 = encoder_block(l6, starting_filters*16)

    r5 = tf.keras.layers.Conv2DTranspose(starting_filters*8, (3,2), (2,2), padding='same')(l6)
    m5 = tf.keras.layers.Concatenate(axis=1)([l5, r5])
    # m5 = tf.keras.layers.Concatenate(axis=1)([m5, r5])
    r5 = encoder_block(m5, starting_filters*8)

    r4 = tf.keras.layers.Conv2DTranspose(starting_filters*4, (2,2), (2,2), padding='same')(r5)
    if input_shape == (468,468): 
        r4 = tf.pad(r4, [[0,0], [0,0], [1,0], [1,0]])
    m4 = tf.keras.layers.Concatenate(axis=1)([l4, r4])
    # m4 = tf.keras.layers.Concatenate(axis=1)([m4, r4])
    r4 = encoder_block(m4, starting_filters*4)

    r3 = tf.keras.layers.Conv2DTranspose(starting_filters*2, (2,2), (2,2), padding='same')(r4)
    if input_shape == (338,338): 
        r3 = tf.pad(r3, [[0,0], [0,0], [1,0], [1,0]])
    m3 = tf.keras.layers.Concatenate(axis=1)([l3, r3])
    # m3 = tf.keras.layers.Concatenate(axis=1)([m3, r3])
    r3 = encoder_block(m3, starting_filters*2)

    r2 = tf.keras.layers.Conv2DTranspose(starting_filters, (2,2), (2,2), padding='same')(r3)
    m2 = tf.keras.layers.Concatenate(axis=1)([l2, r2])
    # m2 = tf.keras.layers.Concatenate()([m2, r2])
    r2 = encoder_block(m2, starting_filters)

    r1 = tf.keras.layers.Conv2D(num_channels, (1,1), activation='sigmoid', padding='same')(r2)

    return tf.keras.models.Model(input, r1)


def down_Unet(starting_filters=32, input_shape=(240,320), num_channels=3):
    input = tf.keras.layers.Input((num_channels, input_shape[0], input_shape[1]))

    l2 = encoder_block(input, starting_filters)

    l3 = tf.keras.layers.MaxPooling2D()(l2)
    l3 = encoder_block(l3, starting_filters*2)

    l4 = tf.keras.layers.MaxPooling2D()(l3)
    l4 = encoder_block(l4, starting_filters*4)

    l5 = tf.keras.layers.MaxPooling2D()(l4)
    l5 = encoder_block(l5, starting_filters*8)

    l6 = tf.keras.layers.MaxPooling2D()(l5)
    l6 = encoder_block(l6, starting_filters*16)

    r5 = tf.keras.layers.Conv2DTranspose(starting_filters*8, (3,3), (2,1), padding='same')(l6)
    ch, cw = get_crop_shape(l5, r5)
    l5 = tf.keras.layers.Cropping2D((ch, cw))(l5)
    m5 = tf.keras.layers.Concatenate(axis=1)([l5, r5])
    r5 = encoder_block(m5, starting_filters*8)

    r5 = tf.pad(r5, [[0,0], [0,0], [1,1], [6,6]])

    r4 = tf.keras.layers.Conv2D(starting_filters*4, (3,3), (1,1), padding='same')(r5)
    ch, cw = get_crop_shape(l4, r4)
    l4 = tf.keras.layers.Cropping2D((ch, cw))(l4)
    m4 = tf.keras.layers.Concatenate(axis=1)([l4, r4])
    r4 = encoder_block(m4, starting_filters*4)

    r3 = tf.keras.layers.Conv2D(starting_filters*2, (3,3), (1,1), padding='same')(r4)
    ch, cw = get_crop_shape(l3, r3)
    l3 = tf.keras.layers.Cropping2D((ch, cw))(l3)
    m3 = tf.keras.layers.Concatenate(axis=1)([l3, r3])
    r3 = encoder_block(m3, starting_filters*2)

    r2 = tf.keras.layers.Conv2D(starting_filters*2, (3,3), (1,1), padding='same')(r3)
    ch, cw = get_crop_shape(l2, r2)
    l2 = tf.keras.layers.Cropping2D((ch, cw))(l2)
    m2 = tf.keras.layers.Concatenate(axis=1)([l2, r2])
    r2 = encoder_block(m2, starting_filters)

    r1 = tf.keras.layers.Conv2D(num_channels, (1,1), activation='sigmoid', padding='same')(r2)

    return tf.keras.models.Model(input, r1)


def encoder_block(input, filters):
    x = tf.keras.layers.Conv2D(filters, (3,3), padding='same')(input)
    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, (3,3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def get_crop_shape(target, refer):
  cw = target.get_shape()[3] - refer.get_shape()[3]
  if cw % 2 != 0:
    cw1, cw2 = int(cw/2), int(cw/2) + 1
  else:
    cw1, cw2 = int(cw/2), int(cw/2)

  ch = target.get_shape()[2] - refer.get_shape()[2]
  if ch % 2 != 0:
    ch1, ch2 = int(ch/2), int(ch/2) + 1
  else:
    ch1, ch2 = int(ch/2), int(ch/2)

  return (ch1, ch2), (cw1, cw2)