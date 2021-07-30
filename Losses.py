import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input


def expand_multiple(tensor):
    return tf.expand_dims(tf.expand_dims(tf.expand_dims(tensor, axis=-1), axis=-1), axis=-1)

def pcc(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    pred_mean = expand_multiple(tf.reduce_mean(y_pred, axis=[1,2,3]))
    y_mean = expand_multiple(tf.reduce_mean(y_true, axis=[1,2,3]))
    cov = tf.reduce_sum((y_pred - pred_mean) * (y_true - y_mean), axis=[1,2,3])
    std_x = tf.sqrt(tf.reduce_sum(tf.square(y_true-y_mean), axis=[1,2,3]))
    std_y = tf.sqrt(tf.reduce_sum(tf.square(y_pred-pred_mean), axis=[1,2,3]))
    return cov / (std_x*std_y)
  
def dist_npcc(y_true, y_pred):
    return tf.nn.compute_average_loss(-pcc(y_true, y_pred))
    


def dist_compute_metrics(x, fy):
    nhwc_x = tf.transpose(x, [0, 2, 3, 1])
    nhwc_fy = tf.transpose(fy, [0, 2, 3, 1])
    f_ssim = tf.nn.compute_average_loss(tf.image.ssim(nhwc_x, nhwc_fy, 1.))
    f_psnr = tf.nn.compute_average_loss(tf.image.psnr(nhwc_x, nhwc_fy, 1.))
    return f_ssim, f_psnr

def dist_mae_loss(y_true, y_pred, axis=[1,2,3]):
    return tf.nn.compute_average_loss(tf.reduce_mean(tf.abs(y_true - y_pred), axis=axis))

def dist_mse_loss(y_true, y_pred, axis=[1,2,3]):
    return tf.nn.compute_average_loss(tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=axis))

def dist_ls_gan_loss_gen(disc_out_fake):
    """
    (D_x(F(y)) - 1)^2 or (D_y(G(x)) - 1)^2

    disc_out_fake: D_x(F(y)) or D_y(G(x))
    """
    return tf.nn.compute_average_loss(tf.reduce_mean(tf.math.squared_difference(disc_out_fake, 1), axis=[1,2,3]))

def dist_ls_gan_loss_disc(disc_out_real, disc_out_fake):
    """
    (D_x(x) - 1)^2 + D_x(F(y))^2 or (D_y(y) - 1)^2 + D_y(G(x))^2

    disc_out_real: D_x(x), and disc_x_out_fake: D_x(F(y)) 
    or disc_out_real: D_y(y) and disc_x_out_fake: D_y(G(x))
    """
    disc_loss = tf.reduce_mean(tf.math.squared_difference(disc_out_real, 0.9), axis=[1,2,3]) + tf.reduce_mean(tf.math.squared_difference(disc_out_fake, 0), axis=[1,2,3])
    # disc_loss = tf.reduce_mean(tf.math.squared_difference(disc_out_real, tf.random.uniform((), minval=0.7, maxval=1.)), axis=[1,2,3]) + tf.reduce_mean(tf.math.squared_difference(disc_out_fake, tf.random.uniform((), minval=0., maxval=0.3)), axis=[1,2,3])
    # disc_loss = tf.reduce_mean(tf.math.squared_difference(disc_out_real, 1), axis=[1,2,3]) + tf.reduce_mean(tf.math.squared_difference(disc_out_fake, 0), axis=[1,2,3])
    return 0.5 * tf.nn.compute_average_loss(disc_loss)

def dist_cycle_perceptual_loss(gen_latents, true_latents, gamma):
    """
    gen_latents: [] of hidden layer outputs from f(y) or g(f(y))
    true_latents: [] of hidden layer outputs from g(x)
    """
    total_loss = 0
    for i in range(len(gen_latents)):
        total_loss += dist_mae_loss(tf.math.l2_normalize(true_latents[i], axis=1), tf.math.l2_normalize(gen_latents[i], axis=1))
        # total_loss += dist_mae_loss(true_latents[i], gen_latent)
        # total_loss += dist_mae_loss(gram_matrix(g_latents[i]), gram_matrix(f_latent), axis=[1,2]) # Style
    return total_loss * gamma #/ cycle_perceptual_norm_term


tf.keras.backend.set_image_data_format('channels_first')
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(3,224,224))
VGG = tf.keras.models.Model(vgg.input, vgg.get_layer('block3_conv4').output)
VGG.trainable = False
# norm_term = 256*56*56
def dist_vgg_perceptual_loss(x, fy):
    x_perceptual = VGG(preprocess_input(x * 255), training=False)
    fy_perceptual = VGG(preprocess_input(fy * 255), training=False)
    return dist_mae_loss(tf.math.l2_normalize(x_perceptual, axis=1), tf.math.l2_normalize(fy_perceptual, axis=1)) # / norm_term


def dist_ce_loss(y_true, y_pred):
    # z * -log(x) + (1 - z) * -log(1 - x) for x = preds, z = labels
    ce = y_true * (-tf.math.log(y_pred + 1e-7)) + (1-y_true) * (-tf.math.log(1 - y_pred + 1e-7))
    return tf.nn.compute_average_loss(tf.math.reduce_mean(ce, axis=(1,2,3)))




def compute_metrics(x, fy):
    nhwc_x = tf.transpose(x, [0, 2, 3, 1])
    nhwc_fy = tf.transpose(fy, [0, 2, 3, 1])
    f_ssim = tf.reduce_mean(tf.image.ssim(nhwc_x, nhwc_fy, 1.))
    f_psnr = tf.reduce_mean(tf.image.psnr(nhwc_x, nhwc_fy, 1.))
    return f_ssim, f_psnr

def mae_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

def ce_loss(y_true, y_pred):
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    sce = tf.math.reduce_sum(ce, axis=(1,2,3))
    return tf.math.reduce_mean(sce)


def reconstruction_loss(real, gen, loss_func):
    """
    real: x and gen: F(y)
    or real: y and gen: G(x)
    """
    return loss_func(real, gen)


def cycle_loss(real, cycled, loss_func, LAMBDA):
    """
    real: x and cycled: F(G(x))
    or real: y and cycled: G(F(y))
    """
    return LAMBDA * loss_func(real, cycled)


def identity_loss(real, gen_real, loss_func, LAMBDA):
    """
    real: x and gen_real: F(x)
    or real: y and gen_real: G(y)
    """
    return 0.5 * LAMBDA * loss_func(real, gen_real)


def latent_loss(f_latents, g_latents, gamma):
    """
    f_latent: [] of hidden layers from f
    g_latent: [] of hidden layers from g
    """
    total_loss = 0
    for i in range(len(f_latents)):
        f_latent = f_latents[-(i+1)]
        total_loss += mse_loss(g_latents[i], f_latent)
        # total_loss += mae_loss(gram_matrix(g_latents[i]), gram_matrix(f_latent))
    # return (total_loss / len(f_latents)) * gamma
    return total_loss * gamma


# https://www.tensorflow.org/api_docs/python/tf/einsum
# https://www.tensorflow.org/tutorials/generative/style_transfer
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bcij,bdij->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


def ls_gan_loss_gen(disc_out_fake):
    """
    (D_x(F(y)) - 1)^2 or (D_y(G(x)) - 1)^2

    disc_out_fake: D_x(F(y)) or D_y(G(x))
    """
    return tf.reduce_mean(tf.math.squared_difference(disc_out_fake, 1))

def ls_gan_loss_disc(disc_out_real, disc_out_fake):
    """
    (D_x(x) - 1)^2 + D_x(F(y))^2 or (D_y(y) - 1)^2 + D_y(G(x))^2

    disc_out_real: D_x(x), and disc_x_out_fake: D_x(F(y)) 
    or disc_out_real: D_y(y) and disc_x_out_fake: D_y(G(x))
    """
    disc_loss = tf.reduce_mean(tf.math.squared_difference(disc_out_real, 0.9)) + tf.reduce_mean(tf.math.squared_difference(disc_out_fake, 0.1))
    # disc_loss = tf.reduce_mean(tf.math.squared_difference(disc_out_real, 1)) + tf.reduce_mean(tf.math.squared_difference(disc_out_fake, 0))
    return 0.5 * disc_loss

