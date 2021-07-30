import time
import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from scipy import ndimage

from Data import *
from Losses import *
from ResNetGANModel import *
from PatchGAN import *
from CycleReconstructor import *
from Unet import *

def get_and_increment_chpt_num(filename="chpt.txt"):
    with open(filename, "a+") as f:
        f.seek(0)
        val = int(f.read() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(str(val))
        return val

PARAMS = {
    'change': 'Cifar 10mm. Test_and_generate.py. Load and test then generate 100 images. Batch size 8',

    'dir': '200x200-10mm-cifar10-2', 'num_channels': 3, 'raw_input_shape': (180,240), 'target_input_shape': (32,32),
    # 'dir': '200x200-10mm-cifar10-2', 'num_channels': 3, 'raw_input_shape': (240,320), 'target_input_shape': (200,200),
    # 'dir': '200x200-10mm-cifar10-2', 'num_channels': 3, 'raw_input_shape': (240,320), 'target_input_shape': (32,32),
    # 'dir': 'lensless-cifar', 'num_channels': 3, 'raw_input_shape': (180,240), 'target_input_shape': (32,32),
    # 'dir': 'lensless-cifar', 'num_channels': 3, 'raw_input_shape': (240,320), 'target_input_shape': (32,32),
    # 'dir': 'imagenet', 'num_channels': 3, 'raw_input_shape': (240,320), 'target_input_shape': (224,224),
    # 'dir': 'imagenet', 'num_channels': 3, 'raw_input_shape': (180,240), 'target_input_shape': (224,224),

    'generator': cycle_reconstructor,
    'rec_loss': dist_mae_loss, # mse_loss, # ce_loss, #
    'cycle_loss': dist_mae_loss, # ce_loss, # mse_loss, #
    'chpt': get_and_increment_chpt_num(),
    'channel_format': 'NCHW',
    'latent': True,

    'load_chpt': "124/60", # None - new model, chpt number/string - single model, array of chpt numbers/strings - ensemble
    'save_models': False,

    'epochs': 80, # 200, # 30, #
    'batch_size': 8, # 16, 
    'learning_rate': 2e-4, # 2e-4, # 1e-3
    'disc_lr': 2e-5, # 2e-5,
    'lambda': 0.1, # 10,
    'lambda_b': 1,
    'gamma': 0,
    'beta': 100,

    'generator_to_discriminator_steps': 0, # 0: update both each step, 1: 1 generator step then 1 discriminator, 2: 2 generator steps then 1 discriminator, ...

    'type': 'double',
    # 'type': 'double_no_disc',
    # 'type': 'single_no_disc',
    # 'type': 'unet',

    'F_PARAMS': {
        'filters': {'down': [64, 128, 256, 256], 'up': [256, 128, 64]},
        'dropout': {'down': [0.0, 0.0, 0.0, 0.0], 'up': [0.0, 0.0, 0.0]},
        'kernels': [5,5], 'dilation_rate': 2,
        'res_depth': {'down':1, 'bottom':2, 'up':1},
        'norm': 'batch', # 'instance', # 'spectral', # 'batch',
        'activation': 'relu',
        'compression': True,
    },
    'G_PARAMS': {
        'filters': {'down':[64, 128, 256, 256], 'up':[256, 128, 64]},
        'dropout': {'down':[0.0, 0.0, 0.0, 0.0], 'up':[0.0, 0.0, 0.0]},
        'kernels': [5,5], 'dilation_rate': 2,
        # 'kernels': [3,3], 'dilation_rate': 1,
        'res_depth': {'down':1, 'bottom':2, 'up':1},
        'norm': 'batch', # 'instance', # 'spectral', # 'batch', 
        'activation': 'relu',
        'compression': True,
    },
}
# NCHW
# https://forums.developer.nvidia.com/t/tensorflow-op-spacetobatchnd-does-not-work-correctly-on-tx2/67671
# https://stackoverflow.com/questions/37689423/convert-between-nhwc-and-nchw-in-tensorflow
# https://github.com/keras-team/keras/issues/12656
if PARAMS['channel_format'] == 'NCHW': tf.keras.backend.set_image_data_format('channels_first')
PARAMS['F_PARAMS']['num_channels'] = PARAMS['G_PARAMS']['num_channels'] = PARAMS['num_channels']
PARAMS['F_PARAMS']['channel_format'] = PARAMS['G_PARAMS']['channel_format'] = PARAMS['channel_format']
PARAMS['F_PARAMS']['input_shape'] = PARAMS['G_PARAMS']['output_shape'] =  PARAMS['raw_input_shape']
PARAMS['F_PARAMS']['output_shape'] = PARAMS['G_PARAMS']['input_shape'] =  PARAMS['target_input_shape']

reset_epoch_losses = {'dx_loss': [], 'dy_loss': [], 'f_loss': [], 'f_rec_loss': [], 'resized_mae': [], 'f_cycle': [], 'f_latent': [], 'f_cycle_perceptual': [], 'f_ssim': [],
                        'f_psnr': [], 'g_loss': [], 'g_rec_loss': [], 'g_cycle': [], 'g_cycle_perceptual': []}
val_reset_epoch_losses = {'val_dx_loss': [], 'val_dy_loss': [], 'val_f_loss': [], 'val_resized_mae': [], 'val_f_cycle': [], 'val_f_ssim': [], 'val_f_psnr': [], 
                        'val_g_loss': [], 'val_g_cycle': []}
total_losses = {'dx_loss': [], 'dy_loss': [], 'f_loss': [], 'f_rec_loss': [], 'resized_mae': [], 'f_cycle': [], 'f_latent': [], 'f_cycle_perceptual': [], 
                'f_ssim': [], 'f_psnr': [], 'g_loss': [], 'g_rec_loss': [], 'g_cycle': [], 'g_cycle_perceptual': [], 'val_dx_loss': [], 'val_dy_loss': [], 'val_f_loss': [], 'val_resized_mae': [], 'val_f_cycle': [], 'val_f_ssim': [], 'val_f_psnr': [],  'val_g_loss': [], 'val_g_cycle': []}


class Train:

    def log_parameters(self):
        print()
        print('PARAMETERS')
        for key,value in PARAMS.items(): print(key + ': ' + str(value))
        print("\n\n")
        if not (PARAMS['load_chpt'] and type(PARAMS['load_chpt']) is list):
            self.F.summary()
        elif PARAMS['load_chpt'] and type(PARAMS['load_chpt']) is list:
            self.F[0].summary
        if PARAMS['type'] == 'double' and not (PARAMS['load_chpt'] and type(PARAMS['load_chpt']) is list): 
            print()
            self.Dx.summary()
            print()
            self.Dy.summary()
        if (PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double') and PARAMS['raw_input_shape'] != PARAMS['target_input_shape'] and not (PARAMS['load_chpt'] and type(PARAMS['load_chpt']) is list): 
            print() 
            self.G.summary()
        elif PARAMS['load_chpt'] and type(PARAMS['load_chpt']) is list:
            print() 
            self.G[0].summary()
        print("\n\n")
        print('PARAMETERS')
        for key,value in PARAMS.items(): print(key + ': ' + str(value))
        print("\n\n")

    def get_val_func(self):
        # Double ensemble
        if PARAMS['load_chpt'] and type(PARAMS['load_chpt']) is list: 
            self.num_nets = len(self.F)
            self.val_step = self.validate_two_gen_step_ensemble
        elif PARAMS['type'] == 'double': 
            self.val_step = self.validate_two_gen_step
        elif PARAMS['type'] == 'double_no_disc': 
            self.val_step = self.validate_two_gen_no_disc_step
        elif PARAMS['type'] == 'single_no_disc': 
            self.val_step = self.validate_one_gen_no_disc_step
        elif PARAMS['type'] == 'unet':
            self.val_step = self.validate_unet_step

    def setup_models_and_optimizers(self):
        self.g = PARAMS['gamma']
        self.b = PARAMS['beta']
        self.l = PARAMS['lambda']
        self.lb = PARAMS['lambda_b']

        # Ensemble - only setup for full model (PARAMS['type'] = double)
        if PARAMS['load_chpt'] and type(PARAMS['load_chpt']) is list:
            self.F, self.G, self.Dx, self.Dy = [], [], [], []
            for chpt in PARAMS['load_chpt']:
                self.F.append(tf.keras.models.load_model('./models/' + str(chpt) + '/F'))
                self.G.append(tf.keras.models.load_model('./models/' + str(chpt) + '/G'))
                # self.Dx.append(tf.keras.models.load_model('./models/' + str(chpt) + '/Dx'))
                # self.Dy.append(tf.keras.models.load_model('./models/' + str(chpt) + '/Dy'))

        else:
            if PARAMS['type'] == 'unet':
                if PARAMS['load_chpt']:
                    self.F = tf.keras.models.load_model('./models/' + str(PARAMS['load_chpt']) + '/F')
                else:
                    self.F = full_resize_Unet(input_shape=PARAMS['F_PARAMS']['input_shape'], num_channels=PARAMS['F_PARAMS']['num_channels'])
                self.f_optimizer = tf.keras.optimizers.Adam(PARAMS['learning_rate'])
                self.f_optimizer._create_all_weights(self.F.trainable_weights)

            else:
                if PARAMS['load_chpt']:
                    self.F = tf.keras.models.load_model('./models/' + str(PARAMS['load_chpt']) + '/F')
                else:
                    self.F = PARAMS['generator'](PARAMS['F_PARAMS'])
                self.f_optimizer = tf.keras.optimizers.Adam(PARAMS['learning_rate'], beta_1=0.5)
                self.f_optimizer._create_all_weights(self.F.trainable_weights)

            if PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double':
                if PARAMS['load_chpt']:
                    self.G = tf.keras.models.load_model('./models/' + str(PARAMS['load_chpt']) + '/G')
                else:
                    self.G = PARAMS['generator'](PARAMS['G_PARAMS'])
                self.g_optimizer = tf.keras.optimizers.Adam(PARAMS['learning_rate'], beta_1=0.5)
                self.g_optimizer._create_all_weights(self.G.trainable_weights)
                    
            if PARAMS['type'] == 'double':
                if PARAMS['load_chpt']:
                    self.Dx = tf.keras.models.load_model('./models/' + str(PARAMS['load_chpt']) + '/Dx')
                    self.Dy = tf.keras.models.load_model('./models/' + str(PARAMS['load_chpt']) + '/Dy')
                else:
                    self.Dx = patch_gan(PARAMS['F_PARAMS']['output_shape'], PARAMS['F_PARAMS']['num_channels'], PARAMS['F_PARAMS']['norm'])
                    self.Dy = patch_gan(PARAMS['G_PARAMS']['output_shape'], PARAMS['G_PARAMS']['num_channels'], PARAMS['G_PARAMS']['norm'])
                self.dx_optimizer = tf.keras.optimizers.Adam(PARAMS['disc_lr'], beta_1=0.5)
                self.dy_optimizer = tf.keras.optimizers.Adam(PARAMS['disc_lr'], beta_1=0.5)
                self.dx_optimizer._create_all_weights(self.Dx.trainable_weights)
                self.dy_optimizer._create_all_weights(self.Dy.trainable_weights)

    def load_old_optimizers_if_necessary(self):
        if PARAMS['load_chpt'] is None: return
        self.load_optimizer_state(self.f_optimizer, 'F', self.F)
        if PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double':
            self.load_optimizer_state(self.g_optimizer, 'G', self.G)
        if PARAMS['type'] == 'double':
            self.load_optimizer_state(self.dx_optimizer, 'Dx', self.Dx)
            self.load_optimizer_state(self.dy_optimizer, 'Dy', self.Dy)

    def load_optimizer_state(self, optimizer, opt_name, model):
        '''
        Loads keras.optimizers object state.

        Arguments:
        optimizer --- Optimizer object to be loaded.
        load_path --- Path to save location.
        load_name --- Name of the .npy file to be read.
        model_train_vars --- List of model variables (obtained using Model.trainable_variables)
        '''
        opt_weights = np.load('./models/' + str(PARAMS['load_chpt']) + '/' + opt_name + '-opt.npy', allow_pickle=True)
        optimizer.set_weights(opt_weights)

    
    def display_test_images(self, epoch, test_gen):
        if PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double':
            os.mkdir('./figures/' + str(PARAMS['chpt']) + '/epoch_' + str(epoch))
            if PARAMS['load_chpt'] and type(PARAMS['load_chpt']) is list:
                generate_images_single_gen_ensemble(self.F, test_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=100, num_channels=PARAMS['num_channels'], train_str='test', latent=True)
            else:
                # generate_images(self.G, self.F, test_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=10000, num_channels=PARAMS['num_channels'], train_str='test')
                generate_images_single_gen(self.F, test_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=100, num_channels=PARAMS['num_channels'], train_str='test', latent=True)
        elif PARAMS['type'] == 'single_no_disc' or PARAMS['type'] == 'unet':
            if PARAMS['type'] == 'single_no_disc': gen_images = generate_images_single_gen
            else: gen_images = generate_images_unet
            os.mkdir('./figures/' + str(PARAMS['chpt']) + '/epoch_' + str(epoch))
            gen_images(self.F, test_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=100, num_channels=PARAMS['num_channels'], train_str='test')


    def display_all_images(self, train_gen, val_gen, test_gen):
        if PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double':
            generate_all_images(self.G, self.F, train_gen, val_gen, test_gen, PARAMS['chpt'], PARAMS['batch_size'])

        elif PARAMS['type'] == 'single_no_disc' or PARAMS['type'] == 'unet':
            generate_all_images_single_gen(self.F, train_gen, val_gen, test_gen, PARAMS['chpt'], PARAMS['batch_size'])

    
    def log_epoch_stats(self, epoch, start, batch_losses, val_losses, epoch_losses, epoch_val_losses):
        if epoch is not None:
            print('E{}, {}s:'.format(epoch, int(time.time()-start)), end='')
        if batch_losses is not None:
            for key,_ in batch_losses.items(): 
                print(" {}: {:.4f}".format(key, np.average(epoch_losses[key])), end=',')
                total_losses[key].append(np.average(epoch_losses[key]))
        if val_losses is not None:
            for key,_ in val_losses.items(): 
                print(" {}: {:.3f}".format(key, np.average(epoch_val_losses[key])), end=',')
                total_losses[key].append(np.average(epoch_val_losses[key]))
        print('\n', end='', flush=True)


    @tf.function
    def distributed_val_step(self, x, y):
        per_replica_losses = self.mirrored_strategy.run(self.val_step, args=(x,y))
        replica_losses = self.mirrored_strategy.experimental_local_results(per_replica_losses)
        return replica_losses


    @tf.function
    def validate_two_gen_step_ensemble(self, x, y):
        fy, f_gx = tf.zeros_like(x), tf.zeros_like(x)
        gx, g_fy = tf.zeros_like(y), tf.zeros_like(y)
        for i in range(self.num_nets):
            F = self.F[i]
            G = self.G[i]
            fy += F(y, training=True)[0]
            gx += G(x, training=True)[0]
            f_gx += F(gx, training=True)[0]
            g_fy += G(fy, training=True)[0]

        fy /= self.num_nets
        gx /= self.num_nets
        f_gx /= self.num_nets
        g_fy /= self.num_nets
        f_loss = reconstruction_loss(x, fy, dist_mae_loss)
        f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l) 

        g_loss = reconstruction_loss(y, gx, dist_mae_loss) # PARAMS['rec_loss'])
        g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l) 

        f_ssim, f_psnr = dist_compute_metrics(x, fy)

        return {'val_f_loss': f_loss, 'val_f_cycle': f_cycle_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr, 'val_g_loss': g_loss, 'val_g_cycle': g_cycle_loss}
        # return {'val_dx_loss': dx_loss, 'val_dy_loss': dy_loss, 'val_f_loss': f_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr, 'val_g_loss': g_loss}

    
    @tf.function
    def validate_two_gen_step(self, x, y):
        fy = self.F(y, training=True)
        # fy_latent = fy[1:]
        fy = fy[0]
        gx = self.G(x, training=True)
        # gx_latent = gx[1:]
        gx = gx[0]
        f_gx = self.F(gx, training=True)[0]
        g_fy = self.G(fy, training=True)[0]

        # dx_x = self.Dx(x, training=True)
        # dx_fy = self.Dx(fy, training=True)
        # dy_gx = self.Dy(gx, training=True)
        # dy_y = self.Dy(y, training=True)

        # dx_loss = dist_ls_gan_loss_disc(dx_x, dx_fy)
        f_loss = reconstruction_loss(x, fy, dist_mae_loss) # PARAMS['rec_loss'])
        f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l) 

        # dy_loss = dist_ls_gan_loss_disc(dy_y, dy_gx)
        g_loss = reconstruction_loss(y, gx, dist_mae_loss) # PARAMS['rec_loss'])
        g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l) 

        f_ssim, f_psnr = dist_compute_metrics(x, fy)

        return {'val_f_loss': f_loss, 'val_f_cycle': f_cycle_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr, 'val_g_loss': g_loss, 'val_g_cycle': g_cycle_loss}
        # return {'val_dx_loss': dx_loss, 'val_dy_loss': dy_loss, 'val_f_loss': f_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr, 'val_g_loss': g_loss}


    @tf.function
    def validate_two_gen_no_disc_step(self, x, y):
        fy = self.F(y, training=True)
        # fy_latent = fy[1:]
        fy = fy[0]
        gx = self.G(x, training=True)
        # gx_latent = gx[1:]
        gx = gx[0]
        f_gx = self.F(gx, training=True)[0]
        g_fy = self.G(fy, training=True)[0]

        f_loss = reconstruction_loss(x, fy, dist_mae_loss) #PARAMS['rec_loss'])
        f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l) 

        g_loss = reconstruction_loss(y, gx, dist_mae_loss) #PARAMS['rec_loss'])
        g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l) 

        f_ssim, f_psnr = dist_compute_metrics(x, fy)

        return {'val_f_loss': f_loss, 'val_f_cycle': f_cycle_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr, 'val_g_loss': g_loss, 'val_g_cycle': g_cycle_loss}
        # return {'val_f_loss': f_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr, 'val_g_loss': g_loss}


    @tf.function
    def validate_one_gen_no_disc_step(self, x, y):
        fy = self.F(y, training=True)[0]
        f_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])
        f_ssim, f_psnr = dist_compute_metrics(x, fy)
        return {'val_f_loss': f_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr}


    @tf.function
    def validate_unet_step(self, x, y):
        fy = self.F(y, training=False)
        f_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])

        f_ssim, f_psnr = dist_compute_metrics(x, fy)

        return {'val_f_loss': f_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr}
        # return {'val_f_loss': f_loss, 'val_resized_mae': resized_mae, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr}


    def load_and_test(self):
        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        with self.mirrored_strategy.scope():
            self.setup_models_and_optimizers()
        self.log_parameters()

        # Setup datasets
        ds = DataSetup(PARAMS['dir'], PARAMS['batch_size'], PARAMS['raw_input_shape'], PARAMS['type'])
        train_gen, val_gen, test_gen = ds.setup_datasets(shuffle=False)
        dist_test_gen = self.mirrored_strategy.experimental_distribute_dataset(test_gen)

        # Setup training/validation functions
        self.get_val_func()

        os.mkdir('./figures/' + str(PARAMS['chpt']))
        
        # Test loop
        test_losses = dict(val_reset_epoch_losses)
        for x, y in dist_test_gen:
            step_losses = self.distributed_val_step(x, y)[0]
            for key,value in step_losses.items(): 
                test_losses[key].append(self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, value, axis=None))

        self.log_epoch_stats(None, None, None, step_losses, None, test_losses)
        self.display_test_images(0, test_gen)
        # self.display_all_images(train_gen, val_gen, test_gen)


def main():
    train = Train()
    train.load_and_test()


if __name__ == '__main__':
    main()