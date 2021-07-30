import time
import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from Data import *
from Losses import *
from ResNetGANModel import *
from PatchGAN import *
from CycleReconstructor import *
from Unet import *

def get_and_increment_chpt_num(filename="chpt.txt"):
    # Loads and increments chpt number stored in chpt.txt which is used for figures and models directory names. 
    # If no file exists it creates one.
    with open(filename, "a+") as f:
        f.seek(0)
        val = int(f.read() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(str(val))
        return val

PARAMS = {
    'change': 'Cifar 10mm. F = GAN + 100(MAE + 1(Forward + 0.1*Backward)). G = GAN + 100(MAE + 1(Forward + 0.1*Backward)). Save models. Training=True, Batch Norm. Train-5.py. 3200 shuffle buffer size and 16 batch size. without non local - normal patchgan (*with* dropout). normalize 255',

    # 'dir': 'hamrick', 'num_channels': 3, 'raw_input_shape': (64,64), 'target_input_shape': (64,64),
    # 'dir': '200x200-10mm-cifar10-2', 'num_channels': 3, 'raw_input_shape': (180,240), 'target_input_shape': (32,32),
    # 'dir': '200x200-10mm-cifar10-2', 'num_channels': 3, 'raw_input_shape': (240,320), 'target_input_shape': (200,200),
    # 'dir': '200x200-10mm-cifar10-2', 'num_channels': 3, 'raw_input_shape': (240,320), 'target_input_shape': (32,32),
    # 'dir': 'lensless-cifar', 'num_channels': 3, 'raw_input_shape': (60,80), 'target_input_shape': (32,32),
    # 'dir': 'lensless-cifar', 'num_channels': 3, 'raw_input_shape': (120,160), 'target_input_shape': (32,32),
    'dir': 'lensless-cifar', 'num_channels': 3, 'raw_input_shape': (180,240), 'target_input_shape': (32,32),
    # 'dir': 'lensless-cifar', 'num_channels': 3, 'raw_input_shape': (240,320), 'target_input_shape': (32,32),
    # 'dir': 'combined-lensless-cifar', 'num_channels': 3, 'raw_input_shape': (180,240), 'target_input_shape': (32,32),
    # 'dir': 'imagenet', 'num_channels': 3, 'raw_input_shape': (240,320), 'target_input_shape': (224,224),
    # 'dir': 'imagenet', 'num_channels': 3, 'raw_input_shape': (180,240), 'target_input_shape': (224,224),
    'dir': 'imagenet', 'num_channels': 3, 'raw_input_shape': (180,240), 'target_input_shape': (32,32),

    'generator': cycle_reconstructor,
    'rec_loss': dist_mae_loss, # dist_mae_loss,
    'cycle_loss': dist_mae_loss, # dist_mae_loss,
    'chpt': get_and_increment_chpt_num(),
    'channel_format': 'NCHW',
    'latent': True,

    'load_chpt': None,
    'save_models': True,

    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 2e-4, # 2e-4, # 1e-3
    'disc_lr': 2e-5,
    'lambda': 1,
    'lambda_b': 0.1,
    'gamma': 0,
    'beta': 100,

    'linear_decay_lr': None, # None for no decay, or integer for number of epochs to start decaying after
    'generator_to_discriminator_steps': 0, # 0: update both each step, 1: 1 generator step then 1 discriminator, 2: 2 generator steps then 1 discriminator, ...

    # Uncomment for training method
    'type': 'double', # Self Consistent Supervised
    # 'type': 'double_no_disc', # Self Consistent Supervised without GAN loss
    # 'type': 'single_no_disc', # Supervised only F, current model
    # 'type': 'unet', # Supervised only F, standard unet

    'F_PARAMS': {
        'filters': {'down': [64, 128, 256, 256], 'up': [256, 128, 64]},
        'dropout': {'down': [0.0, 0.0, 0.0, 0.0], 'up': [0.0, 0.0, 0.0]},
        'kernels': [5,5], 'dilation_rate': 2,
        'res_depth': {'down':1, 'bottom':2, 'up':1},
        'norm': 'batch', # 'instance', # 'batch',
        'activation': 'relu',
        'compression': True,
    },
    'G_PARAMS': {
        'filters': {'down':[64, 128, 256, 256], 'up':[256, 128, 64]},
        'dropout': {'down':[0.0, 0.0, 0.0, 0.0], 'up':[0.0, 0.0, 0.0]},
        'kernels': [5,5], 'dilation_rate': 2,
        'res_depth': {'down':1, 'bottom':2, 'up':1},
        'norm': 'batch', # 'instance', # 'batch',
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

# Helper functions for saving / setting up models

    def log_parameters(self):
        print()
        print('PARAMETERS')
        for key,value in PARAMS.items(): print(key + ': ' + str(value))
        print("\n\n")
        self.F.summary()
        if PARAMS['type'] == 'double': 
            print()
            self.Dx.summary()
            print()
            self.Dy.summary()
        if (PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double') and PARAMS['raw_input_shape'] != PARAMS['target_input_shape']: 
            print() 
            self.G.summary()
        print("\n\n")
        print('PARAMETERS')
        for key,value in PARAMS.items(): print(key + ': ' + str(value))
        print("\n\n")

    def get_train_val_funcs(self):
        if PARAMS['type'] == 'double': 
            self.train_step = self.train_two_gen_step
            self.val_step = self.validate_two_gen_step
        elif PARAMS['type'] == 'double_no_disc': 
            self.train_step = self.train_two_gen_no_disc_step
            self.val_step = self.validate_two_gen_no_disc_step
        elif PARAMS['type'] == 'single_no_disc': 
            self.train_step = self.train_one_gen_no_disc_step
            self.val_step = self.validate_one_gen_no_disc_step
        elif PARAMS['type'] == 'unet':
            self.train_step = self.train_unet_step
            self.val_step = self.validate_unet_step

    def setup_linear_decay_lr(self):
        # Sets up decay schedule.
        self.lr_gen = PARAMS['learning_rate']
        self.lr_disc = PARAMS['disc_lr']
        self.lr_drop_gen = self.lr_gen / PARAMS['epochs']
        self.lr_drop_disc = self.lr_disc / PARAMS['epochs']

    def linear_decay_lr(self):
        # Decays learning rate. To be called each epoch if using.
        self.lr_gen -= self.lr_drop_gen
        self.lr_disc -= self.lr_drop_disc
        self.f_optimizer.learning_rate.assign(self.lr_gen)
        self.g_optimizer.learning_rate.assign(self.lr_gen)
        self.dx_optimizer.learning_rate.assign(self.lr_disc)
        self.dy_optimizer.learning_rate.assign(self.lr_disc)

    def setup_models_and_optimizers(self):
        self.g = PARAMS['gamma']
        self.b = PARAMS['beta']
        self.l = PARAMS['lambda']
        self.lb = PARAMS['lambda_b']
        self.decay = False # to start

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
        self.load_optimizer_state(self.f_optimizer, 'F')
        if PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double':
            self.load_optimizer_state(self.g_optimizer, 'G')
        if PARAMS['type'] == 'double':
            self.load_optimizer_state(self.dx_optimizer, 'Dx')
            self.load_optimizer_state(self.dy_optimizer, 'Dy')

    def save_models(self, epoch=None):
        if not os.path.exists('./models/' + str(PARAMS['chpt'])):
            os.mkdir('./models/' + str(PARAMS['chpt']))
        if epoch is not None:
            if epoch % 10 == 0 and epoch > 30:
                path = './models/' + str(PARAMS['chpt']) + '/' + str(epoch)
                os.mkdir(path)
            else:
                return
        else:
            path = './models/' + str(PARAMS['chpt'])

        print('saving', flush=True)

        self.F.save(path + '/F')
        self.save_optimizer_state(self.f_optimizer, 'F', path)
        if PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double':
            self.G.save(path + '/G')
            self.save_optimizer_state(self.g_optimizer, 'G', path)
        if PARAMS['type'] == 'double':
            self.Dx.save(path + '/Dx')
            self.save_optimizer_state(self.dx_optimizer, 'Dx', path)
            self.Dy.save(path + '/Dy')
            self.save_optimizer_state(self.dy_optimizer, 'Dy', path)

    def turn_off_dropout(self):
        for layer in self.F.layers:
            if layer.__class__.__name__ == 'Dropout':
                layer.training = False
        for layer in self.G.layers:
            if layer.__class__.__name__ == 'Dropout':
                layer.training = False

    def turn_on_dropout(self):
        for layer in self.F.layers:
            if layer.__class__.__name__ == 'Dropout':
                layer.training = True
        for layer in self.G.layers:
            if layer.__class__.__name__ == 'Dropout':
                layer.training = True

    # https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state/49504376
    def save_optimizer_state(self, optimizer, opt_name, path):
        '''
        Saves keras.optimizers object state.

        Arguments:
        optimizer --- Optimizer object to be loaded.
        opt_name --- "Dx" or "Dy".
        path --- path to chpt.
        '''
        if os.path.exists(path + '/' + opt_name + '-opt.npy'):
            os.remove(path + '/' + opt_name + '-opt.npy')
        np.save(path + '/' + opt_name + '-opt', optimizer.get_weights())

    def load_optimizer_state(self, optimizer, opt_name):
        '''
        Loads keras.optimizers object state.

        Arguments:
        optimizer --- Optimizer object to be loaded.
        opt_name --- "Dx" or "Dy".
        '''
        # Load optimizer weights
        opt_weights = np.load('./models/' + str(PARAMS['load_chpt']) + '/' + opt_name + '-opt.npy', allow_pickle=True)
        # Set the weights of the optimizer
        optimizer.set_weights(opt_weights)


# Helper functions for logging / plotting / displaying images

    def display_images(self, epoch, train_gen, val_gen, epoch_step=None):
        if PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double':
            if epoch_step is not None:
                os.mkdir('./figures/' + str(PARAMS['chpt']) + '/epoch_' + str(epoch) + '-' + str(epoch_step))
            else:
                os.mkdir('./figures/' + str(PARAMS['chpt']) + '/epoch_' + str(epoch))
            generate_images(self.G, self.F, train_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=5, num_channels=PARAMS['num_channels'], train_str='train', latent=PARAMS['latent'], epoch_step=epoch_step)
            val_num = 40 if epoch > 20 else 20
            generate_images(self.G, self.F, val_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=val_num, num_channels=PARAMS['num_channels'], train_str='val', latent=PARAMS['latent'], epoch_step=epoch_step)

        elif PARAMS['type'] == 'single_no_disc' or PARAMS['type'] == 'unet':
            if PARAMS['type'] == 'single_no_disc': 
                gen_images = generate_images_single_gen
            else: 
                gen_images = generate_images_unet
            os.mkdir('./figures/' + str(PARAMS['chpt']) + '/epoch_' + str(epoch))
            gen_images(self.F, train_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=10, num_channels=PARAMS['num_channels'], train_str='train', latent=PARAMS['latent'])
            gen_images(self.F, val_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=10, num_channels=PARAMS['num_channels'], train_str='val', latent=PARAMS['latent'])

    def display_test_images(self, epoch, test_gen):
        if PARAMS['type'] == 'double_no_disc' or PARAMS['type'] == 'double':
            os.mkdir('./figures/' + str(PARAMS['chpt']) + '/epoch_' + str(epoch))
            generate_images(self.G, self.F, test_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=1000, num_channels=PARAMS['num_channels'], train_str='test')
        elif PARAMS['type'] == 'single_no_disc' or PARAMS['type'] == 'unet':
            if PARAMS['type'] == 'single_no_disc': gen_images = generate_images_single_gen
            else: gen_images = generate_images_unet
            os.mkdir('./figures/' + str(PARAMS['chpt']) + '/epoch_' + str(epoch))
            gen_images(self.F, test_gen, PARAMS['chpt'], epoch, PARAMS['batch_size'], num=1000, num_channels=PARAMS['num_channels'], train_str='test')

    def plot(self, epoch):
        if PARAMS['type'] == 'single_no_disc':
            vals = [total_losses['f_loss'], total_losses['val_f_loss']]
            names = ['Train', 'Validation']
            plot(vals, names, 'MAE', 'MAE', PARAMS['chpt'], epoch)
        elif PARAMS['type'] == 'double_no_disc':
            vals = [total_losses['f_loss'], total_losses['val_f_loss'], total_losses['g_loss'], total_losses['val_g_loss']]
            names = ['F Train', 'F Validation', 'G Train', 'G Validation']
            plot(vals, names, 'Losses', 'Loss', PARAMS['chpt'], epoch)
        elif PARAMS['type'] == 'double':
            vals = [total_losses['f_rec_loss'], total_losses['val_f_loss']]
            names = ['Train', 'Validation']
            plot(vals, names, 'MAE', 'MAE', PARAMS['chpt'], epoch)
            vals = [total_losses['f_ssim'], total_losses['val_f_ssim']]
            names = ['Train', 'Validation']
            plot(vals, names, 'SSIM', 'SSIM', PARAMS['chpt'], epoch)
    
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


# Training

    def train(self):
        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        with self.mirrored_strategy.scope():
            self.setup_models_and_optimizers()
        self.log_parameters()

        # Setup datasets
        ds = DataSetup(PARAMS['dir'], PARAMS['batch_size'], PARAMS['raw_input_shape'], PARAMS['type'])
        train_gen, val_gen, test_gen = ds.setup_datasets()
        dist_train_gen = self.mirrored_strategy.experimental_distribute_dataset(train_gen)
        dist_val_gen = self.mirrored_strategy.experimental_distribute_dataset(val_gen)
        dist_test_gen = self.mirrored_strategy.experimental_distribute_dataset(test_gen)
        os.mkdir('./figures/' + str(PARAMS['chpt']))

        # Setup training/validation functions
        self.get_train_val_funcs()

        
        for epoch in range(1, PARAMS['epochs'] + 1):
            start = time.time()
            epoch_losses = dict(reset_epoch_losses)
            step, img_count = 0, 0

            # Setup lr decaying
            if PARAMS['linear_decay_lr'] and epoch == PARAMS['linear_decay_lr'] + 1:
                self.setup_linear_decay_lr()
                self.decay = True

            # Epoch Train Loop
            for x, y in dist_train_gen:
                if PARAMS['generator_to_discriminator_steps'] == 0:
                    batch_losses = self.distributed_train_step(x, y)[0]
                else:
                    if step % (PARAMS['generator_to_discriminator_steps'] + 1) == 0:
                        batch_losses = self.distributed_train_disc_step(x, y)[0]
                    else:
                        batch_losses = self.distributed_train_gen_step(x, y)[0]

                for key,value in batch_losses.items():
                    epoch_losses[key].append(self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, value, axis=None))

                step += 1

                # get validation stats and images on Imagenet more frequent than once per epoch
                if step % 10000 == 0 and PARAMS['dir'] == 'imagenet':
                    epoch_val_losses = dict(val_reset_epoch_losses)
                    for x, y in dist_val_gen:
                        val_losses = self.distributed_val_step(x, y)[0]
                        for key,value in val_losses.items(): 
                            epoch_val_losses[key].append(self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, value, axis=None))
                    self.log_epoch_stats(epoch, start, batch_losses, val_losses, epoch_losses, epoch_val_losses)
                    self.display_images(epoch, train_gen, val_gen, epoch_step=img_count)
                    img_count += 1
                
            # Validation Loop
            epoch_val_losses = dict(val_reset_epoch_losses)
            for x, y in dist_val_gen:
                val_losses = self.distributed_val_step(x, y)[0]
                for key,value in val_losses.items(): 
                    epoch_val_losses[key].append(self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, value, axis=None))

            # Log / Plot stats and display images
            self.log_epoch_stats(epoch, start, batch_losses, val_losses, epoch_losses, epoch_val_losses)
            if epoch % 5 == 0 or PARAMS['dir'] == 'imagenet': 
                self.display_images(epoch, train_gen, val_gen)
                self.plot(epoch)

            # Save models
            if PARAMS['save_models']:
                self.save_models(epoch)

            # Decay learning rate
            if self.decay:
                with self.mirrored_strategy.scope():
                    self.linear_decay_lr()

        # generate_latent_images(G, F, val_gen, PARAMS['chpt'], PARAMS['batch_size'], PARAMS['epochs'])
        
        # Test loop
        test_losses = dict(val_reset_epoch_losses)
        for x, y in dist_test_gen:
            step_losses = self.distributed_val_step(x, y)[0]
            for key,value in step_losses.items(): 
                test_losses[key].append(self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, value, axis=None))

        self.log_epoch_stats(None, None, None, step_losses, None, test_losses)
        self.display_test_images(0, test_gen)


# Distributed functions to compile losses across GPUs
    @tf.function
    def distributed_train_gen_step(self, x, y):
        per_replica_losses = self.mirrored_strategy.run(self.train_two_gen_gen_step, args=(x,y))
        replica_losses = self.mirrored_strategy.experimental_local_results(per_replica_losses)
        return replica_losses

    @tf.function
    def distributed_train_disc_step(self, x, y):
        per_replica_losses = self.mirrored_strategy.run(self.train_two_gen_disc_step, args=(x,y))
        replica_losses = self.mirrored_strategy.experimental_local_results(per_replica_losses)
        return replica_losses

    @tf.function
    def distributed_train_step(self, x, y):
        per_replica_losses = self.mirrored_strategy.run(self.train_step, args=(x,y))
        replica_losses = self.mirrored_strategy.experimental_local_results(per_replica_losses)
        return replica_losses

    @tf.function
    def distributed_val_step(self, x, y):
        per_replica_losses = self.mirrored_strategy.run(self.val_step, args=(x,y))
        replica_losses = self.mirrored_strategy.experimental_local_results(per_replica_losses)
        return replica_losses


# Train steps: Double - Alternating Generator and Discriminator updates
    @tf.function
    def train_two_gen_gen_step(self, x, y):
        # persistent is True because tape is used more than once to calculate gradients
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fy = self.F(y, training=True)
            fy_latent = fy[1:]
            fy = fy[0]
            gx = self.G(x, training=True)
            gx_latent = gx[1:]
            gx = gx[0]
            g_fy = self.G(fy, training=True)
            g_fy_latent = g_fy[1:]
            g_fy = g_fy[0]
            f_gx = self.F(gx, training=True)
            f_gx_latent = f_gx[1:]
            f_gx = f_gx[0]
            
            dx_x = self.Dx(x, training=True)
            dx_fy = self.Dx(fy, training=True)
            dy_gx = self.Dy(gx, training=True)
            dy_y = self.Dy(y, training=True)

            # Discriminator losses
            dy_loss = dist_ls_gan_loss_disc(dy_y, dy_gx)
            dx_loss = dist_ls_gan_loss_disc(dx_x, dx_fy)

            # Generator losses
            g_loss = dist_ls_gan_loss_gen(dy_gx)
            g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l)
            g_rec_loss = reconstruction_loss(y, gx, PARAMS['rec_loss'])

            f_loss = dist_ls_gan_loss_gen(dx_fy) 
            f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l)
            f_rec_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])

            total_f_loss = f_loss + self.b * (f_rec_loss + f_cycle_loss + self.lb*g_cycle_loss)
            total_g_loss = g_loss + self.b * (g_rec_loss + g_cycle_loss + self.lb*f_cycle_loss)

        # Must be outside of 'with gradientTape'
        # Calculate gradients
        f_gradients = tape.gradient(total_f_loss, self.F.trainable_variables)
        g_gradients = tape.gradient(total_g_loss, self.G.trainable_variables)

        # Apply gradients
        self.g_optimizer.apply_gradients(zip(g_gradients, self.G.trainable_variables))
        self.f_optimizer.apply_gradients(zip(f_gradients, self.F.trainable_variables))

        f_ssim, f_psnr = dist_compute_metrics(x, fy)
        
        return {'dx_loss': dx_loss, 'dy_loss': dy_loss, 'f_loss': f_loss, 'f_rec_loss': f_rec_loss, 'f_cycle': f_cycle_loss, 'f_ssim': f_ssim, 'f_psnr': f_psnr, 'g_loss': g_loss, 'g_rec_loss': g_rec_loss, 'g_cycle': g_cycle_loss}

    @tf.function
    def train_two_gen_disc_step(self, x, y):
        # persistent is True because tape is used more than once to calculate gradients
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fy = self.F(y, training=True)
            fy_latent = fy[1:]
            fy = fy[0]
            gx = self.G(x, training=True)
            gx_latent = gx[1:]
            gx = gx[0]
            g_fy = self.G(fy, training=True)
            g_fy_latent = g_fy[1:]
            g_fy = g_fy[0]
            f_gx = self.F(gx, training=True)
            f_gx_latent = f_gx[1:]
            f_gx = f_gx[0]
            
            dx_x = self.Dx(x, training=True)
            dx_fy = self.Dx(fy, training=True)
            dy_gx = self.Dy(gx, training=True)
            dy_y = self.Dy(y, training=True)

            # Discriminator losses
            dy_loss = dist_ls_gan_loss_disc(dy_y, dy_gx)
            dx_loss = dist_ls_gan_loss_disc(dx_x, dx_fy)

            # Generator losses
            g_loss = dist_ls_gan_loss_gen(dy_gx)
            g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l)
            g_rec_loss = reconstruction_loss(y, gx, PARAMS['rec_loss'])

            f_loss = dist_ls_gan_loss_gen(dx_fy) 
            f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l)
            f_rec_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])

        # Must be outside of 'with gradientTape'
        # Calculate gradients
        dx_gradients = tape.gradient(dx_loss, self.Dx.trainable_variables)
        dy_gradients = tape.gradient(dy_loss, self.Dy.trainable_variables)

        # Apply gradients
        self.dx_optimizer.apply_gradients(zip(dx_gradients, self.Dx.trainable_variables))
        self.dy_optimizer.apply_gradients(zip(dy_gradients, self.Dy.trainable_variables))

        f_ssim, f_psnr = dist_compute_metrics(x, fy)
        
        return {'dx_loss': dx_loss, 'dy_loss': dy_loss, 'f_loss': f_loss, 'f_rec_loss': f_rec_loss, 'f_cycle': f_cycle_loss, 
                'f_ssim': f_ssim, 'f_psnr': f_psnr, 'g_loss': g_loss, 'g_rec_loss': g_rec_loss, 'g_cycle': g_cycle_loss}


# Train step: Double - Generator and Discriminator update same step
    @tf.function
    def train_two_gen_step(self, x, y):
            # persistent is True because tape is used more than once to calculate gradients
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fy = self.F(y, training=True)
            fy_latent = fy[1:]
            fy = fy[0]
            gx = self.G(x, training=True)
            gx_latent = gx[1:]
            gx = gx[0]
            g_fy = self.G(fy, training=True)
            g_fy_latent = g_fy[1:]
            g_fy = g_fy[0]
            f_gx = self.F(gx, training=True)[0]
            
            dx_x = self.Dx(x, training=True)
            dx_fy = self.Dx(fy, training=True)
            dy_gx = self.Dy(gx, training=True)
            dy_y = self.Dy(y, training=True)

            # Discriminator losses
            dy_loss = dist_ls_gan_loss_disc(dy_y, dy_gx)
            dx_loss = dist_ls_gan_loss_disc(dx_x, dx_fy)

            # Generator losses
            g_loss = dist_ls_gan_loss_gen(dy_gx)
            g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l)
            g_rec_loss = reconstruction_loss(y, gx, PARAMS['rec_loss'])

            f_loss = dist_ls_gan_loss_gen(dx_fy) 
            f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l)
            # f_cycle_perceptual_loss = dist_cycle_perceptual_loss(gx_latent, fy_latent, self.g)
            # f_cycle_perceptual_loss = dist_cycle_perceptual_loss(gx_latent, g_fy_latent, self.g)
            f_rec_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])
            # f_rec_loss = dist_vgg_perceptual_loss(x, fy)

            total_f_loss = f_loss + self.b * (f_rec_loss + f_cycle_loss + self.lb*g_cycle_loss)
            total_g_loss = g_loss + self.b * (g_rec_loss + g_cycle_loss + self.lb*f_cycle_loss)
            # total_f_loss = f_loss + self.b * (f_cycle_loss + self.lb*g_cycle_loss)
            # total_g_loss = g_loss + self.b * (g_cycle_loss + self.lb*f_cycle_loss)

        # Must be outside of 'with gradientTape'
        # Calculate gradients
        f_gradients = tape.gradient(total_f_loss, self.F.trainable_variables)
        g_gradients = tape.gradient(total_g_loss, self.G.trainable_variables)
        dx_gradients = tape.gradient(dx_loss, self.Dx.trainable_variables)
        dy_gradients = tape.gradient(dy_loss, self.Dy.trainable_variables)

        # Apply gradients
        self.g_optimizer.apply_gradients(zip(g_gradients, self.G.trainable_variables))
        self.f_optimizer.apply_gradients(zip(f_gradients, self.F.trainable_variables))
        self.dx_optimizer.apply_gradients(zip(dx_gradients, self.Dx.trainable_variables))
        self.dy_optimizer.apply_gradients(zip(dy_gradients, self.Dy.trainable_variables))

        f_ssim, f_psnr = dist_compute_metrics(x, fy)
        
        return {'dx_loss': dx_loss, 'dy_loss': dy_loss, 'f_loss': f_loss, 'f_rec_loss': f_rec_loss, 'f_cycle': f_cycle_loss, 'f_ssim': f_ssim, 'f_psnr': f_psnr, 'g_loss': g_loss, 'g_rec_loss': g_rec_loss, 'g_cycle': g_cycle_loss}
        # return {'dx_loss': dx_loss, 'dy_loss': dy_loss, 'f_loss': f_loss, 'f_rec_loss': f_rec_loss, 'f_cycle': f_cycle_loss, 'f_cycle_perceptual': f_cycle_perceptual_loss, 'f_ssim': f_ssim, 'f_psnr': f_psnr, 'g_loss': g_loss, 'g_rec_loss': g_rec_loss, 'g_cycle': g_cycle_loss}
        # return {'dx_loss': dx_loss, 'dy_loss': dy_loss, 'f_loss': f_loss, 'f_rec_loss': f_rec_loss, 'f_cycle': f_cycle_loss, 'f_latent': f_latent_loss, 'f_ssim': f_ssim, 'f_psnr': f_psnr, 'g_loss': g_loss, 'g_rec_loss': g_rec_loss, 'g_cycle': g_cycle_loss}

# Validation step: Double
    @tf.function
    def validate_two_gen_step(self, x, y):
        fy = self.F(y, training=True)[0]
        gx = self.G(x, training=True)[0]
        f_gx = self.F(gx, training=True)[0]
        g_fy = self.G(fy, training=True)[0]

        f_loss = reconstruction_loss(x, fy, dist_mae_loss)
        f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l) 

        g_loss = reconstruction_loss(y, gx, dist_mae_loss)
        g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l) 

        f_ssim, f_psnr = dist_compute_metrics(x, fy)

        return {'val_f_loss': f_loss, 'val_f_cycle': f_cycle_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr, 'val_g_loss': g_loss, 'val_g_cycle': g_cycle_loss}


# Steps: Double_no_disc
    @tf.function
    def train_two_gen_no_disc_step(self, x, y):
        # persistent is True because tape is used more than once to calculate gradients
        # with tf.GradientTape(persistent=True) as tape:
        with tf.GradientTape() as g_tape, tf.GradientTape() as f_tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fy = self.F(y, training=True)
            fy_latent = fy[1:]
            fy = fy[0]
            gx = self.G(x, training=True)
            gx_latent = gx[1:]
            gx = gx[0]
            g_fy = self.G(fy, training=True)
            g_fy_latent = g_fy[1:]
            g_fy = g_fy[0]
            f_gx = self.F(gx, training=True)[0]

            g_loss = reconstruction_loss(y, gx, PARAMS['rec_loss'])
            g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l) 
            
            f_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])
            f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l) 
            # f_latent_loss = dist_vgg_perceptual_loss(x, fy) * self.g
            # f_latent_loss = dist_latent_loss(gx_latent, fy_latent, self.g)
            # f_cycle_perceptual_loss = dist_cycle_perceptual_loss(gx_latent, g_fy_latent, self.g)
            
            total_g_loss = g_loss + g_cycle_loss + self.lb*f_cycle_loss
            total_f_loss = f_loss + f_cycle_loss+ self.lb*g_cycle_loss

        # Must be outside of 'with gradientTape'
        # Calculate gradients
        f_gradients = f_tape.gradient(total_f_loss, self.F.trainable_variables)
        g_gradients = g_tape.gradient(total_g_loss, self.G.trainable_variables)

        # Apply gradients
        self.g_optimizer.apply_gradients(zip(g_gradients, self.G.trainable_variables))
        self.f_optimizer.apply_gradients(zip(f_gradients, self.F.trainable_variables))

        # Calculate metrics
        f_ssim, f_psnr = dist_compute_metrics(x, fy)

        return {'f_loss': f_loss, 'f_cycle': f_cycle_loss, 'f_ssim': f_ssim, 'f_psnr': f_psnr, 'g_loss': g_loss, 'g_cycle': g_cycle_loss}

    @tf.function
    def validate_two_gen_no_disc_step(self, x, y):
        fy = self.F(y, training=True)
        fy = fy[0]
        gx = self.G(x, training=True)
        gx = gx[0]
        f_gx = self.F(gx, training=True)[0]
        g_fy = self.G(fy, training=True)[0]

        f_loss = reconstruction_loss(x, fy, dist_mae_loss)
        f_cycle_loss = cycle_loss(x, f_gx, PARAMS['cycle_loss'], self.l) 

        g_loss = reconstruction_loss(y, gx, dist_mae_loss)
        g_cycle_loss = cycle_loss(y, g_fy, PARAMS['cycle_loss'], self.l) 

        f_ssim, f_psnr = dist_compute_metrics(x, fy)

        return {'val_f_loss': f_loss, 'val_f_cycle': f_cycle_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr, 'val_g_loss': g_loss, 'val_g_cycle': g_cycle_loss}


# Steps: Single_no_disc
    @tf.function
    def train_one_gen_no_disc_step(self, x, y):
        with tf.GradientTape() as f_tape:
            # Generator F translates Y -> X.
            fy = self.F(y, training=True)[0]
            f_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])
        # Calculate gradients
        f_gradients = f_tape.gradient(f_loss, self.F.trainable_variables)
        # Apply gradients
        self.f_optimizer.apply_gradients(zip(f_gradients, self.F.trainable_variables))

        f_ssim, f_psnr = dist_compute_metrics(x, fy)

        return {'f_loss': f_loss, 'f_ssim': f_ssim, 'f_psnr': f_psnr}

    @tf.function
    def validate_one_gen_no_disc_step(self, x, y):
        fy = self.F(y, training=True)[0]
        f_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])
        f_ssim, f_psnr = dist_compute_metrics(x, fy)
        return {'val_f_loss': f_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr}


# Steps: U-net
    @tf.function
    def train_unet_step(self, x, y):
        with tf.GradientTape() as f_tape:
            # Generator F translates Y -> X.
            fy = self.F(y, training=True)
            f_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])
        # Calculate gradients
        f_gradients = f_tape.gradient(f_loss, self.F.trainable_variables)
        # Apply gradients
        self.f_optimizer.apply_gradients(zip(f_gradients, self.F.trainable_variables))

        f_ssim, f_psnr = dist_compute_metrics(x, fy)
        return {'f_loss': f_loss, 'f_ssim': f_ssim, 'f_psnr': f_psnr}

    @tf.function
    def validate_unet_step(self, x, y):
        fy = self.F(y, training=False)
        f_loss = reconstruction_loss(x, fy, PARAMS['rec_loss'])
        f_ssim, f_psnr = dist_compute_metrics(x, fy)
        return {'val_f_loss': f_loss, 'val_f_ssim': f_ssim, 'val_f_psnr': f_psnr}


def main():
    train = Train()
    train.train()


if __name__ == '__main__':
    main()