import shutil
import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import skimage
from skimage.io import imread #, imsave
from skimage.transform import resize
import scipy.io
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from Losses import gram_matrix



class DataSetup:
    def __init__(self, path, batch_size, size, type):
        self.data_path = path
        self.batch_size = batch_size
        self.image_size = size
        self.type = type

    def __get_cifar_img(self, img_data):
        img_channel_len = 1024
        red = np.array(img_data)[:img_channel_len].reshape(32,32,1)
        green = np.array(img_data)[img_channel_len:2*img_channel_len].reshape(32,32,1)
        blue = np.array(img_data)[2*img_channel_len:].reshape(32,32,1)
        return np.concatenate([red, green, blue], axis=-1)

    def __get_cifar_10_imgs(self, t=1):
        if t == 1:
            lensless_path = './datasets/lensless-cifar/20x20-cifar10/'
        elif t == 10:
            lensless_path = './datasets/lensless-cifar/200x200-10mm-cifar10-2/'
        cifar_10_path = './datasets/lensless-cifar/cifar10/data_batch_'
        cifar_10_batches = [cifar_10_path + str(i) for i in range(1,6)] + ['./datasets/lensless-cifar/cifar10/test_batch']

        x, y = [], []
        for batch_path in cifar_10_batches:
            if 'test_batch' in batch_path: lensless_path = lensless_path + 'test-'
            with open(batch_path, 'rb') as fo: 
                batch_dict = pickle.load(fo, encoding='bytes')

            for i in range(len(batch_dict[b'filenames'])):
                x.append(self.__get_cifar_img(batch_dict[b'data'][i]))
                lensless_file = lensless_path + batch_dict[b'filenames'][i].decode("utf-8")[:-3] + 'jpeg'
                y.append(lensless_file)
        
        return np.array(x), y

    def __get_imagenet_imgs(self):
        lensless_path = './datasets/ILSVRC2012/ILSVRC2012_img_train_optics-free/'
        gt_path = './datasets/ILSVRC2012/ILSVRC2012_img_train_224x224/'
        x, val_x, test_x, y, val_y, test_y = [], [], [], [], [], []
        for dir in os.listdir(gt_path):
            if '.DS_Store' in dir: continue
            val_count, test_count = 0,0
            for file in os.listdir(gt_path + dir):
                if '.DS_Store' in file: continue
                if val_count <= 8: # 48:
                    val_x.append(gt_path + dir + '/' + file)
                    val_y.append(lensless_path + dir + '/' + file)
                    val_count += 1
                elif test_count <= 32:
                    test_x.append(gt_path + dir + '/' + file)
                    test_y.append(lensless_path + dir + '/' + file)
                    test_count += 1
                else:
                    x.append(gt_path + dir + '/' + file)
                    y.append(lensless_path + dir + '/' + file)
        # Must be multiple of batch size for multi-gpu
        excess_fat = len(x) % self.batch_size
        if excess_fat > 0:
            x = x[:-excess_fat]
            y = y[:-excess_fat]

        # Shuffle Train
        p = np.random.permutation(len(x))
        x, y = np.array(x)[p], np.array(y)[p]

        # Shuffle Validation
        p = np.random.permutation(len(val_x))
        val_x, val_y = np.array(val_x)[p], np.array(val_y)[p]

        # Shuffle Test
        p = np.random.permutation(len(test_x))
        test_x, test_y = np.array(test_x)[p], np.array(test_y)[p]

        return x, val_x, test_x, y, val_y, test_y


    def setup_datasets(self, shuffle=True):
        # Gather images and set up preprocessing function based on self.data_path
        if self.data_path == 'lensless-cifar':
            x, y = self.__get_cifar_10_imgs()

            train_x = x[:45000,:,:,:]
            val_x = x[45000:50000,:,:,:]
            test_x = x[50000:,:,:,:]
            train_y = y[:45000]
            val_y = y[45000:50000]
            test_y = y[50000:]
            
            if self.type == 'unet':
                preprocess = preprocess_lensless_cifar_unet
            elif self.image_size == (180,240):
                preprocess = preprocess_lensless_cifar_resized_normalize_255
            elif self.image_size == (120,160):
                preprocess = preprocess_lensless_cifar_resized_120x160_normalize_255
            else:
                preprocess = preprocess_lensless_cifar

        elif '200x200-10mm-cifar10' in self.data_path:
            x, y = self.__get_cifar_10_imgs(10)

            train_x = x[:45000,:,:,:]
            val_x = x[45000:50000,:,:,:]
            test_x = x[50000:,:,:,:]
            train_y = y[:45000]
            val_y = y[45000:50000]
            test_y = y[50000:]
            
            if self.type == 'unet':
                preprocess = preprocess_lensless_cifar_unet
            elif self.image_size == (180,240):
                preprocess = preprocess_lensless_cifar_resized_normalize_255
            elif self.image_size == (120,160):
                preprocess = preprocess_lensless_cifar_resized_120x160_normalize_255
            else:
                preprocess = preprocess_lensless_cifar

        elif self.data_path == 'combined-lensless-cifar':
            x_1mm, y_1mm = self.__get_cifar_10_imgs()
            x_10mm, y_10mm = self.__get_cifar_10_imgs(10)

            # make datasets every other image
            train_idx = 0
            train_x, train_y = [], []
            for i in range(90000):
                if i % 2 == 0:
                    train_x.append(x_1mm[train_idx])
                    train_y.append(y_1mm[train_idx])
                else:
                    train_x.append(x_10mm[train_idx])
                    train_y.append(y_10mm[train_idx])
                    train_idx += 1
            train_x, train_y = np.array(train_x), train_y

            val_idx = 45000
            val_x, val_y = [], []
            for i in range(10000):
                if i % 2 == 0:
                    val_x.append(x_1mm[val_idx])
                    val_y.append(y_1mm[val_idx])
                else:
                    val_x.append(x_10mm[val_idx])
                    val_y.append(y_10mm[val_idx])
                    val_idx += 1
            val_x, val_y = np.array(val_x), val_y

            test_idx = 50000
            test_x, test_y = [], []
            for i in range(20000):
                if i % 2 == 0:
                    test_x.append(x_1mm[test_idx])
                    test_y.append(y_1mm[test_idx])
                else:
                    test_x.append(x_10mm[test_idx])
                    test_y.append(y_10mm[test_idx])
                    test_idx += 1
            test_x, test_y = np.array(test_x), test_y
            
            if self.type == 'unet':
                preprocess = preprocess_lensless_cifar_unet
            elif self.image_size == (180,240):
                preprocess = preprocess_lensless_cifar_resized_normalize_255

        elif 'hamrick' in self.data_path:
            x = np.array([imread('./datasets/Hamrick/expanded_64_gt/example' + str(i) + '.png') for i in range(16585)])
            y = np.array([imread('./datasets/Hamrick/expanded_64_cap/example' + str(i) + '.png') for i in range(16585)])
            p = np.random.permutation(len(x))
            x, y = x[p], y[p]

            # 64x64 and 0-255
            train_x = x[:14928,:,:,:]
            val_x = x[14928:15760,:,:,:] # 5%
            test_x = x[15760:,:,:,:] # 5%

            train_y = y[:14928,:,:,:]
            val_y = y[14928:15760,:,:,:]
            test_y = y[15760:,:,:,:]

            preprocess = preprocess_hamrick

        elif self.data_path == 'imagenet':
            train_x, val_x, test_x, train_y, val_y, test_y = self.__get_imagenet_imgs()
            if self.image_size == (180,240):
                preprocess = preprocess_imagenet_resized
            else:
                preprocess = preprocess_imagenet

        # Setup TF datasets
        train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        if shuffle:
            train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(3200).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        else:
            train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return train_ds, val_ds, test_ds



def preprocess_lensless_cifar(x, y):
    y = tf.io.read_file(y)
    y = tf.image.decode_jpeg(y, channels=3)
    y = tf.cast(y, tf.float32)
    y = normalize_255(y)

    x = tf.cast(x, tf.float32)
    x = normalize_255(x)

    ##### NCHW
    x = tf.transpose(x, [2, 0, 1])
    y = tf.transpose(y, [2, 0, 1])
    #####

    return x, y

def preprocess_lensless_cifar_resized_normalize_255(x, y):
    y = tf.io.read_file(y)
    y = tf.image.decode_jpeg(y, channels=3)
    y = tf.image.resize(y, [180, 240])
    y = tf.cast(y, tf.float32)
    y = normalize_255(y)

    x = tf.cast(x, tf.float32)
    x = normalize_255(x)

    # x, y = random_flip(x,y)

    ##### NCHW
    x = tf.transpose(x, [2, 0, 1])
    y = tf.transpose(y, [2, 0, 1])
    return x, y

def preprocess_lensless_cifar_resized_120x160_normalize_255(x, y):
    y = tf.io.read_file(y)
    y = tf.image.decode_jpeg(y, channels=3)
    y = tf.image.resize(y, [120, 160])
    y = tf.cast(y, tf.float32)
    y = normalize_255(y)

    x = tf.cast(x, tf.float32)
    x = normalize_255(x)

    ##### NCHW
    x = tf.transpose(x, [2, 0, 1])
    y = tf.transpose(y, [2, 0, 1])
    return x, y

def preprocess_lensless_cifar_resized(x, y):
    y = tf.io.read_file(y)
    y = tf.image.decode_jpeg(y, channels=3)
    y = tf.image.resize(y, [180, 240])
    y = tf.cast(y, tf.float32)
    y = normalize_max(y)

    x = tf.cast(x, tf.float32)
    x = normalize_255(x)

    # x, y = random_flip(x,y)

    ##### NCHW
    x = tf.transpose(x, [2, 0, 1])
    y = tf.transpose(y, [2, 0, 1])
    return x, y

def preprocess_lensless_cifar_unet(x, y):
    y = tf.io.read_file(y)
    y = tf.image.decode_jpeg(y, channels=3)
    y = tf.cast(y, tf.float32)
    y = normalize_255(y)
    
    # x = tf.image.resize(x, [240, 320]) # U-net
    x = tf.cast(x, tf.float32)
    x = normalize_255(x)

    ##### NCHW
    x = tf.transpose(x, [2, 0, 1])
    y = tf.transpose(y, [2, 0, 1])
    return x, y

def preprocess_hamrick(x,y):
    y = tf.cast(y, tf.float32)
    y = normalize_255(y)
    
    x = tf.cast(x, tf.float32)
    x = normalize_255(x)

    ##### NCHW
    x = tf.transpose(x, [2, 0, 1])
    y = tf.transpose(y, [2, 0, 1])
    #####

    return x, y

def preprocess_imagenet(x, y):
    y = tf.io.read_file(y)
    y = tf.image.decode_jpeg(y, channels=3)
    y = tf.cast(y, tf.float32)
    y = normalize_255(y)

    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.cast(x, tf.float32)
    x = normalize_255(x)

    ##### NCHW
    x = tf.transpose(x, [2, 0, 1])
    y = tf.transpose(y, [2, 0, 1])
    #####
    return x, y

def preprocess_imagenet_resized(x, y):
    y = tf.io.read_file(y)
    y = tf.image.decode_jpeg(y, channels=3)
    y = tf.image.resize(y, [180, 240])
    y = tf.cast(y, tf.float32)
    y = normalize_255(y)

    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize(x, [32, 32])
    x = tf.cast(x, tf.float32)
    x = normalize_255(x)

    ##### NCHW
    x = tf.transpose(x, [2, 0, 1])
    y = tf.transpose(y, [2, 0, 1])
    #####
    return x, y

def normalize_max(x):
    return x / tf.reduce_max(x)

def normalize_255(x):
    return x / 255.



def generate_all_images(G, F, train_gen, val_gen, test_gen, chpt, batch_size):
    gen_strs = ['train', 'val', 'test']
    gens = [train_gen, val_gen, test_gen]
    for i in range(len(gens)):
        os.mkdir('./figures/' + str(chpt) + '/' + gen_strs[i])
        for dir in ['x', 'y', 'f_y', 'g_x']: os.mkdir('./figures/' + str(chpt) + '/' + gen_strs[i] + '/' + dir)

        displayed = 0
        for x, y in gens[i]:
            f_pred = F(y, training=True)[0]
            g_pred = G(x, training=True)[0]

            ##### NCHW
            x = tf.transpose(x, [0, 2, 3, 1])
            y = tf.transpose(y, [0, 2, 3, 1])
            f_pred = tf.transpose(f_pred, [0, 2, 3, 1])
            g_pred = tf.transpose(g_pred, [0, 2, 3, 1])
            #####


            for j in range(batch_size):
                display_list = [x[j], y[j], f_pred[j], g_pred[j]]
                dir = ['x', 'y', 'f_y', 'g_x']

                for k in range(len(display_list)):
                    img = display_list[k]
                    skimage.io.imsave('./figures/' + str(chpt) + '/' + gen_strs[i] + '/' + dir[k] + '/' + str(displayed) + '.png', tf.cast(img*255, tf.uint8))

                displayed += 1


def generate_all_images_single_gen(F, train_gen, val_gen, test_gen, chpt, batch_size):
    # Turn off any shuffling

    gen_strs = ['train', 'val', 'test']
    gens = [train_gen, val_gen, test_gen]
    for i in range(len(gens)):
        os.mkdir('./figures/' + str(chpt) + '/' + gen_strs[i])
        for dir in ['x', 'y', 'f_y']: os.mkdir('./figures/' + str(chpt) + '/' + gen_strs[i] + '/' + dir)

        displayed = 0
        for x, y in gens[i]:
            f_pred = F(y, training=True)

            ##### NCHW
            x = tf.transpose(x, [0, 2, 3, 1])
            y = tf.transpose(y, [0, 2, 3, 1])
            f_pred = tf.transpose(f_pred, [0, 2, 3, 1])
            #####


            for j in range(batch_size):
                display_list = [x[j], y[j], f_pred[j]]
                dir = ['x', 'y', 'f_y']

                for k in range(len(display_list)):
                    img = display_list[k]
                    skimage.io.imsave('./figures/' + str(chpt) + '/' + gen_strs[i] + '/' + dir[k] + '/' + str(displayed) + '.png', tf.cast(img*255, tf.uint8))

                displayed += 1



def generate_images(G, F, data_gen, chpt, epoch, batch_size, num=1, num_channels=3, train_str='train', latent=False, plot_filter=None, epoch_step=None):
    displayed = 0
    for x, y in data_gen:
        if plot_filter == 'high_pass':
            filter_y = y - tfa.image.gaussian_filter2d(y, 3)
            filter_x = x - tfa.image.gaussian_filter2d(x, 3)
        elif plot_filter == 'fft':
            filter_y = tf.signal.fft(tf.cast(y, tf.complex64))
            filter_x = tf.signal.fft(tf.cast(x, tf.complex64))
        else:
            filter_y = y
            filter_x = x
        f_prediction = F(filter_y, training=True)[0]
        g_prediction = G(filter_x, training=True)[0]

        if plot_filter == 'high_pass':
            filter_f_pred = f_prediction - tfa.image.gaussian_filter2d(f_prediction, 3)
            filter_g_pred = g_prediction - tfa.image.gaussian_filter2d(g_prediction, 3)
        elif plot_filter == 'fft':
            filter_f_pred = tf.signal.fft(tf.cast(f_prediction, tf.complex64))
            filter_g_pred = tf.signal.fft(tf.cast(g_prediction, tf.complex64))
        else:
            filter_f_pred = f_prediction
            filter_g_pred = g_prediction
        f_g_x_prediction = F(filter_g_pred, training=True)[0]
        g_f_y_prediction = G(filter_f_pred, training=True)[0]

        ##### NCHW
        x = tf.transpose(x, [0, 2, 3, 1])
        y = tf.transpose(y, [0, 2, 3, 1])
        f_prediction = tf.transpose(f_prediction, [0, 2, 3, 1])
        g_prediction = tf.transpose(g_prediction, [0, 2, 3, 1])
        f_g_x_prediction = tf.transpose(f_g_x_prediction, [0, 2, 3, 1])
        g_f_y_prediction = tf.transpose(g_f_y_prediction, [0, 2, 3, 1])
        #####

        for j in range(batch_size):
            plt.figure(figsize=(12, 12))

            display_list = [x[j], y[j], f_prediction[j], g_prediction[j], f_g_x_prediction[j], g_f_y_prediction[j]]
            title = ['Ground Truth', 'Input Image', 'F Predicted Image', 'G Predicted Image', 'F Cycled Image', 'G Cycled Image']
            fig_title = ['x', 'y', 'f_y', 'g_x', 'f_g_x', 'g_f_y']

            for i in range(6):
                plt.subplot(1, 6, i+1)
                plt.title(title[i])
                if epoch_step is not None:
                    fp = './figures/' + str(chpt) + '/epoch_' + str(epoch) + '-' + str(epoch_step) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png'
                else:
                    fp = './figures/' + str(chpt) + '/epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png'
                if num_channels == 1:
                    img = np.squeeze(display_list[i])
                    imsave(fp, img)
                elif 'pre' in train_str:
                    img = display_list[i]
                    skimage.io.imsave('./figures/' + str(chpt) + '/pre-epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', tf.cast(img*255, tf.uint8))
                else:
                    img = display_list[i]
                    skimage.io.imsave(fp, tf.cast(img*255, tf.uint8))
                plt.imshow(img) # * 0.5 + 0.5) # tanh loss
                plt.axis('off')
            if epoch_step is not None: 
                fp = './figures/' + str(chpt) + '/epoch_' + str(epoch) + '-' + str(epoch_step) + '-' + train_str + '-' + str(displayed)
            else:
                fp = './figures/' + str(chpt) + '/epoch_' + str(epoch) + '-' + train_str + '-' + str(displayed)
            plt.savefig(fp, bbox_inches='tight', pad_inches=0)
            displayed += 1
            if displayed == num: return


def generate_images_single_gen(F, data_gen, chpt, epoch, batch_size, num=1, num_channels=3, train_str='train', latent=False):
    displayed = 0
    for x, y in data_gen:
        # f_prediction = F(tf.signal.fft(tf.cast(y, tf.complex64)))
        f_prediction = F(y, training=True)
        if latent: f_prediction = f_prediction[0]

        ##### NCHW
        x = tf.transpose(x, [0, 2, 3, 1])
        y = tf.transpose(y, [0, 2, 3, 1])
        f_prediction = tf.transpose(f_prediction, [0, 2, 3, 1])
        #####

        for j in range(batch_size):
            plt.figure(figsize=(12, 12))

            display_list = [x[j], y[j], f_prediction[j]]
            title = ['Ground Truth', 'Input Image', 'F Predicted Image']
            fig_title = ['x', 'y', 'f_y']

            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(title[i])
                if num_channels == 1:
                    img = np.squeeze(display_list[i])
                    imsave('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', img)
                elif 'pre' in train_str:
                    img = display_list[i]
                    skimage.io.imsave('./figures/' + str(chpt) + '/pre-epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', tf.cast(img*255, tf.uint8))
                else:
                    img = display_list[i]
                    skimage.io.imsave('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', tf.cast(img*255, tf.uint8))
                plt.imshow(img) # * 0.5 + 0.5) # tanh loss
                plt.axis('off')
            if 'pre' in train_str:
                plt.savefig('./figures/' + str(chpt) + '/pre-epoch_' + str(epoch) + '-' + train_str + '-' + str(displayed), bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '-' + train_str + '-' + str(displayed), bbox_inches='tight', pad_inches=0)
            displayed += 1
            if displayed == num: return


def generate_images_unet(F, data_gen, chpt, epoch, batch_size, num=1, num_channels=3, train_str='train', latent=False):
    displayed = 0
    for x, y in data_gen:
        f_prediction = F(y, training=False)

        ##### NCHW
        # x = tf.image.resize(tf.transpose(x, [0, 2, 3, 1]), [32, 32])
        # y = tf.transpose(y, [0, 2, 3, 1])
        # f_prediction = tf.image.resize(tf.transpose(f_prediction, [0, 2, 3, 1]), [32, 32])
        x = tf.transpose(x, [0, 2, 3, 1])
        y = tf.transpose(y, [0, 2, 3, 1])
        f_prediction = tf.transpose(f_prediction, [0, 2, 3, 1])
        #####

        for j in range(batch_size):
            plt.figure(figsize=(12, 12))

            display_list = [x[j], y[j], f_prediction[j]]
            title = ['Ground Truth', 'Input Image', 'F Predicted Image']
            fig_title = ['x', 'y', 'f_y']

            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(title[i])
                if num_channels == 1:
                    img = np.squeeze(display_list[i])
                    imsave('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', img)
                else:
                    img = display_list[i]
                    skimage.io.imsave('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', tf.cast(img*255, tf.uint8))
                plt.imshow(img) # * 0.5 + 0.5) # tanh loss
                plt.axis('off')
            plt.savefig('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '-' + train_str + '-' + str(displayed), bbox_inches='tight', pad_inches=0)
            displayed += 1
            if displayed == num: return


def generate_images_single_gen_ensemble(F, data_gen, chpt, epoch, batch_size, num=1, num_channels=3, train_str='train', latent=False):
    displayed = 0
    for x, y in data_gen:
        f_prediction = tf.zeros_like(x)
        for f in F:
            f_prediction += f(y, training=True)[0]
        f_prediction /= len(F)

        ##### NCHW
        x = tf.transpose(x, [0, 2, 3, 1])
        y = tf.transpose(y, [0, 2, 3, 1])
        f_prediction = tf.transpose(f_prediction, [0, 2, 3, 1])
        #####

        for j in range(batch_size):
            plt.figure(figsize=(12, 12))

            display_list = [x[j], y[j], f_prediction[j]]
            title = ['Ground Truth', 'Input Image', 'F Predicted Image']
            fig_title = ['x', 'y', 'f_y']

            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(title[i])
                if num_channels == 1:
                    img = np.squeeze(display_list[i])
                    imsave('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', img)
                elif 'pre' in train_str:
                    img = display_list[i]
                    skimage.io.imsave('./figures/' + str(chpt) + '/pre-epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', tf.cast(img*255, tf.uint8))
                else:
                    img = display_list[i]
                    skimage.io.imsave('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '/' + train_str + '-' + str(displayed) + '-' + fig_title[i] + '.png', tf.cast(img*255, tf.uint8))
                plt.imshow(img) # * 0.5 + 0.5) # tanh loss
                plt.axis('off')
            if 'pre' in train_str:
                plt.savefig('./figures/' + str(chpt) + '/pre-epoch_' + str(epoch) + '-' + train_str + '-' + str(displayed), bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '-' + train_str + '-' + str(displayed), bbox_inches='tight', pad_inches=0)
            displayed += 1
            if displayed == num: return


def plot(vals, names, title, y_label, chpt, epoch):
    plt.rcParams.update({'font.family': 'serif'})
    plt.figure()
    for val in vals:
        plt.plot(val)
    plt.title(title)
    plt.legend(names)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.savefig('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '-' + title + '.png')
    plt.show()




def generate_latent_images(G, F, data_gen, chpt, batch_size, epoch):
    num = 7
    displayed = 0
    for x, y in data_gen:
        f_prediction = F(y)
        g_prediction = G(x)
        
        # Content
        for j in range(batch_size):
            for i,latent_layer in enumerate(f_prediction[1:]):
                display_latent_layer('F-C', i, j, latent_layer, chpt, displayed, epoch)
            for i,latent_layer in enumerate(g_prediction[1:]):
                display_latent_layer('G-C', i, j, latent_layer, chpt, displayed, epoch)
            displayed += 1
            if displayed == num: break

        # Style
        for i,latent_layer in enumerate(f_prediction[1:]):
            displayed = 0
            gram = gram_matrix(latent_layer)
            for j in range(batch_size):
                imsave('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '-F-S-' + str(displayed) + '-' + str(i) + '.png', gram[j])
                displayed += 1
                if displayed == num: break

        for i,latent_layer in enumerate(g_prediction[1:]):
            displayed = 0
            gram = gram_matrix(latent_layer)
            for j in range(batch_size):
                imsave('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '-G-S-' + str(displayed) + '-' + str(i) + '.png', gram[j])
                displayed += 1
                if displayed == num: break
    
        
def display_latent_layer(model_str, i, j, latent_layer, chpt, displayed, epoch):
    images_per_row = 16
    ##### NCHW
    latent_layer = tf.transpose(latent_layer, [0, 2, 3, 1])

    n_features = latent_layer.shape[-1]
    h, w = latent_layer.shape[1], latent_layer.shape[2]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((w * n_cols, images_per_row * h))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = latent_layer[j,:,:,col * images_per_row + row]
            # Post-processes the feature
            channel_image -= tf.math.reduce_mean(channel_image)
            channel_image /= tf.math.reduce_std(channel_image)
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * w : (col + 1) * w, row * h : (row + 1) * h] = channel_image
    
    plt.figure(figsize=(1./w * display_grid.shape[1],
                        1./h * display_grid.shape[0]))
    plt.title(model_str + '-' + str(i))
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig('./figures/' + str(chpt) + '/epoch_' + str(epoch) + '-' + model_str + '-' + str(displayed) + '-' + str(i), bbox_inches='tight', pad_inches=0)

