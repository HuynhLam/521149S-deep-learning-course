from __future__ import division
import argparse, sys, os, errno, torch
import keras
from keras.callbacks import TensorBoard
import numpy as np
from scipy.stats import multivariate_normal as mvn

import tensorflow as tf
import speech_recognition as sr
import pyaudio
from sklearn.mixture import BayesianGaussianMixture
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging


def write_log(callback, names, logs, batch_no):
    '''
    Simple Tensorboard logs
    '''
    summary = tf.Summary()
    
    for name, value in zip(names, logs):
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        
    callback.writer.flush()

class Logger:
    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)

    def log(self, d_error, g_error, epoch, n_batch, num_batches):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/D_error'.format(self.comment), d_error, step)
        self.writer.add_scalar(
            '{}/G_error'.format(self.comment), g_error, step)

    def log_images(self, images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):
        '''
        input images are expected in format (NCHW)
        '''
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        
        if format=='NHWC':
            images = images.transpose(1,3)
        

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '')

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True)
        # Make vertical grid from image tensor
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(
            images, nrow=nrows, normalize=True, scale_each=True)

        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)

        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, 'hori')
        plt.close()

        # Save squared
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,
                                                         comment, epoch, n_batch))

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):
        
        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        if isinstance(d_pred_real, torch.autograd.Variable):
            d_pred_real = d_pred_real.data
        if isinstance(d_pred_fake, torch.autograd.Variable):
            d_pred_fake = d_pred_fake.data
        
        
        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch,num_epochs, n_batch, num_batches)
             )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

    def save_models(self, generator, discriminator, epoch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(),
                   '{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),
                   '{}/D_epoch_{}'.format(out_dir, epoch))

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

              

def save_images_to_disk(model, epoch, output):
        if not os.path.exists(output):
            os.makedirs(output)

        r, c = 8, 8
        plt.figure(figsize = (r, c))
        gs1 = gridspec.GridSpec(r, c, wspace=0.0, hspace=0.0)

        noise = np.random.normal(0, 1, (r * c, model.latent_dim))
        gen_imgs = model.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        cnt = 0

        for i in range(r*c):
            ax1 = plt.subplot(gs1[i])
            ax1.imshow(gen_imgs[cnt,:,:,0], cmap='gray', aspect='auto')
            ax1.axis('off')
            cnt += 1

        plt.savefig(os.path.join(output, 'mnist_{:09d}.png'.format(epoch)))
        plt.show()
            
def load_mnist():
    # Load MNIST from built-in keras
    mnist = keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    # Flatten and normalize
    X_train = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[1] * X_train.shape[2])) / 255.0
    X_test = np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[1] * X_test.shape[2])) / 255.0

    return X_train, Y_train, X_test, Y_test

def analyze_text(text):
    if 'stop' in text:
        return 'stop'
    else:
        chars = []
        for char in text:
            if char.isdigit():
                number = int(char)
                print('***** CONTAIN number {} in text {}'.format(number, text))
                chars.append(number)
                if number >= 0 and number <= 9:
                    return number
    return 'no'

def get_speech():
    print('***** YES, still running, now test the speech recognition!! {}'.format(pyaudio.pa.__file__))
    r = sr.Recognizer()

    with sr.Microphone(device_index=0) as source:
        print('PLEASE, SAY SOMETHING ..')
        audio = r.record(source, duration = 5)
    try:
        print("OK, DONE RECORED!")

        audio_text = r.recognize_google(audio, language = "en-US")

        print('TEXT : {}'.format(audio_text))
        number = analyze_text(audio_text)
        return number

    except:
        print("ERROR, CAN NOT RECOGNIZE THE SPEECH!")