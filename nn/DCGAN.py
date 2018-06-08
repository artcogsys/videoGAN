"""
__author__: Florian Mahner
__email__: fmahner@uos.de
__status__: Development
__date__: 11-05-2018

This file defines the general architecture or skeleton of the Deep Convolutional Generative Adversarial Network for 3D
Video generation.
"""

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import cuda


def convolution3d(in_channel, out_channel, weights, k_size=(4,4,4), stride=(2,2,2), pad=1, direction='forward'):
    """ Convolutions - A convolutional block consists of: (n_dim, in_channel, out_channel, ksize, stride, pad).
    n_dim is set to 3 for 3D videos. Depending on the direction, each convolution propagates from the dimension of
    the input channel to the output channel"""
    if direction == 'forward':
        return L.ConvolutionND(3, in_channel, out_channel, k_size, stride, pad=pad, initialW=weights)
    elif direction == 'backward':
        return L.DeconvolutionND(3, in_channel, out_channel, k_size, stride, pad=pad, initialW=weights)
    else:
        raise NameError('Wrong direction specified, please choose between {forward, backward}')


class VideoGAN(chainer.Chain):
    """ Superclass that holds shared attributes and parameters for the Discriminator and Generator """
    def __init__(self, **kwargs):
        super(VideoGAN, self).__init__()
        self._batch_size = kwargs.get('batch_size', 64)
        self._latent_dim = kwargs.get('latent_dim', 100)
        self._n_frames = kwargs.get('n_frames', 32)
        self._weight_scale = kwargs.get('weight_scale', .001)
        self._frame_size = kwargs.get('frame_size', 64)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """ Setter method to allow sampling of single video from generator with ongoing training iterations. Without
            changing the batch_size, the generator will always produce the same number of videos as initialized with the
            batch size during training, which is redundant for visualization """
        self._batch_size = value

    @property
    def n_frames(self):
        """ Getter method for number of frames"""
        return self._n_frames


class Discriminator(VideoGAN):
    """ The discriminator network consists of five convolutional layers. The first convolutional layer has three input
    dimensions (RGB) and propagates forward to the given intial number of feature channels in the first hidden layer.
    With each subsequent convolution the amount of channels is doubled (512 in last hidden layer).
    The last layer is a fully connected linear layer. Due to the Wasserstein constraint, batch normalization is
    ommited. The discriminator is in this case a critic network, since it is not trained to classify but only to
    give good gradient information onto the generator. Therefore using softmax to connect to acutal choice probabilities
    is also omitted.
    See http://carlvondrick.com/tinyvideo/discriminator.png for a visualization of tensor sizes with increasing layer
    order."""

    def __init__(self, ch = 64, **kwargs):
        # Initial channel is the number of feature channels in the first convolutional layer.
        self.ch = ch
        super(Discriminator, self).__init__(**kwargs)

        # Initialize weights with HeNormal Distribution and scale inherited from superclass
        self.w = chainer.initializers.HeNormal(self._weight_scale)

        with self.init_scope():

            # Convolutions. A kernel size of 4, stride 2 and pad 1 is used for forward propagation. The padding is
            # required to fit dimensionality. Propagate from 64 to 128 to 256 to 512 channels.
            self.conv1 = convolution3d(3, ch, self.w)
            self.conv2 = convolution3d(ch, ch * 2, self.w)
            self.conv3 = convolution3d(ch * 2, ch * 4, self.w)
            self.conv4 = convolution3d(ch * 4, ch * 8, self.w)

            # Convolute from 4x4x2 to 1x1x1 tensor in the last layer by changing the ksize and removing the padding
            self.conv5 = convolution3d(ch * 8, 1, self.w, k_size=(4,4,2), pad=0)
            self.linear5 = L.Linear(1, 1, initialW=self.w)

    def __call__(self, input):
        """ Procedure: Convolute forward from each hidden unit amd then activate (no pooling!).
        
        input shape: (batch size, 3(RGB), 64, 64, 32)
        return shape: (batch size, 1)
        """
        hidden1 = F.leaky_relu(self.conv1(input))
        hidden2 = F.leaky_relu(self.conv2(hidden1))
        hidden3 = F.leaky_relu(self.conv3(hidden2))
        hidden4 = F.leaky_relu(self.conv4(hidden3))
        hidden5 = F.leaky_relu(self.conv5(hidden4))
        return self.linear5(hidden5)


class Generator(VideoGAN):
    """ A video generator linearly upsamples from a n-dimensional latent space onto the first layer. A series of
    fractionally stride convolutions or backward convolutions and then produces a video directly from the noise input.
    For a visualization, please see: http://carlvondrick.com/tinyvideo/network.png.
    """

    def __init__(self, ch = 512, **kwargs):
        self.ch = ch
        super(Generator, self).__init__(**kwargs)
        self.w = chainer.initializers.HeNormal(self._weight_scale)
        self._up_sample_dim = tuple([ch, 4, 4, 2])

        with self.init_scope():

            # Linear Block. Linearly up sample latent space to first hidden layer in four-dimensional space producing
            # a feature space of size (512, 4, 4, 2).
            self.linear1 = L.Linear(self._latent_dim, np.prod(self._up_sample_dim), initialW = self.w)

            # Convolutions. Perform a series of four fractionally-strided convolutions which map to 256,128,64,3
            # feature channels, producing of video of size (64,64,32,3)
            self.deconv1 = convolution3d(ch, ch // 2, weights = self.w, direction = 'backward', pad=1)
            self.deconv2 = convolution3d(ch // 2, ch // 4, weights = self.w, direction = 'backward', pad=1)
            self.deconv3 = convolution3d(ch // 4, ch // 8, weights = self.w, direction = 'backward', pad=1)
            self.deconv4 = convolution3d(ch // 8, 3, weights = self.w, direction = 'backward', pad=1)

            # Batch normalizations for all layers except last one
            self.batch_norm1 = L.BatchNormalization(np.prod(self._up_sample_dim))
            self.batch_norm2 = L.BatchNormalization(ch // 2)
            self.batch_norm3 = L.BatchNormalization(ch // 4)
            self.batch_norm4 = L.BatchNormalization(ch // 8)

    def __call__(self, latent_z):
        """
            input shape: (batch size, 100)
            return shape: (batch size, 3, 64, 64, 32) - A full video is produced from a 1D noise input!
        """
        hidden1 = F.leaky_relu(self.batch_norm1(self.linear1(latent_z)))
        hidden1 = F.reshape(hidden1, tuple([self._batch_size]) + self._up_sample_dim)
        hidden2 = F.leaky_relu(self.batch_norm2(self.deconv1(hidden1)))
        hidden3 = F.leaky_relu(self.batch_norm3(self.deconv2(hidden2)))
        hidden4 = F.leaky_relu(self.batch_norm4(self.deconv3(hidden3)))

        # Use hyperbolic tanget function in last layer to normalize generated vids from -1 to 1 (same as training vids)
        # Prior to activation, pass from 64 onto three feature channels
        return F.tanh(self.deconv4(hidden4))

    def sample_hidden(self):
        """ Sample latent space from a spherical uniform distribution.
        For details please see: https://arxiv.org/pdf/1609.04468.pdf """
        z_layer = np.random.uniform(-1, 1, (self._batch_size, self._latent_dim)).astype(np.float32)

        # Normalize z-layer manually (identical to :func:`~chainer.functions.normalization.l2_normalization.forward()
        # or F.normalize()) to avoid Variable conversion
        norm = np.sqrt(np.sum(np.square(z_layer), axis=1, keepdims=True)) + 1e-5

        return z_layer / norm



