"""
__author__: Florian Mahner
__email__: fmahner@uos.de
__status__: Development
__date__: 11-05-2018

This file defines the general architecture or skeleton of the Deep Convolutional Generative Adversarial Network for 3D
Video generation
"""

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


def convolution3d(in_channel, out_channel, weights, k_size=(4,4,4), stride=(2,2,2), direction='forward'):
    """ Convolutions - A convolutional block consists of: (n_dim, in_channel, out_channel, ksize, stride).
    n_dim is set to 3 for 3D videos. Depending on the direction, each convolution propagates from the dimension of
    the input channel to the output channel"""
    if direction == 'forward':
        return L.ConvolutionND(3, in_channel, out_channel, k_size, stride, initialW=weights)
    elif direction == 'backward':
        return L.DeconvolutionND(3, in_channel, out_channel, k_size, stride, initialW=weights)
    else:
        raise NameError('Wrong direction specified, please choose between {forward, backward}')


class VideoGenerator(chainer.Chain):
    """ A video generator consists of: ..."""

    def __init__(self, batch_size=64, frame_size=32, latent_dim=100, w_scale=.001):
        super(VideoGenerator, self).__init__()
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.latent_dim = latent_dim


# TODO It is important that biases are initialized with 0 - Krazwald adds trainable bias (how?)
class Discriminator(VideoGenerator):
    """ The discriminator is in this case a critic network, since it is not trained to classify but only to
    give good gradient information onto the generator. Therefore using softmax to connect to acutal choice probabilities
    is omitted """

    def __init__(self, channels=64):

        # Initial channel is the number of feature channels in the first convolutional layer.
        self.channels = channels

        # Initialize weights with HeNormal Distribution and scale inherited from <class> VideoGenerator
        self.w = chainer.initializers.HeNormal(self.weight_scale)
        super(Discriminator, self).__init__()

        with self.init_scope():

            """ Convolutions - The first convolutional layer has three input dimensions (RGB) and propagates forward to 
            the given intial number of channels in the first layer. Each convolution to higher layer doubles the amount
            of channels, such that the last convolutional layer has eight times more channels (for five layers). 
            Pass initial weights drawn from HeNormal for every convolutional layers"""
            self.conv1 = convolution3d(3, channels, self.w)
            self.conv2 = convolution3d(channels, channels * 2, self.w)
            self.conv3 = convolution3d(channels * 2, channels * 4, self.w)
            self.conv4 = convolution3d(channels * 4, channels * 8, self.w)
            self.conv5 = convolution3d(channels * 8, 1, self.w) # TODO Check if propagates to 1 or 512!

            """ Linear Block - Fully connected, takes input from last convolutional layer """
            self.linear5 = L.Linear(1, 1, initialW=self.w)

            """ Layer normalizations instead of batch normalizationm for every convolutional layer except for last one.
            For details see: Kratzwald et al."""
            self.layer_norm1 = L.LayerNormalization(channels)
            self.layer_norm2 = L.LayerNormalization(channels * 2)
            self.layer_norm3 = L.LayerNormalization(channels * 4)
            self.layer_norm4 = L.LayerNormalization(channels * 8)

    def __call__(self, input):
        """ Procedure: Convolute forward from each hidden unit, layer normalization and then activate (no pooling!).
        The return Variable from theforward computation __call_ has backward() method to compute its gradients
        afterwards!"""

        # TODO Look for which convolution layer normalization is applied and if ReLu or PReLu should be used! Should we
        # add noise on input? e.g. leakyrelu(self.conv(add_noise(x))" - PreLu available as function in chainer
        hidden1 = F.leaky_relu(self.conv1(input))
        hidden2 = F.leaky_relu(self.layer_norm1(self.conv2(hidden1)))
        hidden3 = F.leaky_relu(self.layer_norm2(self.conv3(hidden2)))
        hidden4 = F.leaky_relu(self.layer_norm3(self.conv4(hidden3)))
        hidden5 = F.leaky_relu(self.layer_norm4(self.conv5(hidden4)))
        return self.linear5(hidden5)


# TODO Check if latent_z is passed or just the dimension of it, which can be obtained from superclass!
class Generator(VideoGenerator):

    def __init__(self, latent_z, ch=512):
        super(Generator, self).__init__()
        self.latent_z = latent_z
        self.ch = ch # Amnount of feature channels in the first layer
        self.w = chainer.initializers.HeNormal(self.weight_scale)

        with self.init_scope():

            """ Linear Block. Linearly up sample latent space to first hidden layer in four-dimensional space producing
            a feature space of size (4,4,2,512) """
            up_sample = (4 * 4 * 2 * ch) # TODO Change to constants
            self.linear1 = L.Linear(latent_z, up_sample)

            """ Convolutions. Perform a series of four fractionally-strided convolutions which map to 256,128,64,and 3
            feature channels, producing a video of size(64,64,32,3)"""
            self.deconv1 = convolution3d(ch, ch // 2, weights=self.w, direction='backward') # tensor=(8,8,4,256)
            self.deconv2 = convolution3d(ch // 2, ch // 4, weights=self.w, direction='backward') # tensor=(16,16,8,128)
            self.deconv3 = convolution3d(ch // 4, ch // 8, weights=self.w, direction='backward') # tensor=(32,32,16,64)
            self.deconv4 = convolution3d(ch // 8, 3, weights=self.w, direction='backward') # tensor = (64,64,32,3)

            """ Batch normalizations (Batch for Generator, Layer for Discriminator). All but last layer!"""
            self.batch_norm1 = L.BatchNormalization(ch)
            self.batch_norm2 = L.BatchNormalization(ch // 2)
            self.batch_norm3 = L.BatchNormalization(ch // 4)
            self.batch_norm4 = L.BatchNormalization(ch // 8)

    def __call__(self):
        # TODO Check if relu activation has specific slope specifed for convergence (e.g. .02 Kratzwald)?
        hidden1 = F.leaky_relu(self.batch_norm1(self.linear1(self.latent_z)))
        hidden2 = F.leaky_relu(self.batch_norm2(self.deconv1(hidden1)))
        hidden3 = F.leaky_relu(self.batch_norm3(self.deconv2(hidden2)))
        hidden4 = F.leaky_relu(self.batch_norm4(self.deconv3(hidden3)))

        return F.tanh(hidden4) # Last layer has tanh activation instead of leaky relu!

    def sample_hidden(self):
        # TODO Implement cupy differentiation!
        """ Sample latent space from a spherical uniform distribution """
        z_layer = np.random.uniform(-1, 1, (self.batch_size, self.latent_dim)).astype(np.float32)
        return F.normalize(z_layer)
