"""
__author__: Florian Mahner
__email__: fmahner@uos.de
__status__: Development
__date__: 11-05-2018

This file updates the GAN network following the Wasserstein framework
For details, please see:

https://arxiv.org/abs/1701.07875 (original paper by Arjovski)
https://arxiv.org/abs/1704.00028 (follow up paper)

For mathematical explanations and loss functions for GANs in general and WGANs, please see this excellent summary:
https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html
"""

import chainer
import chainer.functions as F
from chainer import Variable
import numpy as np

class GANUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, models, iterator, optimizer, batch_size, device=0, penalty_coeff=10, critic_iter=5):
        super(GANUpdater, self).__init__(iterator, optimizer)
        self.generator, self.discriminator = models
        self.critic_iter = critic_iter
        self.penalty_coeff = penalty_coeff
        self.batch_size = batch_size
        self.iterator = iterator
        self.optimizers = optimizer
        self.device = device
        self.loss_history = {'gen': [], 'disc': []}

    def update_core(self):

        # Get all models from inherited from superclass after passing at initialization
        generator_opt = self.get_optimizer('gen-opt')
        discriminator_opt = self.get_optimizer('disc-opt')

        # The Generator is updated once every set of critic iterations and the Discriminator at each iteration
        for i in range(self.critic_iter):

            # Sample a new point from the real distribution every critic update
            videos_true = self.get_iterator('main').next()

            # Wrap training batch with chainer.variable and send to gpu using built-in converter function from updater
            videos_true = Variable(self.converter(videos_true, self.device))

            # Feed current batch into discriminator and determine if fake or real
            eval_true = self.discriminator(videos_true)

            # Feed 100-dimensional z-space into generator and produce a video
            latent_z = self.generator.sample_hidden(self.batch_size)
            videos_fake = self.generator(latent_z)

            # Feed generated image into discriminator and determine if fake or real
            eval_fake = self.discriminator(videos_fake)

            # Calculate the gradient penalty added to the loss function by enforcing the Lipschitz constraint on the critic network.
            gradient_penalty = self._gradient_penalty(self.discriminator, videos_true, videos_fake)

            # Update the discriminator and generator (at last) with the defined loss functions
            discriminator_opt.update(self.discriminator_loss, self.discriminator, eval_true, eval_fake, gradient_penalty)

        # Update generator at last
        latent_z = self.generator.sample_hidden(self.batch_size)
        videos_fake = self.generator(latent_z)
        eval_fake = self.discriminator(videos_fake)
        generator_opt.update(self.generator_loss, self.generator, eval_fake)


    def generator_loss(self, generator, eval_fake):
        # The goal of the generator is to minimize the mean loss
        gen_loss = - F.sum(eval_fake) / self.batch_size
        chainer.report({'loss': gen_loss}, generator)
        return gen_loss

    def discriminator_loss(self, discriminator, eval_true, eval_fake, gradient_penalty):
        # Calculate the discriminator loss by enforcing the gradient penalty on the summed difference of real and fake
        disc_loss = F.sum(eval_fake) / self.batch_size
        disc_loss += F.sum(-eval_true) / self.batch_size
        disc_loss += gradient_penalty
        chainer.report({'loss': disc_loss}, discriminator)
        return disc_loss

    def _gradient_penalty(self, discriminator, real_video, fake_video):
        """ For details and background, please see the algorithm on page 4 (line 4-8) and the corresponding equation
            (3) of the gradient penalty on: https://arxiv.org/abs/1704.00028. The loss of the discriminator network
            enforces the Lipschitz constraint on its loss, by interpolating a real and a fake video, feeding it as
            input into the discriminator network and thereby restricitng the gradient norm of the critics output
            with regard to its input"""

        def l2norm(vec):
            # Calculate the l2norm (or euclidean norm), which is the (absolute) square root of squared summed inputs
            if vec.ndim > 1:
                # Add epsilon to avoid problems of square root derivative close to zero. Since f(x + ε) = f(x),
                # it follows that f(x + ε) - f(x) = 0
                vec = F.sqrt(F.sum(vec * vec, axis=(1,2,3,4)) + 1e-12)
            return abs(vec)

        # Interpolation creates new data points within range of discrete data points
        xp = self.generator.xp
        epsilon = xp.random.uniform(low=0, high=1, size=(self.batch_size,1,1,1,1)).astype(xp.float32)
        interpolates = (1. - epsilon) * fake_video + epsilon * real_video

        # Feed interpolated sample into discriminator and compute gradients
        eval_interpolate = discriminator(interpolates)
        gradients = chainer.grad([eval_interpolate], [interpolates], enable_double_backprop=True)[0]
        slopes = l2norm(gradients)

        # Penalty coefficient is a hyperparameter, where 10 was found to be working best (eq. 7)
        gradient_penalty = (self.penalty_coeff * (slopes - 1.) ** 2)[:, xp.newaxis]

        # Expected gradient penalty
        gradient_penalty = F.sum(gradient_penalty) / self.batch_size

        chainer.report({'gp' : gradient_penalty})

        return gradient_penalty

