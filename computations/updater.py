"""
__author__: Florian Mahner
__email__: fmahner@uos.de
__status__: Development
__date__: 11-05-2018

This file updates the GAN network following the Wasserstein framework
For details, please see:

https://arxiv.org/abs/1701.07875 (original paper by Arjovski)
https://arxiv.org/abs/1704.00028 (follow up paper)
"""

import chainer
import chainer.functions as F
import numpy as np

class GANUpdater(chainer.training.updaters.StandardUpdater):
    """
    Every Updater consists of an <class>
    Iterator that continously feeds batches and an <class> Optimizer for the loss function. An instance of
    <class> chainer.training.Trainer is connected to <class> GANUpdater, which calls the <method> update_core() to
    calculate the loss of the Generator and Discriminator. This loss is passed onto the individual optimizers of both
    networks and reported to the Trainer.
    """

    def __init__(self, **kwargs):
        """
        :param kwargs: Every Updater requires a model to be trained, an iterator for batches and an optimizer for the
                       loss function
        """
        self.generator, self.discriminator = kwargs.pop('models')
        self.critic_iter = kwargs.pop('critic_iter', 5)
        self.penalty_coeff = kwargs.pop('penalty_coeff', 10)
        self.batch_size = kwargs.pop('batch_size')
        super(GANUpdater, self).__init__(**kwargs)
        self.iterator = kwargs.pop('iterator')
        self.optimizers = kwargs.pop('optimizer')

    def update_core(self):

        # Get all models from inherited from superclass after passing at initialization
        generator_opt = self.get_optimizer('gen-opt')
        discriminator_opt = self.get_optimizer('disc-opt')

        # Get next batch
        videos_true = np.asarray(self.get_iterator('main').next())

        # The Generator is updated once every set of critic iterations and the Discriminator at each iteration
        for i in range(self.critic_iter):

            # Feed current batch into discriminator and determine if fake or real
            eval_true = self.discriminator(videos_true)

            # Feed 100-dimensional z-space into generator and produce a video
            latent_z = self.generator.sample_hidden()
            videos_fake = self.generator(latent_z)

            # Feed generated image into discriminator and determine if fake or real
            eval_fake = self.discriminator(videos_fake)

            # Calculate the gradient penalty added to the loss function by enforcing the Lipschitz constraint on
            # the critic network.
            gradient_penalty = self._gradient_penalty(self.discriminator, videos_true, videos_fake)

            # Update the discriminator and generator with the defined loss functions
            if i == 0:
                generator_opt.update(self.generator_loss, self.generator, eval_fake)

            discriminator_opt.update(self.discriminator_loss, self.discriminator, eval_true, eval_fake, gradient_penalty)

    def generator_loss(self, generator, eval_fake):
        # The goal of the generator is to maximize the mean loss
        gen_loss = F.sum(-eval_fake) / self.batch_size
        chainer.report({'loss': gen_loss}, generator)
        return gen_loss

    def discriminator_loss(self, discriminator, eval_true, eval_fake, gradient_penalty):
        # Calculate the discriminator loss by enforcing the gradient penalty on the summed difference of real and fake
        # videos
        disc_loss = F.sum(eval_true - eval_fake + gradient_penalty)
        disc_loss /= self.batch_size
        chainer.report({'loss': disc_loss}, discriminator)
        return disc_loss

    def _gradient_penalty(self, discriminator, real_video, fake_video):
        """ For details and backgroundm, please see the algorithm on page 4 (line 4-8) and the corresponding equation
            (3) of the gradient penalty on: https://arxiv.org/abs/1704.00028. The loss of the discriminator network
            enforces the Lipschitz constraint on its loss, by interpolating a real and a fake video, feeding it as
            input into the discriminator network and thereby restricitng the gradient norm of the critics output
            with regard to its input"""

        def l2norm(vec):
            # Calculate the l2norm (or euclidean norm), which is the (absolute) square root of squared summed inputs
            if vec.ndim > 1:
                # Add epsilon to avoid problems of square root derivative close to zero. Since f(x + epsilon) = f(x),
                # it follows that f(x + epsilon) - f(x) = 0
                vec = F.sqrt(F.sum(vec * vec, axis=(1,2,3,4)) + 1e-12)
            return abs(vec)

        # Interpolation creates new data points within range of discrete data points
        epsilon = np.random.uniform(low=0, high=1, size=self.batch_size).astype(np.float32)[:, None, None, None, None]
        interpolates = (1. - epsilon) * real_video + epsilon * fake_video

        # Feed interpolated sample into discriminator and compute gradients
        eval_interpolate = discriminator(interpolates)
        gradients = chainer.grad([eval_interpolate], [interpolates], enable_double_backprop=True)[0]
        slopes = l2norm(gradients)

        # Penalty coefficient is a hyperparameter, where 10 was found to be working best (eq. 7)
        gradient_penalty = (self.penalty_coeff * (slopes - 1.) ** 2)[:,np.newaxis]

        return gradient_penalty
