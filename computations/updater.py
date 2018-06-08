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
from chainer import Variable


class GANUpdater(chainer.training.updaters.StandardUpdater):
    """
    Every GANUpdater consists of a :class:`~chainer.iterators.MultiProcessIterator` that continously feeds batches for
    training and implements a loss function to update the weights. An instance of :class:`~chainer.training.Trainer` is
    connected to :class:`~computations.GANUpdater`, which calls :func:`~computations.GANUpdater.update_core() to
    calculate the loss for the Generator and Discriminator. This loss is passed onto any optimization approach for both
    networks and reported to the Trainer.
    """

    def __init__(self, **kwargs):
        """
        :param kwargs: Every Updater requires a model to be trained, an iterator for batches and an optimizer for the
                       loss function
        """
        self._generator, self._discriminator = kwargs.pop('models')
        self._critic_iter = kwargs.pop('critic_iter', 5)
        self._penalty_coeff = kwargs.pop('penalty_coeff', 10)
        self._batch_size = kwargs.pop('batch_size')
        super(GANUpdater, self).__init__(**kwargs)#, converter=chainer.dataset.convert.ConcatWithAsyncTransfer)
        self._iterator = kwargs.pop('iterator')
        self._optimizers = kwargs.pop('optimizer')
        self.device = kwargs.pop('device')

    def update_core(self):

        # Get all models from inherited from superclass after passing at initialization
        generator_opt = self.get_optimizer('gen-opt')
        discriminator_opt = self.get_optimizer('disc-opt')
        xp = self._generator.xp

        videos_true = self.get_iterator('main').next()

        # Wrap training batch with chainer.variable and send to gpu using built-in converter function from updater
        videos_true = Variable(self.converter(videos_true, self.device))

        # The Generator is updated once every set of critic iterations and the Discriminator at each iteration
        for i in range(self._critic_iter):

            # Feed current batch into discriminator and determine if fake or real
            eval_true = self._discriminator(videos_true)

            # Feed 100-dimensional z-space into generator and produce a video
            latent_z = Variable(xp.asarray(self._generator.sample_hidden()))
            videos_fake = self._generator(latent_z)

            # Feed generated image into discriminator and determine if fake or real
            eval_fake = self._discriminator(videos_fake)

            # Calculate the gradient penalty added to the loss function by enforcing the Lipschitz constraint on
            # the critic network.
            gradient_penalty = self._gradient_penalty(self._discriminator, videos_true, videos_fake)

            # Update the discriminator and generator with the defined loss functions
            if i == 0:
                generator_opt.update(self.generator_loss, self._generator, eval_fake)

            discriminator_opt.update(self.discriminator_loss, self._discriminator, eval_true, eval_fake, gradient_penalty)

    def generator_loss(self, generator, eval_fake):
        # The goal of the generator is to maximize the mean loss
        gen_loss = F.sum(-eval_fake) / self._batch_size
        chainer.report({'loss': gen_loss}, generator)
        return gen_loss

    def discriminator_loss(self, discriminator, eval_true, eval_fake, gradient_penalty):
        # Calculate the discriminator loss by enforcing the gradient penalty on the summed difference of real and fake
        # videos
        disc_loss = F.sum(eval_true - eval_fake + gradient_penalty)
        disc_loss /= self._batch_size
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
                # Add epsilon to avoid problems of square root derivative close to zero. Since f(x + ε) = f(x),
                # it follows that f(x + ε) - f(x) = 0
                vec = F.sqrt(F.sum(vec * vec, axis=(1,2,3,4)) + 1e-12)
            return abs(vec)

        # Interpolation creates new data points within range of discrete data points
        xp = self._generator.xp
        epsilon = xp.random.uniform(low=0, high=1, size=self._batch_size).astype(xp.float32)[:, None, None, None, None]
        interpolates = (1. - epsilon) * real_video + epsilon * fake_video

        # Feed interpolated sample into discriminator and compute gradients
        eval_interpolate = discriminator(interpolates)
        gradients = chainer.grad([eval_interpolate], [interpolates], enable_double_backprop=True)[0]
        slopes = l2norm(gradients)

        # Penalty coefficient is a hyperparameter, where 10 was found to be working best (eq. 7)
        gradient_penalty = (self._penalty_coeff * (slopes - 1.) ** 2)[:, xp.newaxis]

        return gradient_penalty

    @property
    def generator(self):
        """ Getter for :attrbiute: generator for current state of training and video generation from that state """
        return self._generator

    @property
    def discriminator(self):
        """ Getter for :attribute: discriminator to retrieve current state of training """
        return self._discriminator
