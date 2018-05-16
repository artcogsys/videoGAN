import chainer
import chainer.functions as F
from chainer import Variable
import numpy as np

"""
What has to be done:

1. Implement an Iterator that takes batches (random?) from training data and feeds it into Updater
2. Updater calls Optimizer and updates the weights with gradient info
    -> This optimization has to be implemented for generator and discriminator!
3. Connect an instance of <class> Trainer to <class> Updater, which runs <method> update() and reports the 
   computations (e.g. loss etc.)
"""


class GANUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.generator, self.discriminator = kwargs.pop('models')
        super(GANUpdater, self).__init__(*args, **kwargs)
        self.iterator = kwargs.pop('iterator')
        self.optimizers = kwargs.pop('optimizers')
        self.critic_iter = kwargs.pop('critic_iter') # Number of critic iterations

    def update_core(self):

        # TODO- Include n_dis that differentiates the updating iteration between iterator and discriminator!
        # In paper the critic network is optimized 5 times!! every discriminator update!

        # Get all models from  Standardupdater after passing at initialization
        generator_opt = self.get_optimizer('generator') # Get generator optimizer from superclass
        discriminator_opt = self.get_optimizer('discriminator')
        current_batch = self.get_iterator('iter').next()

        # Get batch size
        batch_size = np.size(current_batch, 1)

        # Feed current batch into discriminator and determine if fake or real
        disc_real = self.discriminator(current_batch)

        # Feed 100-dimensional z-space into generator and produce a video
        latent_z = self.generator.sample_hidden
        gen_fake = self.generator(latent_z)

        # Feed generated image into discriminator and determine if fake or real
        disc_fake = self.discriminator(gen_fake)

        # TODO: Update the discriminator and generator with gradient info here! -Only works with correct loss function!
        discriminator_opt.update(self._discriminator_loss, disc_real, disc_fake)
        generator_opt.update(self._generator_loss(), disc_fake)


    """ Define loss functions for Generator and Discriminator"""
    @staticmethod
    def generator_loss(generator, fake_video):
        """ Calculate the loss function here for adam optimization! Difficult part! """

        # TODO Either attach a trainer here (planned) or I have to clear the grads and update manually! MAKE COMMENTS!
        batch_size = len(fake_video)
        gen_loss = F.sum(- fake_video) / batch_size
        chainer.report({'gen_loss': gen_loss}, generator)
        return gen_loss

    @staticmethod
    def discriminator_loss(discriminator, real_video, fake_video, critic_iter):
        """Code taken from: https://github.com/mr4msm/wgan_gp_chainer/blob/master/train_wgan_gp.py """

        def _l2norm(var):
            if var.ndim > 1:
                if np.asarray(var.shape[1:]).prod() > 1:
                    return F.sqrt(F.sum(var * var, axis=tuple(range(1, var.ndim))))
                else:
                    var = F.reshape(var, (-1,))

            return abs(var)

        # Get loss of discriminator network
        batch_size = len(fake_video)
        disc_loss= F.sum(fake_video) - F.sum(real_video)

        # Add gradient penalty to loss
        alpha = np.random.uniform(low= 0., high= 1., size=((batch_size,) + (1,) * (real_video.ndim - 1))).astype('float32')
        interpolates = Variable((1. - alpha) * real_video.data + alpha * fake_video.data)
        gradients = chainer.grad([discriminator(interpolates)],
                                 [interpolates],
                                 enable_double_backprop=True)[0]
        slopes = _l2norm(gradients)

        gradient_penalty = critic_iter * F.sum((slopes - 1.) * (slopes - 1.))
        disc_loss = (disc_loss + gradient_penalty) / batch_size

        return disc_loss

