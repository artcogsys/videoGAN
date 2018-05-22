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
        self.optimizers = kwargs.pop('optimizer')
        self.critic_iter = kwargs.pop('critic_iter') # Number of critic iterations

    def update_core(self):

        # TODO- Include n_dis that differentiates the updating iteration between iterator and discriminator!
        # In paper the critic network is optimized 5 times!! every discriminator update!

        # Get all models from  Standardupdater after passing at initialization
        generator_opt = self.get_optimizer('generator') # Get generator optimizer from superclass
        discriminator_opt = self.get_optimizer('discriminator')
        current_batch = self.get_iterator('iter').next()

        for i in range(self.critic_iter):

            # Feed current batch into discriminator and determine if fake or real
            disc_real = self.discriminator(current_batch)

            # Feed 100-dimensional z-space into generator and produce a video
            latent_z = self.generator.sample_hidden
            gen_fake = self.generator(latent_z)

            # Feed generated image into discriminator and determine if fake or real
            disc_fake = self.discriminator(gen_fake)

            # TODO: Update the discriminator and generator with gradient info here! -Only works with correct loss function!
            # The generator is updated once every <attrbiute> critic_iter of discriminator updatesr
            # with using_config('enable_backprop', False): Include?
            if i == 0:
                discriminator_opt.update(self.discriminator_loss, disc_real, disc_fake)

            generator_opt.update(self.generator_loss, disc_fake)


    """ Define loss functions for Generator and Discriminator"""
    @staticmethod
    def generator_loss(generator, fake_video):

        # TODO Either attach a trainer here (planned) or I have to clear the grads and update manually! MAKE COMMENTS!
        batch_size = len(fake_video)
        gen_loss = F.sum(- fake_video) / batch_size
        chainer.report({'generator_loss': gen_loss}, generator)
        return gen_loss

    @staticmethod
    def discriminator_loss(discriminator, real_video, fake_video, penalty_coef = 10):
        """ For details please see the algorithm on page 4 (line 4-8) and the corresponding equation (3) of the
        gradient penalty on: https://arxiv.org/abs/1704.00028 """

        def _l2norm(vec):
            # Calculate the l2norm (or euclidean norm), which is the (absolute) square root of squared summed inputs
            if vec.ndim > 1:
                # Add epsilon to avoid problems of square root derivative close to zero. Since f(x + epsilon) = f(x),
                # it follows that f(x + epsilon) - f(x) = 0
                vec = F.sqrt(F.sum(vec * vec, dim = 1), + 1e-12)
            return abs(vec)

        """ The loss of the discriminator network enforces the Lipschitz-constraint on the critic loss, by interpolating
        a real and a fake video tbc."""

        batch_size = len(fake_video)
        epsilon = Variable(np.random.uniform(low=0, high=1, size=(batch_size,1)).astype(np.float32))

        # Interpolation creates new data points within range of discrite data points
        interpolates = (1. - epsilon) * real_video.data + epsilon * fake_video.data

        # Feed interpolated sample into discriminator and compute gradients
        # ∇x_hat * Discriminator(x_hat)
        gradients = chainer.grad([discriminator(interpolates)], [interpolates], enable_double_backprop=True)[0]
        slopes = _l2norm(gradients)

        # Penalty coefficient is a hyperparameter, where 10 was found to be working best
        gradient_penalty = penalty_coef * (slopes - 1.) ** 2

        # Calculate the discriminator loss by enforcing the gradient penalty on the summed difference of real and fake
        # videos
        disc_loss= F.sum(fake_video) - F.sum(real_video)
        disc_loss = disc_loss + gradient_penalty # Some have divided by batch size here, why?

        chainer.report({'discriminator_loss': disc_loss}, discriminator)

        return disc_loss