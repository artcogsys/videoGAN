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
        # Pop information not accepted by superclass constructor
        self.generator, self.discriminator = kwargs.pop('models')
        self.critic_iter = kwargs.pop('critic_iter', 5)
        self.penalty_coeff = kwargs.pop('penalty_coeff', 10)

        super(GANUpdater, self).__init__(*args, **kwargs)
        self.iterator = kwargs.pop('iterator')
        self.optimizers = kwargs.pop('optimizer')


    def update_core(self):

        # Get all models from  Standardupdater after passing at initialization
        generator_opt = self.get_optimizer('gen-opt') # Get generator optimizer from superclass
        discriminator_opt = self.get_optimizer('disc-opt')
        current_batch = np.asarray(self.get_iterator('main').next())

        for i in range(self.critic_iter):

            print(np.shape(current_batch))
            # Feed current batch into discriminator and determine if fake or real
            disc_real = self.discriminator(current_batch)

            # Feed 100-dimensional z-space into generator and produce a video
            latent_z = self.generator.sample_hidden()
            gen_fake = self.generator(latent_z)

            # Feed generated image into discriminator and determine if fake or real
            disc_fake = self.discriminator(gen_fake)

            # TODO: Update the discriminator and generator with gradient info here! -Only works with correct loss function!
            # TODO: with using_config('enable_backprop', False): Include?
            # The generator is updated once every <attrbiute> critic_iter of discriminator updater. The function
            # optimizer.update clears the gradients and computes backwards gradients to update parameters!
            # Perform clearing of gradients and forward / backward computations by calling update method of optimizer
            # ' batch = self._iterators['main'].next()' This is also called in updaterm but it is outside for loop,
            # pay attention though
            if i == 0:
                generator_opt.update(self.generator_loss, disc_fake)

            discriminator_opt.update(self.discriminator_loss, disc_real, disc_fake, self.penalty_coeff)


    def generator_loss(self, generator, fake_video):
        batch_size = len(fake_video)
        gen_loss = F.sum(- fake_video) / batch_size
        chainer.report({'generator_loss': gen_loss}, generator)
        return gen_loss


    def discriminator_loss(self, discriminator, real_video, fake_video, penalty_coef = 10):
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
        disc_loss= F.sum(fake_video) - F.sum(real_video) + gradient_penalty
        disc_loss /= batch_size # Divide by batch size (why?)

        chainer.report({'discriminator_loss': disc_loss}, discriminator)

        return disc_loss