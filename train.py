import argparse
import chainer
import nn.DCGAN as GAN
from utils import get_mnist
from computations.updater import GANUpdater
from chainer.training import extensions

#---------------------------------------------------------------------------------------------------------------------#
#                                                   BEGIN CODE                                                        #
#-------------------------------------------------------------------------------------------------------------------- #

def get_arguments():
    parser = argparse.ArgumentParser(description='Wasserstein DCGAN')

    """ Model/Computation specifics"""
    parser.add_argument('--batchsize', '-b', type=int, default=64, help='Size of each mini-batch fed into thr network')
    parser.add_argument('--epoch', '-e', type=int, default=1000, help='Number of training loops')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='Using GPU? (negative value stands for CPU)')
    parser.add_argument('--n_hidden', '-n', type=int, default=100, help='Number of hidden units (z)')
    parser.add_argument('--lambda', '-l', type=int, default=100, help='Gradient penality coefficient')
    parser.add_argument('critic_iter', '-c', type=int, default=100, help='Number of critic iter. per generator iter.')

    """ Directories """
    parser.add_argument('--dir', '-i', default='', help='Directory for videos')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')

    """ Saving specifics """
    parser.add_argument('--snapshot_interval', type=int, default=1000, help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100, help='Interval of displaying log to console')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')

    args = parser.parse_args()
    return args


def create_optimizer(model, alpha=.001, beta1=.5, beta2=.99):
    """ Create an optimizer from a model of <class> {discriminator, generator}"""
    opt = chainer.optimizers.Adam(alpha, beta1, beta2)
    opt.setup(model)
    # opt.add_hook(chainer.optimizer.WeightDecay(0.1)) - Lets weights exponentially decay to zero (regularization).
    return opt


# DEFINE CONSTANTS
def main():

    """ Testing parameters """
    disc_n_hidden = 20
    gen_n_hidden = 20
    n_latent_z = 25
    n_epochs = 100
    batch_size = 5
    mnist_dim = 28

    #args = get_arguments()

    Generator = GAN.Generator(100)
    Discriminator = GAN.Discriminator()

    # Load the MNIST dataset and reshape
    train_data, test_data = get_mnist(n_train=350, n_test=100, with_label=False, classes=[0])
    train_data = train_data.reshape([-1, 1, mnist_dim, mnist_dim])
    test_data = test_data.reshape([-1, 1, mnist_dim, mnist_dim])

    # Define data iterators (testing not needed yet)
    train_data_iter = chainer.iterators.SerialIterator(train_data, batch_size=batch_size)
    test_data_iter = chainer.iterators.SerialIterator(test_data, batch_size=batch_size)

    # Create optimizers
    optimizers = {}
    optimizers['gen-opt'] = create_optimizer(Generator)
    optimizers['disc_opt'] = create_optimizer(Discriminator)

    """ Start training"""
    # Create new updater!
    updater = GANUpdater(models=(Generator, Discriminator), optimizer=optimizers, iterator=train_data_iter)

    # Build trainer and extend with log info that is send to outfile -out-. This trainer also runs epoch etc.!
    trainer = chainer.training.trainer(updater, (n_epochs, 'epoch'), stop_trigger=None, out='GAN-test')

    trainer.extend(extensions.LogReport()) # Add log report (loss/accuracy) and append every epoch to out of trainer
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch')) # TODO change saving to every x epochs
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.ProgressBar()) # Gives us a progress bar !

    trainer.run() # Run job

# TODO Write a function recovering/loading the trained model afterwards!
# TODO Write snapshot of model from generator (random z in model und output anschauen!)


if __name__ == '__main__':
    main()



