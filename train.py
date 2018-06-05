"""
__author__: Florian Mahner
__email__: fmahner@uos.de
__status__: Development
__date__: 31-05-2018
"""
# TODO See if add-hook with weight decay in create optimizer - Lets weights decay to zero (regularization).
# TODO Stil need to implement things like dropout, randomization, serialization!
# TODO Write backup/save for parameter configuration to txt file prior to run
# TODO GO THROUGH BERNHARDS PARAMETERS IN DETAIL BEFORE RUN!!
# TODO Write function that loads discriminator and generator model from npz file!
# TODO Normal iterator or multi process iterator? if multy, how many processes?
# TODO Scale input?! -> Kratzwald mentions but where is it done?!
# TODO Multiprocess or serialiterator??

import chainer
import nn.DCGAN as GAN
from utils import config_parse as cp
from computations.updater import GANUpdater
from computations.trainer import GANTrainer
from chainer import cuda
from datastream import framereader

conf_parser = cp.Config('setup.ini')
params = conf_parser.get_params()

""" Define constants and load parameters from .ini file """

ROOT_DIR = params['Data']['root_dir']
INDEX_DIR = params['Data']['index_dir']

LEARNING_RATE = params['Adam']['learning_rate']
BETA1 = params['Adam']['beta1']
BETA2 = params['Adam']['beta2']
PENALTY_COEFF = params['Adam']['penalty_coeff']
CRITIC_ITER = params['Adam']['critic_iter']

EPOCHS = params['Model']['epochs']
USE_GPU = params['Model']['use_gpu']
BATCH_SIZE = params['Model']['batch_size']
N_FRAMES = params['Model']['n_frames']
FRAME_SIZE = params['Model']['frame_size']


def create_optimizer(model, learning_rate=.0001, beta1=.9, beta2=.99):
    """ Create an optimizer from a model of :class: {discriminator, generator}"""
    opt = chainer.optimizers.Adam(learning_rate, beta1, beta2)
    opt.setup(model)
    return opt


def deserialize_model(file_path, model):
    return chainer.serializers.load_npz(file_path, model)


def main():

    generator = GAN.Generator(**params['Model'])
    discriminator = GAN.Discriminator(**params['Model'])

    if USE_GPU >= 0:
        chainer.cuda.get_device(USE_GPU).use()
        generator.to_gpu()
        discriminator.to_gpu()
        print('Discriminator and Generator passed onto GPU\n')

    print('\nStart loading data from {0} and {1} ...'.format(ROOT_DIR, INDEX_DIR))
    input_pipeline = framereader.FrameReader(ROOT_DIR, INDEX_DIR, n_frames=N_FRAMES, frame_size=FRAME_SIZE)
    train_data = input_pipeline.load_dataset()
    print('... Done')
    assert len(train_data) > 0, 'Problem with datastream, empty training set'

    with chainer.using_config('train', True):

        # Define data iterator -> Define n_processs here? (num CPU by default)
        train_data_iter = chainer.iterators.MultiprocessIterator(train_data, batch_size=BATCH_SIZE)

        # Create optimizers
        optimizers = {}
        optimizers['gen-opt'] = create_optimizer(generator, learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2)
        optimizers['disc-opt'] = create_optimizer(discriminator, learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2)

        """ Start training by passing an <class> updater to a <class> trainer and extract log information """
        # Create new updater!
        updater = GANUpdater(models=(generator, discriminator), optimizer=optimizers, iterator=train_data_iter,
                             critic_iter=CRITIC_ITER, penalty_coeff=PENALTY_COEFF, batch_size=BATCH_SIZE,
                             device=USE_GPU)

        trainer = GANTrainer(updater, epochs=EPOCHS, **params['Saving'])
        trainer.write_report()
        trainer.run()


if __name__ == '__main__':
    main()