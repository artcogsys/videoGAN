"""
__author__: Florian Mahner
__email__: fmahner@uos.de
__status__: Development
__date__: 31-05-2018

Main configuration file for the DCGAN and computation anker.
"""

import chainer
import nn.DCGAN as GAN
from utils import config_parse as cp
from computations.updater import GANUpdater
from computations.trainer import GANTrainer
from datastream import framereader
import logging

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

LOG_FILE = params['Saving']['log_file']


def main():

    logger = build_logger(log_file=LOG_FILE)
    logger.info('Hyperparameters of computation: %s', params)

    generator = GAN.Generator(**params['Model'])
    discriminator = GAN.Discriminator(**params['Model'])

    if USE_GPU >= 0:
        chainer.cuda.get_device(USE_GPU).use()
        generator.to_gpu()
        discriminator.to_gpu()
        logger.info('Discriminator and Generator are passed onto GPU')

    logger.info('Start Loading Data from {0} and {1} ...'.format(ROOT_DIR, INDEX_DIR))
    input_pipeline = framereader.FrameReader(ROOT_DIR, INDEX_DIR, n_frames=N_FRAMES, frame_size=FRAME_SIZE)

    with chainer.using_config('train', True):
        # Iterator takes an :class:`~datastream.FrameReader` with :func:`~FrameReader.get_example() implementation as
        # primary argument. Batches are then retrieved by calling get_example() repetitively and sent to GPU by
        # :class:`~computations.GANUpdater', on which the iterator acts for training
        train_data_iter = chainer.iterators.MultiprocessIterator(input_pipeline, batch_size=BATCH_SIZE, repeat=True,
                                                                 shuffle=True)
        # Create optimizers
        optimizers = {}
        optimizers['gen-opt'] = create_optimizer(generator, learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2)
        optimizers['disc-opt'] = create_optimizer(discriminator, learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2)

        # Create new updater and train!
        logger.info('Start Training')
        updater = GANUpdater(models=(generator, discriminator), optimizer=optimizers, iterator=train_data_iter,
                             critic_iter=CRITIC_ITER, penalty_coeff=PENALTY_COEFF, batch_size=BATCH_SIZE,
                             device=USE_GPU)

        trainer = GANTrainer(updater, epochs=EPOCHS, **params['Saving'])
        trainer.write_report()
        trainer.run()


def create_optimizer(model, learning_rate=.0001, beta1=.9, beta2=.99):
    """ Create an optimizer from a model of :class: {discriminator, generator}"""
    opt = chainer.optimizers.Adam(learning_rate, beta1, beta2)
    opt.setup(model)
    return opt


def build_logger(log_file=None):
    """ Create a logger that stores computation information for different classes with traceback and print its output
    either to STODU or a given log-file"""
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)

    if not log_file:
        fh = logging.StreamHandler()
    else:
        fh = logging.FileHandler(log_file, mode='a')
        print('\nLogging information passed onto log file with name: %s' % log_file)

    formatter = logging.Formatter('\n%(name)s.%(levelname)s -> %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def deserialize_model(file_path, model):
    return chainer.serializers.load_npz(file_path, model)


if __name__ == '__main__':
    main()
