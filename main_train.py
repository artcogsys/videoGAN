"""
__author__: Florian Mahner
__email__: fmahner@uos.de
__status__: Development
__date__: 31-05-2018
"""

import chainer
from chainer.serializers import load_npz
import nn.DCGAN as GAN
from utils import config_parse as cp
from computations.updater import GANUpdater
from computations.trainer import GANTrainer
from datastream import framereader
import logging
import os

conf_parser = cp.Config('setup.ini')
params = conf_parser.get_params()

def main():

    logger = build_logger(log_file=params.log_file)
    logger.info('Hyperparameters of computation: %s', params)

    generator = GAN.Generator(params.ch, params.latent_dim)
    discriminator = GAN.Discriminator(params.ch)

    if params.use_gpu >= 0:
        chainer.cuda.get_device(params.use_gpu).use()
        generator.to_gpu()
        discriminator.to_gpu()
        logger.info('Discriminator and Generator are passed onto GPU')

    logger.info('Start Loading Data from {0} and {1} ...'.format(params.root_dir, params.index_dir))

    input_pipeline = framereader.FrameReader(params.root_dir,
                                             params.index_dir,
                                             n_frames=params.n_frames,
                                             frame_size=params.frame_size)

    with chainer.using_config('train', True):
        train_data_iter = chainer.iterators.SerialIterator(input_pipeline, params.batch_size)

        # Create optimizers
        opts = {}
        opts['gen-opt'] = create_optimizer(generator, params.learning_rate, params.beta1, params.beta2)
        opts['disc-opt'] = create_optimizer(discriminator, params.learning_rate, params.beta1, params.beta2)

        # Create new updater and train
        logger.info('Start Training')
        models = (generator, discriminator)
        updater = GANUpdater(models,
                             train_data_iter,
                             opts,
                             params.batch_size,
                             params.use_gpu,
                             params.penalty_coeff,
                             params.critic_iter)

        trainer = GANTrainer(updater,
                             params.epochs,
                             params.plot_interval,
                             params.disp_interval,
                             params.snap_interval,
                             params.out_dir)

        if params.resume:
            file_dir = params.out_dir + '/' + params.filename
            print('Load trainer from ', file_dir)
            load_npz(file_dir, trainer)

        trainer.write_report()
        trainer.run()

def create_optimizer(model, learning_rate=.0001, beta1=.9, beta2=.99):
    opt = chainer.optimizers.Adam(learning_rate, beta1, beta2)
    opt.setup(model)

    return opt

def build_logger(log_file=None):
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)

    if not log_file:
        fh = logging.StreamHandler()
    else:
        fh = logging.FileHandler(log_file, mode='a+')
        print('\nLogging information passed onto log file with name: %s' % log_file)

    formatter = logging.Formatter('\n%(name)s.%(levelname)s -> %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

if __name__ == '__main__':
    main()
