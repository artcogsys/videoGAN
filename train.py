import argparse
import chainer
import nn.DCGAN as GAN
from utils import config_parse as cp
from computations.updater import GANUpdater
from chainer.training import extensions
#from datastream import input_pipeline
from datastream import framereader

import numpy as np

# TODO WRITE SERIALIZER!!!
# TODO MAKE SURE THAT INPUT OF TRAIN AND TEST IS ALWAYS OF TYPE VARIABLE! - Change in DCGAN most likely to enforce!
# TODO Write a function recovering/loading the trained model afterwards!
# TODO See if add-hook with weight decay in create optimizer - Lets weights decay to zero (regularization).
# TODO Stil need to implement things like dropout, randomization, serialization!
# TODO Write backup/save for parameter configuration to txt file prior to run

# TODO Dropout?!
# TODO GO THROUGH BERNHARDS PARAMETERS IN DETAIL BEFORE RUN!!
# TODO WRAP train into main() method!
# TODO Write trainer that captures object state and generates a video for visual convergence determination every x epoch
# TODO Write snapshot of model from generator (random z in model und output anschauen!) (extension above)

# TODO Write function that loads discriminator and generator model from npz file!
# TODO Normal iterator or multi process iterator? if multy, how many processes (4?)
#---------------------------------------------------------------------------------------------------------------------#
#                                                   BEGIN CODE                                                        #
#-------------------------------------------------------------------------------------------------------------------- #


def create_optimizer(model, learning_rate=.0001, beta1=.9, beta2=.99):
    """ Create an optimizer from a model of <class> {discriminator, generator}"""
    opt = chainer.optimizers.Adam(learning_rate, beta1, beta2)
    #optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec') Needed?
    opt.setup(model)
    return opt


conf_parser = cp.Config('setup.ini')
params = conf_parser.get_params()

# DEFINE CONSTANTS
ROOT_DIR = params['Directories']['root_dir']
INDEX_DIR = params['Directories']['index_dir']
OUT_DIR = params['Directories']['out_dir']
BATCH_SIZE = params['Model']['batch_size']
N_FRAMES = params['Model']['n_frames']
FRAME_SIZE = params['Model']['frame_size']
LEARNING_RATE = params['Adam']['learning_rate']
BETA1 = params['Adam']['beta1']
BETA2 = params['Adam']['beta2']
PENALTY_COEFF = params['Adam']['penalty_coeff']
CRITIC_ITER = params['Adam']['critic_iter']
OUT_DIR = params['Directories']['out_dir']
EPOCHS = params['Model']['epoch']
USE_GPU = params['Model']['use_gpu']

SNAP_INTERVAL = (params['Saving']['snap_interval'], 'epoch')
DISP_INTERVAL = (params['Saving']['display_interval'], 'epoch')
PRINT_INTERVAL = (params['Saving']['print_interval'], 'iteration')
PLOT_INTERVAL = (params['Saving']['plot_interval'], 'iteration')


generator = GAN.Generator(**params['Model'])
discriminator = GAN.Discriminator(**params['Model'])

if USE_GPU >= 0:
    chainer.cuda.get_device(USE_GPU).use()
    generator.to_gpu()
    discriminator.to_gpu()
    print('Discriminator and Generator passed onto GPU')

input_pipeline = framereader.FrameReader(ROOT_DIR, INDEX_DIR, batch_size=BATCH_SIZE, n_frames=N_FRAMES,
                                         frame_size=FRAME_SIZE)
train_data = input_pipeline.load_dataset()

# Set training environment for dropout
chainer.using_config('train', True)

# Define data iterator
train_data_iter = chainer.iterators.MultiprocessIterator(train_data, batch_size=BATCH_SIZE, n_processes=4)

# Create optimizers
optimizers = {}
optimizers['gen-opt'] = create_optimizer(generator, learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2)
optimizers['disc-opt'] = create_optimizer(discriminator)

""" Start training"""
# Create new updater!
updater = GANUpdater(models=(generator, discriminator), optimizer=optimizers, iterator=train_data_iter,
                     critic_iter=CRITIC_ITER, penalty_coeff=PENALTY_COEFF, batch_size=BATCH_SIZE, device=USE_GPU)

""" TRAINER """

# Build trainer and extend with log info that is send to outfile -out-. This trainer also runs epoch etc.!
# Specify the stop trigger as the maximum number of epochs
trainer = chainer.training.Trainer(updater, (EPOCHS, 'epoch'), out=OUT_DIR)

# Not working nicely yet!
snapshot_interval = (1, 'iteration')
display_interval = (1, 'iteration')
trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),trigger=snapshot_interval)
trainer.extend(extensions.snapshot_object(generator, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
trainer.extend(extensions.snapshot_object(discriminator, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
trainer.extend(extensions.LogReport(trigger=display_interval))

trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'gen-opt/loss', 'disc-opt/loss',]), trigger=display_interval)

trainer.extend(extensions.ProgressBar(update_interval=1))



trainer.run()



#if __name__ == '__main__':
#    main()



