import argparse
import chainer
import nn.DCGAN as GAN
from utils import config_parse as cp
from computations.updater import GANUpdater
from chainer.training import extensions
from datastream import input_pipeline

import numpy as np

# TODO WRITE SERIALIZER!!!
# TODO MAKE SURE THAT INPUT OF TRAIN AND TEST IS ALWAYS OF TYPE VARIABLE! - Change in DCGAN most likely to enforce!
# TODO Write a function recovering/loading the trained model afterwards!
# TODO Write snapshot of model from generator (random z in model und output anschauen!)
# TODO Implement number of epochs somewhere (in Trainer?!)
# TODO See if add-hook with weight decay in create optimizer - Lets weights decay to zero (regularization).
# TODO Stil need to implement things like dropout, randomization, serialization!
# TODO Write backup/save for parameter configuration to txt file prior to run

# TODO Dropout?!
# TODO WRITE INPUT PIPELINE FOR DATA AND GO THROUGH BERNHARDS PARAMETERS IN DETAIL BEFORE RUN!!
# TODO WRAP train into main() method!
# TODO Write trainer that captures object state and generates a video for visual convergence determination every x epoch

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
BATCH_SIZE = params['Model']['batch_size']
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

train_data = np.random.randint(1,255,(BATCH_SIZE,3,64,64,32)).astype(np.float32)
test_data = np.random.randint(1,255,(BATCH_SIZE,3,64,64,32)).astype(np.float32)

pipe = input_pipeline.InputPipeline('/Users/florianmahner/Desktop/Donders_Internship/Programming/videoGAN/VideosSample',
                                    'job-list.txt', 1, BATCH_SIZE)
import tensorflow as tf

batch = pipe.input_pipeline()

# Set training environment for dropout
chainer.using_config('train', True)


# Define data iterators (testing not needed yet)
train_data_iter = chainer.iterators.SerialIterator(train_data, batch_size=BATCH_SIZE)
test_data_iter = chainer.iterators.SerialIterator(test_data, batch_size=BATCH_SIZE)

# Create optimizers
optimizers = {}
optimizers['gen-opt'] = create_optimizer(generator, learning_rate=LEARNING_RATE,
                                        beta1=BETA1, beta2=BETA2)
optimizers['disc-opt'] = create_optimizer(discriminator)

""" Start training"""
# Create new updater!
updater = GANUpdater(models=(generator, discriminator), optimizer=optimizers, iterator=train_data_iter,
                     critic_iter=CRITIC_ITER, penalty_coeff=PENALTY_COEFF, batch_size=BATCH_SIZE)

""" TRAINER """

# Build trainer and extend with log info that is send to outfile -out-. This trainer also runs epoch etc.!
# Specify the stop trigger as the maximum number of epochs
trainer = chainer.training.Trainer(updater, (EPOCHS, 'epoch'), out=OUT_DIR)


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



