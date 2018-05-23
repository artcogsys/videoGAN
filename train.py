import argparse
import chainer
import nn.DCGAN as GAN
from utils import config_parse as cp
from computations.updater import GANUpdater
from chainer.training import extensions

import numpy as np

# TODO WRITE SERIALIZER!!!
# TODO MAKE SURE THAT INPUT OF TRAIN AND TEST IS ALWAYS OF TYPE VARIABLE! - Change in DCGAN most likely to enforce!
# TODO Write a function recovering/loading the trained model afterwards!
# TODO Write snapshot of model from generator (random z in model und output anschauen!)
# TODO Implement number of epochs somewhere (in Trainer?!)
# TODO See if add-hook with weight decay in create optimizer - Lets weights decay to zero (regularization).
# TODO Stil need to implement things like dropout, randomization, serialization!
# TODO Write backup/save for parameter configuration to txt file prior to run

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
OUT_DIR = params['Directories']['out_dir']
EPOCHS = params['Model']['epoch']
USE_GPU = params['Model']['use_gpu']

SNAP_INTERVAL = (params['Saving']['snap_interval'], 'epoch')
DISP_INTERVAL = (params['Saving']['display_interval'], 'epoch')
PRINT_INTERVAL = (params['Saving']['print_interval'], 'iteration')
PLOT_INTERVAL = (params['Saving']['plot_interval'], 'iteration')


generator = GAN.Generator(**params['Model'])
discriminator = GAN.Discriminator(**params['Model'])

train_data = np.random.randint(1,255,(1,3,64,64,32)).astype(np.float32)
test_data = np.random.randint(1,255,(1,3,64,64,32)).astype(np.float32)

# Set training environment for dropout
chainer.using_config('train', True)


# Define data iterators (testing not needed yet)
train_data_iter = chainer.iterators.SerialIterator(train_data, batch_size=params['Model']['batch_size'])
test_data_iter = chainer.iterators.SerialIterator(test_data, batch_size=params['Model']['batch_size'])

# Create optimizers
optimizer = {}
optimizer['gen-opt'] = create_optimizer(generator, learning_rate=params['Adam']['learning_rate'],
                                        beta1=params['Adam']['beta1'], beta2=params['Adam']['beta2'])
optimizer['disc-opt'] = create_optimizer(discriminator)

""" Start training"""
# Create new updater!
updater = GANUpdater(models=(generator, discriminator), optimizer=optimizer, iterator=train_data_iter,
                     critic_iter=params['Adam']['critic_iter'], penalty_coeff=params['Adam']['penalty_coeff'])

""" TRAINER """

# Build trainer and extend with log info that is send to outfile -out-. This trainer also runs epoch etc.!
# Specify the stop trigger as the maximum number of epochs
trainer = chainer.training.Trainer(updater, (EPOCHS, 'epoch'), out=OUT_DIR)
trainer.extend(extensions.LogReport(trigger=DISP_INTERVAL))
#trainer.extend(extensions.PrintReport(trigger=DISP_INTERVAL))

# SNAPTSHOT OF OBJECT STATE!
trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=SNAP_INTERVAL)
trainer.extend(extensions.snapshot_object(generator, 'g_epoch_{.updater.epoch}.npz'), trigger=SNAP_INTERVAL)
trainer.extend(extensions.snapshot_object(discriminator, 'd_epoch_{.updater.epoch}.npz'), trigger=SNAP_INTERVAL)

# Display
trainer.extend(extensions.LogReport(trigger=PRINT_INTERVAL))
trainer.extend(extensions.PrintReport(['iteration', 'main/loss', 'D/loss', 'D/loss_real', 'D/loss_fake']),
               trigger=PRINT_INTERVAL)
trainer.extend(extensions.ProgressBar(update_interval=PRINT_INTERVAL))

trainer.extend(extensions.dump_graph('D/loss', out_name='TrainGraph.dot'))

# Plotting
trainer.extend(extensions.PlotReport(['main/loss'], 'iteration', file_name='Loss.png', trigger=PLOT_INTERVAL),
               trigger=PLOT_INTERVAL)
trainer.extend(extensions.PlotReport(['D/loss'], 'iteration', file_name='D_Loss.png', trigger=PLOT_INTERVAL),
               trigger=PLOT_INTERVAL)

trainer.extend(extensions.PlotReport(['D/loss_real'], 'iteration', file_name='Loss_Real.png', trigger=PLOT_INTERVAL),
               trigger=PLOT_INTERVAL)

trainer.extend(extensions.PlotReport(['D/loss_fake'], 'iteration', file_name='Loss_Fake.png', trigger=PLOT_INTERVAL),
               trigger=PLOT_INTERVAL)


# Execute!
trainer.run()

#if __name__ == '__main__':
#    main()



