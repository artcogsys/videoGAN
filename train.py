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

#---------------------------------------------------------------------------------------------------------------------#
#                                                   BEGIN CODE                                                        #
#-------------------------------------------------------------------------------------------------------------------- #

def create_optimizer(model, learning_rate=.0001, beta1=.9, beta2=.99):
    """ Create an optimizer from a model of <class> {discriminator, generator}"""
    opt = chainer.optimizers.Adam(learning_rate, beta1, beta2)
    opt.setup(model)
    return opt


conf_parser = cp.Config('setup.ini')
params = conf_parser.get_params()

generator = GAN.Generator(**params['Model'])
discriminator = GAN.Discriminator(**params['Model'])


train_data = chainer.Variable(np.random.randint(1,255,(1,3,64,64,32)).astype(np.float32))
test_data = chainer.Variable(np.random.randint(1,255,(1,3,64,64,32)).astype(np.float32))


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

# Build trainer and extend with log info that is send to outfile -out-. This trainer also runs epoch etc.!
trainer = chainer.training.Trainer(updater, stop_trigger=None, out='GAN-test')

#trainer.extend((n_epochs, 'epoch'))
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


#if __name__ == '__main__':
#    main()



