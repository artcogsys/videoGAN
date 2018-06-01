"""
__author__: Florian Mahner
__email__: fmahner@uos.de
__status__: Development
__date__: 31-05-2018
"""

import chainer
import nn.DCGAN as GAN
from utils import config_parse as cp
from computations.updater import GANUpdater
from chainer.training import extensions
from chainer import cuda
import os
from datastream import framereader
import numpy as np
import imageio

# TODO WRITE SERIALIZER!!!
# TODO MAKE SURE THAT INPUT OF TRAIN AND TEST IS ALWAYS OF TYPE VARIABLE! - Change in DCGAN most likely to enforce!
# TODO Write a function recovering/loading the trained model afterwards!
# TODO See if add-hook with weight decay in create optimizer - Lets weights decay to zero (regularization).
# TODO Stil need to implement things like dropout, randomization, serialization!
# TODO Write backup/save for parameter configuration to txt file prior to run

# TODO Dropout?!
# TODO GO THROUGH BERNHARDS PARAMETERS IN DETAIL BEFORE RUN!!
# TODO Write trainer that captures object state and generates a video for visual convergence determination every x epoch
# TODO Write snapshot of model from generator (random z in model und output anschauen!) (extension above)

# TODO Write function that loads discriminator and generator model from npz file!
# TODO Normal iterator or multi process iterator? if multy, how many processes (4?)
#---------------------------------------------------------------------------------------------------------------------#
#                                                   BEGIN CODE                                                        #
#-------------------------------------------------------------------------------------------------------------------- #

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

SNAP_INTERVAL = (params['Saving']['snap_interval'], 'iteration')
DISP_INTERVAL = (params['Saving']['display_interval'], 'iteration')
PRINT_INTERVAL = (params['Saving']['print_interval'], 'iteration')
PLOT_INTERVAL = (params['Saving']['plot_interval'], 'iteration')




def create_optimizer(model, learning_rate=.0001, beta1=.9, beta2=.99):
    """ Create an optimizer from a model of <class> {discriminator, generator}"""
    opt = chainer.optimizers.Adam(learning_rate, beta1, beta2)
    #optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec') Needed?
    opt.setup(model)
    return opt


def generate_training_video(generator, save_path = ''):

    @chainer.training.make_extension()
    def get_video(trainer):

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        original_batch_size = generator.get_sample_size()
        generator.set_sample_size(1)
        latent_z = generator.sample_hidden()

        with chainer.using_config('train', False) and chainer.using_config('enable_backprop', False):

            vid = generator(latent_z).data

        #filename = os.path.join(save_path, "iter_{}.jpg").format(trainer.updater.iteration)
        with imageio.get_writer('test_' + str(trainer.updater.iteration) + '.gif', mode='I') as writer:
            for i in range(N_FRAMES):
                frame = np.swapaxes(np.squeeze(vid[:, :, :,:, i]), 0, 2)
                writer.append_data(frame.astype(np.uint8))

        generator.set_sample_size(original_batch_size)

    return get_video

def deserialize_model(file_path, model):
    return chainer.serializers.load_npz(file_path, model)




#def main():
generator = GAN.Generator(**params['Model'])
discriminator = GAN.Discriminator(**params['Model'])


if USE_GPU >= 0:
    chainer.cuda.get_device(USE_GPU).use()
    generator.to_gpu()
    discriminator.to_gpu()
    print('Discriminator and Generator passed onto GPU')

input_pipeline = framereader.FrameReader(ROOT_DIR, INDEX_DIR, n_frames=N_FRAMES, frame_size=FRAME_SIZE)
train_data = input_pipeline.load_dataset()
#train_data = np.random.randint(1,255,(4,3,64,64,32)).astype(np.float32)

# Set training environment for dropout
chainer.using_config('train', True)

# Define data iterator -> Define n_processs here? (num CPU by default)
train_data_iter = chainer.iterators.MultiprocessIterator(train_data, batch_size=BATCH_SIZE)

# Create optimizers
optimizers = {}
optimizers['gen-opt'] = create_optimizer(generator, learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2)
optimizers['disc-opt'] = create_optimizer(discriminator, learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2)

""" Start training"""
# Create new updater!
updater = GANUpdater(models=(generator, discriminator), optimizer=optimizers, iterator=train_data_iter,
                     critic_iter=CRITIC_ITER, penalty_coeff=PENALTY_COEFF, batch_size=BATCH_SIZE, device=USE_GPU)

""" TRAINER """

# Build trainer and extend with log info that is send to outfile -out-. This trainer also runs epoch etc.!
# Specify the stop trigger as the maximum number of epochs
trainer = chainer.training.Trainer(updater, (EPOCHS, 'epoch'), out=OUT_DIR)

# Change!
#trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),trigger=SNAP_INTERVAL)
#trainer.extend(extensions.snapshot_object(generator, 'gen_iter_{.updater.iteration}.npz'), trigger=SNAP_INTERVAL)
#trainer.extend(extensions.snapshot_object(discriminator, 'dis_iter_{.updater.iteration}.npz'), trigger=SNAP_INTERVAL)
trainer.extend(extensions.LogReport(trigger=DISP_INTERVAL))

trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'gen-opt/loss', 'disc-opt/loss',]), trigger=DISP_INTERVAL)

trainer.extend(extensions.ProgressBar(update_interval=1))
trainer.extend(generate_training_video(generator, save_path=OUT_DIR), trigger=PLOT_INTERVAL)

trainer.run()



if __name__ == '__main__':
    pass
    #main()



