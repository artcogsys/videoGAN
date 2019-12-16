"""
__author__: Florian Mahner
__email__: fmahner@uos.de
__status__: Development
__date__: 03-06-2018
"""
from chainer.training import extension, extensions
from chainer.training import trigger as trigger_module
from chainer import serializers
import chainer
import imageio
import os
import numpy as np
import logging
import matplotlib.pyplot as plt

class GANTrainer(chainer.training.Trainer):

    def __init__(self, updater, epochs=15, plot_interval=1, disp_interval=1, snap_interval=1, out_dir='', n_frames=32):

        self.updater = updater
        self.epochs = (epochs, 'epoch')
        self.plot_interval = (plot_interval, 'iteration')
        self.disp_interval = (disp_interval, 'iteration')
        self.snap_interval = (snap_interval, 'iteration')
        self.out_dir = out_dir
        self.log_stream = logging.getLogger('main').handlers[0].stream # STDOUT if no file specified
        self.n_frames = n_frames
        super(GANTrainer, self).__init__(updater, stop_trigger=self.epochs, out=self.out_dir)

    def write_report(self):
        """ Extend the trainer to write object snapshots and log reports"""

        generator = self.updater.generator
        discriminator = self.updater.discriminator

        # Get snapshots of generator, discriminator and general trainer objects
        self.extend(extensions.snapshot_object(generator, 'gen_iter_{.updater.iteration}.npz'), trigger=self.snap_interval)
        self.extend(extensions.snapshot_object(discriminator, 'dis_iter_{.updater.iteration}.npz'),trigger=self.snap_interval)
        self.extend(extensions.snapshot(filename='trainer_iter_{.updater.iteration}.npz'),trigger=self.snap_interval)

        self.extend(extensions.LogReport(trigger=self.disp_interval, log_name='loss.json'))

        # Log performances and error during training
        self.extend(extensions.PrintReport(['epoch', 'iteration', 'gen-opt/loss', 'disc-opt/loss'],
                                            out=self.log_stream), trigger=self.disp_interval)

        self.extend(extensions.ProgressBar(update_interval=self.disp_interval[0], out=self.log_stream))

        # Every plot_interval, generate a new video and save
        self.extend(self.__get_video(generator), trigger=self.plot_interval)

        # Plot loss
        self.extend(extensions.PlotReport(['gen-opt/loss', 'disc-opt/loss'], 'iteration', grid=True,
                                          trigger=self.disp_interval, file_name='loss.pdf', marker='.'))


    def __get_video(self, generator, num_videos=1):
        @chainer.training.make_extension()
        def make(trainer):
            out_dir = os.path.join(self.out_dir, "gifs")
            filename = os.path.join(out_dir, "vid-epoch{}-iter{}.gif").format(trainer.updater.epoch,
                                                                                  trainer.updater.iteration)
            generate_training_video(generator, out_dir, filename, num_videos)

        return make

def generate_training_video(generator, out_dir='/.', filename='abc.gif', num_videos=1):

    xp = generator.xp
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    latent_z = generator.sample_hidden(num_videos+1)

    # Generate a new video and retrieve only the data
    with chainer.using_config('train', False) and chainer.using_config('enable_backprop', False):

        vid = generator(latent_z)

        if chainer.backends.cuda.get_device_from_array(vid.array).id >= 0:
            vid = xp.asnumpy(vid.data)
        else:
            vid = vid.data

        frame_size = vid.shape[2]
        n_frames = vid.shape[-1]

        for i in range(num_videos):
            with imageio.get_writer(filename, mode='I') as writer:
                    for j in range(n_frames):
                        frame = vid[i,:,:,:,j]
                        frame = np.transpose(frame) # Tranpose image from (3 x nframes x nframes) to (nframes x nframes x 3)

                        # Interpolate back to uint-8 format
                        frame = np.interp(frame, (frame.min(), frame.max()), (0, 255)).astype(np.uint8)
                        writer.append_data(frame)
