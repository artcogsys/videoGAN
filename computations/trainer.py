"""
__author__: Florian Mahner
__email__: fmahner@uos.de
__status__: Development
__date__: 03-06-2018

This file defines the training process of the Generative Adversarial Network and extracts relevant information of
computations
"""

import chainer
import imageio
import os
import numpy as np
from chainer.training import extensions


class GANTrainer(chainer.training.Trainer):
    """ A GANTrainer is initialized with a :class:`~chainer.training.updater.StandardUpdater' and the corresponding
    loss functions and update rules are used by the trainer. The trainer writes reports at specific iteration intervals
    of the state of GAN training. Every couple of iterations, the trainer creates a video from the learned
    representation of the trainer, to obtain visual feedback of convergence """

    def __init__(self, updater, epochs=1000, **kwargs):
        """
        :param updater: Custom or normal updater, as in chainer.training.StandardUpdater()
        :param epochs: number of training epochs
        :param kwargs: plotting specifics, including saving intervals and directories
        """
        self.updater = updater
        self.epochs = (epochs, 'epoch')
        self._plot_interval = (kwargs.pop('plot_interval', 1), 'iteration')
        self._disp_interval = (kwargs.pop('disp_interval', 1), 'iteration')
        self._snap_interval = (kwargs.pop('snap_interval', 1), 'iteration')
        self._out_dir = kwargs.pop('out_dir', '')
        super(GANTrainer, self).__init__(updater, stop_trigger=self.epochs, out=self._out_dir)

    def write_report(self):
        """ Extend the trainer to write object snapshots and log reports"""

        generator = self.updater.generator
        discriminator = self.updater.discriminator

        # Get snapshots of generator, discriminator objects
        self.extend(extensions.snapshot_object(generator, 'gen_iter_{.updater.iteration}.npz'),
                    trigger=self._snap_interval)
        self.extend(extensions.snapshot_object(discriminator, 'dis_iter_{.updater.iteration}.npz'),
                    trigger=self._snap_interval)
        self.extend(extensions.LogReport(trigger=self._disp_interval, log_name='log'))

        # Log performances and error during training
        self.extend(extensions.PrintReport(['epoch', 'iteration', 'gen-opt/loss', 'disc-opt/loss', ]),
                    trigger=self._disp_interval)
        self.extend(extensions.ProgressBar(update_interval=self._disp_interval[0]))

        # Every plot_interval, generate a new video and save
        self.extend(self.__generate_training_video(generator),trigger=self._plot_interval)

    def __generate_training_video(self, generator, num_videos=2):
        """ Extension from to generate videos during training """

        @chainer.training.make_extension()
        def get_video(trainer):

            xp = generator.xp
            if not os.path.exists(self._out_dir):
                os.makedirs(self._out_dir)

            # Set batch size to 2, to create a single video instead of multiple ones
            original_batch_size = generator.batch_size
            generator.batch_size = num_videos
            latent_z = generator.sample_hidden()

            # Generate a new video and retrieve only the data
            with chainer.using_config('train', False) and chainer.using_config('enable_backprop', False):

                vid = generator(latent_z)

                if chainer.backends.cuda.get_device_from_array(vid.array).id >= 0:
                    vid = xp.asnumpy(vid.data)
                else:
                    vid = vid.data

                for i in range(num_videos):
                    # Write the videos as .gif by writing individual frames to imageio package writer
                    filename = os.path.join(self._out_dir, "vid{}_epoch_{}_iter{}.gif").format(
                                            i, trainer.updater.epoch, trainer.updater.iteration)
                    with imageio.get_writer(filename, mode='I') as writer:
                            for j in range(generator.n_frames):
                                frame = np.swapaxes(np.squeeze(vid[i, :, :, :, j]), 0, 2)
                                frame = imageio.core.image_as_uint(frame)# Rescale to uint8 range
                                writer.append_data(frame)

            # Reset original batch_size for further training
            generator.batch_size = original_batch_size
        return get_video
