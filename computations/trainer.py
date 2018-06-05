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
        self.plot_interval = (kwargs.pop('plot_interval', 1), 'iteration')
        self.disp_interval = (kwargs.pop('disp_interval', 1), 'iteration')
        self.snap_interval = (kwargs.pop('snap_interval', 1), 'iteration')
        self.out_dir = kwargs.pop('out_dir', '')
        super(GANTrainer, self).__init__(updater, stop_trigger=self.epochs, out=self.out_dir)

    def write_report(self):
        """ Extend the trainer to write object snapshots and log reports"""

        generator = self.updater.generator
        discriminator = self.updater.discriminator

        # Get snapshots of generator, discriminator objects
        self.extend(extensions.snapshot_object(generator, 'gen_iter_{.updater.iteration}.npz'),
                    trigger=self.snap_interval)
        self.extend(extensions.snapshot_object(discriminator, 'dis_iter_{.updater.iteration}.npz'),
                    trigger=self.snap_interval)
        self.extend(extensions.LogReport(trigger=self.disp_interval))

        # Log performances and error during training
        self.extend(extensions.PrintReport(['epoch', 'iteration', 'gen-opt/loss', 'disc-opt/loss', ]),
                    trigger=self.disp_interval)
        self.extend(extensions.ProgressBar(update_interval=2))

        # Every <attribute> plot_interval, generate a new video and save
        self.extend(self.__generate_training_video(generator),trigger=self.plot_interval)

    def __generate_training_video(self, generator):
        """ Extension from to generate videos during training """

        @chainer.training.make_extension()
        def get_video(trainer):

            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

            # Set batch size to 1, to create a single video instead of multiple ones
            original_batch_size = generator.batch_size
            generator.batch_size = 1
            latent_z = generator.sample_hidden()

            # Generate a new video and retrieve only the data
            with chainer.using_config('train', False) and chainer.using_config('enable_backprop', False):
                vid = generator(latent_z).data

            # Write the videos as .gif by writing individual frames to imageio package writer
            filename = os.path.join(self.out_dir, "vid_iter_{}.gif").format(trainer.updater.iteration)
            with imageio.get_writer(filename, mode='I') as writer:
                for i in range(generator.n_frames):
                    frame = np.swapaxes(np.squeeze(vid[:, :, :, :, i]), 0, 2)
                    writer.append_data(frame.astype(np.uint8))

            # Reset original batch_size for further training
            generator.batch_size = original_batch_size
        return get_video