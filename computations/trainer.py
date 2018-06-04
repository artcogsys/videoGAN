"""
__author__: Florian Mahner
__email__: fmahner@uos.de
__status__: Development
__date__: 11-05-2018
"""

import chainer
import imageio
import os
import numpy as np
from chainer.training import extensions


class GANTrainer(chainer.training.Trainer):

    def __init__(self, updater, epochs=1000, **kwargs):

        self.updater = updater
        self.epochs = (epochs, 'epoch')
        self.n_frames = kwargs.pop('n_frames', 32)
        self.epochs = kwargs.pop('epoch', 100)
        self.plot_interval = (kwargs.pop('plot_interval', 1), 'iteration')
        self.disp_interval = (kwargs.pop('disp_interval', 1), 'iteration')
        self.snap_interval = (kwargs.pop('snap_interval', 1), 'iteration')
        self.out_dir = kwargs.pop('out_dir', '')


    def write_report(self):

        # Get <protected> attributes from updarter instance
        generator = self.updater.generator
        discriminator = self.updater.discriminator

        self.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),trigger=self.snap_interval)
        self.extend(extensions.snapshot_object(generator, 'gen_iter_{.updater.iteration}.npz'), trigger=self.snap_interval)
        self.extend(extensions.snapshot_object(discriminator, 'dis_iter_{.updater.iteration}.npz'), trigger=self.snap_interval)
        self.extend(extensions.LogReport(trigger=self.disp_interval))

        self.extend(extensions.PrintReport(['epoch', 'iteration', 'gen-opt/loss', 'disc-opt/loss', ]),
                    trigger=self.disp_interval)
        self.extend(extensions.ProgressBar(update_interval=1))
        self.extend(self.__generate_training_video(generator, save_path=self.out_dir),
                    trigger=self.plot_interval)


    def __generate_training_video(self, generator, n_frames, save_path=''):

        @chainer.training.make_extension()
        def get_video(trainer):

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            original_batch_size = generator.get_batch_size()
            generator.get_sample_size(1)
            latent_z = generator.sample_hidden()

            with chainer.using_config('train', False) and chainer.using_config('enable_backprop', False):

                vid = generator(latent_z).data

            # filename = os.path.join(save_path, "iter_{}.jpg").format(trainer.updater.iteration)
            with imageio.get_writer('test_' + str(trainer.updater.iteration) + '.gif', mode='I') as writer:
                for i in range(n_frames):
                    frame = np.swapaxes(np.squeeze(vid[:, :, :, :, i]), 0, 2)
                    writer.append_data(frame.astype(np.uint8))

            generator.set_sample_size(original_batch_size)

        return get_video