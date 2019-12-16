"""
__author__: Florian Mahner
__email__: fmahner@uos.de
__status__: Development
__date__: 30-05-2018

This file creates an input pipeline to load the videos from a given root-directory using the built-in dataset
abstraction :class:`~chainer.dataset.DatasetMixin`
"""
from chainer import cuda
from PIL import Image
import glob
import os
import numpy as np
from tqdm import tqdm
import chainer
import logging

class FrameReader(chainer.dataset.DatasetMixin):
    """ Every FrameReader uses an index file (e.g. job-list.txt) with directory locations to videos. Videos are assumed
    to be stored asvertically stacked frames for every second of video. The number of frames and frame sizes are then
    reshaped to equal sizes. A video of 16 seconds would then be of size: [16, 3 (RGB), frame_size, frame_size, n_frames]
    Mini-batches for training are obtained by a :class:`~chainer.iterators.MutliprocessIterator, that calls
    :func:`~FrameReader.get_example()` n-times for a given batch size
    """

    def __init__(self, root_dir, index_file, n_frames=32, frame_size=64, ext='.jpg'):
        """
        :param root_dir: Base directory where all videos are stored in sub-directories. contains index-file
        :param index_file: index-file with explicit directories to video locations
        :param n_frames: number of frames all videos are supposed to have
        :param frame_size: size of a frame or crop size (e.g. 64x64)
        :param ext: file extension of vertically stacked frames. All files need to have the same extension
        """
        self.root_dir = root_dir
        self.index_file = index_file
        self.ext = ext
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.logger = logging.getLogger('main.framereader')
        self.content, self.data_size = self.__get_paths()
        assert self.data_size > 0, 'Dataset is empty. Have a look at the input paths.'

    def get_example(self, idx):
        """ Returns a preprocessed example from the path directory. If an exception occurs and the preprocess fails,
        report that exception and fetch recursively a random new sample """
        if idx > self.data_size:
            raise IndexError('Index out of range. Not able to retrieve example')

        try:
            batch = Image.open(self.content[idx])
            batch = self.__preprocess_vid(batch)

        except ValueError:
            self.logger.error('Could not preprocess {}. Take another random sample from set'.format(self.content[idx]))
            return self.get_example(np.random.randint(0, self.data_size))

        else:
            return batch

    def __len__(self):
        """ Get datasize to set maximum indexing for iterator """
        return self.data_size

    def __get_paths(self):
        """ Make a directory list from text file to read all files in directory with given :attr:`~extension`,
            irrespective of the filename. """
        with open(os.path.join(self.root_dir, self.index_file)) as f:
            paths = f.readlines()

        # Get all path from index file and save all file paths entailed in its subdirectory
        all_paths = []
        for path in tqdm(paths, file=self.logger.parent.handlers[0].stream):
            if path.strip():  # Ignore empty lines from filereader
                file_path = os.path.join(path.strip(), ('*' + self.ext))
                all_paths.extend(glob.glob(file_path))

        return all_paths, len(all_paths)

    def __preprocess_vid(self, video):

        xp = cuda.get_array_module()

        # Rescale the image to fit the actual frame size. A video of e.g. of original size {128 x 128} is then
        # reduced to {frame_size x frame_size}
        r = self.frame_size / video.size[0] # Scaling factor
        dim = (self.frame_size, (int(video.size[1] * r)))

        # We want to resize (not center crop) and thereby maintain the aspect ratio!
        # !Changing to numpy array also changes dimension order!
        video = xp.array(video.resize(dim))
        frames_video = video.shape[0] // self.frame_size # determine the actual number of frames in the vid


        video = video.reshape((frames_video, self.frame_size, self.frame_size, 3))
        video = xp.transpose(video) # make last dimension first

        # Either take the first n_frames, or if the frame length is too short, create repeated copies of last frame
        # and append
        if frames_video >= self.n_frames:
            video = video[:, :, :, 0 : self.n_frames]
        else:
            last_frame = video[:,:,:,-1]
            last = xp.repeat(last_frame[:,:,:,None], self.n_frames - frames_video, axis=3)
            video = xp.concatenate((video, last), axis=3)

        # Recale to -1 / 1 for tanh activation
        video = xp.interp(video, (video.min(), video.max()), (-1, +1)).astype(xp.float32)

        return video
