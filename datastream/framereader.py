"""
__author__: Florian Mahner
__email__: fmahner@uos.de
__status__: Development
__date__: 30-05-2018

This file creates an input pipeline to load the videos from a given root-directory. The resulting data is passed onto
an <instance> of a SerialIterator to obtain mini-batches for training.
"""

from chainer import cuda
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
import chainer
from chainer import Variable


class FrameReader(chainer.dataset.DatasetMixin):
    """
    Every FrameReader uses an index file (e.g. job-list.txt) or similar that contains direcotry locations to videos.
    The Videos are supposed to be stored as vertically stacked frames for every second of video. The number of frames
    and frame sizes are then fit into a common framework and each second of the video returned as an individual batch of
    size. A video of 16 seconds would then be of size: [16, 3 (RGB), frame_size, frame_size, n_frames]
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
        self.content, self.data_size = self.__get_paths()

    def __len__(self):
        return self.data_size

    def get_example(self, idx):
        if idx > self.data_size:
            raise IndexError('Index out of range. Not able to retrieve example')

        batch = cv2.imread(self.content[idx])
        # Get preprocessed batch
        return self.__preprocess_vid(batch)

    # Is it important that the Framereader reads all batches from a path distinct from each other?!

    def __get_paths(self):
        """ Make a directory list from text file to read all files in directory with given :attribute: extension, no
            irrespective of the filenam. Rename to get_paths!"""

        with open(os.path.join(self.root_dir, self.index_file)) as f:
            paths = f.readlines()

        all_paths = []
        # Get all path from index file and save all file path in its subdirs
        for path in tqdm(paths):
            if path.strip():  # Ignore empty lines from filereader
                file_paths = os.path.join(path.strip(), ('*' + self.ext))
                all_paths.extend(glob.glob(file_paths))

        return all_paths, len(all_paths)

    def __preprocess_vid(self, video):

        xp = cuda.get_array_module()

        frame_size = self.frame_size
        n_frames = self.n_frames

        r = frame_size / video.shape[1] # Scaling factor
        dim = (frame_size, (int(video.shape[0] * r)))

        # Rescale the image to fit the actual frame size. A video of e.g. of original size {128 x 128} is then
        # reduced to {frame_size x frame_size}
        video = cv2.resize(video, dim)
        frames_video = video.shape[0] // frame_size
        video = xp.reshape(video, [3, 64, 64, frames_video])

        # Either take the first n_frames, or if the frame length is too short, create repeated copies of last frame
        # and append
        if frames_video >= n_frames:
            video = video[:, :, :, 0 : n_frames]
        else:
            last = xp.tile(video[-1], (1, 1, 1, n_frames - frames_video))
            video = xp.concatenate((video, last))

        # Rescale video from -1 to 1 due to tanh activation function of generator
        #video = video.astype(np.float32)
        video = np.interp(video, (video.min(), video.max()), (-1, +1)).astype(np.float32)

        return video
