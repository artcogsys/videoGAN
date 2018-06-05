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
from multiprocessing import Pool


class FrameReader(object):
    """
    Every FrameReader uses an index file (e.g. job-list.txt) or similar that contains direcotry locations to videos.
    The Videos are supposed to be stored as vertically stacked frames for every second of video. The number of frames
    and frame sizes are then fit into a common framework and each second of the video returned as an individual batch of
    size. A video of 16 seconds would then be of size: {16, 3 (RGB), n_frames, frame_size, frame_size}
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

    def load_dataset(self):
        """ Loads all data from index file in a multithreaded fashion.
        :return <list> entire dataset """

        with open(os.path.join(self.root_dir, self.index_file)) as f:
            content = f.readlines()

        # Make a directory list from text file to read all files in directory with given <attribute> extension, no
        # irrespective of the filename
        content = [os.path.join(x.strip(), ('*' + self.ext)) for x in content]

        # Load all images from directories and pass onto thread pool for parallel processing. Images are first loaded
        # and then reshaped into a common dimensionality framework
        pool = Pool()
        images = pool.map(self._load_imgs, content)
        batch = pool.map(self._preprocess_imgs, [images[i] for i in range(len(images))])
        batch = np.concatenate(batch)
        pool.close()
        pool.join()

        return batch

    def _load_imgs(self, relative_path):
        """ Load all images as array into single list """
        image_list = list(map(cv2.imread, glob.glob(relative_path)))
        return image_list

    def _preprocess_imgs(self, images):

        xp = cuda.get_array_module()

        # Preallocate data for computational speed and retrieve attributes once
        img_proc = np.empty([len(images), 3, self.frame_size, self.frame_size, self.n_frames], dtype=np.float32)
        frame_size = self.frame_size
        n_frames = self.n_frames

        for idx, img in enumerate(images):

            r = frame_size / img.shape[1] # Scaling factor
            dim = (frame_size, (int(img.shape[0] * r)))
            # Rescale the image to fit the actual frame size. A video of e.g. of original size {128 x 128} is then
            # reduced to {frame_size x frame_size}
            img = cv2.resize(img, dim)
            frames_video = img.shape[0] // frame_size
            img = xp.reshape(img, [3, 64, 64, frames_video])

            # Either take the first n_frames, or if the frame length is too short, create repeated copies of last frame
            # and append
            if frames_video >= n_frames:
                img = img[:, :, :, 0 : n_frames]
            else:
                last = xp.tile(img[-1], (1, 1, 1, n_frames - frames_video))
                img = xp.concatenate((img, last))

            img_proc[idx] = img

        return img_proc
