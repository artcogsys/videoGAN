import os

import chainer
import numpy as np
from PIL import Image
import glob
import os
from scipy.misc import imresize
import cv2
import numpy as np
from multiprocessing import Pool


class FrameReader(chainer.dataset.DatasetMixin):

    def __init__(self, root_dir, index_file, batch_size=5, n_frames=32, frame_size=64, ext='.jpg'):

        self.root_dir = root_dir
        self.index_file = index_file
        self.ext = ext
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.frame_size = frame_size
        super(FrameReader, self).__init__()

    def __len__(self):
        return

    def get_example(self, i):
        pass

    def load_imgs(self, relative_path):
        # Load all images as array into single list
        image_list = list(map(cv2.imread, glob.glob(relative_path)))
        return image_list


    def read_imgs(self):
        """return video shape: (frame, height, width, ch)"""

        with open(os.path.join(self.root_dir, self.index_file)) as f:
            content = f.readlines()

        # Make a directory list from text file to read all files in directory with <attribute> extension, no matter
        # which filename
        content = [os.path.join(x.strip(), ('*' + self.ext)) for x in content]

        # Pool for multithreading
        pool = Pool()
        images = pool.map(self.load_imgs, content)
        num_videos = len(images)
        batch = pool.map(self._preprocess_imgs, [images[i] for i in range(num_videos)])

        batch = chainer.dataset.concat_examples(batch)

        pool.close()
        pool.join()

        return batch

    def _preprocess_imgs(self, images):

        img_proc = np.empty([len(images), 64, 64, 32, 3], dtype=np.float32) # Preallocate data for computational speed

        for idx, img in enumerate(images):

            r = 64.0 / img.shape[1] # Scaling factor
            dim = (64, (int(img.shape[0] * r)))
            img_scaled = cv2.resize(img, dim) # Rescal the image to fit 64 width

            frames_video = img_scaled.shape[0] // 64
            img = np.reshape(img_scaled, [64, 64, frames_video, 3])

            if frames_video >= self.n_frames:
                img = img[0:self.n_frames]
            else:
                last = np.tile(img[-1], (self.n_frames - frames_video,1,1,1)) # Created repeated copies of last frame
                img = np.concatenate((img, last)) # Join frames!

            img_proc[idx] = img

        return img_proc


import time
start = time.time()

a = FrameReader('/Users/florianmahner/Desktop/Donders_Internship/Programming/videoGAN/VideosSample/', 'job-list.txt')
batches = a.read_imgs()

print('Time: %s' %(time.time() - start))

#path = '/Users/florianmahner/Desktop/Donders_Internship/Programming/videoGAN/VideosSample'


