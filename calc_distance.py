import chainer
from nn.DCGAN import Generator, Discriminator
from chainer import serializers
from utils import config_parse as cp
from computations.trainer import generate_training_video
import glob
import os
from PIL import Image
import numpy as np

def preprocess_vid(video):

    #xp = cuda.get_array_module()

    frame_size = 64
    n_frames = 32

    # Rescale the image to fit the actual frame size. A video of e.g. of original size {128 x 128} is then
    # reduced to {frame_size x frame_size}
    r = frame_size / video.size[0]  # Scaling factor
    dim = (frame_size, (int(video.size[1] * r)))

    # We want to resize (not center crop) and thereby maintain the aspect ratio!
    # !Changing to numpy array also changes dimension order!
    video = np.array(video.resize(dim))
    # determine the actual number of frames in the vid
    frames_video = video.shape[0] // frame_size

    video = video.reshape(
          (frames_video, frame_size, frame_size, 3))
    video = np.transpose(video)  # make last dimension first

    # Either take the first n_frames, or if the frame length is too short, create repeated copies of last frame
    # and append
    if frames_video >= n_frames:
            video = video[:, :, :, 0: n_frames]
    else:
        last_frame = video[:, :, :, -1]
        last = np.repeat(last_frame[:, :, :, None],
                        n_frames - frames_video, axis=3)
        video = np.concatenate((video, last), axis=3)

    # Recale to -1 / 1 for tanh activation
    video = np.interp(video, (video.min(), video.max()),
                    (-1, +1)).astype(np.float32)

    return video

MODE = 'cat'

if MODE == 'cat':
    discriminator_path = './wdistance_truck/dis_iter_cats.npz'
    generator_path = './wdistance_truck/gen_iter_cats.npz'
    video_path = './wdistance_truck/videos_cats'

else:
    discriminator_path = './wdistance_truck/dis_iter_truck.npz'
    generator_path = './wdistance_truck/gen_iter_truck.npz'
    video_path = './wdistance_truck/videos_trucks'


paths = os.listdir(video_path)
all_paths = glob.glob(video_path + '/**/0*3.jpg', recursive=True) # filenames

nvideos = len(all_paths)

conf_parser = cp.Config('setup.ini')
params = conf_parser.get_params()

generator = Generator(params.ch, params.latent_dim)
discriminator = Discriminator(params.ch)

serializers.load_npz(generator_path, generator)
serializers.load_npz(discriminator_path, discriminator)

wdistance = []


for i in range(0, nvideos-1, 2):

    real_vid1 = Image.open(all_paths[i])
    real_vid2 = Image.open(all_paths[i+1])

    real_vid1 = preprocess_vid(real_vid1)
    real_vid2 = preprocess_vid(real_vid2)

    real_vid = np.array([real_vid1, real_vid2])

    latent_z = generator.sample_hidden(2)
    fake_vid = generator(latent_z)


    # Generate a new video and retrieve only the data
    with chainer.using_config('train', False) and chainer.using_config('enable_backprop', False):
        eval_real = chainer.as_array(discriminator(real_vid))
        eval_fake = chainer.as_array(discriminator(fake_vid))

    dist_real_fake = np.abs(eval_real - eval_fake)
    wdistance.append(dist_real_fake)

    print(dist_real_fake)


wdistance = np.array(wdistance).flatten()
print(f'Average wasserstein distance for {MODE} and {nvideos} test samples = {np.mean(wdistance)}')
