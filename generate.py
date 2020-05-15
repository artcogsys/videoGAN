import chainer
from chainer import serializers
import nn.DCGAN import Generator
from utils import config_parse as cp
from computations.trainer import generate_training_video

MODEL_PATH = './models/generator.npz'
N_GENERATIONS = 100

conf_parser = cp.Config('setup.ini')
params = conf_parser.get_params()
generator = Generator(params.ch, params.latent_dim)
serializers.load_npz(MODEL_PATH, generator)

for i in range(N_GENERATIONS):
    print(f'Generator video {i} / {N_GENERATIONS}', end='\r')
    generate_training_video(generator, filename=f'./gifs/gif_{i}.gif')
