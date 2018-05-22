import sys
import configparser
from optparse import OptionParser

class Config(object):
    ''' Reading videoGAN parameters'''

    def __init__(self, cfgfile):
        self.parser = configparser.ConfigParser()
        self.parser.read(cfgfile)

    def get_params(self):

        params = {}
        try:
            params['in_dir'] = self.parser.get('DIRECTORIES', 'root_dir')
            params['out_dir'] = self.parser.get('DIRECTORIES', 'out_dir')

            params['Model'] = {}
            params['Model']['batch_size'] = self.parser.getint('MODEL', 'batch_size')
            params['Model']['epoch'] = self.parser.getint('MODEL', 'epoch')
            params['Model']['use_gpu'] = self.parser.getint('MODEL', 'use_gpu')
            params['Model']['gradient_penalty'] = self.parser.getint('MODEL', 'gradient_penalty')
            params['Model']['n_latent_z'] = self.parser.getint('MODEL', 'latent_dim')
            params['Model']['weight_scale'] = self.parser.getint('MODEL', 'weight_scale')

            params['snap_interval'] = self.parser.getint('SAVING', 'snap_interval')
            params['display_interval'] = self.parser.getint('SAVING', 'display_interval')

            print(params)

        except configparser.NoOptionError as er:
            print('Option could not be loaded, following error: {}'.format(er))

        return params
