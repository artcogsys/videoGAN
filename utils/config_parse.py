"""
__author__: Florian Mahner
__email__: fmahner@uos.de
__status__: Development
__date__: 11-05-2018
"""
import configparser

class Config(object):
    """ Reading parameters for videoGAN from setup.ini file and return as parameter dictionary """

    def __init__(self, cfgfile):
        self.parser = configparser.ConfigParser()
        self.parser.read(cfgfile)

    def get_params(self):

        params = {}
        try:
            params['Directories'] = {}
            params['Directories']['root_dir'] = self.parser.get('DIRECTORIES', 'root_dir')
            params['Directories']['index_dir'] = self.parser.get('DIRECTORIES', 'index_dir')
            params['Directories']['out_dir'] = self.parser.get('DIRECTORIES', 'out_dir')

            params['Model'] = {}
            params['Model']['batch_size'] = self.parser.getint('MODEL', 'batch_size')
            params['Model']['n_frames'] = self.parser.getint('MODEL', 'n_frames')
            params['Model']['frame_size'] = self.parser.getint('MODEL', 'frame_size')
            params['Model']['epoch'] = self.parser.getint('MODEL', 'epoch')
            params['Model']['use_gpu'] = self.parser.getint('MODEL', 'use_gpu')
            params['Model']['latent_dim'] = self.parser.getint('MODEL', 'latent_dim')
            params['Model']['weight_scale'] = self.parser.getfloat('MODEL', 'weight_scale')

            params['Saving'] = {}
            params['Saving']['snap_interval'] = self.parser.getint('SAVING', 'snap_interval')
            params['Saving']['display_interval'] = self.parser.getint('SAVING', 'display_interval')
            params['Saving']['print_interval'] = self.parser.getint('SAVING', 'print_interval')
            params['Saving']['plot_interval'] = self.parser.getint('SAVING', 'plot_interval')

            params['Adam'] = {}
            params['Adam']['learning_rate'] = self.parser.getfloat('ADAM', 'learning_rate')
            params['Adam']['beta1'] = self.parser.getfloat('ADAM', 'beta1')
            params['Adam']['beta2'] = self.parser.getfloat('ADAM', 'beta2')
            params['Adam']['weight_decay'] = self.parser.getfloat('ADAM', 'weight_decay')
            params['Adam']['critic_iter'] = self.parser.getint('ADAM', 'critic_iter')
            params['Adam']['penalty_coeff'] = self.parser.getint('ADAM', 'penalty_coeff')

        except configparser.NoOptionError as er:
            print('Option could not be loaded, following error: {}'.format(er))

        return params
