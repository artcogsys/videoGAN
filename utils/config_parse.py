import configparser

class Config(configparser.ConfigParser):
    """ Reading parameters from setup.ini file """

    def __init__(self, cfgfile):
        super().__init__({}, dict)
        super().read(cfgfile)

    def get_params(self):
        d = {}
        for section in self._sections:
            for key, value in self.items(section):
                # Return value as original class type if possible, otherwise as string
                try:
                    d[key] = eval(value)
                except NameError:
                    d[key] = value

        return AttrDict(d)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
