# dev_net.py
# TODO
# [ ] implement reset option with cache
# [ ] implement reload option with cache
# [ ] Implement logging
# [ ] Implement initializer
# [ ] Figure out better network instantiation:
#       Do I want to have the user specify the network architecture exactly?
#           Probably yes.
from torch import nn
#import os
import util
import ubelt as ub
from collections import Iterable
import ipdb  # NOQA
import IPython  # NOQA


class DevelopmentNetwork(nn.Module):
    # Constraints on any hyperparameters
    # TODO do I want to bring verbosity, workdir, reset, and models_dpath outside this
    #   hyperparameters list? Technically, they don't affect the model, so that
    #   makes sense
    # Default values for hyperparameters
    def __init__(self, **kwargs):
        """
            Example:
                >>> self = DevelopmentNetwork(input_shape=(1, 1, 5, 5),
                ...                           output_len=2,
                ...                           initializer=None,
                ...                           verbosity=1,
                ...                           nice_name='untitled',
                ...                           work_dir='./models')
                DevelopmentNetwork: models/untitled/model.pkl
        """
        super().__init__()

        # If constraints to hyperparameters are defined by subclass, add to them
        try:
            constraints = {'verbosity': int,
                           'input_shape': Iterable,
                           'output_len': (int, '>0'),
                           'nice_name': str,
                           'work_dir': str,
                           'initializer': lambda x: x is None or isinstance(x, str),
                           'reset': bool,
                           'models_dpath': str,
                           }
            self.__getattr__('constraints')
        except AttributeError:
            self.constraints = constraints
        else:
            self.constraints.update(constraints)

        # If defaults to hyperparameters  are defined by subclass, add to them
        try:
            defaults = {'verbosity': 0,
                        'input_shape': None,
                        'output_len': None,
                        'nice_name': 'untitled',
                        'work_dir': '.',
                        'initializer': None,
                        'reset': False,
                        'models_dpath': './models',
                        }
            self.__getattr__('defaults')
        except AttributeError:
            self.defaults = defaults
        else:
            self.defaults.update(defaults)

        self.hyperparams = util.ParameterRegister(self.constraints, self.defaults)

        # Check if hyperparameters are properly specified
        kwarg_isgood = self.hyperparams.check_kwargs(**kwargs)
        if not all(kwarg_isgood.values()):
            msg = self._bad_init_msg(kwarg_isgood, kwargs)
            raise ValueError("\n" + msg)
        else:
            # Set hyperparameters
            for k, v in sorted(kwargs.items()):
                self.hyperparams[k] = v

        self.hyperparams.set_uninitialized_params(self.defaults)

        self._v = self.hyperparams['verbosity']
        self._cache_name = ub.hash_data(self.hyperparams.hashable_str, base='abc')
        self._cacher = ub.Cacher(fname=self.hyperparams['nice_name'],
                                 cfgstr=self._cache_name,
                                 dpath=self.hyperparams['work_dir'])
        self._cacher_params = ub.Cacher(fname=self.hyperparams['nice_name'] + '_params',
                                        cfgstr=self._cache_name,
                                        dpath=self.hyperparams['work_dir'])

        self._best_val_error = None
        self.epoch = None

        if self._v > 0:
            print('Cache location: {}'.format(self._cacher.get_fpath()))

        # Set all unitialized hyperparameters to their default value

    def parameters(self):
        #return ((p for p in self._net[i].parameters()) for i in range(len(self._net)))
        #return(p for component in self._net for p in component.parameters())
        return super().parameters()

    def to(self, device):
        #self._net = self._net.to(device)
        #return self
        return super().to(device)

    def on_epoch(self, epoch, error):
        self.epoch = epoch
        if self._best_val_error is None or error < self._best_val_error:
            self._best_val_error = error
            self.cache()

    def cache(self):
        cached_data = self.state_dict()
        cached_data['epoch'] = self.epoch
        cached_data['best_val_error'] = self._best_val_error
        self._cacher.save(cached_data)
        self._cacher_params.save(self.hyperparams.hashable_str)

    def load_cache(self):
        data = self._cacher.tryload()
        if data is None:
            print('Cacher did not find a model at {}'.format(self._cacher.get_fpath()))
        else:
            self._best_val_error = data.pop('best_val_error')
            self.epoch = data.pop('epoch')
            self.load_state_dict(data)
            if self._v > 0:
                print('Loaded model from {}\n  epoch: {}\n  error: {}'.format(
                    self._cacher.get_fpath(),
                    self.epoch,
                    self._best_val_error))


# The sequential subnet
# This may become deprecated, since it hides the network architecture and is
# somewhat immutable
# TODO this is deprecated
class Subnet(nn.Sequential):
    def __init__(self, input_shape, *args):
        super().__init__()
        self._components = []
        for i in range(len(args)):
            self._components.append(args[i])

        self._output_shape = [util.output_shape_for(input_shape,
                                                    self._components[0])]

        self._output_shape.extend([util.output_shape_for(self._output_shape[i - 1],
                                                         self._components[i])
                                   for i in range(1, len(self) - 1)])

    def forward(self, x):
        y = x
        for i in range(len(self)):
            y = self[i](y)
        return y

    def state_dict(self):
        components = {i: self[i].state_dict() for i in range(len(self))}
        return components

    def load_state_dict(self, state):
        (self._components.load_state_dict(state[i]) for i in range(len(state)))

    def __len__(self):
        return len(self._components)

    def __getitem__(self, idx):
        return self._components[idx]

    def __repr__(self):
        string = '{} of length {}\n'.format(self.__class__.__name__, len(self))
        for i in range(len(self)):
            string += '[{}] '.format(i)
            string += self[i].__repr__()
        return string

    def to(self, device):
        for i in range(len(self)):
            self._components[i].to(device)
        return self


if __name__ == "__main__":
    import doctest
    doctest.testmod()
