# dev_net.py
# TODO
# [x] Bring items like verbosity, reset, etc. out of ParameterRegister because
#     I don't want the hash to be dependent on that - it's not really a hyperparam
# [ ] Reload on instantiation if cache exists, unless reset is set
# [ ] implement reset option with cache
# [ ] implement reload option with cache
# [ ] Implement logging
# [ ] Implement initializer
# [x] Figure out better network instantiation:

from torch import nn
from .utils.general_utils import ParameterRegister
import ubelt as ub
from collections import Iterable


class NetworkSkeleton(nn.Module):
    def __init__(self, **kwargs):
        """
            Example:
                >>> self = NetworkSkeleton(input_shape=(1, 1, 5, 5),
                ...                        output_len=2,
                ...                        initializer=None,
                ...                        verbosity=1,
                ...                        nice_name='untitled',
                ...                        work_dir='./models')
                NetworkSkeleton: models/untitled/model.pkl
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
                           'models_dpath': str,  # TODO deprecated, right?
                           }
            constraints.update(self.constraints)
        except AttributeError:
            pass
        finally:
            self.constraints = constraints

        # If defaults to hyperparameters  are defined by subclass, add to them
        try:
            defaults = {'verbosity': 0,
                        'input_shape': None,
                        'output_len': None,
                        'nice_name': 'untitled',
                        'work_dir': '.',
                        'initializer': None,
                        'reset': False,
                        'models_dpath': './models',  # TODO deprecated, right?
                        }
            defaults.update(self.defaults)
        except AttributeError:
            pass
        finally:
            self.defaults = defaults

        self.hyperparams = ParameterRegister(self.constraints, self.defaults)

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

        # We don't want these to be included in the caching configuration
        #  string, since they're not really hyperparameters
        self._v = self.hyperparams['verbosity']
        self._reset = self.hyperparams['reset']
        self._work_dir = self.hyperparams['work_dir']

        self.hyperparams.unset('verbosity')
        self.hyperparams.unset('reset')
        self.hyperparams.unset('work_dir')

        self._cache_name = ub.hash_data(self.hyperparams.hashable_str, base='abc')
        self._cacher = ub.Cacher(fname=self.hyperparams['nice_name'],
                                 cfgstr=self._cache_name,
                                 dpath=self._work_dir)
        self._cacher_params = ub.Cacher(fname=self.hyperparams['nice_name'] + '_params',
                                        cfgstr=self._cache_name,
                                        dpath=self._work_dir)

        self._best_val_error = None
        self._best_loss = None
        self.epoch = None

        if self._v > 0:
            print('Cache location: {}'.format(self._cacher.get_fpath()))

    def parameters(self):
        return super().parameters()

    def to(self, device):
        #self._net = self._net.to(device)
        #return self
        return super().to(device)

    def on_epoch(self, epoch=None, error=None, loss=None):
        assert loss is not None
        if epoch is None:
            self.epoch += 1
        else:
            self.epoch = epoch
        if self._best_loss is None or loss < self._best_loss:
            self._best_val_error = error
            self._best_loss = loss
            self.cache()

    def cache(self):
        cached_data = self.state_dict()
        cached_data['epoch'] = self.epoch
        cached_data['best_val_error'] = self._best_val_error
        cached_data['best_loss'] = self._best_loss
        self._cacher.save(cached_data)
        self._cacher_params.save(self.hyperparams.hashable_str)

    def load_cache(self):
        if self._v > 0:
            print("Attempting cache load at {}".format(self._cacher.get_fpath()))
        data = self._cacher.tryload()
        if data is None:
            if self._v > 0:
                print('Cacher did not find a model at {}'.format(self._cacher.get_fpath()))
        else:
            self._best_loss = data.pop('best_loss')
            self._best_val_error = data.pop('best_val_error')
            self.epoch = data.pop('epoch')
            self.load_state_dict(data)
            if self._v > 0:
                print('Loaded model from {}\n  epoch: {}\n  error: {}\n  loss: {}'.format(
                    self._cacher.get_fpath(),
                    self.epoch,
                    self._best_val_error,
                    self._best_loss))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
