from .utils.general_utils import ParameterRegister
import ubelt as ub
import torch


class NetworkSchool(object):
    def __init__(self, **kwargs):
        try:
            constraints = {
                'verbosity': int,
                'epochs': int,
                'model': None,
                'objective': None,
                'optimizer': None,
                'datasets': None,
                'loaders': None,
                'work_dir': str,
                'device': str,
                'hash_on': dict,
                'nice_name': str,
            }
            constraints.update(self.constraints)
        except AttributeError:
            pass
        finally:
            self.constraints = constraints

        # If defaults to hyperparameters  are defined by subclass, add to them
        try:
            defaults = {
                'verbosity': 0,
                'epochs': 1,
                'model': None,
                'objective': None,
                'optimizer': None,
                'datasets': None,
                'loaders': None,
                'work_dir': './models',
                'device': 'cpu',
                'hash_on': None,
                'nice_name': self.__class__.__name__,
            }
            defaults.update(self.defaults)
        except AttributeError:
            pass
        finally:
            self.defaults = defaults

        self.params = ParameterRegister(self.constraints, self.defaults)

        # Check if hyperparameters are properly specified
        kwarg_isgood = self.params.check_kwargs(**kwargs)
        if not all(kwarg_isgood.values()):
            msg = self._bad_init_msg(kwarg_isgood, kwargs)
            raise ValueError("\n" + msg)
        else:
            # Set hyperparameters
            for k, v in sorted(kwargs.items()):
                self.params[k] = v

        self.params.set_uninitialized_params(self.defaults)

        self.epochs = self.params['epochs']
        self.sequential_log = [None for i in range(self.epochs)]

        self.cache_name = None
        self.cacher = None
        self.location = None

        if self.params['hash_on'] is not None:
            self.init_cacher(**self.params['hash_on'])

        return

    def init_cacher(self, **kwargs):
        hashable = '_'.join(['{}:{}'.format(k, v) for k, v in kwargs.items()])
        self.cache_name = ub.hash_data(hashable, base='abc')
        self.cacher = ub.Cacher(fname=self.params['nice_name'],
                                cfgstr=self.cache_name,
                                dpath=self.params['work_dir'])
        self.location = self.cacher.get_fpath()

    def train(self):
        for e in range(self.epochs):
            self.epoch = e
            for i, data in enumerate(self.loaders['train']):
                loss_dict = self.forward(data)

                self.log_it(loss_dict)

                self.backward(loss_dict['loss'])

            for i, data in enumerate(self.loaders['val']):
                with torch.no_grad():
                    loss_dict_val = self.forward_val(data)
                    self.log_it(loss_dict_val)

            self.on_epoch()
        return

    def on_epoch(self):
        for k in self.sequential_log.keys():
            self.sequential_log[k] /= self.params['batch_size']
        self.model.on_epoch(epoch=self.epoch,
                            loss=self.sequential_log['loss_val'][self.epoch],
                            error=self.sequential_log['error_val'][self.epoch])

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def log_it(self, **kwargs):
        for k, v in kwargs.items():
            self.sequential_log[k][self.epoch] += v

    def test(self):
        raise NotImplementedError
        return

    def reset(self):
        raise NotImplementedError
        return
