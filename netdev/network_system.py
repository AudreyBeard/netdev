from .utils.general_utils import ParameterRegister
import ubelt as ub
import torch
# TODO:
# [ ] logging
# [ ]


class SystemHyperparameters(object):
    def __init__(self, **kwargs):
        


class NetworkSystem(object):
    def __init__(self, verbosity=1, epochs=1, work_dir='./models',
                 device='cpu', hash_on=None, nice_name='untitled',
                 batch_size=1, **kwargs):
        """
            Network training and testing system, designed for modules to plug
            into for simple part swapping. Several system parameters are simply
            for housekeeping and therefore do not need to be considered when
            saving the model, others affect the performance of the model and
            should therefore be used in differentiating between different
            models when saving.
        """
        try:
            constraints = {
                'model': None,
                'objective': None,
                'optimizer': None,
                'datasets': None,
                'loaders': None,
            }
            constraints.update(self.constraints)
        except AttributeError:
            pass
        finally:
            self.constraints = constraints

        # If defaults to hyperparameters  are defined by subclass, add to them
        try:
            defaults = {
                'model': None,
                'objective': None,
                'optimizer': None,
                'datasets': None,
                'loaders': None,
            }
            defaults.update(self.defaults)
        except AttributeError:
            pass
        finally:
            self.defaults = defaults

        self.modules = ParameterRegister(self.constraints, self.defaults)

        # Check if hyperparameters are properly specified
        kwarg_isgood = self.modules.check_kwargs(**kwargs)
        if not all(kwarg_isgood.values()):
            msg = self._bad_init_msg(kwarg_isgood, kwargs)
            raise ValueError("\n" + msg)
        else:
            # Set hyperparameters
            for k, v in sorted(kwargs.items()):
                self.modules[k] = v

        self.modules.set_uninitialized_params(self.defaults)

        self._v = verbosity
        self.epochs = epochs
        self.dir = work_dir
        self._dev = device
        self.nice_name = nice_name
        self.batch_size = batch_size

        self.cache_name = None
        self.cacher = None
        self.location = None

        self.sequential_log = [None for i in range(self.epochs)]

        # If user specified parameters to use for hash cfgstr, use them
        if hash_on is not None:
            self.init_cacher(hash_on=hash_on)
        # Default: hash on modules
        else:
            self.init_cacher(hash_on=self.modules)

        # Attach all passed-in parameters to the system
        for k, v in self.modules.items():
            self.__setattr__(k, v)

        return
        self.status = None

    def init_cacher(self, hash_on=dict()):
        hashable = '_'.join(['{}:{}; '.format(k, v) for k, v in hash_on.items()])
        self.cache_name = ub.hash_data(hashable, base='abc')
        self.cacher = ub.Cacher(fname=self.nice_name,
                                cfgstr=self.cache_name,
                                dpath=self.dir)
        self.location = self.cacher.get_fpath()

    def train(self):
        self.status = 'training'
        for e in range(self.epochs):
            self.epoch = e
            for i, data in enumerate(self.loaders['train']):
                batch_stats_dict = self.forward(data)

                self.log_it(batch_stats_dict)

                self.backward(batch_stats_dict['loss'])

            for i, data in enumerate(self.loaders['val']):
                with torch.no_grad():
                    loss_dict_val = self.forward_val(data)
                    self.log_it(loss_dict_val)

            self.on_epoch()
        return

    def forward(self):
        """ Analogous to the torch forward method - feed-forward component
            Implement this in all subclasses
            Should return a dictionary with at least a 'loss' key corresponding
            to the scalar loss value returned by self.objective
        """
        raise NotImplementedError

    def on_epoch(self):
        # TODO Handle all caching logic here, not in the network
        for k in self.sequential_log.keys():
            self.sequential_log[k] /= self.batch_size
        self.model.on_epoch(epoch=self.epoch,
                            loss=self.sequential_log['loss_val'][self.epoch],
                            error=self.sequential_log['error_val'][self.epoch])

    def backward(self, loss):
        """ Analog to the torch backward method - backpropagation
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def log_it(self, **kwargs):
        for k, v in kwargs.items():
            self.sequential_log[k][self.epoch] += v

    def test(self):
        self.status = 'testing'
        # TODO should I implement the skeleton of this?
        raise NotImplementedError

    def reset(self):
        # TODO implement this
        raise NotImplementedError
