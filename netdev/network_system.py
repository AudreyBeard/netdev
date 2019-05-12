#from collections import Iterable
import ubelt as ub
import torch
from .utils.general_utils import ParameterRegister
# TODO:
# [ ] logging
# [x] caching
# [ ] epoch_summary()

__all__ = ['NetworkSystem']


class NetworkSystem(object):
    def __init__(self, verbosity=1, epochs=1, work_dir='./models',
                 device='cpu', hash_on=None, nice_name='untitled',
                 batch_size=1, reset=False, scale_metrics=True,
                 selection_metric='error_val',
                 metrics=['loss_train', 'error_train', 'loss_val', 'error_val'],
                 **kwargs):

        """
            Network training and testing system, designed for modules to plug
            into for simple part swapping. Several system parameters are simply
            for housekeeping and therefore do not need to be considered when
            saving the model, others affect the performance of the model and
            should therefore be used in differentiating between different
            models when saving.
        """
        # These define some important items that impact the model
        try:
            constraints = {
                'model': None,
                'objective': None,
                'optimizer': None,
                'loaders': None,
            }
            constraints.update(self.constraints)
        except AttributeError:
            pass
        finally:
            self.constraints = constraints
        try:
            defaults = {
                'model': None,
                'objective': None,
                'optimizer': None,
                'loaders': None,
            }
            defaults.update(self.defaults)
        except AttributeError:
            pass
        finally:
            self.defaults = defaults

        self.modules = ParameterRegister(self.constraints, self.defaults)

        # Check if modules are properly specified
        poorly_specced = self.modules.try_set_params(**kwargs)
        if poorly_specced:
            raise ValueError(poorly_specced)

        self.modules.set_uninitialized_params(self.defaults)

        self._v = verbosity
        self.epochs = epochs
        self.dir = work_dir
        self.device = device
        self.nice_name = nice_name
        self.batch_size = batch_size
        self.scale_metrics = scale_metrics

        self.cache_name = None
        self.cacher = None
        self.location = None
        self.status = None
        self.epoch = -1

        # Keep a tensor for tracking required metrics
        self.journal = {k: torch.zeros(self.epochs) for k in metrics}
        self.selection_metric = selection_metric
        self.best_metrics = None

        # If user specified parameters to use for hash cfgstr, use them
        if hash_on is not None:
            self.init_cacher(hash_on=hash_on)
        # Default: hash on modules
        else:
            self.init_cacher(hash_on=self.modules)

        # Attach all passed-in parameters to the system
        for k, v in self.modules.items():
            self.__setattr__(k, v)

        if reset:
            self.load()

        return

    def init_cacher(self, hash_on=dict()):
        """ Initialize cacher for saving and loading models
        """
        hashable = '_'.join(['{}:{}; '.format(k, v) for k, v in hash_on.items()])
        self.cache_name = ub.hash_data(hashable, base='abc')
        self.cacher = ub.Cacher(fname=self.nice_name,
                                cfgstr=self.cache_name,
                                dpath=self.dir)
        info_cacher = ub.Cacher(fname=self.nice_name,
                                cfgstr='info',
                                dpath=self.dir)
        info_cacher.save(hashable)
        self.location = self.cacher.get_fpath()

    def train(self, n_epochs=None):
        """ Called once, to train model over the number of defined epochs
        """

        if n_epochs is not None:
            self.epochs = self.epoch + n_epochs

        while self.epoch < self.epochs:
            self.epoch += 1

            self.status = 'train'
            for i, data in enumerate(self.loaders['train']):

                # Feed forward
                batch_stats_dict = self.forward(data)

                # Log requisite information
                self.log_it(partition='train', to_log=batch_stats_dict)

                self.backward(batch_stats_dict['loss'])

            self.status = 'val'
            for i, data in enumerate(self.loaders['val']):
                with torch.no_grad():
                    loss_dict_val = self.forward(data)
                    self.log_it(partition='val', to_log=loss_dict_val)

            self.on_epoch()
        return

    def forward(self, data):
        """ Analogous to the torch forward method - feed-forward component
            Implement this in all subclasses
            Should return a dictionary with at least a 'loss' key corresponding
            to the scalar loss value returned by self.objective

            Should return a dictionary with at least a 'loss' key and whatever
            other keys included in self.journal, without the `_train` or
            `_val` suffixes.
        """
        raise NotImplementedError

    def forward_val(self, data):
        """ Called during self.train()
            This exists in case of very special actions that must be taken
            during validation, that otherwise couldn't (or shouldn't) be
            handled with an if-statement
        """
        return self.forward(data)

    # TODO test
    @property
    def last_metrics(self):
        """ Returns most recent journal entries for each trackable metric as a
            dictionary
        """
        return {k: v[self.epoch] for k, v in self.journal.items()}

    # TODO test
    def epoch_summary(self, precision=5):
        def fmt_key(string, width):
            return '{0:{width}}'.format(string, width=width)

        def fmt_val(number, precision, width):
            return '{0:{width}.{precision}g}'.format(number, precision=precision, width=width)

        # TODO implement
        summary = 'Epoch {}:\n'.format(self.epoch)
        max_width = max([len(k) for k in self.journal.keys()])
        max_width = max(max_width, precision)
        metrics = self.last_metrics
        summary += ' | '.join([fmt_key(k, max_width) for k in metrics.keys()])
        summary += '\n'
        summary += ' | '.join([fmt_val(v, precision, max_width) for v in metrics.values()])
        summary += '\n'
        return summary

    # TODO test
    def _scale_last_journal_entry(self):
        """ Iterates through journal metrics and divides by batch size to make
            comparisons across loader batch sizes easier
            NOTE: This may not work if batch loading is not used
        """
        for k in self.journal.keys():
            self.journal[k][self.epoch] /= self.loaders['train'].batch_size

    # TODO test
    def _check_model_improved(self):
        """ Checks to see if the model has improved since last epoch
        """

        # The model is considered to have improved if:
        # a) It is being trained for the first time
        if self.best_metrics is None:
            model_improved = True

        # b) Its last selection metric is lower than the previous best
        else:
            model_improved = self.last_metrics[self.selection_metric] < \
                self.best_metrics[self.selection_metric]
        return model_improved

    def on_model_improved(self):
        """ What should the system do if the model has improved?
        """
        self.best_metrics = self.last_metrics
        self.save_model()

    def on_epoch(self):
        """ Actions to take on each epoch
        """
        # If we're scaling each journal entry, do so now
        if self.scale_metrics:
            self._scale_last_journal_entry()

        print(self.epoch_summary())

        # If model has improved, take requisite actions
        if self._check_model_improved():
            self.on_model_improved()

    def save_model(self):
        """ Save model parameters and some useful training information for
            future training or testing
        """
        cache_data = {'model': self.model.state_dict(),
                      'journal': self.journal,
                      'epoch': self.epoch}

        if self._v > 0:
            key_str = ', '.join(list(cache_data.keys()))
            print('Saving {}:\n  {}'.format(self.nice_name, key_str))

        self.cacher.save(cache_data)

    def load(self):
        """ Load a previously-saved model and information to resume training or
            testing
        """
        if self._v > 0:
            print('Attempting cache load at {}'.format(self.location))
        cache_data = self.cacher.try_load()
        if cache_data:
            self.journal = cache_data['journal']
            self.epoch = cache_data['epoch']
            self.model = self.model.load_state_dict(cache_data['model'])
        else:
            print('No cache data found!')

    def backward(self, loss):
        """ Analog to the torch backward method - backpropagation
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def log_it(self, to_log=dict(), partition='train'):
        suffix = '_' + partition if partition else ''
        for k, v in to_log.items():
            # Make sure item is in journal
            if self.journal.get(k + suffix) is not None:
                self.journal[k + suffix][self.epoch] += v

    def test(self):
        self.status = 'test'
        # TODO should I implement the skeleton of this?
        raise NotImplementedError

    def reset(self):
        # TODO implement this
        raise NotImplementedError
