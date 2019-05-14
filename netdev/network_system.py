#from collections import Iterable
import time

import ubelt as ub
import torch

from .utils.general_utils import ParameterRegister, pretty_repr
# TODO:
# [ ] logging
# [x] caching
# [x] epoch_summary()
# [ ] fix cacher cfgstr initialization to not be reliant on memory address
#     - The way I'm handling this now is by specifying hash_on - There should
#       be a more robust way of handling this

__all__ = ['NetworkSystem']


class NetworkSystem(object):
    def __init__(self, verbosity=1, epochs=1, work_dir='./models',
                 device='cpu', hash_on=None, nice_name='untitled',
                 reset=False, scale_metrics=True, selection_metric='error_val',
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
        self.scale_metrics = scale_metrics

        self.cache_name = None
        self.cacher = None
        self.location = None
        self.status = None
        self.epoch = -1
        self.time_epoch = -1
        self.time_total = -1
        self._hash_on = hash_on

        # Keep a tensor for tracking required metrics
        self.journal = {k: torch.zeros(self.epochs) for k in metrics}
        self.selection_metric = selection_metric
        self.best_metrics = None

        # If user specified parameters to use for hash cfgstr, use them
        self.init_cacher(hash_on=self.hash_on)

        # Attach all passed-in parameters to the system
        if self._v > 0:
            print('Initialized ', end='')
            print(self)

        for k, v in self.modules.items():
            # Move to specified device, if applicable
            try:
                self.__setattr__(k, v.to(self.device))
            except AttributeError:
                self.__setattr__(k, v)

        if reset:
            self.load()

        return

    @property
    def hash_on(self):
        if self._hash_on:
            return self._hash_on
        else:
            return dict(self.modules.items())

    def __repr__(self):
        def fmt(s, w):
            return pretty_repr(s, base_indent=w, indent_first=False)
        rep = '<{}> {}\n  '.format(self.__class__.__name__, self.nice_name)
        rep += '\n  '.join(['{}: {}'.format(k, fmt(v, len(k) + 2))
                            for k, v in self.hash_on.items()])
        return rep

    # TODO this needs to be reformatted, since the current implementation uses
    # the __repr__ for each class, which sometimes defaults to a class's memory
    # address, which is changes on different executions, regardless of
    # parameters
    def init_cacher(self, hash_on=dict()):
        """ Initialize cacher for saving and loading models
            The cachers don't need to be loud, so we just suppress them somewhat
        """
        hashable = '{' + '_'.join(['{}:{}; '.format(k, v) for k, v in hash_on.items()]) + '}'
        self.cache_name = ub.hash_data(hashable, base='abc')
        self.cacher = ub.Cacher(fname=self.nice_name,
                                cfgstr=self.cache_name,
                                dpath=self.dir,
                                verbose=max(self._v - 1, 0))
        info_cacher = ub.Cacher(fname=self.nice_name + '_config',
                                cfgstr='string',
                                dpath=self.dir,
                                verbose=max(self._v - 1, 0))
        info_cacher.save(hashable)
        self.location = self.cacher.get_fpath()

    def train(self, n_epochs=None):
        """ Main training loop
            Iterates over training data to feed forward, log metrics, and
            backpropagate. Iterates over validation for metric logging and
            (probably) model selection
        """
        # Training time
        t_total = time.time()

        # If given a set number of epochs, override the one given in __init__
        if n_epochs is not None:
            self.epochs = self.epoch + n_epochs

        while self.epoch < (self.epochs - 1):
            t_epoch = time.time()
            self.epoch += 1

            self.status = 'train'
            for i, data in enumerate(self.loaders['train']):

                # Feed forward
                batch_stats_dict = self.forward(data)

                # Log requisite information
                self.log_it(partition=self.status, to_log=batch_stats_dict)

                # Backpropagate
                self.backward(batch_stats_dict['loss'])

            # Validation
            self.status = 'val'
            for i, data in enumerate(self.loaders['val']):
                with torch.no_grad():
                    batch_stats_dict = self.forward(data)
                    self.log_it(partition=self.status, to_log=batch_stats_dict)

            self.time_epoch = time.time() - t_epoch
            self.on_epoch()

        self.time_total = time.time() - t_total
        self.on_train()
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

    # TODO test
    @property
    def last_metrics(self):
        """ Returns most recent journal entries for each trackable metric as a
            dictionary
        """
        return {k: v[self.epoch] for k, v in self.journal.items()}

    # TODO test
    def epoch_summary(self, precision=5, metrics=None, t=None):
        def fmt_key(string, width):
            return '{0:{width}}'.format(string, width=width)

        def fmt_val(number, precision, width):
            return '{0:{width}.{precision}g}'.format(number, precision=precision, width=width)

        if metrics is None:
            metrics = self.last_metrics

        if t is None:
            t = self.time_epoch

        summary = 'Epoch {} ({:d} seconds):\n'.format(self.epoch, int(t))
        summary += ' | '.join([fmt_key(k, max(precision, len(k)))
                              for k in metrics.keys()])
        summary += '\n'
        summary += ' | '.join([fmt_val(v, precision, max(precision, len(k)))
                              for k, v in metrics.items()])
        return summary

    # TODO test
    def _scale_last_journal_entry(self):
        """ Iterates through journal metrics and divides by batch size to make
            comparisons across loader batch sizes easier
            NOTE: This may not work if batch loading is not used
        """
        for k in self.journal.keys():
            # self.journal[k][self.epoch] /= \
            #     (self.loaders['train'].batch_size * len(self.loaders['train']))
            self.journal[k][self.epoch] /= len(self.loaders['train'])

    # TODO test
    def _check_set_model_improved(self):
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

        print(pretty_repr(self.epoch_summary(), indent_first=False))

        # If model has improved, take requisite actions
        if self._check_set_model_improved():
            self.on_model_improved()

    def on_train(self):
        """ Actions to take when done training
        """
        print('Done training')
        print(self.epoch_summary(precision=5, metrics=self.best_metrics, t=self.time_total))

    def save_model(self):
        """ Save model parameters and some useful training information for
            future training or testing
        """
        cache_data = {'model': self.model.state_dict(),
                      'journal': self.journal,
                      'epoch': self.epoch}

        if self._v > 0:
            key_str = ', '.join(list(cache_data.keys()))
            print('  Saving {} ({})'.format(self.nice_name, key_str))

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


class ClassificationSystem(NetworkSystem):
    def forward(self, data):
        inputs, labels = data
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(inputs)
        loss = self.objective(outputs, labels)

        error = self.compute_error(outputs, labels)

        loss_dict = {
            'loss': loss,
            'error': error,
        }

        return loss_dict

    def compute_error(self, outputs, labels):
        prediction = outputs.argmax(dim=1)
        error = (prediction != labels).float().sum() / labels.shape[0]
        return error
