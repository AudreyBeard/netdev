import os

from tqdm import tqdm
import ubelt as ub
import torch
try:
    import tensorboard_logger
    tensorboard = True
except ImportError:
    tensorboard = False

from .utils.general_utils import ParameterRegister, pretty_repr, Cache
# TODO:
# [ ] logging
# [ ] caching
#     [x] First pass with ubelt
#     [x] Implement my own for fewer dependencies
#     [ ] Eliminate dependency on ubelt
# [x] epoch_summary()
# [ ] fix cacher cfgstr initialization to not be reliant on memory address
#     - The way I'm handling this now is by specifying hash_on - There should
#       be a more robust way of handling this

__all__ = ['NetworkSystem', 'ClassificationSystem']


class NetworkSystem(object):
    def __init__(self, verbosity=1, epochs=1, work_dir='./models',
                 device='cpu', hash_on=None, nice_name='untitled',
                 reset=False, scale_metrics=True, selection_metric='error_val',
                 selection_metric_goal='min', eval_val_every=10, save_every=10,
                 metrics=['loss_train', 'error_train', 'loss_val', 'error_val'],
                 **kwargs):

        """
            Network training and testing system, designed for modules to plug
            into for simple part swapping.

            Design concepts:
                Several system parameters are simply for housekeeping and
                therefore do not need to be considered when saving the model,
                others affect the performance of the model and should therefore
                be used in differentiating between different models when
                saving.

            Parameters:
                verbosity (int): Level of verbosity desired. 0 is silent,
                    higher numbers means more verbosity
                epochs (int): number of epochs to train past its current point
                work_dir (str): Location for saved models and tensorboard logs
                device (str): 'cpu' or specified device
                hash_on (dict or None): dictionary specifying parameters and
                    values to hash on when generating the nice name. If none,
                    it uses whatever is passed in **kwargs
                nice_name (str): Prefix for saving name, and a good way to do
                    human-readable scanning
                reset (bool): Starts the model fresh, regardless of whether
                    another model exists with the same name
                scale_metrics (bool): Do we scale all user-defined metrics at
                    the end of an epoch? Usually the answer is yes, so probably
                    keep this as True
                metrics (list of str): List of metrics to track. MUST follow
                    the convention of "metricname_split", where split is one of
                    ['train', 'val']
                selection_metric (str): Metric (from metrics parameter) by
                    which we determine if the model has improved or not. This
                    determines when we save.
                selection_metric_goal (str): What we want from the
                    selection_metric parameter. If it's accuracy, we want
                    'max', if it's error, we want 'min'
                eval_val_every (int): How often to we evaluate the validation
                    data? This also determines how often we save a "better
                    model", since I assume the model should be selected based
                    on its performance on validation data
                save_every (int): How often we save, regardless of its performance.
                **kwargs (dict): Additional parameters that need to be given,
                    especially network modules like "model", "objective",
                    "optimizer", "loaders", etc.
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
        self.eval_val_every = eval_val_every
        self.save_every = save_every

        self.cache = None
        self.cache_name = None
        self.cacher = None
        self.status = None
        self.training = False
        self.epoch = 0
        self.best_epoch = -1
        self._hash_on = hash_on

        # Keep a tensor for tracking required metrics
        self.journal = {k: torch.zeros(self.epochs, requires_grad=False) for k in metrics}
        self.selection_metric = selection_metric
        if selection_metric_goal.lower().startswith('min'):
            self.is_improvement = lambda new, old: new < old
        else:
            self.is_improvement = lambda new, old: new > old

        self.best_metrics = None

        # If user specified parameters to use for hash cfgstr, use them
        self.init_cacher(self.hash_on)

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

        if not reset:
            self.load()

        if tensorboard:
            tb_dir = os.path.join(os.path.split(self.cache.dpath)[0],
                                  'logs',
                                  self.nice_name)
            tensorboard_logger.configure(tb_dir, flush_secs=5)
            print("Start up a tensorboard server with the following command:")
            print("  tensorboard --logdir {}".format(tb_dir))

        return

    @property
    def hash_on(self):
        """ Returns a dictionary of keys and values from which we create a
            unique hash. If self._hash_on is set, return that. Otherwise, fall
            back to set of user-defined properties that control behavior of
            model
        """
        if self._hash_on:
            return self._hash_on
        else:
            return dict(self.modules.items())

    @hash_on.setter
    def hash_on(self, hash_on):
        self._hash_on = hash_on

    def __repr__(self):
        def fmt(s, w):
            return pretty_repr(s, base_indent=w, indent_first=False)
        rep = '<{}> {}\n  '.format(self.__class__.__name__, self.nice_name)
        rep += '\n  '.join(['{}: {}'.format(k, fmt(self.hash_on[k], len(k) + 2))
                            for k in sorted(self.hash_on)])
        return rep

    # TODO this needs to be reformatted, since the current implementation uses
    # the __repr__ for each class, which sometimes defaults to a class's memory
    # address, which is changes on different executions, regardless of
    # parameters
    def init_cacher(self, hash_on):
        """ Initialize cacher for saving and loading models
            The cachers don't need to be loud, so we just suppress them somewhat
        """
        hashable = '{' + '___'.join(['{}:{};'.format(k, hash_on[k]) for k in sorted(hash_on)]) + '}'
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

        cache_directory = os.path.join(self.dir,
                                       self.__class__.__name__,
                                       self.nice_name)
        self.cache = Cache(cache_directory, self._v - 1)
        self.cache.write_str(hashable, "{}_hash-on_string.txt".format(self.nice_name))

    @property
    def location(self):
        # TODO implement this with self.cache
        if self.cacher is None:
            return None
        else:
            return self.cacher.get_fpath()

    def single_epoch_train(self):
        """ Training steps for a single epoch
        """
        self.status = 'train'
        self.training = True
        self.model.train()
        for i, data in enumerate(self.loaders['train']):

            # Feed forward
            batch_stats_dict = self.forward(data)

            # Log requisite information
            self.log_it(partition=self.status, to_log=batch_stats_dict)

            # Backpropagate
            self.backward(batch_stats_dict['loss'])

        if tensorboard:
            self.tensorboard_log(self.status)

    def single_epoch_val(self):
        """ Validation steps for a single epoch
        """
        self.status = 'val'
        self.training = False
        self.model.eval()
        for i, data in enumerate(self.loaders['val']):
            with torch.no_grad():
                batch_stats_dict = self.forward(data)
                self.log_it(partition=self.status, to_log=batch_stats_dict)

        if tensorboard:
            self.tensorboard_log(self.status)

    def tensorboard_log(self, partition=None):
        if partition is not None:
            keys = [k for k in self.journal.keys() if k.endswith(partition)]
        else:
            keys = list(self.journal.keys())

        for metric_name in keys:
            tensorboard_logger.log_value(
                metric_name,
                self.journal[metric_name][self.epoch] / len(self.loaders[partition]),
                self.epoch
            )

    def train(self, n_epochs=None):
        """ Main training loop
            Iterates over training data to feed forward, log metrics, and
            backpropagate. Iterates over validation for metric logging and
            (probably) model selection
        """
        # Training time
        #t_total = time.time()

        # If given a set number of epochs, override the one given in __init__
        if n_epochs is not None:
            self.epochs = self.epoch + n_epochs
            self.journal = {k: torch.cat((v, torch.zeros(self.epochs)))
                            for k, v in self.journal.items()}

        start_epoch = self.epoch

        self.single_epoch_val()
        for _ in tqdm(range(start_epoch, self.epochs)):
            epoch_done = False

            self.single_epoch_train()
            if self.epoch % self.eval_val_every == (self.eval_val_every - 1):
                self.single_epoch_val()

                self.on_epoch()
                epoch_done = True

                # If model has improved, take requisite actions
                if self._check_set_model_improved():
                    self.on_model_improved()

                # If model hasn't improved, save if it's time to do so
                elif self.epoch % self.save_every == (self.save_every - 1):
                    self.save_model()

            if not epoch_done:
                self.on_epoch()

            self.epoch += 1

        self.on_train()
        return

    def forward(self, data):
        """ Analogous to the torch forward method - feed-forward component
            Implement this in all subclasses.
            Should be used for training and evaluation, and may be switched on
            self.training (bool) or self.status (str).
            Should return a dictionary with at least a 'loss' key corresponding
            to the scalar loss value returned by self.objective and whatever
            other keys included in self.journal, without the `_train` or
            `_val` suffixes.
        """
        raise NotImplementedError

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
                              for k in sorted(metrics)])
        summary += '\n'
        summary += ' | '.join([fmt_val(metrics[k], precision, max(precision, len(k)))
                              for k in sorted(metrics)])
        return summary

    # TODO test
    def _scale_last_journal_entry(self):
        """ Iterates through journal metrics and divides by number of batches to make
            comparisons across loader batch sizes easier
            NOTE: This may not work if batch loading is not used
            NOTE: If the last batch is incomplete, this will yield slightly inaccurate resultss.
        """
        for k in self.journal.keys():
            self.journal[k][self.epoch] /= len(self.loaders[k.split('_')[-1]])

    # TODO test
    def _check_set_model_improved(self):
        """ Checks to see if the model has improved since last epoch
        """

        # The model is considered to have improved if:
        # a) It is being trained for the first time
        # b) Its last selection metric is lower than the previous best
        improved = self.is_improvement(
            self.last_metrics[self.selection_metric],
            self.best_metrics[self.selection_metric]
        ) if self.best_metrics is not None else True
        return improved

    def on_model_improved(self):
        """ What should the system do if the model has improved?
        """
        self.best_metrics = self.last_metrics
        self.best_epoch = self.epoch
        self.save_model()

    def on_epoch(self):
        """ Actions to take on each epoch
        """
        self.status = 'on_epoch'
        self.training = False
        # If we're scaling each journal entry, do so now
        if self.scale_metrics:
            self._scale_last_journal_entry()

        if not tensorboard:
            print(pretty_repr(self.epoch_summary(), indent_first=False))

    def on_train(self):
        """ Actions to take when done training
        """
        print('Done training')
        print(self.epoch_summary(precision=5, metrics=self.best_metrics, t=self.time_total))

    def save_model(self, **kwargs):
        """ Save model parameters and some useful training information for
            future training or testing
        """
        cache_data = {'model': self.model.state_dict(),
                      'journal': self.journal,
                      'epoch': self.epoch,
                      'best_metrics': self.best_metrics,
                      'best_epoch': self.best_epoch,
                      }

        # If anything is specified, add it to the data to be saved
        cache_data.update(kwargs)

        if self._v > 0:
            key_str = ', '.join(list(cache_data.keys()))
            print('  Saving {} ({})'.format(self.nice_name, key_str))

        #self.cacher.save(cache_data)

        # This is faster than pickling
        torch.save(cache_data, self.checkpoint_name)

    @property
    def checkpoint_name(self, **kwargs):
        # TODO extend this
        name = "E={}".format(self.epoch)
        for k in sorted(kwargs):
            # List each kwarg in order
            name += "_{}".format(k)
            name += "={}".format(kwargs[k])

        name = self.cache.fpath(name)

        name += '.t7'
        return name

    def load(self, nice_name=None, verbosity=1):
        """ Load a previously-saved model and information to resume training or
            testing.
            Note, this systems's model must be the same model as was saved
        """

        # If nice_name is not given, fall back to originally-given nice name
        if nice_name is None:
            nice_name = self.nice_name

        # Try to get the original hashable string
        info_cacher = ub.Cacher(fname=nice_name + '_config',
                                cfgstr='string',
                                dpath=self.dir,
                                verbose=max(verbosity - 1, 0))
        orig_hashable_str = info_cacher.tryload()

        if not orig_hashable_str:
            if verbosity > 0:
                print('No configuration data found at {}'.format(info_cacher.get_fpath()))
                print('Falling back on system metadata')
            cacher = self.cacher

        else:
            if verbosity > 0:
                print('Found hashable configuration string at:\n  {}\n  {}'
                      .format(info_cacher.get_fpath(), orig_hashable_str))
                print('Instantiating new cacher')

            cache_name = ub.hash_data(orig_hashable_str, base='abc')
            cacher = ub.Cacher(fname=nice_name,
                               cfgstr=self.cache_name,
                               dpath=self.dir,
                               verbose=max(verbosity - 1, 0))

        # Try to load from whatever cacher we're using
        if verbosity > 0:
            print('Attempting cache load at {}'.format(cacher.get_fpath()))
        cache_data = cacher.tryload()

        if not cache_data:
            print('No cache data found!')
        else:
            print('Cache data found, loading data now...', end='')
            self.cache_name = cache_name
            self.cacher = cacher
            self.nice_name = nice_name
            self.journal = cache_data.get('journal')
            self.epoch = cache_data.get('epoch')
            self.epochs = cache_data.get('epoch')
            self.best_metrics = cache_data.get('best_metrics')
            self.best_epoch = cache_data.get('best_epoch')
            self.model.load_state_dict(cache_data['model'])
            print(' done')

        return cache_data is not None

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
        self.training = False
        self.model.training = False
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
