import collections
from warnings import warn
import os
import shutil
import pickle

from torch import no_grad
import torch
import numpy as np
# TODO
# [ ] refactor check_constraints to match what is in youtill, since
# that is more well-though-out


__all__ = [
    'no_grad_if',
    'deprecated',
    'isiterable',
    'check_constraints',
    'ParameterRegister',
    'pretty_repr',
    'rand_select',
    'rand_select_n',
    'seed_rng'
]


def seed_rng(seed=None):
    """ Seeds numpy and torch
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)


def rand_select(tensor, exclude=None, do_squeeze=True):
    """ Randomly selects one item from the given tensor
    """
    index = torch.randint(tensor.size()[0], (1, 1))
    selection = tensor[index]

    # If we want to exclude certain values, grab a different one (rolls backwards)
    if exclude is not None:
        selection = tensor[index - 1]

    if do_squeeze:
        return selection.squeeze()
    else:
        return selection


def rand_select_n(tensor, n_samples=1):
    """ Randomly choose some number of samples from an input tensor.
        If n_samples > len(tensor), output will contain duplicates. If so,
        items from tensor will appear at most once more than any other -
        meaning roughly equal representation.
    """
    def choicemax(t, n):
        """ Helper function that samples as many as possible
        """
        return np.random.choice(t, min(n, t.size()[0]), replace=False)

    #try:
    #    selection = np.random.choice(tensor, size=n_samples, replace=False)
    #except ValueError:

    # Grab as many samples as possible, up to n_samples
    selection = choicemax(tensor, n_samples)

    # If selection is too small, add to it until it's large enough
    while selection.shape[0] < n_samples:
        selection = np.concatenate((selection,
                                    choicemax(tensor, n_samples - selection.shape[0])))

    # Cast as a tensor of correct type
    return torch.Tensor(selection).type(tensor.dtype)


def pretty_repr(thing, base_indent=0, nested_indent=2, indent_first=True):
    """ Returns a better-formatted __repr__ for easier-to-read diagnostic messages
        Parameters:
            - thing: any object to print
            - base_indent (int): number of spaces to prepend to each line of
              string-representation of thing
            - nested_indent (int): number of spaces to append to each newline
              of string-representation of thing
            - indent_first (bool): Whether to prepend base_indent to first line
              of string, turn this off if you want to put first line on same
              line as another string of length base_indent
        Returns:
            - (str): properly-indented thing.__repr__() (or thing if type(thing) == str)
    """
    if isinstance(thing, str):
        thing_string = thing
    else:
        thing_string = thing.__repr__()

    thing_string = ('\n' + ' ' * (base_indent + nested_indent)).join(thing_string.split('\n'))
    if indent_first:
        thing_string = ' ' * base_indent + thing_string
    return thing_string


class no_grad_if(no_grad):
    """ Context manager that turns off gradient computations if initial condition is true
    """
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        if self.status:
            super().__enter__()

        return

    def __exit__(self, *args):
        if self.status:
            super().__exit__(*args)

        return False


def deprecated(f):
    """ Allows function decoration to raise a warning when called
        Raises a UserWarning instead of DeprecationWarning so that it's
        detectable in an IPython session.
        For use, see examples/deprecated.py
    """
    def deprec_warn():
        deprecation_msg = '{} is deprecated - consider replacing it'.format(f.__name__)
        warn(deprecation_msg)
        f()
    return deprec_warn


def isiterable(item):
    return isinstance(item, collections.Iterable)


def check_constraints(param_value, param_name, constraints):
    """ function for checking if a parameter value is valid within constraints.
        Parameters:
            param_value: The value of the parameter to check
            param_name: The name of the parameter
            constraints: The dictionary containing all constraints on this
                parameter, including type or numerical constraints
        Returns:
            bool: whether the param_value is properly constrained
        Example:
            >>> v = dict(s=('a', 'b', None), n=(float, int, None))
            >>> check_constraints('a', 's', v)
            True
            >>> check_constraints(1, 's', v)
            False
            >>> check_constraints(1, 'n', v)
            True
            >>> check_constraints('a', 'n', v)
            False
    """

    def _check_constraint(item, c, item_name=None):
        """ helper to determine if item is correctly constrained by c
            Parameters:
                item: the value to check
                c: constraint to check against
        """

    # Helper to determine if constraint (c) is the same as operation string (opstr)
        def _isop(c, opstr):
            return c.startswith(opstr) and not c.split(opstr)[1].startswith('=')

        # Helper to get the value on the RHS of the constraint
        def _getval(c, opstr):
            return float(c.split(opstr)[1])

    # No constraint
        if c is None:
            return item is None

    # Constrained by type
        elif type(c) == type or c == collections.Iterable:
            return isinstance(item, c)

    # Item should be a number and constrained by some operation
        elif isinstance(c, str):
            op_dict = collections.OrderedDict()
            op_dict['<='] = lambda x, y: x <= y
            op_dict['>='] = lambda x, y: x >= y
            op_dict['=='] = lambda x, y: x == y
            op_dict['!='] = lambda x, y: x != y
            op_dict['>'] = lambda x, y: x > y
            op_dict['<'] = lambda x, y: x < y

            is_op = any([_isop(c, opstr) for opstr in op_dict.keys()])
            if is_op:
                correct_or_dontcare = [
                    op(item, _getval(c, opstr))        # Check operation
                    if _isop(c, opstr)                 # Only if operation is constraint
                    else True                          # OTW, don't care
                    for opstr, op in op_dict.items()]  # For all operations given
                is_valid = all(correct_or_dontcare)
            else:
                is_valid = c.lower() == item.lower() if isinstance(item, str) else False

            return is_valid

        # If constraint is, for instance, a lambda function verifying that a
        # parameter is either None or satisfies some other constraint
        # e.g. (lambda x: x is None or isinstance(x, str))
        else:
            print('param {}\n  {}({}) evaluates to {}'.format(
                item_name,
                c.__name__,
                item,
                c(item) if c is not None else 'N/A'))
            return c(item)

    # First get the actual constraint(s)
    constraint = constraints.get(param_name)
    #import ipdb
    #ipdb.set_trace()

    # If multiple constraints, check all of them
    if isiterable(constraint):
        # Mark all constraints that specify types (except None) - these
        # requirements are mutually exclusive
        is_type = [isinstance(c, type) and c is not None for c in constraint]
        # Check all parameters
        checked = [_check_constraint(param_value, c, param_name) for c in constraint]
        # Is this item one of the specified types?
        correct_type = [
            v for v, t in zip(checked, is_type) if t
        ]
        # Is this item of the specified values?
        correct_value = [
            v
            for i, (v, t) in enumerate(zip(checked, is_type))
            if not t and constraint[i] is not None
        ]
        # Make sure it's the correct type and value if specified
        is_good = ((any(correct_type) or            # NOQA
                    len(correct_type) == 0) and     # NOQA
                   (any(correct_value) or           # NOQA
                    len(correct_value) == 0))
        return is_good
        #return all([_check_constraint(param_value, c, param_name) for c in constraint])

    else:
        return _check_constraint(param_value, constraint, param_name) or constraint is None


@deprecated
def check_parameter(param_value, param_name, valid_options_dict):
    """ function for checking if a parameter value is valid in the options dictionary.
        Parameters:
            param_value: The value of the param that you wish to allow
            param_name: The name of the parameter
            valid_options_dict: The dictionary containing all valid values XOR types
        Returns:
            None, or raises a ValueError
        Example:
            >>> v = dict(s=('a', 'b', None), n=(float, int, None))
            >>> check_parameter('a', 's', v)
            >>> try:
            ...     check_parameter(1, 's', v)
            ... except ValueError:
            ...     print("'b' is not a valid value for s")
            'b' is not a valid value for s
            >>> check_parameter(1, 'n', v)
            >>> try:
            ...     check_parameter('a', 'n', v)
            ... except ValueError:
            ...     print("'s' is not a valid value for n")
            's' is not a valid value for n
    """
    from numpy import ndarray
    if valid_options_dict.get(param_name) is None and param_value is not None:
        raise ValueError('Did not find {} in valid options dict'.format(param_name))

    is_valid = [type(param_value) == o if type(o) == type else
                0 if type(param_value) == ndarray else
                param_value == o for o in valid_options_dict.get(param_name)]

    if sum(is_valid) < 1:
        raise ValueError("'{}' must be one of {}".format(param_name, valid_options_dict[param_name]))


class ParameterRegister(collections.OrderedDict):
    def __init__(self, constraints=None, defaults=None):
        super().__init__()
        self._constraints = constraints
        self._defaults = defaults

    def register(self, kwarg_name, constraints=None, default=None):
        self._constraints[kwarg_name] = constraints
        self._defaults[kwarg_name] = default

    def unregister(self, kwarg):
        if kwarg in self.keys():
            del self[kwarg]
            del self._constraints[kwarg]
            del self._defaults[kwarg]

    def unset(self, kwarg):
        if kwarg in self.keys():
            del self[kwarg]

    def check_kwargs(self, **kwargs):
        #(check_parameter(kwargs[key], key, self) for key in kwargs)
        valid = {k: check_constraints(v, k, self._constraints) for k, v in kwargs.items()}
        return valid

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return self[attr]

    @property
    def bad_param_string(self, **kwargs):
        """ Checks parameters and returns a string of poorly-specified
            parameters. Useful for error message prining
        """
        def fmt(key, expected, actual):
            return '{}: expects {}, got {}'.format(key, expected, actual)

        valid = self.check_kwargs(**kwargs)
        if not all(valid.values()):
            bad_string = ''
            for k, v in valid.items():
                if not v:
                    bad_string += fmt(k, self._constraints, kwargs[k])
        else:
            bad_string = None

        return bad_string

    def try_set_params(self, **kwargs):
        """ Checks given kwargs against registered constraints. If all kwargs
            are properly specified, sets the internal dictionary. If not,
            returns a string denoting the poorly-specced kwargs, useful for
            quick setting and error printing.
        """
        valid = self.check_kwargs(**kwargs)
        if not all(valid.values()):
            ec = '; '.join(['{}: expects {}, got {}'.format(k,
                                                            self._constraints[k],
                                                            kwargs[k])
                            for k, v in valid if not v])
            raise ValueError(ec)
        else:
            for k, v in sorted(kwargs.items()):
                self[k] = v
            return None

    def set_uninitialized_params(self, defaults=None):
        if defaults is not None:
            defaults_notset = {key: defaults[key]
                               for key in defaults
                               if self.get(key) is None}
        elif self._defaults is not None:
            defaults_notset = {key: self._defaults[key]
                               for key in self._defaults
                               if self.get(key) is None}
        else:
            print('No defaults dictionary given - cannot set')

        self.update(defaults_notset)

    @property
    def hashable_str(self):
        return ','.join(['{}:{}'.format(k, v) for k, v in self.items()])


class Cache(object):
    """ Simple caching mechanism
    """
    def __init__(self, dpath='~/.cache/actions', verbosity=1):
        """
            Example:
                >>> self = Cache()
                >>> os.path.exists(self.dpath)
                True
        """
        self.verbosity = verbosity

        # Expand variables and store this path
        self.dpath = os.path.realpath(
            os.path.expanduser(
                os.path.expandvars(dpath)))
        if not os.path.exists(self.dpath):
            os.mkdirs(self.dpath)
            if self.verbosity > 0:
                print("[cache] Init at {} - created".format(self.dpath))
        elif self.verbosity > 0:
            print("[cache] Init at {}".format(self.dpath))

    def exists(self, fname=''):
        """
            Example:
                >>> self = Cache()
                >>> self.exists()
                True
                >>> self.exists('TEST_FILE_THAT_SHOULD_NOT_EXIST')
                False
        """
        found = os.path.exists(self.fpath(fname))
        if self.verbosity > 0:
            if found:
                print("[cache] {} found".format(fname))
            else:
                print("[cache] {} NOT found".format(fname))
        return found

    def fpath(self, fname):
        """
            Example:
                >>> self = Cache()
                >>> dpath, fname = os.path.split(self.fpath('test'))
                >>> assert dpath == self.dpath
                >>> assert fname == 'test'
        """
        return os.path.join(self.dpath, fname)

    def pickle(self, item, fname):
        """
            Example:
                >>> self = Cache()
                >>> test_dict = {'a':-1, 'b':0, 'c':1}
                >>> self.pickle(test_dict, 'test_dict.pkl')
                >>> os.path.exists(self.fpath('test_dict.pkl'))
                True
        """
        # If cache path is given for some reason:
        if os.path.split(fname)[0] == self.dpath:
            fname = os.path.split(fname)[1]

        if self.verbosity > 0:
            print("[cache] pickling {} . . . ".format(fname), end='')

        with open(self.fpath(fname), 'wb') as fid:
            pickle.dump(item, fid)

        print("DONE")

    def unpickle(self, fname):
        """
            Example:
                >>> self = Cache()
                >>> test_dict = {'a':-1, 'b':0, 'c':1}
                >>> self.pickle(test_dict, 'test_dict.pkl')
                >>> test_dict_unpickled = self.unpickle('test_dict.pkl')
                >>> assert len(test_dict) == len(test_dict_unpickled)
                >>> assert all([v1 == v2
                ...            for v1, v2 in zip(
                ...                test_dict.values(),
                ...                test_dict_unpickled.values())])
        """
        # If cache path is given for some reason:
        if os.path.split(fname)[0] == self.dpath:
            fname = os.path.split(fname)[1]

        if not self.exists(fname):
            item = None
        else:
            if self.verbosity > 0:
                print("[cache] unpickling {} . . . ".format(fname), end='')

            with open(self.fpath(fname), 'rb') as fid:
                item = pickle.load(fid)

            if self.verbosity > 0:
                print("DONE")

        return item

    def write_str(self, data, fname):
        # If cache path is given for some reason:
        if os.path.split(fname)[0] == self.dpath:
            fname = os.path.split(fname)[1]
        if self.verbosity > 0:
            print("[cache] writing {}".format(fname))
        with open(self.fpath(fname), 'w') as fid:
            fid.write(str(data))

    def cp(self, fpath_src):
        """ Copies file to cache
        """
        if self.verbosity > 0:
            print("[cache] copying {} to {}".format(fpath_src, self.dpath))
        shutil.copy(fpath_src, self.dpath)

    @property
    def ls(self):
        """ List contents in cache directory
            Example:
                >>> self = Cache()
                >>> self.pickle({'a':1}, 'test_dict.pkl')
                >>> assert 'test_dict.pkl' in len(self.ls())
        """
        return os.listdir(self.dpath)


def TODO(msg="Implement this!"):
    raise NotImplementedError(msg)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
