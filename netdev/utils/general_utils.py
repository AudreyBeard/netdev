import collections
from warnings import warn
from torch import no_grad
# TODO
# [ ] refactor check_constraints to match what is in youtill, since
# that is more well-though-out


__all__ = [
    'no_grad_if',
    'deprecated',
    'isiterable',
    'check_constraints',
    'ParameterRegister',
]


class no_grad_if(no_grad):
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
        deprecation_msg = '{} is deprecated - consider replacing it'.format(f.__name__)  # NOQA
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
            return True

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

            correct_or_dontcare = [op(item, _getval(c, opstr))        # Check operation
                                   if _isop(c, opstr)                 # Only if operation is constraint
                                   else True                          # OTW, don't care
                                   for opstr, op in op_dict.items()]  # For all operations given
            return all(correct_or_dontcare)

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

    # If multiple constraints, check all of them
    if isiterable(constraint):
        return all([_check_constraint(param_value, c, param_name) for c in constraint])

    else:
        return _check_constraint(param_value, constraint, param_name)


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
        self.constraints = constraints
        self.defaults = defaults

    def register(self, kwarg_name, constraints=None, default=None):
        self.constraints[kwarg_name] = constraints
        self.defaults[kwarg_name] = default

    def unregister(self, kwarg):
        if kwarg in self.keys():
            del self[kwarg]
            del self.constraints[kwarg]
            del self.defaults[kwarg]

    def unset(self, kwarg):
        if kwarg in self.keys():
            del self[kwarg]

    def check_kwargs(self, **kwargs):
        (check_parameter(kwargs[key], key, self) for key in kwargs)
        valid = {k: check_constraints(v, k, self.constraints) for k, v in kwargs.items()}
        return valid

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
                    bad_string += fmt(k, self.constraints, kwargs[k])
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
            rc = '; '.join(['{}: expects {}, got {}'.format(k, self.constraints[k], kwargs[k]) for k, v in valid if not v])
        else:
            for k, v in sorted(kwargs.items()):
                self[k] = v
            rc = None
        return rc

    def set_uninitialized_params(self, defaults=None):
        if defaults is not None:
            defaults_notset = {key: defaults[key]
                               for key in defaults
                               if self.get(key) is None}
        elif self.defaults is not None:
            defaults_notset = {key: self.defaults[key]
                               for key in self.defaults
                               if self.get(key) is None}
        else:
            print('No defaults dictionary given - cannot set')

        self.update(defaults_notset)

    @property
    def hashable_str(self):
        return ','.join(['{}:{}'.format(k, v) for k, v in self.items()])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
