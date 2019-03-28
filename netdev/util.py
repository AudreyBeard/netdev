from numpy import prod
import torch
import collections


def to_onehot(vec, n_classes=10):
    if n_classes is None:
        n_classes = vec.unique().shape[0]

    onehot = torch.zeros(vec.shape[0], n_classes)
    onehot[torch.arange(vec.shape[0]), vec] = 1
    return onehot


def isiterable(item):
    return isinstance(item, collections.Iterable)


def shape_for_shape(shape_a, dim_b1, ndims=2, typecast=None):
    """ Given a tensor shape and a single dimension, computes the n-dimensional
    shape that satisfies the given single dimension, cast according to the
    typecast function

        Parameters:
            shape_a <list-like>: dimensionality of tensor
            dim_b1 <int>: single dimension of desired tensor shape
            ndims <int>: determines how many dimensions to expand into, if not 2
            typecast <function>: casts the output shape
        Returns:
            <type(typecast(<list-like>))> if typecast is given,
            <tuple> if typecast is None
        Example:
            >>> from numpy import arange
            >>> a = arange(1, 25).reshape(3, 8)
            >>> shape_for_shape(a.shape, 6)
            (6, 4)
            >>> shape_for_shape(a.shape, 1)
            (1, 24)
            >>> shape_for_shape(a.shape, 24)
            (24, 1)
            >>> shape_for_shape(a.shape, 12)
            (12, 2)
            >>> try:
            ...     shape_for_shape(a.shape, 5)
            ... except ValueError as err:
            ...     print("24 is not divisible by 5")
            24 is not divisible by 5

    """

    dim_b2 = prod(shape_a) / dim_b1
    if int(dim_b2) != dim_b2:
        raise ValueError("{!s} is not divisible by {:d}".format(shape_a, dim_b1))

    if ndims == 3:
        out_shape = (1, dim_b1, int(dim_b2))
    else:
        out_shape = (dim_b1, int(dim_b2))

    if typecast is not None:
        return typecast(out_shape)
    else:
        return out_shape


def output_shape_for(input_shape, seq):
    # NOTE This function is just a placeholder right now - it's not robust, as
    # the caclulations are incorrect, but I just don't have the energy right
    # now to fix it
    if len(input_shape) != 4:
        raise ValueError('input shape must be 4')

    out_shape = input_shape

    # Computes output shape for each layer sequentially, returns final shape
    for layer in seq:
        # if this layer is a convolutional layer
        # TODO this needs work
        if layer.__class__ == torch.nn.modules.Conv2d:
            layer_shape = layer.weight.shape
            out_shape = torch.Size([int(out_shape[0]),
                                    int(layer_shape[0]),
                                    int(out_shape[2] - layer_shape[2] + 1),
                                    int(out_shape[3] - layer_shape[3] + 1)])

        # If this later is a max pooling function
        # TODO this needs work
        elif layer.__class__ == torch.nn.modules.pooling.MaxPool2d:
            out_shape = torch.Size([int(out_shape[0]),
                                    int(out_shape[1]),
                                    int(out_shape[2] / layer.kernel_size),
                                    int(out_shape[3] / layer.kernel_size)])

        # If this layer is an activation function
        elif layer.__repr__().split('(')[0] in dir(torch.nn.modules.activation):
            pass

        elif layer.__class__ == torch.nn.modules.linear.Linear:
            layer_shape = layer.weight.shape
            out_shape = torch.Size([int(out_shape[0]),
                                    int(layer_shape[0])])
        else:
            raise NotImplementedError('Only supports Conv2d, MaxPool2d, Linear, and activation functions right now.')

    return out_shape


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
