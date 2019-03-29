from numpy import prod
from general_utils import deprecated
import torch


def shape_for_shape(shape_a, dim_b1, ndims=2):
    """ Given a tensor shape and a single dimension, computes the n-dimensional
    shape that satisfies the given single dimension

        Parameters:
            shape_a <list-like>: dimensionality of tensor
            dim_b1 <int>: single dimension of desired tensor shape
            ndims <int>: determines how many dimensions to expand into, if not 2
        Returns:
            <tuple>

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
            >>> shape_for_shape(a.shape, 12, ndims=3)
            (1, 12, 2)
            >>> shape_for_shape(a.shape, 1, ndims=5)
            (1, 1, 1, 1, 24)
            >>> try:
            ...     shape_for_shape(a.shape, 5)
            ... except ValueError as err:
            ...     print("24 is not divisible by 5")
            24 is not divisible by 5

    """

    dim_b2 = prod(shape_a) / dim_b1
    if int(dim_b2) != dim_b2:
        raise ValueError("{!s} is not divisible by {:d}".format(shape_a, dim_b1))  # NOQA

    out_shape = [1 for i in range(ndims - 2)]
    out_shape.extend([dim_b1, int(dim_b2)])

    return tuple(out_shape)


@deprecated
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
