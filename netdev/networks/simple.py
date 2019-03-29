from ..utils.general_utils import output_shape_for
from ..netskel import NetworkSkeleton
from torch import nn
import torch
from numpy import prod


class MNISTEmbeddingNet(NetworkSkeleton):
    """
        References:
            https://github.com/adambielski/siamese-triplet/blob/master/networks.py

        Example:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/netharn/examples')
            >>> from mnist_matching import *
            >>> input_shape = (None, 1, 28, 28)
            >>> self = MNISTEmbeddingNet(input_shape)
            >>> print('self = {!r}'.format(self))
            >>> print('flat_shape = {!r}'.format(self._flat_shape))
            >>> print(ub.repr2(self.output_shape_for(input_shape).hidden.shallow(2), nl=2))
            {
                'convnet': {
                    '0': (None, 32, 24, 24),
                    '1': (None, 32, 24, 24),
                    '2': (None, 32, 12, 12),
                    '3': (None, 64, 8, 8),
                    '4': (None, 64, 8, 8),
                    '5': (None, 64, 4, 4),
                },
                'reshape': (
                    None,
                    1024,
                ),
                'fc': {
                    '0': (None, 256),
                    '1': (None, 256),
                    '2': (None, 256),
                    '3': (None, 256),
                    '4': (None, 256),
                },
            }
            >>> input_shape = [4] + list(self.input_shape[1:])
            >>> inputs = torch.rand(input_shape)
            >>> dvecs = self(inputs)['dvecs']
            >>> pdists = torch.nn.functional.pdist(dvecs, p=2)
            >>> pos_dist = pdists[0::2]
            >>> neg_dist = pdists[1::2]
            >>> margin = 1
            >>> x = pos_dist - neg_dist + margin
            >>> loss = torch.nn.functional.softplus(x).mean()
            >>> loss.backward()
    """
    # TODO does this override the super?
    #constraints = {'verbosity': int,
    #               'input_shape': Iterable,
    #               'output_len': (int, '>0'),
    #               'nice_name': str,
    #               'work_dir': str,
    #               'initializer': lambda x: x is None or isinstance(x, str),
    #               'reset': bool,
    #               }
    #defaults = {'verbosity': 0,
    #            'input_shape': None,
    #            'output_len': None,
    #            'nice_name': None,
    #            'work_dir': None,
    #            'initializer': None,
    #            'reset': False,
    #            }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self._v > 0:
            print('input_shape = {!r}'.format(self.hyperparams['input_shape']))

        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self._conv_output_shape = output_shape_for(self.hyperparams['input_shape'], self.convnet)
        self._num_flat = prod(self._conv_output_shape[1:])

        self.fc = nn.Sequential(nn.Linear(self._num_flat, 256), nn.PReLU(),
                                nn.Linear(256, 256), nn.PReLU(),
                                nn.Linear(256, self.hyperparams['output_len']))

        if self.hyperparams['initializer'] is not None:
            # TODO kaimingnormal
            print("Initializer %s is not implemented. Falling back to no initialization")
            pass

        #self._make_network(convnet, fc)

    def reshape(self, t):
        return torch.reshape(t, (-1, self._num_flat))

    def forward(self, inputs):
        #with ipdb.launch_ipdb_on_exception():
        #    outputs = {'dvecs': super().forward(inputs)}
        conv_out = self.convnet(inputs)
        flat_conv = self.reshape(conv_out)
        dvecs = self.fc(flat_conv)
        outputs = {'dvecs': dvecs}
        return outputs

    def get_embedding(self, x):
        return self.forward(x)
