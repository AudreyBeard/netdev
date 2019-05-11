import torch

__all__ = [
    'to_onehot',
    'clf_error',
]


def to_onehot(vec, n_classes=10):
    if n_classes is None:
        n_classes = vec.unique().shape[0]

    onehot = torch.zeros(vec.shape[0], n_classes)
    for i in range(n_classes):
        onehot[:, i] = vec == i
    return onehot


def clf_error(y_pred, y_true):
    y_pred = y_pred.to('cpu')
    y_true = y_true.to('cpu')
    wrong = y_pred != y_true
    return wrong.sum() / y_pred.shape[0]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
