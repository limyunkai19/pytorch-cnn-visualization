import torch
import numpy as np

def mean_std(dataloader_object, axis=None):
    """ Compute the mean and standard deviation of a dataset.
        Useful for test data normalization, when the mean and std are unknown.

    Argument:
        dataloader_object (torch.utils.data.DataLoader) - your dataset
        axis (tuple, default=None) - the axis to compute mean and std

    Return:
        (mean, std)

    TODO:
        try to avoid numpy and use data native (torch.Tensor) operation
    """


    sums = np.float64(0.0)
    square_sums = np.float64(0.0)
    n = np.float64(0.0)

    for batch_idx, (data, target) in enumerate(dataloader_object):
        data = data.numpy()
        sums += np.sum(data, axis=axis)
        square_sums += np.sum(data*data, axis=axis)
        if axis:
            n += np.prod(np.asarray(data.shape)[np.asarray(axis)])
        else:
            n += np.prod(data.shape)

    mean = sums/n
    std = (square_sums/n-mean**2)**0.5

    return mean, std

def one_hot_tensor(idx, length):
    """ Return a one hot tensor (1xlength) with 1 at idx and 0 at other position

    Argument:
        idx (int) - such that one_hot[idx] = 1.0
        length (int) - length of the one hot tensor

    Return:
        torch.FloatTensor of size 1xlength
    """

    one_hot = torch.FloatTensor(1, length).zero_()
    one_hot[0][idx] = 1.0
    return one_hot

def normalize(x):
    """ Normalize (linear normalization), x to 0-1 range

    Argument:
        x (Variable, Tensor or Numpy array) - the data to be normalized

    Return:
        normalized data

    Note:
        datatype of x are supposed to be floating point,
        other datatype such as int or uint may cause unexpected behaeviour
    """


    x = x - x.min()
    x = x / x.max()

    return x
