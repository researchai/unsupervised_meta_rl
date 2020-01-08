"""Utility functions for PyTorch."""
import numpy as np
import os
import torch


def np_to_torch(array_dict):
    """Convert numpy arrays to PyTorch tensors.

    Args:
        array_dict (dict): Dictionary of data in numpy arrays.

    Returns:
        dict: Dictionary of data in PyTorch tensors.

    """
    for key, value in array_dict.items():
        array_dict[key] = torch.FloatTensor(value)
    return array_dict


def torch_to_np(value_in):
    """Convert PyTorch tensors to numpy arrays.

    Args:
        value_in (tuple): Tuple of data in PyTorch tensors.

    Returns:
        tuple[numpy.ndarray]: Tuple of data in numpy arrays.

    """
    value_out = tuple(v.numpy() for v in value_in)
    return value_out


def flatten_batch(tensor):
    """Flatten a batch of observations.

    Reshape a tensor of size (X, Y, Z) into (X*Y, Z)

    Args:
        tensor (torch.Tensor): Tensor to flatten.

    Returns:
        torch.Tensor: Flattened tensor.

    """
    return tensor.reshape((-1, ) + tensor.shape[2:])

"""
GPU wrappers
"""

_use_gpu = False
device = None


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:0" if _use_gpu else "cpu")
    if _use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float()


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return from_numpy(elem_or_tuple).float()


def filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: elem_or_tuple_to_variable(x)
        for k, x in filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }
