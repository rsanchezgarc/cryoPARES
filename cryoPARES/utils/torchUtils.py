import os
import warnings

import numpy as np
import torch


def data_to_numpy(x):
    if isinstance(x, np.ndarray):
        return x  # Do nothing if it's already a NumPy array

    if isinstance(x, torch.Tensor):
        # Move tensor to CPU if it's on GPU
        if x.is_cuda:
            x = x.cpu()

        # Detach it from the computation graph if it's attached
        if x.requires_grad:
            x = x.detach()

        return x.numpy()

    raise TypeError(f"Input type {type(x)} is not supported. Only PyTorch tensors and NumPy arrays are valid.")


def accelerator_selector(use_cuda=True, n_gpus_torch=None, n_cpus_torch=None):

    if not use_cuda or not torch.cuda.is_available():
        accel = 'cpu'
        if n_cpus_torch is None:
            dev_count = os.cpu_count()
        else:
            dev_count = max(1, n_cpus_torch)
        if not use_cuda:
            warnings.warn("GPUs not found!!. %d CPUs will be using instead"%dev_count)
    else:
        accel = 'gpu'
        dev_count = torch.cuda.device_count()
        if n_gpus_torch is not None:
            assert n_gpus_torch <= dev_count, f"Error, there are less available GPUS {dev_count} " \
                                              f"than requested {n_gpus_torch}"
            dev_count = max(1, n_gpus_torch)
            print("USING %d GPUS " % dev_count)
    return accel, dev_count
