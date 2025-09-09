import os
import sys
from functools import partial

from tqdm import tqdm, trange

def getWorkerId():
    worker_idx = os.getenv("CUDA_VISIBLE_DEVICES", "")
    return worker_idx
def _getTqdmPosition():
    worker_idx = getWorkerId()
    position = None if not worker_idx else int(worker_idx)
    return worker_idx, position

def _prepareMyKwargs(desc, file,  leave=True, **kwargs):
    worker_id, position = _getTqdmPosition()
    return dict(file=sys.stdout if file is None else file,
                desc= None if desc is None else desc+ f" (worker {worker_id})",
                leave=leave,
                bar_format= "{desc}: {percentage:.1f}%|{bar}| {n:8d}/{total_fmt} [{elapsed}<{remaining}]",
                position=position, **kwargs)

def myTqdm(*args, desc=None, file=None, **kwargs):
    return tqdm(*args, **_prepareMyKwargs(desc, file, **kwargs))

def myTrange(*args, desc=None, file=None, **kwargs):
    return trange(*args, **_prepareMyKwargs(desc, file, **kwargs))
