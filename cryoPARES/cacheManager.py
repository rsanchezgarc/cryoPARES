import functools
import hashlib
import json
import os
import tempfile
import warnings

import joblib

from cryoPARES.configs.mainConfig import main_config


def get_cache(cache_name, cachedir=None, verbose=0):

    if cachedir is None:
        cachedir = main_config.cachedir
    if cache_name is not None:
        try:
            cache = joblib.Memory(location=os.path.join(cachedir, cache_name + ".joblib"),
                                  verbose=verbose)
            return cache
        except (FileNotFoundError, IOError, PermissionError) as e:
            print(e)
            warnings.warn(f"The CACHE_DIR {main_config.cachedir} is not available ({e}), skipping cache. "
                          f"Some weights will be recomputed in each execution, wasting compute")
    cache = joblib.Memory(location=None, verbose=verbose)
    return cache


def hashVars(name, vars_to_hash):
    _txt = json.dumps(vars_to_hash)
    _hash = hashlib.sha256(_txt.encode(encoding='utf-8')).hexdigest()
    _name = name + f"_{_hash}"
    return _name

class SharedTemporaryDirectory(tempfile.TemporaryDirectory):
    def __init__(self, name, vars_to_hash, rootdir=None):
        """

        This class behaves like TemporaryDirectory but only the process that creates the folder can remove it.
        Useful to use in the rank_0 process in torch.distributed.

        :param name: Root name for the tempfolder that will hang after /rootdir/{name}_HASH
        :param vars_to_hash: A list of hashable vars to create a unique name
        :param rootdir: If not provided, it will use /tmp
        """
        super().__init__()
        import weakref as _weakref
        _name = hashVars(name, vars_to_hash)
        assert _name, "Error, name is empty"
        if rootdir is None:
            rootdir = tempfile.gettempdir()
        self.name = os.path.join(rootdir, _name)
        try:
            os.makedirs(self.name)
            self.should_remove = True
            self._finalizer = _weakref.finalize(
                self, self._cleanup, self.name,
                warn_message="Implicitly cleaning up {!r}".format(self))
        except OSError:
            self.should_remove = False

    def cleanup(self) -> None:
        if self.should_remove:
            super().cleanup()
            # print(f"Directory {self.name} has been deleted by {os.getpid()}!")