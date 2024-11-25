import os
import warnings

import joblib

from cryoPARES.configs.mainConfig import main_config


def get_cache(cache_name, memory_only=False, verbose=0):

    if cache_name is not None and not memory_only:
        try:
            cache = joblib.Memory(location=os.path.join(main_config.cachedir, cache_name + ".joblib"),
                                  verbose=verbose)
            return cache
        except (FileNotFoundError, IOError, PermissionError) as e:
            print(e)
            warnings.warn(f"The CACHE_DIR {main_config.cachedir} is not available ({e}), skipping cache. "
                          f"Some weights will be recomputed in each execution, wasting compute")
    cache = joblib.Memory(location=None, verbose=verbose)
    return cache

