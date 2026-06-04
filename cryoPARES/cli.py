import os
import warnings

_PYTHONWARNINGS = "ignore::FutureWarning,ignore::DeprecationWarning,ignore::PendingDeprecationWarning,ignore::UserWarning"


def _suppress_user_warnings():
    # Set env var so spawned subprocesses (e.g. DDP workers) inherit the filters
    os.environ.setdefault("PYTHONWARNINGS", _PYTHONWARNINGS)
    # Broad suppression for the current process (lower priority — added first, checked second)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    # Re-enable UserWarnings from cryoPARES itself (higher priority — added last, checked first)
    warnings.filterwarnings("always", category=UserWarning, module="cryoPARES.*")


def train():
    _suppress_user_warnings()
    from cryoPARES.train.train import main
    main()


def infer():
    _suppress_user_warnings()
    from cryoPARES.inference.infer import main
    main()


def reconstruct():
    _suppress_user_warnings()
    from cryoPARES.reconstruction.reconstruct import main
    main()


def projmatching():
    _suppress_user_warnings()
    from cryoPARES.projmatching.projmatching import main
    main()


def postprocess():
    _suppress_user_warnings()
    from cryoPARES.postprocessing.postprocess import main
    main()
