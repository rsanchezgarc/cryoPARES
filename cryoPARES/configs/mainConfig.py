import importlib
import inspect
import os
import tempfile
from dataclasses import field, dataclass
from pathlib import Path

from cryoPARES.configs.datamanager_config.datamanager_config import DataManager_config
from cryoPARES.configs.inference_config.inference_config import Inference_config
from cryoPARES.configs.models_config.models_config import Models_config
from cryoPARES.configs.projmatching_config.projmatching_config import Projmatching_config
from cryoPARES.configs.reconstruct_config.reconstruct_config import Reconstruct_config
from cryoPARES.configs.train_config.train_config import Train_config
from cryoPARES.utils.paths import find_project_root


def pyObjectFromStr(full_name, prefix=""):

    if prefix is None:
        prefix = ""

    splitName = full_name.split(".")
    *moduleBits, className = splitName
    moduleName = ""

    if prefix and moduleBits:
        moduleName = prefix + "."
    elif prefix:
        moduleName = prefix
    if moduleBits:
        moduleName += ".".join(moduleBits)

    package = None
    if moduleName.startswith("."):
        stack = inspect.stack()
        # stack[0] is this function
        # stack[1] is the function calling this one
        # stack[2] is the parent/caller we want
        caller_frame = stack[2]
        # Get the file path
        filepath = Path(caller_frame.filename)
        root = find_project_root()
        # Get relative path from project root to module
        package = str(filepath.relative_to(root).parent).replace(os.path.sep, ".").removesuffix(".py")

    module = importlib.import_module(moduleName, package=package)
    c = getattr(module, className)
    return c

configs_root = Path(__file__).parent.parent / "configs"


@dataclass
class MainConfig:
    cachedir: Path = Path(tempfile.gettempdir()) / "cryoPARES_cache"
    models: Models_config = field(default_factory=Models_config)
    datamanager: DataManager_config = field(default_factory=DataManager_config)
    train: Train_config = field(default_factory=Train_config)
    inference: Inference_config = field(default_factory=Inference_config)
    projmatching: Projmatching_config = field(default_factory=Projmatching_config)
    reconstruct: Reconstruct_config = field(default_factory=Reconstruct_config)

# Create an instance
main_config = MainConfig()
# print(main_config)
print(f"Config ({id(main_config)}) was initiated for PID {os.getpid()}")