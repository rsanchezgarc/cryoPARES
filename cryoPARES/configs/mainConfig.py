import importlib
import inspect
import os
from pathlib import Path

from cryoPARES.configManager.config_builder import build_config_structure, find_project_root


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



MainConfig = build_config_structure(configs_root,
                                    dict(
                                        cachedir=(Path, "/tmp/cache") #TODO: this is an ugly place to define where the cachedir should be
                                    )
                                    )

# Create an instance
main_config = MainConfig()
print(main_config)
print()