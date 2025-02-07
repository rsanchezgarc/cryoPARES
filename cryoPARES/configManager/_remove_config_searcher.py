import os
import sys
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar, Union, Optional, List, Callable
from inspect import signature
import functools

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.utils.paths import find_configs_root

T = TypeVar('T')
ConfigType = TypeVar('ConfigType')
F = TypeVar('F', bound=Callable)


def find_config(obj: Any, path: List[str], debug: bool = False) -> Optional[Any]:
    """Traverse config path, checking terminal nodes at each step"""
    current = obj

    for i, part in enumerate(path[:-1]):  # All but last part
        if debug:
            print(f"At level {i}, checking {part}")

        try:
            current = getattr(current, part)
            terminal = getattr(current, path[-1], None)
            if is_dataclass(terminal):
                if debug:
                    print(f"Found terminal config at level {i}")
                return terminal
        except AttributeError:
            return None

    try:
        final = getattr(current, path[-1])
        if is_dataclass(final):
            return final
    except AttributeError:
        return None

    return None


def get_module_path(cls: Type, classname: Optional[str] = None) -> List[str]:
    """
    Get the module path by comparing file location to project root.
    Returns: List of path components, e.g. ['models', 'image2sphere', 'components']
    """
    try:
        actual_path = Path(os.path.abspath(sys.modules[cls.__module__].__file__))
    except (AttributeError, KeyError):
        actual_path = Path(__file__).resolve()
    # Get project root
    root = find_configs_root()

    # Get relative path from project root to module
    rel_path = actual_path.relative_to(root)

    # Convert path to module components, excluding the filename and any __init__.py
    path_parts = list(rel_path.parts)
    # if "Image2Sphere" in str(cls): breakpoint()
    # if "DataManager" in str(cls): breakpoint()

    if path_parts[0] == '.':
        path_parts = path_parts[1:]

    # Remove common Python package markers
    # path_parts = [p.removesuffix(".py") for p in path_parts if p not in {'src', 'lib', '__pycache__'}]
    path_parts = [p for p in path_parts if p not in {'src', 'lib', '__pycache__'}]
    assert path_parts
    # Add the class name (lowercase)
    if classname:
        if path_parts[-1].endswith(".py"):
            path_parts[-1] = path_parts[-1].removesuffix(".py")
        path_parts.append(classname)
    else:
        if len(path_parts)>1 and path_parts[-1].removesuffix(".py").lower() == path_parts[-2]: #Somethinkg that looks like name/name.py
            path_parts = path_parts[:-1]
        else:
            path_parts[-1] = path_parts[-1].removesuffix(".py")
        if cls.__name__.lower() != path_parts[-1].lower(): #Somethinkg does it is not like name/name.py
            path_parts.append(cls.__name__)

    path_parts = [p.lower() for p in path_parts]
    # print(cls, path_parts)
    return path_parts

def inject_params_from_config(func: F, config: Any, is_method: bool = False) -> F:
    """Injects parameters from config into a function or method."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        params = list(sig.parameters.items())[1:] if is_method else list(sig.parameters.items())

        new_kwargs = kwargs.copy()
        for param_name, param in params:
            if (param_name not in bound_args.arguments or
                bound_args.arguments[param_name] is None) and \
                    hasattr(config, param_name):
                new_kwargs[param_name] = getattr(config, param_name)

        return func(*args, **new_kwargs)

    return wrapper


def inject_config(
        config_class_or_instance: Union[Type[ConfigType], ConfigType, None] = None,
        methods: Optional[List[str]] = None,
        classname: Optional[str] = None,
):
    """
    Decorator that injects config values, with auto-search in main_config if no config provided.
    Automatically determines module path based on file location relative to project root.
    """
    def decorator(target: Union[Type[T], F]) -> Union[Type[T], F]:
        # Determine config to use
        config = config_class_or_instance
        if config is None:
            # Auto-detect path
            path = get_module_path(target if isinstance(target, type) else target.__class__, classname=classname)
            found_config = find_config(main_config, path)

            if found_config is not None and is_dataclass(found_config):
                config = found_config

            if config is None:
                #breakpoint()
                raise ValueError(
                    f"Could not find matching config for {target.__name__} "
                    f"with path {path}"
                )

        # Get config instance
        config_instance = config if not isinstance(config, type) else config()

        # Case 1: Decorating a class
        if isinstance(target, type):
            methods_to_patch = {'__init__'}.union(methods or set())

            for method_name in methods_to_patch:
                if hasattr(target, method_name):
                    original_method = getattr(target, method_name)
                    patched_method = inject_params_from_config(
                        original_method, config_instance, is_method=True
                    )
                    setattr(target, method_name, patched_method)

            return target

        # Case 2: Decorating a function
        return inject_params_from_config(target, config_instance, is_method=False)

    # Handle case where decorator is used without parameters
    if callable(config_class_or_instance):
        return decorator(config_class_or_instance)

    return decorator