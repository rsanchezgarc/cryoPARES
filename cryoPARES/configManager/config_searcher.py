from dataclasses import is_dataclass
from typing import Any, Type, TypeVar, Union, Optional, List, Callable
from inspect import signature
import functools

from cryoPARES.configManager.config_builder import get_module_path
from cryoPARES.configs.mainConfig import main_config

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