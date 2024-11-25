from dataclasses import dataclass, field, make_dataclass, MISSING, fields
from typing import Dict, Type, Any, Optional, List, Tuple, get_type_hints
from pathlib import Path
import importlib.util
import sys
import inspect


def clean_name(name: str) -> str:
    """Remove _config suffix and convert to lowercase."""
    return name.replace('_config', '').lower()


def load_python_module(file_path: Path) -> Optional[Any]:
    """Load a Python module from file path."""
    try:
        spec = importlib.util.spec_from_file_location(file_path.parent.name, file_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[file_path.parent.name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error loading module {file_path}: {e}")
        return None


def get_module_configs(module: Any) -> Tuple[Dict[str, Type], Dict[str, Any]]:
    """
    Extract both config classes and their fields from a module.
    Returns (nested_configs, direct_fields)
    """
    nested_configs = {}
    direct_fields = {}

    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and
                name.endswith('_config') and
                hasattr(obj, '__dataclass_fields__')):
            # For nested configs (classes ending in _config)
            nested_configs[clean_name(name)] = obj

        elif (inspect.isclass(obj) and
              name.endswith('_fields') and
              hasattr(obj, '__dataclass_fields__')):
            # For direct fields (from classes ending in _fields)
            for field_name, field_type in get_type_hints(obj).items():
                # Get the default value if it exists
                default_value = getattr(obj, field_name, MISSING)
                direct_fields[field_name] = (field_type, default_value)

    return nested_configs, direct_fields


def find_configs_recursive(directory: Path) -> Dict[str, Any]:
    """
    Recursively find all configs and fields in __init__.py files.
    Returns a nested dictionary structure of configs and fields.
    """
    results = {}

    # Check for __init__.py
    init_file = directory / '__init__.py'
    if init_file.exists():
        module = load_python_module(init_file)
        if module:
            nested_configs, direct_fields = get_module_configs(module)
            if nested_configs:
                results['configs'] = nested_configs
            if direct_fields:
                results['fields'] = direct_fields

    # Process subdirectories
    for path in directory.iterdir():
        if path.is_dir() and not path.name.startswith('__'):
            sub_results = find_configs_recursive(path)
            if sub_results:  # Only add non-empty results
                results[clean_name(path.name)] = sub_results

    return results


def create_dataclass_structure(
        name: str,
        config_dict: Dict[str, Any]
) -> Type:
    """
    Recursively create nested dataclass structure from config dictionary.
    Handles both nested configs and direct fields.
    """
    fields: List[Tuple[str, Type, Any]] = []

    # Process direct fields at current level
    direct_fields = config_dict.get('fields', {})
    for field_name, (field_type, default_value) in direct_fields.items():
        if default_value is MISSING:
            fields.append((field_name, field_type))
        else:
            fields.append((field_name, field_type, field(default=default_value)))

    # Process nested configs
    configs = config_dict.get('configs', {})
    for field_name, config_class in configs.items():
        fields.append(
            (field_name, config_class, field(default_factory=config_class))
        )

    # Process nested directories
    for key, value in config_dict.items():
        if key not in {'configs', 'fields'} and isinstance(value, dict):
            nested_class = create_dataclass_structure(
                key,
                value
            )
            if nested_class is not None:
                fields.append(
                    (key, nested_class, field(default_factory=nested_class))
                )

    if not fields:  # No fields found
        return None

    # Use clean name for the class
    clean_class_name = clean_name(name)
    return make_dataclass(
        clean_class_name,
        fields,
        bases=(object,),
        frozen=False
    )


def build_config_structure(project_root: Path, root_config_dict:Optional[Dict[str, Tuple[Type, Any]]]=None) -> Type:
    """
    Build the complete config structure for a project.
    Returns the main config class.
    """
    config_dict = find_configs_recursive(project_root)
    config_dict.update(dict(fields=root_config_dict))

    main_config = create_dataclass_structure('main', config_dict)
    if main_config is None:
        raise ValueError("No config classes found in project")
    return main_config
