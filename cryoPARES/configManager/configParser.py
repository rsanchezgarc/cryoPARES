import argparse
import functools

import yaml
import json
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, List, Optional, Union, Callable, get_origin, get_args
from pathlib import Path
import re
import sys
from enum import Enum

from argParseFromDoc import AutoArgumentParser

from cryoPARES.configs.mainConfig import main_config


class ConfigOverrideSystem:
    """System for overriding config values from command line or YAML files."""

    @staticmethod
    def print_config(config: Any, indent: int = 0, name: str = "config") -> None:
        """Recursively print config structure and values."""
        indent_str = "  " * indent

        if is_dataclass(config):
            print(f"{indent_str}{name}:")
            for field in fields(config):
                field_value = getattr(config, field.name)
                if is_dataclass(field_value):
                    ConfigOverrideSystem.print_config(field_value, indent + 1, field.name)
                else:
                    print(f"{indent_str}  {field.name}: {field_value} ({type(field_value).__name__})")
        else:
            print(f"{indent_str}{name}: {config} ({type(config).__name__})")

    @staticmethod
    def get_all_config_paths(config: Any, prefix: str = "") -> List[str]:
        """Get all valid config paths for documentation."""
        paths = []

        if is_dataclass(config):
            for field in fields(config):
                field_value = getattr(config, field.name)
                field_path = f"{prefix}.{field.name}" if prefix else field.name

                if is_dataclass(field_value):
                    paths.extend(ConfigOverrideSystem.get_all_config_paths(field_value, field_path))
                else:
                    # Format as key=value (type) for display
                    paths.append(f"{field_path} = {repr(field_value)} ({type(field_value).__name__})")

        return paths

    @staticmethod
    def parse_value(value_str: str) -> Any:
        """Parse a string value into appropriate Python type."""
        # Handle quoted strings
        if (value_str.startswith('"') and value_str.endswith('"')) or \
                (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]

        # Handle booleans
        if value_str.lower() == 'true':
            return True
        elif value_str.lower() == 'false':
            return False

        # Handle None
        if value_str.lower() == 'none':
            return None

        # Handle lists (simple parsing)
        if value_str.startswith('[') and value_str.endswith(']'):
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                # If JSON parsing fails, try simple comma separation
                items = value_str[1:-1].split(',')
                return [ConfigOverrideSystem.parse_value(item.strip()) for item in items if item.strip()]

        # Handle paths
        if '/' in value_str or '\\' in value_str:
            return Path(value_str)

        # Try to parse as number
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            # Return as string if all else fails
            return value_str

    @staticmethod
    def parse_config_assignments(assignments: List[str]) -> Dict[str, Any]:
        """Parse command line config assignments like 'train.n_epochs=100'."""
        overrides = {}

        for assignment in assignments:
            # Handle assignment with spaces around =
            match = re.match(r'^(.+?)=(.+)$', assignment)
            if not match:
                raise ValueError(f"Invalid config assignment: {assignment}. Expected format: key=value")

            key, value_str = match.groups()
            key = key.strip()
            value_str = value_str.strip()

            # Parse the value
            value = ConfigOverrideSystem.parse_value(value_str)

            # Handle nested keys
            keys = key.split('.')
            current = overrides
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

        return overrides

    @staticmethod
    def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
        """Load config overrides from a YAML file."""
        with open(yaml_path, 'r') as f:
            return yaml.unsafe_load(f) or {}

    @staticmethod
    def convert_to_enum_if_needed(value: Any, field_type: Any) -> Any:
        """Convert string values to enum if the field type is an enum."""
        # Check if it's a Union type (which includes Optional)
        if get_origin(field_type) is Union:
            # For Optional[EnumType], get the non-None type
            enum_type = None
            for arg in get_args(field_type):
                if arg is not type(None) and hasattr(arg, '__bases__'):
                    if any(issubclass(base, Enum) for base in arg.__bases__ if isinstance(base, type)):
                        enum_type = arg
                        break
            if enum_type:
                field_type = enum_type

        # Check if field_type is an enum class
        if (hasattr(field_type, '__bases__') and
                any(issubclass(base, Enum) for base in field_type.__bases__ if isinstance(base, type))):

            # If value is already an instance of the enum, return it
            if isinstance(value, field_type):
                return value

            # If value is a string, try to convert it to the enum
            if isinstance(value, str):
                try:
                    # Try direct conversion first
                    return field_type(value)
                except ValueError:
                    # Try to find enum member by value
                    for member in field_type:
                        if member.value == value:
                            return member
                    # Try case-insensitive matching
                    for member in field_type:
                        if member.value.lower() == value.lower():
                            return member
                    raise ValueError(f"'{value}' is not a valid {field_type.__name__}")

        return value

    @staticmethod
    def apply_overrides(config: Any, overrides: Dict[str, Any], path: str = "", verbose=True) -> None:
        """Recursively apply overrides to a config object."""
        for key, value in overrides.items():
            current_path = f"{path}.{key}" if path else key

            if hasattr(config, key):
                current_value = getattr(config, key)

                # If value is a dict and current value is a dataclass, recurse
                if isinstance(value, dict) and is_dataclass(current_value):
                    ConfigOverrideSystem.apply_overrides(current_value, value, current_path, verbose=verbose)
                else:
                    # Direct assignment
                    try:
                        # Handle Path type conversion if the target attribute is a Path object
                        # or if its type hint is Path.
                        # This check is slightly more robust by looking at the target's current value
                        # and its field type.
                        field_type = None
                        if is_dataclass(config):
                            for f in fields(config):
                                if f.name == key:
                                    field_type = f.type
                                    break

                        # Handle enum conversion
                        if field_type:
                            value = ConfigOverrideSystem.convert_to_enum_if_needed(value, field_type)

                        if field_type == Path or (isinstance(current_value, Path) and not isinstance(value, Path)):
                            value = Path(value)

                        setattr(config, key, value)
                        if verbose: print(f"Set {current_path} = {value}")
                    except Exception as e:
                        if verbose: print(f"Warning: Failed to set {current_path} to {value} (type {type(value).__name__}): {e}")
            else:
                if verbose: print(
                    f"Warning: Config object '{path or config.__class__.__name__}' has no attribute '{key}'. Full path: '{current_path}'")

    @staticmethod
    def _drop_paths_from_dict(overrides: Dict[str, Any], drop_paths: List[str], verbose: bool = False):
        """
        Removes specified keys from a nested dictionary in-place.

        :param overrides: The dictionary to modify.
        :param drop_paths: A list of dot-separated paths to remove.
        :param verbose: If True, prints warnings for paths that are not found.
        """
        if not drop_paths:
            return

        for path in drop_paths:
            keys = path.split('.')
            parent_dict = overrides
            try:
                # Navigate to the parent dictionary of the key to be deleted
                for key in keys[:-1]:
                    parent_dict = parent_dict[key]
                # Delete the final key if it exists
                if keys[-1] in parent_dict:
                    del parent_dict[keys[-1]]
            except (KeyError, TypeError):
                # This can happen if a parent key doesn't exist or is not a dictionary
                if verbose:
                    print(f"Warning: Could not drop path '{path}' as it was not found.")

    @staticmethod
    def update_config_from_file(config, config_fname, drop_paths=None, verbose=False):
        overrides = ConfigOverrideSystem.load_yaml_config(config_fname)
        ConfigOverrideSystem._drop_paths_from_dict(overrides, drop_paths, verbose)
        ConfigOverrideSystem.apply_overrides(config, overrides, verbose=verbose)
        return config

    @staticmethod
    def update_config_from_dataclass(config, source_dataclass, drop_paths=None, verbose=False):
        """
        Updates a config object with values from a source dataclass object.

        :param config: The config object to update.
        :param source_dataclass: The dataclass object to read values from.
        :param drop_paths: A list of dot-separated paths to not update from the source.
        :param verbose: If True, prints which values are being set.
        """
        # Convert the source dataclass to a dictionary of overrides
        overrides = dataclass_to_dict(source_dataclass)
        ConfigOverrideSystem._drop_paths_from_dict(overrides, drop_paths, verbose)
        # Apply the dictionary of overrides to the target config object
        ConfigOverrideSystem.apply_overrides(config, overrides, verbose=verbose)
        return config


    @staticmethod
    def update_config_from_configstrings(config, source_configstrings:List[str], drop_paths=None, verbose=False):
        """
        Updates a config object with values from a source dataclass object.

        :param config: The config object to update.
        :param source_configstrings: The list of strings of the form xxx.xxx=V to read values from.
        :param drop_paths: A list of dot-separated paths to not update from the source.
        :param verbose: If True, prints which values are being set.
        """
        # Convert the source dataclass to a dictionary of overrides
        overrides = ConfigOverrideSystem.parse_config_assignments(source_configstrings)
        ConfigOverrideSystem._drop_paths_from_dict(overrides, drop_paths, verbose)
        # Apply the dictionary of overrides to the target config object
        ConfigOverrideSystem.apply_overrides(config, overrides, verbose=verbose)
        return config

def merge_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = d1.copy()
    for key, value in d2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def dataclass_to_dict(obj: Any) -> Any:
    """Convert a dataclass (and nested dataclasses) to a dictionary."""
    if is_dataclass(obj):
        result = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def export_config_to_yaml(config: Any, filepath: str) -> None:
    """Export a config object to a YAML file."""
    config_dict = dataclass_to_dict(config)
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


class ConfigArgumentParser(AutoArgumentParser):
    """AutoArgumentParser with integrated config override support."""

    def __init__(self, verbose=False, *args, config_obj: Optional[Any] = main_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.config_obj = config_obj
        self._config_arg_names = []
        self._config_param_mappings = {}  # Maps arg names to config paths

        if config_obj:
            self._add_config_args()

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _add_config_args(self):
        """Add config-related arguments to the parser."""
        config_group = self.add_argument_group('Configuration Overrides')
        config_group.add_argument(
            '--show-config',
            action='store_true',
            help='Show all available config values and exit'
        )

        config_group.add_argument(
            '--config',
            nargs='+',
            metavar='KEY=VALUE or CONFIG_FILE',
            help='Override config values. Can be key=value pairs (e.g., train.n_epochs=100) '
                 'or a YAML config file. Multiple values can be specified.'
        )

        config_group.add_argument(
            '--export-config',
            metavar='FILE',
            help='Export current config (with overrides) to a YAML file'
        )

        # Track which arguments are config-related
        self._config_arg_names = ['show_config', 'config', 'export_config']

    def add_args_from_function(self, callable: Callable, new_group_name: Optional[str] = None,
                               args_to_ignore: List[str] = None,
                               args_to_include: List[str] = None,
                               args_optional: List[str] = None) -> Union["AutoArgumentParser", argparse._ArgumentGroup]:
        """Override to track CONFIG_PARAM arguments and call parent method."""
        # from cryoPARES.configManager.inject_defaults import CONFIG_PARAM # Already imported

        argname_to_configname = getattr(callable, "_argname_to_configname", {})
        for argname, config_param in argname_to_configname.items():
            # Determine the actual config attribute name from CONFIG_PARAM's _name or parameter name
            config_attr_name_for_mapping = config_param._name
            # Find the config path
            if hasattr(config_param, '_config') or config_param._name:  # If explicit config or name is given
                # Prefer the config object specified within CONFIG_PARAM itself
                # Fallback to self.config_obj if not explicitly set in CONFIG_PARAM
                config_obj_for_param = config_param._config or self.config_obj

                if config_obj_for_param:
                    config_path = self._find_config_path(self.config_obj, config_obj_for_param,
                                                         config_attr_name_for_mapping)
                    if config_path:
                        self._config_param_mappings[argname] = config_path
                else:
                    print(
                        f"Warning: CONFIG_PARAM for '{argname}' is not bound to any config object and no default config_obj is set for ConfigArgumentParser.")
            else:
                # If not bound yet (e.g., CONFIG_PARAM() used without explicit config),
                # assume it will use the default config (self.config_obj) and the parameter name.
                config_path = self._find_config_attribute_path(self.config_obj, config_attr_name_for_mapping)
                if config_path:
                    self._config_param_mappings[argname] = config_path
                else:
                    print(
                        f"Warning: Could not find config path for parameter '{argname}' (looking for '{config_attr_name_for_mapping}') in the default config object.")

        # Now call the parent method to add standard arguments
        return super().add_args_from_function(
            callable, new_group_name, args_to_ignore, args_to_include, args_optional
        )

    def _find_config_path(self, root_config: Any, target_config_obj: Any, attr_name: str) -> Optional[str]:
        """
        Find the full dot-separated path from the root_config to target_config_obj.attr_name.
        This is useful when the CONFIG_PARAM is bound to a sub-dataclass.
        """
        if root_config is None or target_config_obj is None:
            return None

        # Base case: if root_config is the target_config_obj itself
        if root_config is target_config_obj:
            if hasattr(root_config, attr_name):
                return attr_name
            return None

        # Recursive case for dataclasses
        if is_dataclass(root_config):
            for field in fields(root_config):
                field_value = getattr(root_config, field.name)
                # If the field value is the target object, then the path is direct
                if field_value is target_config_obj:
                    if hasattr(target_config_obj, attr_name):
                        return f"{field.name}.{attr_name}"
                # If the field value is another dataclass, recurse
                elif is_dataclass(field_value):
                    sub_path = self._find_config_path(field_value, target_config_obj, attr_name)
                    if sub_path:
                        return f"{field.name}.{sub_path}"
        return None

    def _find_config_attribute_path(self, config: Any, attr_name: str, prefix: str = "") -> Optional[str]:
        """
        Find the path to an attribute anywhere in the config hierarchy.
        This is a more general search than _find_config_path.
        """
        # Direct hit
        if hasattr(config, attr_name):
            return f"{prefix}.{attr_name}" if prefix else attr_name

        # Recurse through dataclass fields
        if is_dataclass(config):
            for field in fields(config):
                field_value = getattr(config, field.name)
                field_path = f"{prefix}.{field.name}" if prefix else field.name

                if is_dataclass(field_value):
                    result = self._find_config_attribute_path(field_value, attr_name, field_path)
                    if result:
                        return result
        return None

    def _get_value_from_path(self, config_path: str) -> Any:
        """Navigate a dot-separated path to get a value from the config object."""
        try:
            keys = config_path.split('.')
            return functools.reduce(getattr, keys, self.config_obj)
        except AttributeError:
            return None # Or raise an error

    def _update_defaults_from_config(self, args: argparse.Namespace, sys_argv: List[str]):
        """
        After parsing, update the defaults in the 'args' namespace with any
        values that were changed via --config overrides, but only if the direct
        argument was not provided by the user.
        """
        # self.print("\nRe-checking defaults against config overrides...")
        for arg_name, config_path in self._config_param_mappings.items():
            # Check if the arg exists in the namespace and was NOT provided on the CLI
            if hasattr(args, arg_name) and not self._was_arg_provided(arg_name, sys_argv):
                # Get the potentially updated value from the config object
                new_default = self._get_value_from_path(config_path)
                current_val = getattr(args, arg_name)

                # If the value in the config is different from the stale default, update it
                if new_default is not None and new_default != current_val:
                    setattr(args, arg_name, new_default)
                    self.print(
                        f"Updated default for '{arg_name}' from '{current_val}' to '{new_default}' (from config override)")

    def parse_args(self, args=None, namespace=None):
        """Parse arguments, process config overrides, and return cleaned args."""
        # Capture original sys.argv for _was_arg_provided if args is None
        _original_sys_argv = sys.argv[1:]

        if args is None:
            args_list = sys.argv[1:]
        elif isinstance(args, list):
            args_list = args
        else:
            args_list = list(args)

        # Check for --show-config early, before required arguments are validated
        if '--show-config' in args_list and self.config_obj:
            print("\n=== Available Configuration Options ===\n")
            ConfigOverrideSystem.print_config(self.config_obj)
            print("\n=== Configuration Paths ===")
            print("Use these paths with --config to override values:\n")
            for path_info in ConfigOverrideSystem.get_all_config_paths(self.config_obj):
                print(f"  {path_info}")  # path_info already includes value and type
            print("\nExample: --config train.n_epochs=100 train.batch_size=64")
            print("Or use a YAML file: --config my_config.yaml")
            sys.exit(0)

        # Now do normal parsing
        # Parse once to get all arguments, including config and direct ones.
        # This will populate the `parsed_args` namespace.
        parsed_args = super().parse_args(args_list, namespace)

        # Get the list of args passed to --config, e.g., ["my_file.yaml", "train.n_epochs=20"]
        config_args_list = getattr(parsed_args, 'config', []).copy(
                                ) if hasattr(parsed_args, 'config') and parsed_args.config else []


        # Process --config (YAML files and key=value assignments) first.
        # These are considered lower precedence than direct command-line arguments.
        self._process_config_args(parsed_args)

        # Update the defaults in the parsed_args namespace based on the just-processed config.
        self._update_defaults_from_config(parsed_args, _original_sys_argv)

        # Check for conflicts between direct args and config overrides.
        # This must happen after _process_config_args because we need to know
        # what paths are being set by --config.
        self._check_for_conflicts(parsed_args, _original_sys_argv)

        # Apply direct argument overrides to config. This should happen AFTER --config files,
        # ensuring direct command-line arguments have highest precedence.
        direct_overrides = self._apply_direct_arg_overrides(parsed_args, _original_sys_argv)

        # Remove config arguments from parsed_args so they don't interfere with downstream logic
        # that might not expect them (e.g., if train_model tried to use parsed_args directly for 'config').
        config_args = config_args_list + direct_overrides
        for arg_name in self._config_arg_names:
            if hasattr(parsed_args, arg_name):
                delattr(parsed_args, arg_name)

        return parsed_args, config_args

    def _get_cli_arg_name(self, param_dest_name: str) -> str:
        """Finds the primary option string (e.g., '--batch-size') for a given parameter destination."""
        for action in self._actions:
            if action.dest == param_dest_name:
                # Prefer the first long option string if available
                for opt_string in action.option_strings:
                    if opt_string.startswith('--'):
                        return opt_string
                # Fallback to the first option string or the destination name
                return action.option_strings[0] if action.option_strings else param_dest_name
        return param_dest_name.replace('_', '-')  # Final fallback

    def _check_for_conflicts(self, args: argparse.Namespace, sys_argv: List[str]) -> None:
        """
        Check for conflicts between direct arguments and config overrides.
        A conflict occurs if the same config path is targeted by both a direct CLI argument
        and a --config assignment WITH A DIFFERENT VALUE.
        """
        if not hasattr(args, 'config') or not args.config:
            return

        # Parse config overrides from the --config argument to get all affected paths and values.
        # (This part of the logic remains the same)
        config_overrides_from_cli = {}
        for config_item in args.config:
            if not (config_item.endswith('.yaml') or config_item.endswith('.yml')):
                try:
                    assignment_overrides = ConfigOverrideSystem.parse_config_assignments([config_item])
                    config_overrides_from_cli = merge_dicts(config_overrides_from_cli, assignment_overrides)
                except ValueError as e:
                    print(f"Warning: Could not parse config assignment '{config_item}' for conflict check: {e}")
            else:
                # YAML file loading logic...
                config_path_obj = Path(config_item)
                if config_path_obj.exists():
                    file_overrides = ConfigOverrideSystem.load_yaml_config(config_item)
                    config_overrides_from_cli = merge_dicts(config_overrides_from_cli, file_overrides)
                else:
                    print(f"Warning: Config file {config_item} not found for conflict check.")

        override_paths_from_config_arg = set(self._flatten_dict_keys(config_overrides_from_cli))

        # Iterate and check for conflicts
        for param_name_in_func, config_path_in_config_obj in self._config_param_mappings.items():
            if config_path_in_config_obj in override_paths_from_config_arg:
                if self._was_arg_provided(param_name_in_func, sys_argv):
                    # --- NEW LOGIC STARTS HERE ---

                    # 1. Get the value from the direct argument (e.g., --batch-size 10)
                    direct_arg_value = getattr(args, param_name_in_func)

                    # 2. Get the value from the --config override
                    try:
                        keys = config_path_in_config_obj.split('.')
                        config_override_value = functools.reduce(lambda d, k: d[k], keys, config_overrides_from_cli)
                    except KeyError:
                        # This shouldn't happen if the path is in the set, but as a safeguard:
                        continue

                    # 3. Compare the values and raise an error only if they differ
                    if direct_arg_value != config_override_value:
                        cli_arg_name = self._get_cli_arg_name(param_name_in_func)  # Extracted to a helper for clarity
                        raise ValueError(
                            f"Conflict: The value for '{config_path_in_config_obj}' is ambiguous.\n"
                            f"  - Direct argument '{cli_arg_name}' provides: {direct_arg_value}\n"
                            f"  - A '--config' argument provides: {config_override_value}\n"
                            f"Please provide only one source or ensure the values match."
                        )

    def _was_arg_provided(self, arg_name: str, sys_argv: List[str]) -> bool:
        """
        Check if an argument with `arg_name` (its `dest`) was explicitly provided on the command line.
        This method is robust as it checks against all registered option strings for that argument.
        """
        # Find the action associated with this arg_name (the 'dest' attribute of the argument)
        target_action = None
        for action in self._actions:
            if action.dest == arg_name:
                target_action = action
                break

        if not target_action:
            # This can happen if AutoArgumentParser didn't add an argument for this param_name.
            # For CONFIG_PARAMs used with add_args_from_function, an action *should* exist.
            # print(f"Debug: No argparse action found for destination '{arg_name}'. Cannot reliably check if provided.")
            return False

        # Check all option strings associated with this action (e.g., '--epochs', '-e')
        for option_string in target_action.option_strings:
            # Check for direct match (e.g., '--epochs' or '-e')
            if option_string in sys_argv:
                return True
            # Check for assignment form (e.g., '--epochs=100' or '-e=100')
            for arg_in_sys_argv in sys_argv:
                if arg_in_sys_argv.startswith(f"{option_string}="):
                    return True
        return False

    def _flatten_dict_keys(self, d: Dict[str, Any], prefix: str = "") -> List[str]:
        """Flatten nested dictionary to get all key paths."""
        paths = []
        for key, value in d.items():
            current_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                paths.extend(self._flatten_dict_keys(value, current_path))
            else:
                paths.append(current_path)
        return paths

    def _apply_direct_arg_overrides(self, args: argparse.Namespace, sys_argv: List[str]) -> List[str]:
        """
        Apply direct argument values to their corresponding config parameters.
        Only applies if the argument was explicitly provided by the user.
        """
        overrides_made = []
        for arg_name, config_path in self._config_param_mappings.items():
            if hasattr(args, arg_name) and self._was_arg_provided(arg_name, sys_argv):
                arg_value = getattr(args, arg_name)
                # Check if the argument was actually provided (not just its default)
                if arg_value is not None:
                    # Apply this value to the config
                    keys = config_path.split('.')
                    target = self.config_obj

                    try:
                        # Navigate to the parent object
                        for key_part in keys[:-1]:
                            target = getattr(target, key_part)

                        # Set the value
                        # Need to handle type conversion here for Path, etc., similar to apply_overrides
                        field_type = None
                        if is_dataclass(target):
                            for f in fields(target):
                                if f.name == keys[-1]:
                                    field_type = f.type
                                    break

                        value_to_set = arg_value

                        # Handle enum conversion
                        if field_type:
                            value_to_set = ConfigOverrideSystem.convert_to_enum_if_needed(value_to_set, field_type)

                        if field_type == Path and not isinstance(arg_value, Path):
                            value_to_set = Path(arg_value)

                        setattr(target, keys[-1], value_to_set)
                        overrides_made.append(f"{config_path}={value_to_set}")
                        self.print(
                            f"Set {config_path} = {value_to_set} (from direct arg --{arg_name.replace('_', '-')})")
                    except Exception as e:
                        print(
                            f"Warning: Failed to set config attribute '{config_path}' from direct argument '--{arg_name.replace('_', '-')}' to '{arg_value}': {e}")
        return overrides_made

    def _process_config_args(self, args: argparse.Namespace) -> None:
        """Process config-related arguments and apply overrides from --config."""
        if not self.config_obj:
            return

        # Handle --config overrides
        if hasattr(args, 'config') and args.config:
            overrides = {}

            for config_item in args.config:
                # Check if it's a file
                if config_item.endswith('.yaml') or config_item.endswith('.yml'):
                    config_file_path = Path(config_item)
                    if config_file_path.exists():
                        file_overrides = ConfigOverrideSystem.load_yaml_config(config_item)
                        # Merge file overrides
                        overrides = merge_dicts(overrides, file_overrides)
                        self.print(f"Loaded config from {config_item}")
                    else:
                        print(f"Warning: Config file {config_item} not found. Skipping.")
                else:
                    # It's a key=value assignment
                    try:
                        assignment_overrides = ConfigOverrideSystem.parse_config_assignments([config_item])
                        overrides = merge_dicts(overrides, assignment_overrides)
                    except ValueError as e:
                        print(f"Warning: Invalid config assignment '{config_item}'. Skipping. Error: {e}")

            if overrides:
                self.print("\nApplying config overrides from --config argument:")
                ConfigOverrideSystem.apply_overrides(self.config_obj, overrides, verbose=self.verbose)