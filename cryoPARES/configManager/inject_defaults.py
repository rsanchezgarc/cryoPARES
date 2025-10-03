from dataclasses import dataclass
from functools import wraps
import inspect
from enum import Enum
from typing import Any, get_type_hints, Optional, Callable, Tuple, get_origin, get_args, Union, Literal, List, Dict


class CONFIG_PARAM:
    """Enhanced parameter descriptor for config injection with value tracking."""

    def __init__(
            self,
            validator: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            doc: Optional[str] = None,
            config: Optional[Any] = None
    ):
        self.validator = validator
        self.transform = transform
        self.doc = doc
        self._config = config
        self._name = None

    def bind(self, config: Any, name: str):
        """Bind this parameter to a specific config attribute."""
        if self._config is None:  # Only bind if no config was specified at creation
            self._config = config
        self._name = name

        # Auto-populate doc from config's PARAM_DOCS if not already provided
        if self.doc is None and hasattr(config, 'PARAM_DOCS'):
            self.doc = config.PARAM_DOCS.get(name)

    def validate(self, value: Any) -> bool:
        if self.validator is None:
            return True
        return self.validator(value)

    def transform_value(self, value: Any) -> Any:
        if self.transform is None:
            return value
        return self.transform(value)

    def convert_to_enum_if_needed(self, value: Any, expected_type: Any) -> Any:
        """Convert string values to enum if the expected type is an enum."""
        if (inspect.isclass(expected_type) and
            issubclass(expected_type, Enum) and
            isinstance(value, str)):
            try:
                return expected_type(value)
            except ValueError:
                # Try to find enum member by value
                for member in expected_type:
                    if member.value == value:
                        return member
                raise ValueError(f"'{value}' is not a valid {expected_type.__name__}")
        return value

    def __call__(self) -> Any:
        """Get the current value from the config."""
        return self.get()

    def get(self):
        """Get the current value from the config."""
        if self._config is None or self._name is None:
            raise RuntimeError("CONFIG_PARAM not bound to a config")
        return getattr(self._config, self._name)

    @property
    def value(self):
        return self.get()

    @property
    def is_bound(self) -> bool:
        """Check if the parameter is bound to a config."""
        return self._config is not None


def _check_type_match(expected_type: Any, actual_value: Any) -> bool:
    """Enhanced type checking that handles generic types like Tuple, Optional, Union, and Literal.
    Treats lists and tuples as equivalent for type checking purposes."""
    # Handle None for Optional types
    if actual_value is None:
        # If expected_type is a Union or Optional, check if None is allowed
        if get_origin(expected_type) is Union:
            return type(None) in get_args(expected_type)
        return False

    if inspect.isclass(expected_type) and issubclass(expected_type, Enum):
        # If actual_value is already an instance of the enum, it's valid
        if isinstance(actual_value, expected_type):
            return True
        # If actual_value is a string that matches an enum value, it's valid
        if isinstance(actual_value, str):
            try:
                # Try to create enum from string value
                expected_type(actual_value)
                return True
            except ValueError:
                # Check if the string matches any enum member's value
                return actual_value in [member.value for member in expected_type]
        return False

    # Handle special case where expected_type is Any
    if expected_type is Any:
        return True

    # Get the origin type (e.g., tuple from Tuple[float, float])
    expected_origin = get_origin(expected_type)

    # Handle Literal types
    if expected_origin is Literal:
        if hasattr(actual_value, "value") and not actual_value in get_args(expected_type):
            actual_value = actual_value.value
        return actual_value in get_args(expected_type)

    # If there's no origin type, do a direct comparison
    if expected_origin is None:
        return isinstance(actual_value, expected_type)

    # Handle Union types (including Optional)
    if expected_origin is Union:
        return any(_check_type_match(arg_type, actual_value)
                   for arg_type in get_args(expected_type))

    # Special handling for sequences (list and tuple)
    if expected_origin in (list, tuple):
        if not isinstance(actual_value, (list, tuple)):
            return False

        # Get the expected argument types
        expected_args = get_args(expected_type)

        # If no arguments specified (just List or Tuple), any sequence is fine
        if not expected_args:
            return True

        # Check if it's a variable-length sequence (List[int], Tuple[int, ...])
        if expected_origin is list or (len(expected_args) == 2 and expected_args[1] is Ellipsis):
            element_type = expected_args[0] if expected_origin is list else expected_args[0]
            return all(_check_type_match(element_type, item) for item in actual_value)

        # Fixed-length tuple: check length and types
        if expected_origin is tuple and len(actual_value) != len(expected_args):
            return False

        if expected_origin is tuple:
            return all(_check_type_match(expected_arg, actual_item)
                       for expected_arg, actual_item in zip(expected_args, actual_value))

    # For other generic types, just check the origin type but treat list/tuple as equivalent
    if expected_origin in (list, tuple):
        return isinstance(actual_value, (list, tuple))
    return isinstance(actual_value, expected_origin)


def inject_docs_from_config_params(func):
    """
    Decorator to inject parameter documentation from CONFIG_PARAMs and PARAM_DOCS into function docstrings.

    This decorator should be applied AFTER inject_defaults_from_config, as it relies on
    the _argname_to_configname attribute created by that decorator.

    Usage:
        @inject_docs_from_config_params
        @inject_defaults_from_config(config)
        def my_function(param1: int = CONFIG_PARAM(), param2: str = "default"):
            '''
            My function.

            :param param1: {param1}
            :param param2: {param2}
            '''
            pass

    The {param1} placeholder will be replaced with the documentation from CONFIG_PARAM.doc,
    which is automatically populated from the config's PARAM_DOCS dictionary.

    The {param2} placeholder will be looked up in the config's PARAM_DOCS dictionary directly,
    allowing non-CONFIG_PARAM parameters to also use centralized documentation.
    """
    if not hasattr(func, '_argname_to_configname'):
        # Function not decorated with inject_defaults_from_config, nothing to do
        return func

    if func.__doc__:
        docs_dict = {}

        # First, collect docs from CONFIG_PARAMs
        for param_name, config_param in func._argname_to_configname.items():
            if isinstance(config_param, CONFIG_PARAM) and config_param.doc:
                docs_dict[param_name] = config_param.doc

        # Second, look up remaining parameters from the config's PARAM_DOCS
        # Get the config object from the wrapper's stored reference
        if hasattr(func, '_inject_default_config'):
            config = func._inject_default_config
            sig = inspect.signature(func)

            for param_name in sig.parameters:
                # Skip if we already have docs from CONFIG_PARAM
                if param_name in docs_dict:
                    continue

                # Try to find docs in the config's PARAM_DOCS
                if hasattr(config, 'PARAM_DOCS') and param_name in config.PARAM_DOCS:
                    docs_dict[param_name] = config.PARAM_DOCS[param_name]

        if docs_dict:
            try:
                func.__doc__ = func.__doc__.format(**docs_dict)
            except KeyError as e:
                # Missing placeholder in docstring - that's okay, just skip formatting
                pass

    return func


def inject_defaults_from_config(default_config: Any, update_config_with_args: bool = False):
    """

    :param default_config: The default configuration where the default values will be read. Can be parameter-specific
                            ignored by providing CONFIG_PARAM(config=otherConfig)
    :param update_config_with_args: If true, the config for a parameter will be updated with the new value if it is not a default one.
    :return:
    """
    def decorator(func):
        sig = inspect.signature(func)
        hints = get_type_hints(func)

        lazy_params = {}
        param_processors = {}
        param_configs: Dict[str, Any] = {}

        for name, param in sig.parameters.items():
            if isinstance(param.default, CONFIG_PARAM):
                config_to_use = param.default._config or default_config
                param_configs[name] = config_to_use

                if not hasattr(config_to_use, name):
                    raise ValueError(f"Config missing parameter: {name}")

                config_value = getattr(config_to_use, name)
                expected_type = hints.get(name)

                if not _check_type_match(expected_type, config_value):
                    raise TypeError(
                        f"Type mismatch for {name}: expected {expected_type}, "
                        f"got {type(config_value)} with value {config_value}"
                    )

                param.default.bind(config_to_use, name)
                lazy_params[name] = param.default
                param_processors[name] = param.default

        @wraps(func)
        def wrapper(*args, **kwargs):
            # First, bind positional arguments to their parameter names
            bound_args = sig.bind_partial(*args, **kwargs)

            # Create final kwargs dict starting with positional args
            final_kwargs = {}
            consumed_positional_names = []
            consumed_first_config_param_positional = False

            # Handle the 'self' parameter for methods
            skip_first = inspect.ismethod(func) or (func.__name__ == '__init__' and 'self' in sig.parameters)
            if skip_first and args:
                first_param_name = list(sig.parameters.keys())[0]
                final_kwargs[first_param_name] = args[0]
                args = args[1:]

            # Process positional arguments first
            positional_params = [p for p in sig.parameters.values()
                                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]

            used_positional = 0
            for i, param in enumerate(positional_params[1:] if skip_first else positional_params):
                if i >= len(args):
                    break
                # Decide whether to consume this positional into a named parameter
                take_as_named = True
                if isinstance(param.default, CONFIG_PARAM):
                    # Policy: allow only the FIRST CONFIG_PARAM to be set positionally,
                    # push remaining positionals into *args instead.
                    if consumed_first_config_param_positional:
                        take_as_named = False
                    else:
                        consumed_first_config_param_positional = True
                if not take_as_named:
                    break
                # Use the positional argument value
                value = args[i]
                if param.name in param_processors:
                    processor = param_processors[param.name]
                    expected_type = hints.get(param.name)
                    value = processor.convert_to_enum_if_needed(value, expected_type)
                    value = processor.transform_value(value)
                    if not processor.validate(value):
                        raise ValueError(f"Validation failed for parameter {param.name}")
                    if update_config_with_args:
                        config_to_update = param_configs[param.name]
                        setattr(config_to_update, param.name, value)
                final_kwargs[param.name] = value
                consumed_positional_names.append(param.name)
                used_positional += 1

            # Then process keyword arguments and remaining parameters
            for name, param in sig.parameters.items():
                if name in final_kwargs:  # Skip already processed positional args
                    continue

                if name in kwargs:
                    value = kwargs[name]
                    if name in param_processors:
                        processor = param_processors[name]
                        expected_type = hints.get(param.name)
                        value = processor.convert_to_enum_if_needed(value, expected_type)
                        value = processor.transform_value(value)
                        if not processor.validate(value):
                            raise ValueError(f"Validation failed for parameter {name}")
                        if update_config_with_args:
                            config_to_update = param_configs[name]
                            setattr(config_to_update, name, value)
                    final_kwargs[name] = value
                elif name in lazy_params:
                    final_kwargs[name] = lazy_params[name]()
                elif param.default is not param.empty:
                    final_kwargs[name] = param.default

            for k, v in kwargs.items():
                if k not in final_kwargs and k not in sig.parameters:
                    final_kwargs[k] = v

            # --- Build final call safely ---
            call_args = []
            if 'self' in final_kwargs:
                call_args.append(final_kwargs.pop('self'))
            # Append consumed named positionals in order and remove from kwargs
            for pname in consumed_positional_names:
                call_args.append(final_kwargs.pop(pname))
            # Fill any remaining POSITIONAL_ONLY / POSITIONAL_OR_KEYWORD params positionally
            # so that following extras go into *args (instead of binding to those params).
            for p in (positional_params[1:] if skip_first else positional_params):
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                    if p.name in consumed_positional_names:
                        continue
                    if p.name in final_kwargs:
                        call_args.append(final_kwargs.pop(p.name))
                    # else: if it's not in final_kwargs, it either had no default
                    # (already validated upstream) or will be provided by kwargs later.

            # Any leftover original positionals go to *args if accepted
            extra_positional = args[used_positional:]
            has_var_positional = any(
                p.kind is inspect.Parameter.VAR_POSITIONAL
                for p in sig.parameters.values()
            )
            if has_var_positional:
                call_args.extend(extra_positional)
            # Call with reconstructed args/kwargs
            return func(*call_args, **final_kwargs)

        # Update the signature
        new_params = []
        argname_to_configname = {}
        for param in sig.parameters.values():
            if isinstance(param.default, CONFIG_PARAM):
                config_to_use = param_configs[param.name]
                argname_to_configname[param.name] = param.default
                new_default = getattr(config_to_use, param.name)
                new_params.append(param.replace(default=new_default))
            else:
                new_params.append(param)

        wrapper.__signature__ = sig.replace(parameters=new_params)
        wrapper._argname_to_configname = argname_to_configname #This is used to keep track of the parameters that had configs as defaults
        wrapper._inject_default_config = default_config  # Store config reference for inject_docs_from_config_params
        return wrapper

    return decorator


def test_config_injection():
    """Original test suite for basic config injection functionality."""

    @dataclass
    class ModelConfig:
        hidden_size: int = 256
        num_layers: int = 2
        dropout: float = 0.1

    config = ModelConfig()

    def positive_int(x: int) -> bool:
        return isinstance(x, int) and x > 0

    def float_in_range(x: float) -> bool:
        return isinstance(x, float) and 0 <= x <= 1

    hidden_size_param = CONFIG_PARAM(
        validator=positive_int,
        doc="Number of hidden units per layer"
    )
    dropout_param = CONFIG_PARAM(
        validator=float_in_range,
        transform=float,
        doc="Dropout probability"
    )

    class NeuralNetwork:
        @inject_defaults_from_config(config, update_config_with_args=True)
        def __init__(
                self,
                input_size: int,
                output_size: int,
                hidden_size: int = hidden_size_param,
                num_layers: int = CONFIG_PARAM(
                    validator=positive_int,
                    doc="Number of hidden layers"
                ),
                dropout: float = dropout_param
        ):
            self.input_size = input_size
            self.output_size = output_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout

    # Original tests
    assert hidden_size_param() == 256, "Direct access to hidden_size failed"
    model = NeuralNetwork(input_size=10, output_size=2, hidden_size=512)
    assert hidden_size_param() == 512, "CONFIG_PARAM not tracking config updates"
    model = NeuralNetwork(input_size=10, output_size=2, dropout="0.5")
    assert dropout_param() == 0.5, "CONFIG_PARAM not tracking transformed value"
    config.hidden_size = 1024
    assert hidden_size_param() == 1024, "CONFIG_PARAM not tracking direct config changes"


def test_multi_config_injection():
    """Test suite for multi-config functionality."""

    @dataclass
    class TrainingConfig:
        batch_size: int = 32
        learning_rate: float = 0.001

    @dataclass
    class ModelConfig:
        hidden_size: int = 256
        num_layers: int = 2

    training_config = TrainingConfig()
    model_config = ModelConfig()

    def positive_int(x: int) -> bool:
        return isinstance(x, int) and x > 0

    def positive_float(x: float) -> bool:
        return isinstance(x, float) and x > 0

    class NeuralNetwork:
        @inject_defaults_from_config(model_config, update_config_with_args=True)
        def __init__(
                self,
                input_size: int,
                output_size: int,
                batch_size: int = CONFIG_PARAM(
                    validator=positive_int,
                    config=training_config
                ),
                hidden_size: int = CONFIG_PARAM(
                    validator=positive_int
                ),
                learning_rate: float = CONFIG_PARAM(
                    validator=positive_float,
                    config=training_config
                )
        ):
            self.input_size = input_size
            self.output_size = output_size
            self.batch_size = batch_size
            self.hidden_size = hidden_size
            self.learning_rate = learning_rate

    # Test 1: Default values from different configs
    model = NeuralNetwork(input_size=10, output_size=2)
    assert model.batch_size == 32, "Wrong default from training config"
    assert model.hidden_size == 256, "Wrong default from model config"
    assert model.learning_rate == 0.001, "Wrong default from training config"

    # Test 2: Updating specific configs
    model = NeuralNetwork(
        input_size=10,
        output_size=2,
        batch_size=64,
        hidden_size=512,
        learning_rate=0.01
    )
    assert training_config.batch_size == 64, "Training config not updated"
    assert model_config.hidden_size == 512, "Model config not updated"
    assert training_config.learning_rate == 0.01, "Training config not updated"

    # Test 3: Validation across configs
    try:
        model = NeuralNetwork(input_size=10, output_size=2, batch_size=-1)
        assert False, "Should have raised validation error"
    except ValueError:
        pass

    try:
        model = NeuralNetwork(input_size=10, output_size=2, learning_rate=-0.1)
        assert False, "Should have raised validation error"
    except ValueError:
        pass

def test_args_passthrough():
    """Test that extra *args are forwarded correctly through the decorator."""

    from dataclasses import dataclass

    @dataclass
    class DummyConfig:
        a: int = 1
        b: int = 2

    cfg = DummyConfig()

    class Target:
        @inject_defaults_from_config(cfg, update_config_with_args=True)
        def __init__(self,
                     a: int = CONFIG_PARAM(),
                     b: int = CONFIG_PARAM(),
                     *args):
            self.a = a
            self.b = b
            self.extra_args = args

    t = Target(5, "pos1", "pos2")
    assert t.a == 5, f"Expected a==5, got {t.a}"
    assert t.b == 2, f"Expected default b==2, got {t.b}"
    assert t.extra_args == ("pos1", "pos2"), f"Args not forwarded: {t.extra_args}"
    print(" test_args_passthrough passed.")


def test_kwargs_passthrough():
    """Test that extra **kwargs are forwarded correctly through the decorator."""

    from dataclasses import dataclass

    @dataclass
    class DummyConfig:
        a: int = 1

    cfg = DummyConfig()

    class Target:
        @inject_defaults_from_config(cfg, update_config_with_args=True)
        def __init__(self, a: int = CONFIG_PARAM(), **kwargs):
            self.a = a
            self.extra_kwargs = kwargs

    t = Target(7, foo="bar", z=42)
    assert t.a == 7, f"Expected a==7, got {t.a}"
    assert t.extra_kwargs == {"foo": "bar", "z": 42}, f"Kwargs not forwarded: {t.extra_kwargs}"
    print(" test_kwargs_passthrough passed.")

if __name__ == "__main__":
    test_config_injection()
    test_multi_config_injection()
    test_args_passthrough()
    test_kwargs_passthrough()
    print("All tests passed successfully!")