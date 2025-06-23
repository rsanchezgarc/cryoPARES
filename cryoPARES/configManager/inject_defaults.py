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

            # Handle the 'self' parameter for methods
            skip_first = inspect.ismethod(func) or (func.__name__ == '__init__' and 'self' in sig.parameters)
            if skip_first and args:
                first_param_name = list(sig.parameters.keys())[0]
                final_kwargs[first_param_name] = args[0]
                args = args[1:]

            # Process positional arguments first
            positional_params = [p for p in sig.parameters.values()
                                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]

            for i, param in enumerate(positional_params[1:] if skip_first else positional_params):
                if i < len(args):
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

            return func(**final_kwargs)

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


if __name__ == "__main__":
    test_config_injection()
    test_multi_config_injection()
    print("All tests passed successfully!")