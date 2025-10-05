#!/usr/bin/env python3
"""
Generate CLI documentation from PARAM_DOCS and function signatures.

This script extracts parameter information from config classes and CLI functions,
then generates formatted markdown documentation that can be inserted into README.md
and docs/cli.md.
"""

import sys
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Add parent directory to path to import cryoPARES
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryoPARES.configs.train_config.train_config import Train_config
from cryoPARES.configs.datamanager_config.datamanager_config import DataManager_config
from cryoPARES.configs.datamanager_config.particlesDataset_config import ParticlesDataset_config
from cryoPARES.configs.inference_config.inference_config import Inference_config
from cryoPARES.configs.projmatching_config.projmatching_config import Projmatching_config
from cryoPARES.configs.reconstruct_config.reconstruct_config import Reconstruct_config
from cryoPARES.configManager.inject_defaults import CONFIG_PARAM


@dataclass
class ParamInfo:
    """Information about a CLI parameter."""
    name: str
    type_str: str
    default: Any
    required: bool
    description: str
    is_config_param: bool


def get_type_string(annotation) -> str:
    """Convert type annotation to readable string."""
    if annotation == inspect.Parameter.empty:
        return "Any"

    # Handle string representations
    type_str = str(annotation)

    # Clean up common patterns
    type_str = type_str.replace("typing.", "")
    type_str = type_str.replace("<class '", "").replace("'>", "")

    # Handle Literal types - remove Literal wrapper but keep the content in brackets
    if "Literal[" in type_str:
        type_str = type_str.replace("Literal[", "").rstrip("]")

    return type_str


def collect_param_docs(configs: List[Any]) -> Dict[str, str]:
    """Collect all PARAM_DOCS from multiple config classes."""
    all_docs = {}
    for config in configs:
        if hasattr(config, 'PARAM_DOCS'):
            all_docs.update(config.PARAM_DOCS)
    return all_docs


def extract_function_params(func, param_docs: Dict[str, str]) -> List[ParamInfo]:
    """
    Extract parameter information from a function.

    Args:
        func: Function to extract parameters from
        param_docs: Dictionary of parameter documentation

    Returns:
        List of ParamInfo objects
    """
    sig = inspect.signature(func)
    params = []

    for name, param in sig.parameters.items():
        if name == 'self':
            continue

        # Determine if required
        required = param.default == inspect.Parameter.empty

        # Check if it's a CONFIG_PARAM
        is_config_param = isinstance(param.default, CONFIG_PARAM)

        # Get default value
        if required:
            default_val = None
        elif is_config_param:
            # For CONFIG_PARAM, we need to get the actual default from the config
            default_val = "See config"
        else:
            default_val = param.default

        # Get documentation
        description = param_docs.get(name, "No description available")

        # Get type string
        type_str = get_type_string(param.annotation)

        params.append(ParamInfo(
            name=name,
            type_str=type_str,
            default=default_val,
            required=required,
            description=description,
            is_config_param=is_config_param
        ))

    return params


def generate_parameter_table_readme(params: List[ParamInfo], style: str = "bullets") -> str:
    """
    Generate parameter documentation in README.md style.

    Args:
        params: List of ParamInfo objects
        style: "bullets" or "table"
    """
    lines = []

    # Separate required and optional
    required_params = [p for p in params if p.required]
    optional_params = [p for p in params if not p.required]

    if style == "bullets":
        if required_params:
            lines.append("**Required Parameters:**\n")
            for p in required_params:
                lines.append(f"*   `--{p.name}`: {p.description}\n")
            lines.append("")

        if optional_params:
            lines.append("**Optional Parameters:**\n")
            for p in optional_params:
                default_str = f" (Default: `{p.default}`)" if p.default is not None else ""
                lines.append(f"*   `--{p.name}`: {p.description}{default_str}\n")

    elif style == "table":
        lines.append("| Parameter | Type | Default | Description |")
        lines.append("|-----------|------|---------|-------------|")

        # Required first
        for p in required_params:
            lines.append(f"| `--{p.name}` | {p.type_str} | **Required** | {p.description} |")

        # Then optional
        for p in optional_params:
            default_str = f"`{p.default}`" if p.default is not None else "None"
            lines.append(f"| `--{p.name}` | {p.type_str} | {default_str} | {p.description} |")

    return "\n".join(lines)


def generate_cli_section(
    command_name: str,
    func,
    param_docs: Dict[str, str],
    description: str = ""
) -> str:
    """
    Generate a complete CLI section for docs/cli.md.

    Args:
        command_name: Name of the CLI command (e.g., "cryopares_train")
        func: Function to document
        param_docs: Dictionary of parameter documentation
        description: Brief description of what the command does
    """
    params = extract_function_params(func, param_docs)

    lines = []
    lines.append(f"## `{command_name}`\n")

    if description:
        lines.append(f"{description}\n")

    lines.append("### Usage\n")
    lines.append("```bash")
    lines.append(f"{command_name} [OPTIONS]")
    lines.append("```\n")

    # Generate parameter table
    lines.append("### Parameters\n")
    lines.append(generate_parameter_table_readme(params, style="table"))

    return "\n".join(lines)


def generate_train_docs(output_format: str = "readme") -> str:
    """Generate documentation for cryopares_train."""
    from cryoPARES.train.train import Trainer

    # Collect docs from all relevant configs
    param_docs = collect_param_docs([
        Train_config,
        DataManager_config,
        ParticlesDataset_config
    ])

    params = extract_function_params(Trainer.__init__, param_docs)

    if output_format == "readme":
        return generate_parameter_table_readme(params, style="bullets")
    elif output_format == "cli":
        return generate_cli_section(
            "cryopares_train",
            Trainer.__init__,
            param_docs,
            "Train a CryoPARES model on pre-aligned particle data."
        )

    return ""


def generate_inference_docs(output_format: str = "readme") -> str:
    """Generate documentation for cryopares_infer."""
    from cryoPARES.inference.infer import distributed_inference

    param_docs = collect_param_docs([
        Inference_config,
        Projmatching_config,
        DataManager_config
    ])

    params = extract_function_params(distributed_inference, param_docs)

    if output_format == "readme":
        return generate_parameter_table_readme(params, style="bullets")
    elif output_format == "cli":
        return generate_cli_section(
            "cryopares_infer",
            distributed_inference,
            param_docs,
            "Run inference on new particles using a trained model."
        )

    return ""


def generate_projmatching_docs(output_format: str = "readme") -> str:
    """Generate documentation for cryopares_projmatching."""
    from cryoPARES.projmatching.projmatching import projmatching_starfile

    param_docs = collect_param_docs([Projmatching_config])

    params = extract_function_params(projmatching_starfile, param_docs)

    if output_format == "readme":
        return generate_parameter_table_readme(params, style="bullets")
    elif output_format == "cli":
        return generate_cli_section(
            "cryopares_projmatching",
            projmatching_starfile,
            param_docs,
            "Align particles to a reference volume using projection matching."
        )

    return ""


def generate_reconstruct_docs(output_format: str = "readme") -> str:
    """Generate documentation for cryopares_reconstruct."""
    from cryoPARES.reconstruction.reconstruct import reconstruct_starfile

    param_docs = collect_param_docs([Reconstruct_config])

    params = extract_function_params(reconstruct_starfile, param_docs)

    if output_format == "readme":
        return generate_parameter_table_readme(params, style="bullets")
    elif output_format == "cli":
        return generate_cli_section(
            "cryopares_reconstruct",
            reconstruct_starfile,
            param_docs,
            "Reconstruct a 3D volume from particles with known poses."
        )

    return ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CLI documentation")
    parser.add_argument("--module", choices=["train", "inference", "projmatching", "reconstruct", "all"],
                       default="all", help="Which module to generate docs for")
    parser.add_argument("--format", choices=["readme", "cli"],
                       default="readme", help="Output format")
    parser.add_argument("--output", default="stdout", help="Output file (or 'stdout')")

    args = parser.parse_args()

    generators = {
        "train": generate_train_docs,
        "inference": generate_inference_docs,
        "projmatching": generate_projmatching_docs,
        "reconstruct": generate_reconstruct_docs,
    }

    output_lines = []

    if args.module == "all":
        for name, gen_func in generators.items():
            output_lines.append(f"# {name.upper()}\n")
            output_lines.append(gen_func(args.format))
            output_lines.append("\n" + "="*80 + "\n")
    else:
        output_lines.append(generators[args.module](args.format))

    output = "\n".join(output_lines)

    if args.output == "stdout":
        print(output)
    else:
        Path(args.output).write_text(output)
        print(f"Documentation written to {args.output}")
