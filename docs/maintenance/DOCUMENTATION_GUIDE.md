# Documentation Maintenance Guide

This guide explains how to maintain and update CryoPARES documentation using the centralized documentation system.

## Overview

CryoPARES uses a centralized documentation system where all parameter descriptions are defined once in `PARAM_DOCS` dictionaries within config classes. These descriptions automatically propagate to:

1. **CLI help text** (via `argParseFromDoc`)
2. **Python docstrings** (via `inject_docs_from_config_params` decorator)
3. **README.md** (auto-generated sections)
4. **docs/cli.md** (auto-generated sections)
5. **Sphinx documentation** (via autodoc)

## Quick Start

### Adding or Updating a Parameter Description

1. **Find the appropriate config class** where the parameter is defined or used:
   - Training parameters → `cryoPARES/configs/train_config/train_config.py`
   - Inference parameters → `cryoPARES/configs/inference_config/inference_config.py`
   - Projection matching → `cryoPARES/configs/projmatching_config/projmatching_config.py`
   - Reconstruction → `cryoPARES/configs/reconstruct_config/reconstruct_config.py`
   - Data manager → `cryoPARES/configs/datamanager_config/datamanager_config.py`
   - Particle dataset → `cryoPARES/configs/datamanager_config/particlesDataset_config.py`

2. **Update the `PARAM_DOCS` dictionary** in that config class:

```python
@dataclass
class MyConfig:
    """Configuration parameters."""

    PARAM_DOCS = {
        'my_parameter': 'Clear, concise description of what this parameter does',
        'another_param': 'Another description with usage guidance (Default values shown automatically)',
    }

    my_parameter: int = 42
    another_param: str = "default"
```

3. **Regenerate documentation**:

```bash
cd /home/sanchezg/cryo/myProjects/cryoPARES
python docs/update_docs.py
```

4. **Verify the changes**:

```bash
# Check CLI help
python -m cryoPARES.train.train --help

# Validate documentation is up-to-date
python docs/validate_docs.py
```

That's it! The changes will automatically appear in CLI help, docstrings, and documentation files.

## Documentation System Architecture

### 1. PARAM_DOCS Dictionary

The `PARAM_DOCS` dictionary in each config class is the **single source of truth** for parameter documentation:

```python
@dataclass
class Train_config:
    """Training configuration parameters."""

    PARAM_DOCS = {
        # Config parameters (appear in config files)
        'n_epochs': 'Number of training epochs. More epochs allow better convergence...',
        'batch_size': 'Number of particles per batch. Try to make it as large as possible...',

        # CLI-exposed parameters (not in config dataclass, but used in CLI functions)
        'symmetry': 'Point group symmetry of the molecule (e.g., C1, D7, I, O, T)',
        'particles_star_fname': 'Path(s) to RELION 3.1+ format .star file(s)...',
    }

    # Actual config fields
    n_epochs: int = 100
    batch_size: int = 64
```

### 2. Automatic CLI Help Generation

CLI functions use the `@inject_docs_from_config_params` decorator to automatically format their docstrings:

```python
from cryoPARES.configManager.inject_defaults import inject_docs_from_config_params, inject_defaults_from_config

@inject_docs_from_config_params
@inject_defaults_from_config(main_config.train, update_config_with_args=True)
def __init__(self, symmetry: str, particles_star_fname: List[str], ...):
    """
    Train a model on particle data.

    Args:
        symmetry: {symmetry}
        particles_star_fname: {particles_star_fname}
        n_epochs: {n_epochs}
        batch_size: {batch_size}
    """
```

The decorator replaces `{parameter_name}` placeholders with descriptions from `PARAM_DOCS`.

### 3. Auto-Generated Documentation Sections

README.md and docs/cli.md contain special markers that define auto-generated sections:

```markdown
**Key Arguments:**

<!-- AUTO_GENERATED:train_parameters:START -->
*   `--symmetry`: Point group symmetry of the molecule (e.g., C1, D7, I, O, T)
*   `--particles_star_fname`: Path(s) to RELION 3.1+ format .star file(s)...
<!-- AUTO_GENERATED:train_parameters:END -->

**Additional Information:**
This manual text is preserved and never overwritten.
```

Everything between `START` and `END` markers is replaced when you run `update_docs.py`.

## Documentation Scripts

### docs/generate_cli_docs.py

Generates formatted documentation from PARAM_DOCS and function signatures.

**Usage:**

```bash
# Generate all modules in README format
python docs/generate_cli_docs.py --module all --format readme

# Generate single module in CLI format
python docs/generate_cli_docs.py --module train --format cli

# Output to file instead of stdout
python docs/generate_cli_docs.py --module inference --format readme --output /tmp/output.md
```

**Options:**
- `--module {train,inference,projmatching,reconstruct,all}` - Which module to generate docs for
- `--format {readme,cli}` - Output format (bullet lists for README, tables for CLI docs)
- `--output FILE` - Output file (default: stdout)

### docs/update_docs.py

Updates documentation files in-place by replacing auto-generated sections.

**Usage:**

```bash
# Update all default files (README.md and docs/cli.md)
python docs/update_docs.py

# Dry run (show what would change without modifying files)
python docs/update_docs.py --dry-run

# Update specific files
python docs/update_docs.py README.md

# Don't create backup files
python docs/update_docs.py --no-backup
```

**Options:**
- `--dry-run` - Show what would be changed without actually modifying files
- `--no-backup` - Don't create `.bak` files before updating
- `files` - Specific files to update (default: README.md and docs/cli.md)

**Output:**
```
Processing /path/to/README.md...
  Found 4 section(s) to update:
    - train_parameters: 20 → 42 lines
    - inference_parameters: 22 → 50 lines
  Created backup: /path/to/README.md.bak
  ✓ Updated /path/to/README.md

✓ Documentation updated successfully
```

### docs/validate_docs.py

Validates that auto-generated documentation is up-to-date (for CI/CD).

**Usage:**

```bash
# Validate all default files
python docs/validate_docs.py

# Quiet mode (only errors and summary)
python docs/validate_docs.py --quiet

# Validate specific files
python docs/validate_docs.py README.md docs/cli.md
```

**Exit codes:**
- `0` - All documentation is up-to-date
- `1` - Documentation is out of date (needs regeneration)
- `2` - Script error (missing files, etc.)

**Example output:**
```
Validating /path/to/README.md...
  ✓ Up-to-date (4 sections)
Validating /path/to/docs/cli.md...
  ✗ Out of date - 2 section(s) need updating:
    - train_cli: 73 → 32 lines
    - inference_cli: 64 → 36 lines

✗ Documentation validation failed
  Run 'python docs/update_docs.py' to regenerate documentation
```

## Workflow Examples

### Example 1: Adding a New Parameter

You're adding a new parameter `--enable_awesome_feature` to training:

1. **Add to config class** (`cryoPARES/configs/train_config/train_config.py`):

```python
PARAM_DOCS = {
    # ... existing parameters ...
    'enable_awesome_feature': 'Enable the awesome new feature that improves training by 50%',
}

enable_awesome_feature: bool = False
```

2. **Add to CLI function** (`cryoPARES/train/train.py`):

```python
@inject_docs_from_config_params
@inject_defaults_from_config(main_config.train, update_config_with_args=True)
def __init__(self, symmetry: str, ..., enable_awesome_feature: bool = CONFIG_PARAM()):
    """
    Train a model on particle data.

    Args:
        ...
        enable_awesome_feature: {enable_awesome_feature}
    """
```

3. **Regenerate docs**:

```bash
python docs/update_docs.py
```

Done! The new parameter now appears in CLI help, README.md, and docs/cli.md.

### Example 2: Improving an Existing Description

You want to clarify the `batch_size` parameter description:

1. **Update PARAM_DOCS** (`cryoPARES/configs/train_config/train_config.py`):

```python
PARAM_DOCS = {
    'batch_size': 'Number of particles per batch. Larger batches improve training stability but require more GPU memory. We recommend batch sizes of at least 32 images. If you encounter OOM errors, reduce this value.',
}
```

2. **Regenerate docs**:

```bash
python docs/update_docs.py
```

The improved description automatically appears everywhere!

### Example 3: CI/CD Integration

Add this to your GitHub Actions / CI workflow:

```yaml
- name: Validate documentation is up-to-date
  run: |
    python docs/validate_docs.py

- name: Check for uncommitted doc changes
  run: |
    if [[ -n $(git status -s README.md docs/cli.md) ]]; then
      echo "Error: Documentation is out of sync. Run 'python docs/update_docs.py' and commit changes."
      exit 1
    fi
```

## Best Practices

### Writing Good Parameter Descriptions

1. **Be concise but complete**: One or two sentences is usually enough
2. **Explain the purpose, not just the type**: "Number of training epochs" vs "Maximum iterations for model convergence"
3. **Include guidance**: "Larger values improve accuracy but increase memory usage"
4. **Don't repeat the parameter name**: ❌ "batch_size is the batch size" ✅ "Number of particles processed together"
5. **Mention defaults only for key parameters**: The system automatically adds `(Default: X)` to optional parameters
6. **Use active voice**: "Enables GPU acceleration" vs "GPU acceleration is enabled"

### When to Update Which File

| Change Type | File to Update | Next Step |
|-------------|---------------|-----------|
| Add/modify parameter description | `configs/*/PARAM_DOCS` | Run `update_docs.py` |
| Add narrative documentation | `README.md` or `docs/*.md` | Edit directly (outside markers) |
| Add new auto-generated section | Add markers to docs, update `update_docs.py` | Run `update_docs.py` |
| Fix generated table formatting | `docs/generate_cli_docs.py` | Run `update_docs.py` |

### Marker Guidelines

When adding new auto-generated sections to documentation files:

1. **Use descriptive marker names**: `train_parameters`, not `section1`
2. **Keep markers outside manual content**: Don't put markers inside paragraphs
3. **Register markers** in `docs/update_docs.py`:

```python
GENERATORS: Dict[str, Callable[[], str]] = {
    "my_new_section": lambda: generate_my_docs("format"),
}
```

## Troubleshooting

### Problem: CLI help not updating after changing PARAM_DOCS

**Cause**: Python imports are cached

**Solution**:
```bash
# If using editable install (pip install -e .)
pip install -e . --force-reinstall --no-deps

# Or restart your Python interpreter
```

### Problem: update_docs.py says "No generator found for marker"

**Cause**: Marker name not registered in `GENERATORS` dict

**Solution**: Add the mapping in `docs/update_docs.py`:
```python
GENERATORS = {
    "your_marker_name": lambda: generate_function("format"),
}
```

### Problem: Documentation has wrong default values

**Cause**: CONFIG_PARAM() default doesn't match config value

**Solution**: Ensure the config class has the correct default:
```python
@dataclass
class MyConfig:
    my_param: int = 42  # ← This is the default shown in docs
```

### Problem: Validation fails but files look correct

**Cause**: Whitespace differences or line ending issues

**Solution**:
```bash
# Regenerate to normalize formatting
python docs/update_docs.py

# Validate again
python docs/validate_docs.py
```

## Advanced Topics

### Cross-Config Parameters

Some parameters are defined in one config but used in another CLI. For example, `num_dataworkers` from `DataManager_config` used in `train.py`:

```python
# In datamanager_config.py
PARAM_DOCS = {
    'num_dataworkers': 'Number of parallel data loading workers...',
}

# In train.py
@inject_docs_from_config_params
def __init__(self, num_dataworkers: int = CONFIG_PARAM()):
    """
    Args:
        num_dataworkers: {num_dataworkers}
    """
```

The `CONFIG_PARAM()` knows which config it comes from, so the decorator finds the right `PARAM_DOCS`.

### Custom Documentation Generators

To add a new generator for a different module:

1. **Add generator function** to `docs/generate_cli_docs.py`:

```python
def generate_my_module_docs(output_format: str = "readme") -> str:
    """Generate documentation for my_module."""
    from cryoPARES.my_module.my_cli import my_function

    param_docs = collect_param_docs([MyConfig])
    params = extract_function_params(my_function, param_docs)

    if output_format == "readme":
        return generate_parameter_table_readme(params, style="bullets")
    elif output_format == "cli":
        return generate_cli_section("my_command", my_function, param_docs, "Description")

    return ""
```

2. **Register in GENERATORS** in `docs/update_docs.py`:

```python
GENERATORS = {
    "my_module_params": lambda: generate_my_module_docs("readme"),
    "my_module_cli": lambda: generate_my_module_docs("cli"),
}
```

3. **Add markers** to your documentation files:

```markdown
<!-- AUTO_GENERATED:my_module_params:START -->
<!-- AUTO_GENERATED:my_module_params:END -->
```

## Further Reading

- **`cryoPARES/configManager/PARAM_DOCS_SYSTEM.md`** - Technical details of the PARAM_DOCS system
- **`cryoPARES/configManager/inject_defaults.py`** - Implementation of decorators
- **`docs/generate_cli_docs.py`** - Documentation generator source code
- **`docs/update_docs.py`** - Update script source code

## Summary Checklist

When updating parameter documentation:

- [ ] Update `PARAM_DOCS` in appropriate config class
- [ ] Run `python docs/update_docs.py`
- [ ] Verify with `python docs/validate_docs.py`
- [ ] Test CLI help: `python -m cryoPARES.{module} --help`
- [ ] Check Sphinx build: `cd docs && make html`
- [ ] Commit both config changes and regenerated docs
