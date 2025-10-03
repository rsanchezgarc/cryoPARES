# Building Documentation

This guide explains how to build the CryoPARES documentation locally.

## Prerequisites

Python 3.10+ with pip installed.

## Installation

Install Sphinx and required extensions:

```bash
cd docs
pip install -r requirements.txt
```

## Building HTML Documentation

```bash
# From the docs/ directory
make html
```

The generated documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser.

## Alternative Build Commands

```bash
# Clean previous builds
make clean

# Build and show warnings
sphinx-build -b html . _build/html -W

# Build with verbose output
sphinx-build -b html . _build/html -v

# Live reload during development (requires sphinx-autobuild)
pip install sphinx-autobuild
sphinx-autobuild . _build/html
```

## Checking Documentation

To check for broken links and other issues:

```bash
# Check for broken internal links
make linkcheck

# Build with warnings as errors
sphinx-build -b html . _build/html -W --keep-going
```

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── requirements.txt     # Documentation dependencies
├── api/                 # Auto-generated API docs
│   ├── index.rst
│   ├── training.rst
│   ├── inference.rst
│   └── ...
├── training_guide.md    # User guides (Markdown)
├── configuration_guide.md
├── troubleshooting.md
└── cli.md
```

## Updating API Documentation

The API documentation is **auto-generated** from Python docstrings. To update:

1. Modify docstrings in the source code
2. Rebuild documentation: `make html`

No manual editing of API docs is needed!

## Docstring Format

CryoPARES supports both **Google** and **Sphinx** style docstrings via the Napoleon extension:

### Google Style (Recommended)

```python
def train(symmetry: str, batch_size: int = 32):
    """
    Train a CryoPARES model.

    Args:
        symmetry: Point group symmetry (e.g., C1, D7)
        batch_size: Number of particles per batch

    Returns:
        Path to saved checkpoint
    """
```

### Sphinx Style

```python
def train(symmetry: str, batch_size: int = 32):
    """
    Train a CryoPARES model.

    :param symmetry: Point group symmetry (e.g., C1, D7)
    :param batch_size: Number of particles per batch
    :return: Path to saved checkpoint
    """
```

Both formats work and are automatically parsed by Sphinx.

## GitHub Pages Deployment

Documentation is automatically built and deployed to GitHub Pages when you push to the main branch:

1. GitHub Actions runs (`/.github/workflows/docs.yml`)
2. Sphinx builds the documentation
3. Result is deployed to: https://rsanchezgarc.github.io/cryoPARES/

## Troubleshooting

### Import Errors

If you get import errors when building:

```bash
# Install CryoPARES in the same environment
pip install -e ..
```

### Missing Modules

```bash
# Install all dependencies
pip install -r requirements.txt
pip install -r ../requirements.txt
```

### Warnings About Docstrings

Warnings like "document isn't included in any toctree" are normal and can be ignored.

To suppress them:
```bash
make html 2>/dev/null
```
