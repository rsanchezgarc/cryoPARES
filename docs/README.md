# CryoPARES Documentation

Welcome to the CryoPARES documentation! This directory contains comprehensive guides for using CryoPARES effectively.

## Documentation Index

### Getting Started

- **[Main README](../README.md)** - Project overview, installation, and quick start examples

### Guides

- **[Training Guide](training_guide.md)** - Complete guide for training CryoPARES models
  - Training parameters and best practices
  - Monitoring with TensorBoard
  - Overfitting and underfitting detection and solutions
  - Data preprocessing guidelines
  - Advanced training options (fine-tuning, continuation, simulation)

- **[Configuration Guide](configuration_guide.md)** - Comprehensive configuration reference
  - Complete parameter documentation
  - Configuration system overview
  - Common configuration recipes
  - YAML config file usage

- **[Troubleshooting Guide](troubleshooting.md)** - Solutions to common issues
  - Installation problems
  - File system issues
  - Memory and performance problems
  - Training and inference issues
  - Data-related problems
  - GPU/CUDA issues

### Reference

- **[API Reference](https://rsanchezgarc.github.io/cryoPARES/api/)** - Auto-generated API documentation
  - Training API (Trainer class)
  - Inference API (SingleInferencer class)
  - Projection Matching API (ProjectionMatcher class)
  - Reconstruction API (Reconstructor class)
  - Data Management API (ParticlesDataset class)
  - Model API (PlModel class)
  - Utilities and helper functions

- **[CLI Reference](cli.md)** - Command-line interface documentation
  - `cryopares_train` - Train models
  - `cryopares_infer` - Run inference
  - `cryopares_reconstruct` - Reconstruct volumes
  - `cryopares_projmatching` - Projection matching
  - `compactify_checkpoint` - Package checkpoints

## Quick Links by Task

### I want to train a model
→ Start with [Training Guide](training_guide.md) and refer to [Configuration Guide](configuration_guide.md) for parameters

### I want to run inference
→ See the Inference section in [Main README](../README.md) and [API Reference](api_reference.md)

### I'm getting an error
→ Check [Troubleshooting Guide](troubleshooting.md)

### I need to understand a specific parameter
→ See [Configuration Guide](configuration_guide.md)

### I want to use CryoPARES programmatically
→ See [API Reference](https://rsanchezgarc.github.io/cryoPARES/api/)

### My model is overfitting/underfitting
→ See "Overfitting and Underfitting" section in [Training Guide](training_guide.md)

### Training is too slow
→ See "Performance Issues" in [Troubleshooting Guide](troubleshooting.md) and "Advanced Training Options" in [Training Guide](training_guide.md)

## Document Overview

| Document | Purpose | Target Audience |
|----------|---------|-----------------|
| Training Guide | Learn how to train models effectively | Users training models |
| Configuration Guide | Understand all configuration parameters | All users |
| Troubleshooting Guide | Fix common problems | Users encountering issues |
| API Reference (auto-generated) | Programmatic usage with type hints | Developers, advanced users |
| CLI Reference | Command-line tool usage | All users |

## Contributing to Documentation

If you find errors or have suggestions for improving the documentation, please:

1. Open an issue on [GitHub](https://github.com/rsanchezgarc/cryoPARES/issues)
2. Submit a pull request with your improvements
3. Contact the maintainers

## Additional Resources

- **Paper:** [Supervised Deep Learning for Efficient Cryo-EM Image Alignment in Drug Discovery](https://www.biorxiv.org/content/10.1101/2025.03.04.641536v2)
- **GitHub Repository:** https://github.com/rsanchezgarc/cryoPARES
- **Issue Tracker:** https://github.com/rsanchezgarc/cryoPARES/issues
