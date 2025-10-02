#!/usr/bin/env python
"""
Compactify a CryoPARES checkpoint directory by removing unnecessary files
and packaging it into a single ZIP file for easy distribution and inference.

This script creates a minimal checkpoint archive containing only the files
required for inference, significantly reducing storage and transfer size.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import argparse
import yaml


def validate_checkpoint_dir(checkpoint_dir: Path) -> Tuple[bool, List[str]]:
    """
    Validate that the checkpoint directory contains all required files.

    Args:
        checkpoint_dir: Path to the checkpoint directory (version_X)

    Returns:
        Tuple of (is_valid, list_of_missing_files)
    """
    missing_files = []

    # Check for half directories
    has_half1 = (checkpoint_dir / "half1").exists()
    has_half2 = (checkpoint_dir / "half2").exists()
    has_allParticles = (checkpoint_dir / "allParticles").exists()

    if not (has_half1 or has_half2 or has_allParticles):
        missing_files.append("No half directories found (half1, half2, or allParticles)")
        return False, missing_files

    # Check each half directory
    for half_dir_name in ["half1", "half2", "allParticles"]:
        half_dir = checkpoint_dir / half_dir_name
        if not half_dir.exists():
            continue

        # Check for model file (best_script.pt preferred, best.ckpt as fallback)
        best_script = half_dir / "checkpoints" / "best_script.pt"
        best_ckpt = half_dir / "checkpoints" / "best.ckpt"

        if not best_script.exists() and not best_ckpt.exists():
            missing_files.append(f"{half_dir_name}/checkpoints/best_script.pt or best.ckpt")

        # Check for hparams.yaml
        hparams = half_dir / "hparams.yaml"
        if not hparams.exists():
            missing_files.append(f"{half_dir_name}/hparams.yaml")

    # Check for config file
    config_files = list(checkpoint_dir.glob("configs_*.yml"))
    if not config_files:
        missing_files.append("configs_*.yml")

    return len(missing_files) == 0, missing_files


def get_required_files(checkpoint_dir: Path, include_reconstructions: bool = True) -> List[Tuple[Path, str]]:
    """
    Get list of required files for inference.

    Args:
        checkpoint_dir: Path to the checkpoint directory
        include_reconstructions: Whether to include reconstruction files

    Returns:
        List of (source_path, archive_path) tuples
    """
    files_to_include = []

    # Get the most recent config file
    config_files = sorted(checkpoint_dir.glob("configs_*.yml"))
    if config_files:
        most_recent_config = config_files[-1]
        files_to_include.append((most_recent_config, most_recent_config.name))

    # Process each half directory
    for half_dir_name in ["half1", "half2", "allParticles"]:
        half_dir = checkpoint_dir / half_dir_name
        if not half_dir.exists():
            continue

        checkpoints_dir = half_dir / "checkpoints"

        # Add model file (prefer best_script.pt)
        best_script = checkpoints_dir / "best_script.pt"
        best_ckpt = checkpoints_dir / "best.ckpt"

        if best_script.exists():
            archive_path = f"{half_dir_name}/checkpoints/best_script.pt"
            files_to_include.append((best_script, archive_path))
        elif best_ckpt.exists():
            archive_path = f"{half_dir_name}/checkpoints/best.ckpt"
            files_to_include.append((best_ckpt, archive_path))

        # Add directional normalizer (optional but recommended)
        normalizer = checkpoints_dir / "best_directional_normalizer.pt"
        if normalizer.exists():
            archive_path = f"{half_dir_name}/checkpoints/best_directional_normalizer.pt"
            files_to_include.append((normalizer, archive_path))

        # Add hparams.yaml
        hparams = half_dir / "hparams.yaml"
        if hparams.exists():
            archive_path = f"{half_dir_name}/hparams.yaml"
            files_to_include.append((hparams, archive_path))

        # Add reconstruction (optional)
        if include_reconstructions:
            reconstruction = half_dir / "reconstructions" / "0.mrc"
            if reconstruction.exists():
                archive_path = f"{half_dir_name}/reconstructions/0.mrc"
                files_to_include.append((reconstruction, archive_path))

    return files_to_include


def calculate_size_savings(checkpoint_dir: Path, files_to_include: List[Tuple[Path, str]]) -> Tuple[int, int, float]:
    """
    Calculate how much space will be saved.

    Returns:
        Tuple of (original_size_bytes, compact_size_bytes, savings_percentage)
    """
    # Calculate original size (all files in checkpoint_dir)
    original_size = 0
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.is_file():
                original_size += file_path.stat().st_size

    # Calculate compact size
    compact_size = sum(src_path.stat().st_size for src_path, _ in files_to_include)

    savings_percentage = ((original_size - compact_size) / original_size * 100) if original_size > 0 else 0

    return original_size, compact_size, savings_percentage


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def compactify_checkpoint(
    checkpoint_dir: str,
    output_path: Optional[str] = None,
    include_reconstructions: bool = True,
    compression_level: int = zipfile.ZIP_DEFLATED,
    verbose: bool = True
) -> str:
    """
    Compactify a checkpoint directory into a ZIP archive.

    Args:
        checkpoint_dir: Path to checkpoint directory (version_X)
        output_path: Output ZIP file path (default: checkpoint_dir_name_compact.zip)
        include_reconstructions: Include reconstruction files (can use external reference instead)
        compression_level: ZIP compression level (zipfile.ZIP_STORED or zipfile.ZIP_DEFLATED)
        verbose: Print progress information

    Returns:
        Path to created ZIP file

    Raises:
        ValueError: If checkpoint directory is invalid
        FileNotFoundError: If required files are missing
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Validate checkpoint
    if verbose:
        print(f"Validating checkpoint directory: {checkpoint_dir}")

    is_valid, missing_files = validate_checkpoint_dir(checkpoint_dir)
    if not is_valid:
        error_msg = "Invalid checkpoint directory. Missing required files:\n"
        error_msg += "\n".join(f"  - {f}" for f in missing_files)
        raise ValueError(error_msg)

    if verbose:
        print("✓ Checkpoint validation passed")

    # Get required files
    files_to_include = get_required_files(checkpoint_dir, include_reconstructions)

    if not files_to_include:
        raise ValueError("No files found to include in archive")

    # Calculate size savings
    original_size, compact_size, savings_pct = calculate_size_savings(checkpoint_dir, files_to_include)

    if verbose:
        print(f"\nSize analysis:")
        print(f"  Original size:  {format_size(original_size)}")
        print(f"  Compact size:   {format_size(compact_size)}")
        print(f"  Savings:        {format_size(original_size - compact_size)} ({savings_pct:.1f}%)")
        print(f"  Files included: {len(files_to_include)}")

    # Determine output path
    if output_path is None:
        output_path = checkpoint_dir.parent / f"{checkpoint_dir.name}_compact.zip"
    else:
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.zip')

    # Create ZIP archive
    if verbose:
        print(f"\nCreating ZIP archive: {output_path}")

    with zipfile.ZipFile(output_path, 'w', compression=compression_level) as zipf:
        for src_path, archive_path in files_to_include:
            if verbose:
                print(f"  Adding: {archive_path}")
            zipf.write(src_path, archive_path)

    # Verify ZIP was created successfully
    final_size = output_path.stat().st_size

    if verbose:
        print(f"\n✓ Checkpoint compactified successfully!")
        print(f"  Output file: {output_path}")
        print(f"  Final size:  {format_size(final_size)}")
        print(f"\nYou can now use this ZIP file for inference:")
        print(f"  cryopares_infer --checkpoint_dir {output_path} ...")

    return str(output_path)


def main():
    """Command-line interface for checkpoint compactification."""
    parser = argparse.ArgumentParser(
        description="Compactify a CryoPARES checkpoint for distribution and inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m cryoPARES.scripts.compactify_checkpoint --checkpoint_dir /path/to/version_0

  # Custom output name
  python -m cryoPARES.scripts.compactify_checkpoint \\
      --checkpoint_dir /path/to/version_0 \\
      --output_path my_model_compact.zip

  # Exclude reconstructions (smaller size, requires external reference for inference)
  python -m cryoPARES.scripts.compactify_checkpoint \\
      --checkpoint_dir /path/to/version_0 \\
      --no-reconstructions

  # No compression (faster but larger file)
  python -m cryoPARES.scripts.compactify_checkpoint \\
      --checkpoint_dir /path/to/version_0 \\
      --no-compression
        """
    )

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Path to checkpoint directory (e.g., /path/to/train_output/version_0)'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output ZIP file path. Default: <checkpoint_dir_name>_compact.zip'
    )

    parser.add_argument(
        '--no-reconstructions',
        action='store_true',
        help='Exclude reconstruction files (reduces size, but requires --reference_map during inference)'
    )

    parser.add_argument(
        '--no-compression',
        action='store_true',
        help='Store files without compression (faster but larger file size)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    try:
        compression = zipfile.ZIP_STORED if args.no_compression else zipfile.ZIP_DEFLATED

        output_path = compactify_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output_path,
            include_reconstructions=not args.no_reconstructions,
            compression_level=compression,
            verbose=not args.quiet
        )

        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
