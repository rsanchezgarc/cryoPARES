#!/usr/bin/env python3
"""
Validate that documentation is up-to-date with PARAM_DOCS.

This script checks that all AUTO_GENERATED sections match what would be
generated from the current PARAM_DOCS definitions. It's intended to be run
in CI to catch documentation drift.

Exit codes:
    0: All documentation is up-to-date
    1: Documentation is out of date (needs regeneration)
    2: Script error (missing files, etc.)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from docs.update_docs import update_file_content, find_markers


def validate_file(file_path: Path, verbose: bool = True) -> bool:
    """
    Validate that a file's auto-generated content is up-to-date.

    Args:
        file_path: Path to file to validate
        verbose: Print detailed information

    Returns:
        True if file is up-to-date, False otherwise
    """
    if not file_path.exists():
        if verbose:
            print(f"Error: File not found: {file_path}", file=sys.stderr)
        return False

    if verbose:
        print(f"Validating {file_path}...")

    content = file_path.read_text()

    # Check if file has markers
    markers = find_markers(content)
    if not markers:
        if verbose:
            print(f"  No AUTO_GENERATED markers found - skipping")
        return True

    # Generate what the content should be
    updated_content, changes = update_file_content(content, dry_run=True)

    if not changes:
        if verbose:
            print(f"  ✓ Up-to-date ({len(markers)} sections)")
        return True
    else:
        if verbose:
            print(f"  ✗ Out of date - {len(changes)} section(s) need updating:")
            for change in changes:
                print(f"    - {change['marker']}: {change['old_lines']} → {change['new_lines']} lines")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate that auto-generated documentation is up-to-date"
    )
    parser.add_argument("--quiet", action="store_true",
                       help="Only print errors and summary")
    parser.add_argument("files", nargs="*",
                       help="Specific files to validate (default: all docs)")

    args = parser.parse_args()

    # Determine which files to validate
    docs_dir = Path(__file__).parent
    repo_root = docs_dir.parent

    if args.files:
        files_to_validate = [Path(f) for f in args.files]
    else:
        # Default: validate README.md and docs/cli.md
        files_to_validate = [
            repo_root / "README.md",
            docs_dir / "cli.md",
        ]

    # Validate files
    all_valid = True
    files_checked = 0
    files_with_markers = 0

    for file_path in files_to_validate:
        is_valid = validate_file(file_path, verbose=not args.quiet)

        if not is_valid:
            all_valid = False
            files_with_markers += 1
        elif find_markers(file_path.read_text()):
            files_with_markers += 1

        files_checked += 1

    # Print summary
    if not args.quiet or not all_valid:
        print()
        if all_valid:
            print(f"✓ All documentation is up-to-date ({files_with_markers}/{files_checked} files with auto-generated content)")
        else:
            print(f"✗ Documentation validation failed", file=sys.stderr)
            print(f"  Run 'python docs/update_docs.py' to regenerate documentation", file=sys.stderr)

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
