#!/usr/bin/env python3
"""
Update documentation files with auto-generated content.

This script reads files with AUTO_GENERATED markers and replaces the content
between START and END markers with freshly generated documentation.

Markers format:
    <!-- AUTO_GENERATED:marker_name:START -->
    ... content will be replaced ...
    <!-- AUTO_GENERATED:marker_name:END -->
"""

import sys
import re
from pathlib import Path
from typing import Dict, Callable

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import generate_cli_docs


# Mapping of marker names to generator functions
GENERATORS: Dict[str, Callable[[], str]] = {
    # README.md markers
    "train_parameters": lambda: generate_cli_docs.generate_train_docs("readme"),
    "inference_parameters": lambda: generate_cli_docs.generate_inference_docs("readme"),
    "projmatching_parameters": lambda: generate_cli_docs.generate_projmatching_docs("readme"),
    "reconstruct_parameters": lambda: generate_cli_docs.generate_reconstruct_docs("readme"),

    # docs/cli.md markers
    "train_cli": lambda: generate_cli_docs.generate_train_docs("cli"),
    "inference_cli": lambda: generate_cli_docs.generate_inference_docs("cli"),
    "projmatching_cli": lambda: generate_cli_docs.generate_projmatching_docs("cli"),
    "reconstruct_cli": lambda: generate_cli_docs.generate_reconstruct_docs("cli"),
}


def find_markers(content: str) -> Dict[str, tuple]:
    """
    Find all AUTO_GENERATED markers in content.

    Returns:
        Dict mapping marker_name to (start_pos, end_pos, start_line, end_line)
    """
    pattern = r'<!-- AUTO_GENERATED:(\w+):(START|END) -->'
    markers = {}

    for match in re.finditer(pattern, content):
        marker_name = match.group(1)
        marker_type = match.group(2)

        if marker_name not in markers:
            markers[marker_name] = {}

        markers[marker_name][marker_type] = match.end()

    # Validate that all markers have both START and END
    result = {}
    for name, positions in markers.items():
        if 'START' in positions and 'END' in positions:
            result[name] = (positions['START'], positions['END'])
        else:
            print(f"Warning: Marker {name} is missing START or END", file=sys.stderr)

    return result


def update_file_content(content: str, dry_run: bool = False) -> tuple[str, list]:
    """
    Update content by replacing auto-generated sections.

    Args:
        content: Original file content
        dry_run: If True, don't actually modify, just report what would change

    Returns:
        (updated_content, list_of_changes)
    """
    markers = find_markers(content)
    changes = []

    if not markers:
        return content, changes

    # Sort markers by position (reverse order so we can replace from end to start)
    sorted_markers = sorted(markers.items(), key=lambda x: x[1][0], reverse=True)

    updated_content = content

    for marker_name, (start_pos, end_pos) in sorted_markers:
        if marker_name not in GENERATORS:
            print(f"Warning: No generator found for marker '{marker_name}'", file=sys.stderr)
            continue

        # Generate new content
        try:
            new_content = GENERATORS[marker_name]()
        except Exception as e:
            print(f"Error generating content for '{marker_name}': {e}", file=sys.stderr)
            continue

        # Extract old content for comparison
        old_content = updated_content[start_pos:end_pos - len(f"<!-- AUTO_GENERATED:{marker_name}:END -->")]

        if old_content.strip() != new_content.strip():
            changes.append({
                'marker': marker_name,
                'old_lines': old_content.count('\n'),
                'new_lines': new_content.count('\n')
            })

            if not dry_run:
                # Replace content between markers
                # Add newlines for formatting
                replacement = f"\n{new_content}\n"
                updated_content = (
                    updated_content[:start_pos] +
                    replacement +
                    updated_content[end_pos - len(f"<!-- AUTO_GENERATED:{marker_name}:END -->"):]
                )

    return updated_content, changes


def update_file(file_path: Path, dry_run: bool = False, backup: bool = True) -> bool:
    """
    Update a documentation file with auto-generated content.

    Args:
        file_path: Path to file to update
        dry_run: If True, don't actually write changes
        backup: If True, create .bak file before updating

    Returns:
        True if file was changed, False otherwise
    """
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return False

    print(f"\nProcessing {file_path}...")

    content = file_path.read_text()
    updated_content, changes = update_file_content(content, dry_run=dry_run)

    if not changes:
        print(f"  No changes needed")
        return False

    print(f"  Found {len(changes)} section(s) to update:")
    for change in changes:
        print(f"    - {change['marker']}: {change['old_lines']} → {change['new_lines']} lines")

    if dry_run:
        print(f"  [DRY RUN] Would update file")
        return True

    # Create backup if requested
    if backup:
        backup_path = file_path.with_suffix(file_path.suffix + '.bak')
        backup_path.write_text(content)
        print(f"  Created backup: {backup_path}")

    # Write updated content
    file_path.write_text(updated_content)
    print(f"  ✓ Updated {file_path}")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Update documentation with auto-generated content")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be changed without actually modifying files")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create .bak files")
    parser.add_argument("files", nargs="*",
                       help="Specific files to update (default: all docs)")

    args = parser.parse_args()

    # Determine which files to update
    docs_dir = Path(__file__).parent
    repo_root = docs_dir.parent

    if args.files:
        files_to_update = [Path(f) for f in args.files]
    else:
        # Default: update README.md and docs/cli.md
        files_to_update = [
            repo_root / "README.md",
            docs_dir / "cli.md",
        ]

    # Update files
    any_changes = False
    for file_path in files_to_update:
        if update_file(file_path, dry_run=args.dry_run, backup=not args.no_backup):
            any_changes = True

    if args.dry_run:
        print("\n[DRY RUN] No files were actually modified")
        print("Run without --dry-run to apply changes")
    elif any_changes:
        print("\n✓ Documentation updated successfully")
    else:
        print("\n✓ All documentation is already up to date")

    return 0 if not args.dry_run or not any_changes else 1


if __name__ == "__main__":
    sys.exit(main())
