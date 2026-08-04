"""Locate Kaggle input files across mount layouts.

Kaggle mounts attached datasets and notebook outputs behind symlinked directories, and
newer sessions group them under `/kaggle/input/datasets` and `/kaggle/input/notebooks`
instead of listing dataset slugs directly. `Path.rglob` does not descend into symlinked
directories, so content discovery must walk the tree explicitly.

Both Kaggle notebooks need these helpers before the project is installed, so their
bootstrap cells carry a verbatim copy of the two functions below; the notebook test keeps
the copies identical to this module.
"""

from __future__ import annotations

import os
from pathlib import Path


def iter_input_paths(
    relative: str, root: Path = Path("/kaggle/input"), max_depth: int = 12
) -> list[Path]:
    """Return every existing `<directory>/relative` under `root`, following symlinked mounts."""
    matches: list[Path] = []
    visited: set[str] = set()
    for parent, directories, _ in os.walk(root, followlinks=True):
        real = os.path.realpath(parent)
        if real in visited:
            directories.clear()
            continue
        visited.add(real)
        if len(Path(parent).parts) - len(root.parts) >= max_depth:
            directories.clear()
        candidate = Path(parent) / relative
        if candidate.exists():
            matches.append(candidate)
    return sorted(matches, key=str)


def describe_inputs(root: Path = Path("/kaggle/input"), max_depth: int = 5) -> str:
    """Return a compact inventory of mounted inputs so failures name what is actually attached.

    Kaggle spends three levels on `datasets/<owner>/<slug>` before any content, so the
    default depth has to reach past the mount point itself.
    """
    if not root.is_dir():
        return f"{root} does not exist"
    lines: list[str] = []
    visited: set[str] = set()
    for parent, directories, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(parent)
        if real in visited:
            directories.clear()
            continue
        visited.add(real)
        depth = len(Path(parent).parts) - len(root.parts)
        entries = sorted(filenames)[:4]
        if len(filenames) > 4:
            entries.append(f"+{len(filenames) - 4} more files")
        if depth >= max_depth:
            if directories:
                entries.append(f"+{len(directories)} more directories")
            directories.clear()
        directories.sort()
        lines.append(f"{'  ' * depth}{Path(parent).name or root}/ {entries}")
        if len(lines) >= 80:
            lines.append("... truncated")
            break
    return "\n".join(lines)
