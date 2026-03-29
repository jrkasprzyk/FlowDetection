"""Filesystem utilities for portable path handling.

These helpers make scripts work correctly regardless of which machine or working
directory they are run from, which matters when the same repo is used across
multiple computers (laptop, desktop, HPC, etc.).
"""

import inspect
from os import PathLike
from pathlib import Path


def script_local_path(filename, must_exist=True, caller_file=None):
    """Resolve a path relative to the calling script's directory.

    This means you can reference sibling files by name only, and the path will
    always resolve correctly no matter where Python is invoked from.

    Args:
        filename: File name or relative path from the script directory.
            Absolute paths are returned unchanged.
        must_exist: If True (default), raise FileNotFoundError when the
            resolved path does not exist.
        caller_file: Optional override for the calling script path.
            Defaults to the actual caller's __file__ via the call stack.

    Returns:
        Resolved absolute Path object.
    """
    if not isinstance(filename, (str, PathLike)):
        raise TypeError("Input 'filename' must be a path string or path-like object.")
    if isinstance(filename, str) and not filename.strip():
        raise ValueError("Input 'filename' cannot be an empty string.")
    if not isinstance(must_exist, bool):
        raise TypeError("Input 'must_exist' must be a boolean.")

    candidate = Path(filename)
    if candidate.is_absolute():
        resolved = candidate
    else:
        if caller_file is None:
            # inspect.stack()[1] is the frame of whoever called this function
            caller_file = inspect.stack()[1].filename
        script_dir = Path(caller_file).resolve().parent
        resolved = script_dir / candidate

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    return resolved


def ensure_dir(path):
    """Create a directory (and any missing parents) if it doesn't exist.

    Safe to call when the directory already exists — it will not raise an error.

    Args:
        path: Directory path to create (str or Path).

    Returns:
        The directory as a resolved Path object.
    """
    if not isinstance(path, (str, PathLike)):
        raise TypeError("Input 'path' must be a path string or path-like object.")
    if isinstance(path, str) and not path.strip():
        raise ValueError("Input 'path' cannot be an empty string.")

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)  # parents=True creates intermediate dirs
    return directory
