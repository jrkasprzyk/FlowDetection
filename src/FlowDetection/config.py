import inspect
from pathlib import Path
from yaml import safe_load

from FlowDetection.filesystem import script_local_path


# These mappings separate each machine's root folder from the shared repo/data
# suffixes. That avoids bugs where a joined path accidentally repeats a segment
# like "OneDrive - UCB-O365" twice.
SUPERVISOR_ROOTS = {
    "hpc": Path("/projects/joka0958/supervisor/"),
    "2020laptop": Path("C:/Users/josep/OneDrive - UCB-O365/"),
    "seecdesktop": Path("C:/Users/joka0958/OneDrive - UCB-O365/"),
    "CEAE-L-042": Path("C:/Users/joka0958/OneDrive - UCB-O365/"),
    "Marguerite": Path("/Users/malo1039/Library/CloudStorage/OneDrive-UCB-O365/"),
}

OUTPUT_ROOTS = {
    "hpc": Path("/projects/joka0958/MLoutput/"),
    "2020laptop": Path("C:/Users/josep/Documents/"),
    "seecdesktop": Path("//files.colorado.edu/CEAE/users/joka0958/Documents/"),
    "CEAE-L-042": Path("C:/"),
    "Marguerite": Path("/Users/malo1039/Documents/"),
}

SUPERVISOR_RELATIVE_PATH = Path("Datasets/flow_detection_labeled_camera-b/")
OUTPUT_RELATIVE_PATH = Path("GitHub/FlowDetection/models/")


def _require_machine_root(computer, roots, description):
    if computer not in roots:
        raise ValueError(
            f"Unknown computer name: {computer!r}. "
            f"Add a {description} entry for this machine in FlowDetection/config.py."
        )
    return roots[computer]


def _resolve_config_path(raw_path, config_dir, key_name):
    if not isinstance(raw_path, (str, Path)):
        raise TypeError(f"Config value {key_name!r} must be a path string or Path-like object.")

    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return config_dir / candidate


def set_supervisor_path(computer) -> Path:
    """Return the labeled-image dataset path for the given machine name.

    To add a new machine, add an elif branch with its name and path.
    """
    root = _require_machine_root(computer, SUPERVISOR_ROOTS, "supervisor path")
    if computer == "hpc":
        return root
    return root / SUPERVISOR_RELATIVE_PATH


def set_output_path(computer) -> Path:
    """Return the model-output directory path for the given machine name.

    To add a new machine, add an elif branch with its name and path.
    """
    root = _require_machine_root(computer, OUTPUT_ROOTS, "output path")
    if computer == "hpc":
        return root
    return root / OUTPUT_RELATIVE_PATH


def set_custom_path(raw_path) -> Path:
    """Wrap an arbitrary path string as a Path object."""
    return Path(raw_path)


def load_config(config_filename, caller_file=None):
    """Read a YAML config file and resolve machine-specific paths.

    If config_filename is a relative path it is resolved relative to the
    *calling script's directory*, not the working directory.  This means
    load_config("model001.yaml") works correctly no matter where Python
    is invoked from.

    Args:
        config_filename: Path to the YAML config file (str or Path).
            Absolute paths are used as-is.
        caller_file: Optional override for the calling script path;
            defaults to the actual caller's __file__ via the call stack.

    Returns:
        dict with all YAML keys plus 'supervisor_path' and 'output_path'
        resolved either from explicit YAML path overrides or from the
        machine named in the 'computer' key.
    """
    if caller_file is None:
        caller_file = inspect.stack()[1].filename  # __file__ of whoever called load_config

    resolved = script_local_path(config_filename, must_exist=True, caller_file=caller_file)
    config_dir = resolved.parent

    with open(resolved) as f:  # context manager ensures the file handle is always closed
        config = safe_load(f)

    if "supervisor_path" in config:
        config["supervisor_path"] = _resolve_config_path(
            config["supervisor_path"],
            config_dir,
            "supervisor_path",
        )
    else:
        if "computer" not in config:
            raise KeyError(
                "Config must define either 'supervisor_path' or 'computer'."
            )
        config["supervisor_path"] = set_supervisor_path(config["computer"])

    if "output_path" in config:
        config["output_path"] = _resolve_config_path(
            config["output_path"],
            config_dir,
            "output_path",
        )
    else:
        if "computer" not in config:
            raise KeyError(
                "Config must define either 'output_path' or 'computer'."
            )
        config["output_path"] = set_output_path(config["computer"])

    return config
