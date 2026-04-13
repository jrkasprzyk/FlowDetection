# Smoke test: verifies that the FlowDetection package installs correctly and
# that the config loader resolves all required keys and path types.
# Run this after a fresh install or environment change to catch import/config errors early.

from pathlib import Path

import FlowDetection
from FlowDetection import load_config


def main():
    package_dir = Path(FlowDetection.__file__).resolve().parent
    print(f"Package import OK: {package_dir}")

    config = load_config("config_test_config.yaml")

    # Ensure every key that downstream code depends on is present in the loaded config.
    required_keys = {
        "computer",
        "validation_split",
        "batch_size",
        "epochs",
        "trial_label",
        "supervisor_path",
        "output_path",
    }
    missing_keys = sorted(required_keys - set(config))
    if missing_keys:
        raise KeyError(f"Smoke test failed: missing config keys {missing_keys}")

    if not isinstance(config["supervisor_path"], Path):
        raise TypeError("Smoke test failed: supervisor_path is not a Path object")
    if not isinstance(config["output_path"], Path):
        raise TypeError("Smoke test failed: output_path is not a Path object")

    print(f"Config load OK: computer={config['computer']}")
    print(f"Supervisor path: {config['supervisor_path']}")
    print(f"Output path: {config['output_path']}")
    print("Smoke test passed")


if __name__ == "__main__":
    main()