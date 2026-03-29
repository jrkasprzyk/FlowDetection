# FlowDetection

CNN-based image classification for flow detection from camera images. The core library lives in `src/FlowDetection` and is structured as an installable Python package.

## Requirements

- Python 3.12 (required by `tensorflow==2.16.1`; Python 3.13+ is not yet supported by this TensorFlow version)
- All other dependencies are listed in [requirements.txt](requirements.txt)

## Installation

### Option 1: venv (recommended)

Get Python 3.12 from [python.org](https://www.python.org/downloads/), then run the following from the repo root.

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Option 2: Conda / Mamba

```bash
mamba env create -f environment_minimal.yaml
mamba activate flowdetectionminimal
pip install -e .
```

### Verify the installation

```bash
python src/smoke_test.py
```

A successful install prints the resolved config paths and `Smoke test passed`.

## Usage

```python
from FlowDetection import load_config, get_train_val_data, train_model
```

Both short-form and long-form imports work:

```python
from FlowDetection import load_config          # short form
from FlowDetection.config import load_config   # long form
```

## Configuration

Scripts load their settings from YAML files using `load_config`. Paths are resolved relative to the config file's location, so scripts work correctly regardless of the working directory.

```python
from FlowDetection import load_config

config = load_config("model001.yaml")
```

The `computer` key in the YAML selects machine-specific data and output paths defined in [src/FlowDetection/config.py](src/FlowDetection/config.py). Paths can also be specified explicitly in the YAML, which takes precedence over the machine mapping:

```yaml
computer: my-machine
supervisor_path: /data/flow_detection_labeled
output_path: /results/models
```

Relative paths in the YAML are resolved from the config file's own directory.

## Filesystem Utilities

```python
from FlowDetection import script_local_path, ensure_dir
```

- `script_local_path(filename)` — resolves a path relative to the calling script's directory
- `ensure_dir(path)` — creates a directory (and any missing parents) and returns it as a `Path`

## Notes

- TensorFlow INFO-level startup messages are suppressed automatically when the package is imported.
- Apple Silicon: `tensorflow==2.16.1` installs via pip for CPU use. GPU acceleration requires additional setup.