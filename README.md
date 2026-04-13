# FlowDetection

CNN-based image classification for flow detection from camera images. The core library lives in `src/FlowDetection` and is structured as an installable Python package.

## Requirements

- Python 3.12 (required by `tensorflow==2.16.1`; Python 3.13+ is not yet supported by this TensorFlow version)
- All dependencies are declared in [pyproject.toml](pyproject.toml) and installed automatically by `pip install -e .`

## Installation

The codebase is now able to be set up as an installable package, making imports more straightforward.

Development has always used virtual environments; earlier versions used a conda distribution, and later the developers switched to a Python venv.

### Option 1: venv (recommended)

Get Python 3.12 from [python.org](https://www.python.org/downloads/), then run the following from the repo root.

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Windows (verified):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

Some scripts require additional packages (e.g. `pandas`, `openpyxl`). Install the optional extras as needed:

```bash
pip install -e ".[scripts]"           # pandas + openpyxl for evaluation scripts
pip install -e ".[experimental]"      # scikeras + scikit-learn for hyperparameter tuning
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

## What the Model Classifies

The CNN classifies camera images of a flume into flow-condition categories. The training dataset uses the standard Keras subfolder-per-class layout (one subdirectory per label inside `flow_detection_labeled_camera-b/`), and `tf.keras.utils.image_dataset_from_directory` infers class names from the folder names automatically. The number of classes is determined at runtime from the dataset.

## Scripts

Ready-to-run scripts live in `src/`. Each one loads a YAML config with `load_config` and then calls into the `FlowDetection` package.

| Script | Purpose |
|--------|---------|
| [train_and_save.py](src/train_and_save.py) | Full training pipeline: load config, train, save `.keras` model |
| [concise_run.py](src/concise_run.py) | Load a saved model and evaluate on validation data |
| [evaluate_from_file.py](src/evaluate_from_file.py) | Load and evaluate a trained model |
| [evaluate_unlabeled.py](src/evaluate_unlabeled.py) | Predict on an unlabeled image dataset |
| [hyperparam_manual.py](src/hyperparam_manual.py) | Manual hyperparameter sweep |
| [hyperparam_scikeras.py](src/hyperparam_scikeras.py) | GridSearchCV-based hyperparameter tuning (experimental) |

Example:

```bash
python src/train_and_save.py
```

## Import Statements

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

A complete config file ([model001.yaml](src/model001.yaml)) looks like this:

```yaml
# for setting paths
computer: 2020laptop

# data setup
validation_split: 0.2
seed: 234

# hyperparameters
batch_size: 24
edge_size: 128
activation_function: relu

# training
epochs: 10
learning_rate: 1.0e-3
optimizer: adam

# experiment parameters
trial_label: model001
plot_history: True
save_model: True
verbose: True
```

## Filesystem Utilities

```python
from FlowDetection import script_local_path, ensure_dir
```

- `script_local_path(filename)` — resolves a path relative to the calling script's directory
- `ensure_dir(path)` — creates a directory (and any missing parents) and returns it as a `Path`

## Notes

- TensorFlow INFO-level startup messages are suppressed automatically when the package is imported.
- Apple Silicon: `tensorflow==2.16.1` installs via pip for CPU use. GPU acceleration requires additional setup.