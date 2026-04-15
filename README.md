# FlowDetection

CNN-based image classification for flow detection from camera images. The core library lives in `src/FlowDetection` and is structured as an installable Python package.

## Requirements

- Python 3.12 (required by `tensorflow==2.16.1`; Python 3.13+ is not yet supported by this TensorFlow version)
- All dependencies are declared in [pyproject.toml](pyproject.toml) and can be installed with `pip install -e .`

## Installation

The codebase is now able to be set up as an installable package, making imports more straightforward.

Development has always used virtual environments; earlier versions used a conda distribution, and later the developers switched to a Python venv.

### Option 1: venv (recommended)


Install Python 3.12:

macOS / Linux:

1. (Recommended) Install with Homebrew:
```bash
brew install python@3.12
```
Or, download the installer from [python.org](https://www.python.org/downloads/).
2. Verify installation:
```bash
python3.12 --version
```


Windows:

1. (Recommended) Install with winget (Windows Package Manager):
```powershell
winget install --id Python.Python.3.12
```
If winget is not available, download the Python 3.12 installer from [python.org](https://www.python.org/downloads/).
2. If using the installer, ensure you check "Add Python to PATH" during setup.
3. Verify installation:
```powershell
py -3.12 --version
```

Then run the following from the repo root to install the package. The `-e` flag denotes the package as editable, which is helpful during development stage.


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
python.exe -m pip install --upgrade pip
pip install -e .
```
These install instructions have been verified on both Windows and macOS.

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

> **Note:**The environment_minimal.yaml file is not up-to-date and the conda/mamba instructions have not been verified in recent versions of the codebase.

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

Some scripts use a YAML configuration file to make experiment setup easier. The `load_config` file resolves all of the keys available in the YAML schema. However, not all scripts in the repo have made use of the configuration file.

The `computer` key in the YAML selects machine-specific data and output paths defined in [src/FlowDetection/config.py](src/FlowDetection/config.py). The `load_config` function was originally designed to resolve filepaths seamlessly:

```python
from FlowDetection import load_config

config = load_config("model001.yaml")
```

> **Note:** The `computer` key and the machine mapping in `config.py` are legacy features and may be removed in a future version. To avoid confusion and make configuration more flexible, it is recommended to set paths directly in your script using `set_custom_path`.

The safest way to interact with the codebase is to specify Paths explicitly and make use of the `set_custom_path` function. The `set_custom_path` function accepts one argument -- the path that you need to specify for a particular file -- and turns it into a Path object (which resolves differences between Windows and macOS/Linux filepath types). The below example sets explicit paths for both the config file and the model file:

```python
from FlowDetection import load_config, set_custom_path

config = load_config(set_custom_path("/path/to/your/configs/model001.yaml"))

model = tf.keras.models.load_model(set_custom_path("path/to/your/models/model001.keras"))

```

See the example `/src/evaluate_unlabeled.py` for more.

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