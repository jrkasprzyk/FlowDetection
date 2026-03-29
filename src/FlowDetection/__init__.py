"""FlowDetection — CNN-based flow detection from camera images.

Both short-form and long-form imports always work::

    from FlowDetection import load_config, train_model         # short form
    from FlowDetection.training import train_model             # long form
"""

# Import runtime configuration before any TensorFlow-using modules so startup
# logging and similar process-level settings are applied early enough.
from FlowDetection.runtime import configure_tensorflow_environment

configure_tensorflow_environment()

from FlowDetection.config import (
    set_supervisor_path,
    set_output_path,
    set_custom_path,
    load_config,
)
from FlowDetection.data import get_train_val_data
from FlowDetection.evaluation import (
    predict_one_image,
    predict_image_list,
    predict_unlabeled_image_list,
    create_confusion_matrix,
    evaluate_model,
    plot_history,
)
from FlowDetection.training import train_model
from FlowDetection.filesystem import script_local_path, ensure_dir

# __all__ controls what is exported when a user does `from FlowDetection import *`
# and documents the intended public API surface of the package.
__all__ = [
    # config
    "set_supervisor_path",
    "set_output_path",
    "set_custom_path",
    "load_config",
    # data
    "get_train_val_data",
    # evaluation
    "predict_one_image",
    "predict_image_list",
    "predict_unlabeled_image_list",
    "create_confusion_matrix",
    "evaluate_model",
    "plot_history",
    # training
    "train_model",
    # filesystem
    "script_local_path",
    "ensure_dir",
    # runtime
    "configure_tensorflow_environment",
]
