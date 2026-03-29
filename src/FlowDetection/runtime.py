"""Runtime helpers for TensorFlow-related process configuration.

This module is intentionally safe to import before TensorFlow. It centralizes
process-level settings that need to be applied early, before TensorFlow's C++
runtime initializes.
"""

import os


def configure_tensorflow_environment() -> None:
    """Apply default TensorFlow runtime settings for this repository.

    TF_CPP_MIN_LOG_LEVEL=1 hides noisy INFO-level startup messages like the
    oneDNN notice while still allowing warnings and errors through.
    setdefault means a user can still override this outside the repo if they
    intentionally want different logging behavior.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


configure_tensorflow_environment()