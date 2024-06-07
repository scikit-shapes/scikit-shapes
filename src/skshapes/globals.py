"""This modules contains global variables for the skshapes package"""

import os
from warnings import warn

import torch

# float dtype is float32 by default, and can be switch to float64 using the
# SKSHAPES_FLOAT_DTYPE environment variable (before importing skshapes).
admissile_float_dtypes = ["float32", "float64"]
float_dtype = os.environ.get("SKSHAPES_FLOAT_DTYPE", "float32")

if float_dtype in admissile_float_dtypes:
    float_dtype = getattr(torch, float_dtype)

else:
    warn(
        f"Unknown float dtype {float_dtype}. Possible values are"
        + "{admissile_float_dtypes}. Using float32 as default.",
        stacklevel=1,
    )
    float_dtype = torch.float32

# int dtype is int64
int_dtype = torch.int64

#

try:
    import taichi  # noqa: F401

    taichi_available = True

except ImportError:
    taichi_available = False
