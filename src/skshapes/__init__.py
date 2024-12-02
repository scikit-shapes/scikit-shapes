"""Scikit-Shapes: shape analysis in python."""

from .applications import *
from .convolutions import *
from .data import *
from .decimation import *
from .doc import *
from .features import *
from .globals import (
    float_dtype,
    int_dtype,
    taichi_available,
)
from .input_validation import *
from .loss import *
from .morphing import *
from .multiscaling import *
from .optimization import *
from .tasks import *
from .triangle_mesh import *
from .types import *

__all__ = [
    "applications",
    "convolutions",
    "data",
    "decimation",
    "features",
    "input_validation",
    "loss",
    "morphing",
    "multiscaling",
    "optimization",
    "tasks",
    "triangle_mesh",
    "types",
]
