"""Scikit-Shapes: shape analysis in python."""

from ._data import Circle, PolyData, Sphere
from ._neighborhoods import *
from .applications import *
from .convolutions import *
from .doc import *
from .features import *
from .globals import (
    float_dtype,
    int_dtype,
    taichi_available,
)
from .input_validation import *
from .linear_operators import *
from .loss import *
from .morphing import *
from .multiscaling import *
from .optimization import *
from .tasks import *
from .triangle_mesh import *
from .types import *

__version__ = "0.3.1"

__all__ = [
    "PolyData",
    "Sphere",
    "Circle",
    "Neighborhoods",
    "Spectrum",
    "types",
    "features",
    "convolutions",
    "applications",
    "triangle_mesh",
    "loss",
    "morphing",
    "linear_operators",
    "neighborhoods",
    "optimization",
    "multiscaling",
    "tasks",
    "input_validation",
]
