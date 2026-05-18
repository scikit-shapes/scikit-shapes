"""Scikit-Shapes: shape analysis in python."""

from ._data import Circle, Image, Mask, PolyData, SparseImage, Sphere
from ._features import *
from ._neighborhoods import *
from .applications import *
from .convolutions import *
from .doc import *
from .globals import (
    float_dtype,
    int_dtype,
    taichi_available,
)
from .images import *
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
    "Circle",
    "Circle",
    "Image",
    "Mask",
    "Neighborhoods",
    "PolyData",
    "SparseImage",
    "Spectrum",
    "Sphere",
    "applications",
    "convolutions",
    "features",
    "images",
    "input_validation",
    "linear_operators",
    "loss",
    "morphing",
    "multiscaling",
    "neighborhoods",
    "optimization",
    "tasks",
    "triangle_mesh",
    "types",
    "types",
]
