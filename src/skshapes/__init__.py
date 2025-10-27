"""Scikit-Shapes: shape analysis in python."""

from ._data import Circle, Image, Mask, PolyData, SparseImage, Sphere
from .applications import *
from .convolutions import *
from .doc import *
from .features import *
from .globals import (
    float_dtype,
    int_dtype,
    taichi_available,
)
from .images import *
from .input_validation import *
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
    "Image",
    "Mask",
    "PolyData",
    "SparseImage",
    "Sphere",
    "applications",
    "convolutions",
    "features",
    "images",
    "input_validation",
    "loss",
    "morphing",
    "multiscaling",
    "optimization",
    "tasks",
    "triangle_mesh",
    "types",
]
