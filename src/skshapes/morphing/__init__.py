"""
The :mod:`skshapes.morphing` module gathers deformation models.
"""

from typing import Union

from .extrinsic_deformation import ExtrinsicDeformation
from .intrinsic_deformation import IntrinsicDeformation
from .kernels import GaussianKernel
from .metrics import (
    AsIsometricAsPossible,
    ShellEnergyMetric,
)
from .rigid_motion import RigidMotion
from .utils import EulerIntegrator

Model = IntrinsicDeformation | RigidMotion | ExtrinsicDeformation
