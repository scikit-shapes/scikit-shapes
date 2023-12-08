"""
The :mod:`skshapes.morphing` module gathers deformation models.
"""

from .intrinsic_deformation import IntrinsicDeformation
from .extrinsic_deformation import ExtrinsicDeformation
from .rigid_motion import RigidMotion

from .utils import EulerIntegrator
from .kernels import GaussianKernel
from .metrics import ElasticMetric

from typing import Union

Model = Union[IntrinsicDeformation, RigidMotion, ExtrinsicDeformation]
