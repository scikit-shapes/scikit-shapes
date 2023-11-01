"""
The :mod:`skshapes.morphing` module gathers deformation models.
"""

from .vector_field_deformation import VectorFieldDeformation
from .kernel_deformation import KernelDeformation
from .rigid_motion import RigidMotion

from .utils import EulerIntegrator
from .kernels import GaussianKernel
from .metrics import ElasticMetric

from typing import Union

Model = Union[VectorFieldDeformation, RigidMotion, KernelDeformation]
