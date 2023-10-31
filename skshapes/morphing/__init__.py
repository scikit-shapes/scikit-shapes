"""
The :mod:`skshapes.morphing` module gathers deformation models.
"""

from .vector_field import VectorFieldDeformation
from .kernel import KernelDeformation
from .rigid import RigidMotion

from typing import Union

Model = Union[VectorFieldDeformation, RigidMotion, KernelDeformation]
