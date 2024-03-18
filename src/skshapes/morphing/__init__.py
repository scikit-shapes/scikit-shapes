"""
The :mod:`skshapes.morphing` module gathers deformation models.
"""

from .extrinsic_deformation import ExtrinsicDeformation
from .intrinsic_deformation import IntrinsicDeformation
from .rigid_motion import RigidMotion

Model = IntrinsicDeformation | RigidMotion | ExtrinsicDeformation
