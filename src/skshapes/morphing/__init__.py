"""
The :mod:`skshapes.morphing` module gathers deformation models.
"""

from .basemodel import BaseModel
from .extrinsic_deformation import ExtrinsicDeformation
from .intrinsic_deformation import IntrinsicDeformation
from .rigid_motion import RigidMotion
from .validation import validate_polydata_morphing_model
