"""Types aliases and utility functions for scikit-shapes."""

from typing import Literal, NamedTuple

import numpy as np
import torch
from beartype import beartype
from beartype.typing import Annotated
from beartype.vale import Is
from jaxtyping import Float, Float32, Float64, Int, Int32, Int64

from .globals import float_dtype, int_dtype

# Type aliases
Number = int | float

correspondence = {
    torch.float32: Float32,
    torch.float64: Float64,
    torch.int64: Int64,
    torch.int32: Int32,
}


JaxFloat = correspondence[float_dtype]
JaxDouble = Float64
JaxInt = correspondence[int_dtype]

# Numpy array types
FloatArray = Float[np.ndarray, "..."]  # Any float format numpy array
IntArray = Int[np.ndarray, "..."]  # Any int format numpy array
NumericalArray = FloatArray | IntArray
Float1dArray = Float[np.ndarray, "_"]
Int1dArray = Int[np.ndarray, "_"]

# Numerical typestyp
FloatTensor = JaxFloat[
    torch.Tensor, "..."
]  # Only Float32 tensors are FloatTensors
IntTensor = JaxInt[torch.Tensor, "..."]  # Only Int64 tensors are IntTensors
NumericalTensor = FloatTensor | IntTensor
FloatTensorArray = JaxFloat[torch.Tensor, "_"]
IntTensorArray = JaxInt[torch.Tensor, "_"]
Float1dTensor = JaxFloat[torch.Tensor, "_"]
Float2dTensor = JaxFloat[torch.Tensor, "_ _"]
Float3dTensor = JaxFloat[torch.Tensor, "_ _ _"]
FloatScalar = JaxFloat[torch.Tensor, ""]
Int1dTensor = JaxInt[torch.Tensor, "_"]

FloatSequence = (
    Float[torch.Tensor, "_"]
    | Float[np.ndarray, "_"]
    | list[float]
    | list[Number]
)

IntSequence = Int[torch.Tensor, "_"] | Int[np.ndarray, "_"] | list[int]

NumberSequence = FloatSequence | IntSequence

DoubleTensor = JaxDouble[torch.Tensor, "..."]
Double2dTensor = JaxDouble[torch.Tensor, "_ _"]

# Specific numerical types
Points2d = JaxFloat[torch.Tensor, "_ 2"]
Points3d = JaxFloat[torch.Tensor, "_ 3"]

PointMasses = JaxFloat[torch.Tensor, "n_points"]
PointDensities = JaxFloat[torch.Tensor, "n_points"]
PointAnySignals = JaxFloat[torch.Tensor, "n_points ..."]
EdgeLengths = JaxFloat[torch.Tensor, "n_edges"]
EdgeMidpoints = JaxFloat[torch.Tensor, "n_edges dim"]
EdgePoints = JaxFloat[torch.Tensor, "n_edges 2 dim"]
TriangleAreas = JaxFloat[torch.Tensor, "n_triangles"]
TriangleCentroids = JaxFloat[torch.Tensor, "n_triangles dim"]
TriangleNormals = JaxFloat[torch.Tensor, "n_triangles dim"]
TrianglePoints = JaxFloat[torch.Tensor, "n_triangles 3 dim"]

# name the dimension to ensure identical number of points
Points3d_n = JaxFloat[torch.Tensor, "n 3"]

Points = Points2d | Points3d  # 3D or 2D points

# Points sequences for sequence of poses (i.e. sequence of shapes)
# The first dimension is the number of points
# The second dimension is the number of poses
# The third dimension is the dimension of the points
PointsSequence2D = JaxFloat[torch.Tensor, "_ _ 2"]
PointsSequence3D = JaxFloat[torch.Tensor, "_ _ 3"]
PointsSequence = PointsSequence2D | PointsSequence3D

Edges = JaxInt[torch.Tensor, "_ 2"]
Triangles = JaxInt[torch.Tensor, "_ 3"]

# Jaxtyping does not provide annotation for sparse tensors
# Then we use the torch.Tensor type and checks are made at runtime
# with assert statements


Landmarks = Annotated[
    torch.Tensor,
    Is[lambda tensor: tensor.dtype == float_dtype and tensor.is_sparse],
]


# Types for shapes
class polydata_type:
    """Class for polydata shapes."""


class image_type:
    """Class for image shapes."""


shape_type = polydata_type | image_type


@beartype
class FineToCoarsePolicy(NamedTuple):
    """Parameters for the fine to coarse propagation scheme.

    Parameters
    ----------
    reduce : str, default="mean"
        The reduction operation to use when propagating the signal from the
        fine to the coarse resolutions. Possible values are "mean", "max",
        "min" and "sum".
    """

    reduce: Literal["mean", "max", "min", "sum"] = "mean"


@beartype
class CoarseToFinePolicy(NamedTuple):
    """Parameters for the coarse to fine propagation scheme.

    Parameters
    ----------
    smoothing : str, default="constant"
        The smoothing operation to use when propagating the signal from the
        coarse to the fine resolutions. Possible values are "constant",
        "point_convolution" and "mesh_convolution".
    n_smoothing_steps : int, default=1
        The number of smoothing steps to perform when propagating the signal
        from the coarse to the fine resolutions.
    """

    smoothing: Literal[
        "constant",
        "point_convolution",
        "mesh_convolution",
    ] = "constant"
    n_smoothing_steps: int = 1


class MorphingOutput:
    """Class containing the result of the morphing algorithms.

    It acts as a container for the result of the morphing algorithms. It
    contains the morphed shape, the regularization parameter (if any), the path
    (if any), the path length (if any) and eventually other attributes.

    Parameters
    ----------
    morphed_shape
        the morphed shape
    regularization
        the regularization parameter
    path
        the path (list of shapes)
    path_length
        the length of the path
    kwargs
        other attributes (if any)

    """

    def __init__(
        self,
        morphed_shape: shape_type | None = None,
        regularization: FloatScalar | None = None,
        path: list[shape_type] | None = None,
        path_length: FloatScalar | None = None,
        **kwargs,
    ) -> None:
        # Define the attributes (common to all morphing algorithms)
        self.morphed_shape = morphed_shape
        self.regularization = regularization
        self.path = path
        self.path_length = path_length

        # Eventually add other attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
