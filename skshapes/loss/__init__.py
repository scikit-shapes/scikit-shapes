"""
The :mod:`skshapes.loss` module gathers loss functions for shape processing.
"""

from .baseloss import EmptyLoss, SumLoss, ProductLoss
from .lp import L2Loss, LpLoss, LandmarkLoss

from .optimal_transport import OptimalTransportLoss
from .nearest_neighbors import NearestNeighborsLoss


from typing import Union

# The skshapes.Loss type is a union of all the loss functions
# Note that BaseLoss is not included in the union, as it is an abstract class
# Therefore, passing a BaseLoss object to a function expecting a Loss object
# will raise an error
Loss = Union[
    OptimalTransportLoss,
    LandmarkLoss,
    NearestNeighborsLoss,
    L2Loss,
    LpLoss,
    EmptyLoss,
    SumLoss,
    ProductLoss,
]
