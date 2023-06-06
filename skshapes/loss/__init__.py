from .optimal_transport import OptimalTransportLoss
from .landmarks import LandmarkLoss
from .nearest_neighbors import NearestNeighborsLoss


from .._typing import *

# class BaseLoss:

#     def __init__(self, **kwargs):
#         """Initialize the loss function with the given hyperparameters
#         """
#         for key, value in kwargs.items():
#             setattr(self, key, value)

#     def fit(self, *, target: Shape) -> self:
#         """Fit the loss function to the given shapes
#         """
#         # Store the target shape attributes (all shapes, landmarks, etc.)
#         # for later use
#         return self

#     def __call__(self, source: Shape) -> floatScalarType:
#         # Extract the useful attributes from the source shape
#         # and compute the loss
#         raise NotImplementedError
