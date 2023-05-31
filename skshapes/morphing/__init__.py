from .elastic_metric import ElasticMetric, ElasticMetric2
from .rigid import RigidMotion

from .._typing import *

# class BaseMorphing:

#     def __init__(self, **kwargs) -> None:
#         """Initialize the morphing algorithm with the given hyperparameters
#         """
#         for key, value in kwargs.items():
#             setattr(self, key, value)

#     def fit(self, *, source: Shape):
#         """Fit the morphing algorithm to the given source shape
#         """
#         # Store the source shape attributes (all shapes, landmarks, etc.)
#         # for later use
#         return self
    
#     def morph(
#             self,
#             parameter: torch.Tensor,
#             return_path: bool = False,
#             return_regularization: bool = False,
#             ) -> Union[
#                 Shape/float3dTensorType,
#                 List[Shape],
#                 (Shape, floatScalarType),
#                 (List[Shape], floatScalarType),
#                 ]:
        
#         """Morph the source shape given the parameter
#             depending on the values of return_path and return_regularization
#             the method returns different things :
#             - if return_path is False, the method returns the morphed shape
#             - if return_path is True, the method returns a list of shapes
#             - if return_regularization is True, the method returns a tuple
#                 of the morphed shape and the regularization value
#         """