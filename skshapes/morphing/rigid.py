import torch
from pytorch3d.transforms import axis_angle_to_matrix


from .._typing import *


class RigidMotion:
    """
    Rigid motion morphing. The parameter is a (2, 3) tensor, where the first row
    is the rotation axis-angle and the second row is the translation vector.
    """

    @typecheck
    def __init__(self, n_steps: int = 1) -> None:
        self.n_steps = n_steps
        pass

    @typecheck
    def morph(
        self,
        shape: PolyDataType,
        parameter: float2dTensorType,
        return_path: bool = False,
        return_regularization: bool = False,
    ) -> MorphingOutput:

        rotation = parameter[0]
        translation = parameter[1]

        matrix = axis_angle_to_matrix(rotation)

        newpoints = (matrix @ shape.points.T).T + translation

        morphed_shape = shape.copy()
        morphed_shape.points = newpoints

        regularization = torch.tensor(0.0, device=parameter.device)

        path = None
        if return_path:
            path = [shape.copy(), morphed_shape.copy()]

        return MorphingOutput(
            morphed_shape=morphed_shape,
            regularization=regularization,
            path=path,
        )

    @typecheck
    def parameter_shape(self, shape: Shape) -> Tuple[int, int]:

        return (2, 3)


# class RigidMotion:
#     """
#     Rigid motion morphing. The parameter is a (2, 3) tensor, where the first row
#     is the rotation axis-angle and the second row is the translation vector.
#     """

#     def __init__(self) -> None:
#         pass

#     def fit(self, source):
#         self.source_points = source.points.clone()
#         return self

#     def morph(self, parameter):

#         rotation = parameter[0]
#         translation = parameter[1]

#         matrix = axis_angle_to_matrix(rotation)

#         return (matrix @ self.source_points.T).T + translation

#     def regularization(self, parameter):
#         return torch.tensor(0.0)

#     @property
#     def parameter_template(self):
#         return torch.zeros(
#             (2, 3), dtype=torch.float32, device=self.source_points.device
#         )
