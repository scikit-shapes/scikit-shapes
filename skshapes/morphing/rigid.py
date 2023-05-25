import torch
from pytorch3d.transforms import axis_angle_to_matrix


class RigidMotion:
    """
    Rigid motion morphing. The parameter is a (2, 3) tensor, where the first row
    is the rotation axis-angle and the second row is the translation vector.
    """

    def __init__(self) -> None:
        pass

    def fit(self, source):
        self.source_points = source.points.clone()

    def morph(self, parameter):

        rotation = parameter[0]
        translation = parameter[1]

        matrix = axis_angle_to_matrix(rotation)

        return (matrix @ self.source_points.T).T + translation

    def regularization(self, parameter):
        return torch.tensor(0.0)

    @property
    def parameter_template(self):
        return torch.zeros(
            (2, 3), dtype=torch.float32, device=self.source_points.device
        )
