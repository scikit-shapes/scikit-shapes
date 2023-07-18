import torch


from ..types import (
    typecheck,
    MorphingOutput,
    PolyDataType,
    Float2dTensor,
    ShapeType,
    Tuple,
    Morphing,
)


class RigidMotion(Morphing):
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
        parameter: Float2dTensor,
        return_path: bool = False,
        return_regularization: bool = False,
    ) -> MorphingOutput:

        if parameter.device != shape.device:
            parameter = parameter.to(shape.device)

        rotation_angles = parameter[0]
        matrix = axis_angle_to_matrix(rotation_angles)

        translation = parameter[1]
        center = shape.points.mean(dim=0)

        newpoints = (matrix @ (shape.points - center).T).T + center + translation

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
    def parameter_shape(self, shape: ShapeType) -> Tuple[int, int]:
        return (2, 3)


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))
