"""Rigid motion model.

This module contains the implementation of the rigid motion model. This model
is described by a translation and a rotation. The morphing is not regularized.
"""

from .basemodel import BaseModel
import torch

from ..types import (
    typecheck,
    Float2dTensor,
    polydata_type,
    shape_type,
    convert_inputs,
)

from .utils import MorphingOutput


class RigidMotion(BaseModel):
    """Rigid motion morphing."""

    @typecheck
    def __init__(self, n_steps: int = 1) -> None:
        """Initialize the model

        Args:
            n_steps (int, optional): number of steps. Defaults to 1.
        """
        self.n_steps = n_steps
        pass

    @convert_inputs
    @typecheck
    def morph(
        self,
        shape: polydata_type,
        parameter: Float2dTensor,
        return_path: bool = False,
        return_regularization: bool = False,
    ) -> MorphingOutput:
        """Morph a shape using the rigid motion model

        The parameter is a (2, 3) tensor, where the first row is the rotation
        axis-angle and the second row is the translation vector.

        Args:
            shape (polydata_type): shape to morph
            parameter ((2, 3) tensor) : rigid motion parameters
            return_path (bool, optional): True if you want to have access to
                the morphing's sequence of polydatas. Defaults to False.
            return_regularization (bool, optional): True to have access to the
                regularization. Defaults to False.

        Returns:
            MorphingOutput: a named tuple containing the morphed shape, the
                regularization and the path if needed.
        """
        if parameter.device != shape.device:
            parameter = parameter.to(shape.device)

        rotation_angles = parameter[0]
        matrix = axis_angle_to_matrix(rotation_angles)

        translation = parameter[1]
        center = shape.points.mean(dim=0)

        newpoints = (
            (matrix @ (shape.points - center).T).T + center + translation
        )

        morphed_shape = shape.copy()
        morphed_shape.points = newpoints

        regularization = torch.tensor(0.0, device=parameter.device)

        path = None
        if return_path:
            if self.n_steps == 1:
                path = [shape.copy(), morphed_shape.copy()]

            else:
                path = [shape.copy()]
                for i in range(self.n_steps):
                    local_parameter = ((i + 1) / self.n_steps) * parameter
                    translation = local_parameter[1]
                    center = shape.points.mean(dim=0)

                    newpoints = (
                        (matrix @ (shape.points - center).T).T
                        + center
                        + translation
                    )
                    newshape = shape.copy()
                    newshape.points = newpoints
                    path.append(newshape)

        return MorphingOutput(
            morphed_shape=morphed_shape,
            regularization=regularization,
            path=path,
        )

    @typecheck
    def parameter_shape(self, shape: shape_type) -> tuple[int, int]:
        """Return the shape of the parameter

        Args:
            shape (polydata_type): the shape to morph

        Returns:
            tuple[int, int]: the shape of the parameter
        """
        return (2, 3)


@convert_inputs
@typecheck
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
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        dim=-1,
    )
    return quaternions


@convert_inputs
@typecheck
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


@convert_inputs
@typecheck
def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))