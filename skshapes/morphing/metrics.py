"""Metrics."""

from ..types import (
    FloatScalar,
    Edges,
    Triangles,
    Float3dTensor,
)
from ..input_validation import typecheck
from typing import Optional
import torch


class Metric:
    """Base class for all metrics."""

    pass


class AsIsometricAsPossible(Metric):
    """
    As isometric as possible metric.

    This metric penalizes the non-isometricity of the mesh. See
    https://dmg.tuwien.ac.at/geom/ig/publications/oldpub/2007/kilian-2007-gmss/paper_docs/shape_space_sig_07.pdf # noqa E501
    for more details.
    """

    def __init__(self) -> None:
        """Class constructor."""
        pass

    @typecheck
    def __call__(
        self,
        points_sequence: Float3dTensor,
        velocities_sequence: Float3dTensor,
        edges: Optional[Edges] = None,
        triangles: Optional[Triangles] = None,
    ) -> FloatScalar:
        """Compute the mean velocities' metric along the sequence of points.

        Parameters
        ----------
        points_sequence
            Sequence of points
        velocities_sequence
            Sequence of velocities.
        edges
            Edges.
        triangles
            Triangles.

        Raises
        ------
            ValueError: This metric requires edges to be specified

        Returns
        -------
            FloatScalar: the mean velocities metric
        """
        if edges is None:
            raise ValueError("This metric requires edges to be defined")

        n_steps = points_sequence.shape[1]
        e0, e1 = edges[:, 0], edges[:, 1]
        a1 = (
            (velocities_sequence[e0] - velocities_sequence[e1])
            * (points_sequence[e0] - points_sequence[e1])
        ).sum(dim=2)

        scale = (points_sequence[e0] - points_sequence[e1]).norm(dim=2).mean()

        return torch.sum(a1**2) / (2 * n_steps * (scale**4))


class AsRigidAsPossible(Metric):
    """
    As rigid as possible metric.

    This metric penalizes the non-isometricity of the mesh. See
    https://dmg.tuwien.ac.at/geom/ig/publications/oldpub/2007/kilian-2007-gmss/paper_docs/shape_space_sig_07.pdf # noqa E501
    for more details.
    """

    @typecheck
    def __init__(self) -> None:
        """Class constructor."""
        pass

    @typecheck
    def __call__(
        self,
        points_sequence: Float3dTensor,
        velocities_sequence: Float3dTensor,
        edges: Optional[Edges] = None,
        triangles: Optional[Triangles] = None,
    ) -> FloatScalar:
        """Compute the metric along the sequence of points.

        Parameters
        ----------
        points_sequence
            Sequence of points
        velocities_sequence
            Sequence of velocities.
        edges
            Edges.
        triangles
            Triangles.

        """
        from ..tasks import Registration
        from .rigid_motion import RigidMotion
        from ..loss import L2Loss
        from ..data import PolyData
        from ..optimization import SGD

        gpu = True if points_sequence.is_cuda else False

        loss = L2Loss()
        model = RigidMotion()
        registration = Registration(
            model=model,
            loss=loss,
            optimizer=SGD(1e-2),
            n_iter=2,
            verbose=False,
            gpu=gpu,
        )

        rigid_velocity_sequence = torch.zeros_like(velocities_sequence)
        for i in range(velocities_sequence.shape[1] - 1):
            source = PolyData(points=points_sequence[:, i, :])
            target = PolyData(points=points_sequence[:, i + 1, :])

            rigid_morph = registration.fit_transform(
                source=source, target=target
            )

            rigid_velocity_sequence[:, i, :] = (
                rigid_morph.points - source.points
            )

        return torch.sum(
            (rigid_velocity_sequence - velocities_sequence) ** 2
        ) / (2 * velocities_sequence.shape[1])
