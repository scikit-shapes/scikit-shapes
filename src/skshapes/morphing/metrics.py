"""Metrics."""

import torch

from ..input_validation import typecheck
from ..types import (
    Edges,
    Float3dTensor,
    FloatScalar,
    Number,
    Triangles,
)


class Metric:
    """Base class for all metrics."""


class AsIsometricAsPossible(Metric):
    """
    As isometric as possible metric.

    This metric penalizes the non-isometricity of the mesh.

    References
    ----------
    [KILIAN, Martin, MITRA, Niloy J., et POTTMANN, Helmut. Geometric
    modeling in shape space. In : ACM SIGGRAPH 2007 papers. 2007. p. 64-es.](https://dmg.tuwien.ac.at/geom/ig/publications/oldpub/2007/kilian-2007-gmss/paper_docs/shape_space_sig_07.pdf)

    """

    def __init__(self) -> None:
        pass

    @typecheck
    def __call__(
        self,
        points_sequence: Float3dTensor,
        velocities_sequence: Float3dTensor,
        edges: Edges | None = None,
        triangles: Triangles | None = None,  # noqa: ARG002
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
            AttributeError: This metric requires edges to be defined

        Returns
        -------
            FloatScalar: the mean velocities metric
        """
        if edges is None:
            msg = "This metric requires edges to be defined"
            raise AttributeError(msg)

        n_steps = points_sequence.shape[1]
        e0, e1 = edges[:, 0], edges[:, 1]
        a1 = (
            (velocities_sequence[e0] - velocities_sequence[e1])
            * (points_sequence[e0] - points_sequence[e1])
        ).sum(dim=2)

        scale = (points_sequence[e0] - points_sequence[e1]).norm(dim=2).mean()

        return torch.sum(a1**2) / (2 * n_steps * (scale**4))


class ShellEnergyMetric(Metric):
    """Shell energy metric.

    This metric is the sum of a membrane contribution, measuring the local
    stretching and a bending contribution, measuring the change in curvature.
    The parameter weight controls the importance of the bending energy.

    Parameters
    ----------
    weight
        Weight of the bending energy.

    References
    ----------
    [HEEREN, Behrend, RUMPF, Martin, SCHRÖDER, Peter, et al. Exploring the
    geometry of the space of shells. In : Computer Graphics Forum. 2014.
    p. 247-256.](https://ddg.math.uni-goettingen.de/pub/HeRuSc14.pdf)
    """

    @typecheck
    def __init__(self, weight: Number = 0.001) -> None:
        self.weight = weight

    @typecheck
    def __call__(
        self,
        points_sequence: Float3dTensor,
        velocities_sequence: Float3dTensor,
        edges: Edges | None = None,  # noqa: ARG002
        triangles: Triangles | None = None,
    ) -> FloatScalar:
        """Compute shell energy along the sequence of points.

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
            AttributeError: This metric requires triangles to be defined

        Returns
        -------
            FloatScalar: the mean velocities metric
        """
        if triangles is None:
            msg = "This metric requires triangles to be defined"
            raise AttributeError(msg)

        n_steps = points_sequence.shape[1]

        points_undef = points_sequence[:, 0:n_steps, :]
        points_def = points_undef + velocities_sequence

        from ..triangle_mesh import shell_energy

        return shell_energy(
            points_undef=points_undef,
            points_def=points_def,
            triangles=triangles,
            weight=self.weight,
        ).mean()
