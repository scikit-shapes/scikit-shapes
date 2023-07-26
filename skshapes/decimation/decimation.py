import numpy as np
import pyvista

from .utils import _do_decimation, _decimate, _compute_alphas
from ..data import PolyData
import torch

from ..types import typecheck
from typing import Optional, Literal


class QuadricDecimation:
    """
    This class implements the quadric decimation algorithm and make it possible to run it in parallel.
    """

    @typecheck
    def __init__(
        self,
        *,
        target_reduction: Optional[float] = None,
        n_points: Optional[int] = None,
        implementation: Literal["vtk", "sks"] = "sks"
    ) -> None:
        """
        Initialize the quadric decimation algorithm with a target reduction.

        Args:
        """
        if target_reduction is None and n_points is None:
            raise ValueError("Either target_reduction or n_points must be provided.")

        if target_reduction is not None and n_points is not None:
            raise ValueError(
                "Only one of target_reduction or n_points must be provided."
            )

        if target_reduction is not None:
            assert (
                target_reduction > 0 and target_reduction < 1
            ), "The target reduction must be between 0 and 1"
            self.target_reduction = target_reduction

        if n_points is not None:
            self.n_points = n_points

        self.implementation = implementation

    @typecheck
    def fit(self, mesh) -> None:
        """ """

        if hasattr(self, "n_points"):
            self.target_reduction = 1 - self.n_points / mesh.n_points

        assert hasattr(
            mesh, "triangles"
        ), "Quadric decimation only works on meshes with triangles"

        if self.implementation == "vtk":
            return None

        points = mesh.points.clone().cpu().numpy()
        triangles = mesh.triangles.clone().cpu().numpy()

        # Run the quadric decimation algorithm
        decimated_points, collapses_history, newpoints_history = _do_decimation(
            points=points, triangles=triangles, target_reduction=self.target_reduction
        )
        keep = np.setdiff1d(
            np.arange(mesh.n_points), collapses_history[:, 1]
        )  # Indices of the points that must be kept after decimation

        # Compute the alphas
        alphas = _compute_alphas(points, collapses_history, newpoints_history)

        # Compute the mapping from original indices to new indices
        # start with identity mapping
        indice_mapping = np.arange(mesh.n_points, dtype=int)
        # First round of mapping
        origin_indices = collapses_history[:, 1]
        indice_mapping[origin_indices] = collapses_history[:, 0]
        previous = np.zeros(len(indice_mapping))
        while not np.array_equal(previous, indice_mapping):
            previous = indice_mapping.copy()
            indice_mapping[origin_indices] = indice_mapping[
                indice_mapping[origin_indices]
            ]
        application = dict([keep[i], i] for i in range(len(keep)))
        indice_mapping = np.array([application[i] for i in indice_mapping])

        # compute the new triangles
        # TODO avoid torch -> numpy conversion here
        triangles_copy = mesh.triangles.clone().cpu()
        triangles_copy = indice_mapping[triangles_copy]
        keep_triangle = (
            (triangles_copy[0] != triangles_copy[1])
            * (triangles_copy[1] != triangles_copy[2])
            * (triangles_copy[0] != triangles_copy[2])
        )
        new_triangles = torch.from_numpy(triangles_copy[:, keep_triangle])
        self.new_triangles = new_triangles

        # Save the results
        self.keep = keep
        self.indice_mapping = indice_mapping
        self.alphas = alphas
        self.collapses_history = collapses_history
        self.ref_mesh = mesh

    @typecheck
    def transform(self, mesh):
        """ """

        assert (
            self.implementation == "sks"
        ), "The transform method is only available when the decimation implementation is 'sks'"

        assert (
            mesh.n_points == self.ref_mesh.n_points
        ), "The number of points of the mesh to decimate must be the same as the reference mesh"
        assert torch.allclose(
            mesh.triangles.cpu(), self.ref_mesh.triangles.cpu()
        ), "The triangles of the mesh to decimate must be the same as the reference mesh"

        device = mesh.device

        points = mesh.points.clone().cpu().numpy()
        points = _decimate(
            points, collapses_history=self.collapses_history, alphas=self.alphas
        )

        # keep can be saved in the fit method
        points = points[self.keep]

        return PolyData(
            torch.from_numpy(points), triangles=self.new_triangles, device=device
        )

    @typecheck
    def fit_transform(self, mesh: PolyData) -> PolyData:
        """ """

        self.fit(mesh)
        if self.implementation == "vtk":
            device = mesh.device
            return PolyData(
                mesh.to_pyvista().decimate(self.target_reduction), device=device
            )

        else:
            return self.transform(mesh)
