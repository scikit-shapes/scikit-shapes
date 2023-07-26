import numpy as np
import pyvista

from .utils import _do_decimation, _decimate, _compute_alphas
from ..data import PolyData
import torch

from ..types import typecheck
from typing import Optional, Literal


class QuadricDecimation:
    """
    This class implements the quadric decimation algorithm. The goal of decimation is to reduce the number of points of a mesh while preserving its shape.
    The algorithm is based on the following paper : https://www.cs.cmu.edu/~./garland/Papers/quadrics.pdf

    An implementation of the algorithm is available in vtk : https://vtk.org/doc/nightly/html/classvtkQuadricDecimation.html, it is accessible through the
    implementation="vtk" argument. The parameters of the vtk implementation are the same as the default parameters used in pyvista (https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.PolyDataFilters.decimate.html#pyvista.PolyDataFilters.decimate)

    Usage example :
    mesh = sks.PolyData("data/mesh.vtk")
    decimator = sks.QuadricDecimation(target_reduction=0.5, implementation="vtk")
    mesh_decimated = decimator.fit_transform(mesh) # mesh_decimated is a sks.PolyData object with (approximately) half the number of points of mesh


    A drawback of the vtk implementation appears when the decimation is applied to several meshes with the same connectivity (eg the dataset SCAPE consisting of human poses where points and triangles are in correspondence accross poses).
    Indeed, applying the vtk decimation to each mesh independently will result in a loss of correspondence between points and triangles. To avoid this, the implementation="sks" argument can be used.

    The implementation="sks" argument uses a custom implementation of the algorithm that allows to run the same decimation on several meshes with the same connectivity in order to keep the correspondence between points and triangles in low-resolution meshes.
    It is an implementation based on the C++ implementation in vtk but written in python using numba's JIT engine to speed up the computations. In this implementation, we keep track of the collapses that are performed during the decimation in order to be able to apply the same collapses to other meshes with the same connectivity.


    Usage examples :

    # assume that pose1 and pose2 are two meshes with the same connectivity:

    pose1, pose2 = sks.PolyData("data/pose1.vtk","data/pose2.vtk")
    decimator = sks.QuadricDecimation(target_reduction=0.5, implementation="sks")
    decimator.fit(cat1)
    pose1_decimated = decimator.transform(cat1)
    pose2_decimated = decimator.transform(cat2)
    assert torch.allclose(pose1_decimated.triangles, pose2_decimated.triangles) # pose1_decimated and pose2_decimated have the same connectivity

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
        Initialize the quadric decimation algorithm with a target reduction or the desired number of points in lox-resulution mesh and choose between vtk and sks implementations.

        Args:
            target_reduction (float, optional): The target reduction of the number of points in the low-resolution mesh. Must be between 0 and 1. Defaults to None.
            n_points (int, optional): The desired number of points in the low-resolution mesh. Defaults to None.
            implementation (Literal["vtk", "sks"], optional): The implementation of the algorithm. Defaults to "sks".

        Raises:
            ValueError: If both target_reduction and n_points are provided or if none of them is provided.
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
    def fit(self, mesh: PolyData) -> None:
        """
        Fit the decimation algorithm to a mesh. The behavior depends on the implementation of the algorithm.

        If the implementation is "vtk", the fit method does nothing more than checking that the mesh argument is a triangle mesh.
        If the implementation is "sks", the fit method runs the quadric decimation algorithm on the mesh and saves the information needed to apply the same decimation to other meshes inside the object.

        Args:
            mesh (PolyData): The mesh to fit the decimation object to.
        """

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
    def transform(self, mesh: PolyData) -> PolyData:
        """
        Transform a mesh using the decimation algorithm. The behavior depends on the implementation of the algorithm.

        If the implementation is "vtk", the transform method raises an error. It makes no sense to transform another mesh than the one used to fit the decimation object.
        If the implementation is "sks", the transform method applies the decimation process that was fitted to the reference mesh to the mesh argument. It raises an error if the mesh argument does not have the same connectivity as the reference mesh.

        Args:
            mesh (PolyData): The mesh to transform.
        """

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
        """
        Fit and transform a mesh using the decimation algorithm. The behavior depends on the implementation of the algorithm.

        If the implementation is "vtk", the fit_transform method consists in applying the vtkQuadricDecimation algorithm to the mesh argument and returning the decimated mesh.
        If the implementation is "sks", the fit_transform method consists in fitting the decimation object to the mesh argument and then applying the decimation process to the mesh argument and return the decimated mesh. the .transform() method can then be used to apply the same decimation to other meshes with the same connectivity.
        """

        self.fit(mesh)
        if self.implementation == "vtk":
            device = mesh.device
            return PolyData(
                mesh.to_pyvista().decimate(self.target_reduction), device=device
            )

        else:
            return self.transform(mesh)
