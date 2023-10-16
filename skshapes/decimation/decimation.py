import numpy as np
import pyvista

from ..data import PolyData
import torch

from ..types import typecheck, float_dtype, int_dtype
from typing import Optional, Literal


import fast_simplification


class Decimation:
    """
    This class implements the quadric decimation algorithm. The goal of decimation is to reduce the number of points of a mesh while preserving its shape.
    The algorithm is based on the following paper : https://www.cs.cmu.edu/~./garland/Papers/quadrics.pdf

    An implementation of the algorithm is available in vtk : https://vtk.org/doc/nightly/html/classvtkQuadricDecimation.html, it is accessible through the
    method="vtk" argument. The parameters of the vtk implementation are the same as the default parameters used in pyvista (https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.PolyDataFilters.decimate.html#pyvista.PolyDataFilters.decimate)

    Usage example :
    mesh = sks.PolyData("data/mesh.vtk")
    decimator = sks.Decimation(target_reduction=0.5, method="vtk")
    mesh_decimated = decimator.fit_transform(mesh) # mesh_decimated is a sks.PolyData object with (approximately) half the number of points of mesh


    A drawback of the vtk implementation appears when the decimation is applied to several meshes with the same connectivity (eg the dataset SCAPE consisting of human poses where points and triangles are in correspondence accross poses).
    Indeed, applying the vtk decimation to each mesh independently will result in a loss of correspondence between points and triangles. To avoid this, the implementation="sks" argument can be used.
    Another drawback is that if landmarks are present in the mesh, there is no way to keep them after the decimation.

    The method="sks" argument corresponds to a custom implementation of the algorithm that allows to run the same decimation on several meshes with the same connectivity in order to keep the correspondence between points and triangles in low-resolution meshes.
    It also allows to keep landmarks after the decimation when landmarks are stored in barycentric coordinates on the mesh.
    It is an implementation based on the C++ vtk code but written in python using numba's JIT engine to speed up the computations. In this implementation, we keep track of the collapses that are performed during the decimation in order to be able to apply the same collapses to other meshes with the same connectivity.


    Usage examples :

    # assume that pose1 and pose2 are two meshes with the same connectivity:

    pose1, pose2 = sks.PolyData("data/pose1.vtk","data/pose2.vtk")
    decimator = sks.Decimation(target_reduction=0.5, method="sks")
    decimator.fit(cat1)
    pose1_decimated = decimator.transform(cat1)
    pose2_decimated = decimator.transform(cat2)
    assert torch.allclose(pose1_decimated.triangles, pose2_decimated.triangles) # pose1_decimated and pose2_decimated have the same connectivity

    # if landmarks are present in the meshes, they are kept after the decimation
    if pose1.landmarks is not None:
        assert pose1_decimated.landmarks is not None
    if pose2.landmarks is not None:
        assert pose2_decimated.landmarks is not None
    """

    @typecheck
    def __init__(
        self,
        *,
        target_reduction: Optional[float] = None,
        n_points: Optional[int] = None,
    ) -> None:
        """
        Initialize the quadric decimation algorithm with a target reduction or the desired number of points in lox-resulution mesh and choose between vtk and sks implementations.

        Args:
            target_reduction (float, optional): The target reduction of the number of points in the low-resolution mesh. Must be between 0 and 1. Defaults to None.
            n_points (int, optional): The desired number of points in the low-resolution mesh. Defaults to None.

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

    @typecheck
    def fit(self, mesh: PolyData) -> None:
        """
        Fit the decimation algorithm to a mesh.

        Args:
            mesh (PolyData): The mesh to fit the decimation object to.
        """

        if hasattr(self, "n_points"):
            self.target_reduction = 1 - self.n_points / mesh.n_points

        assert hasattr(
            mesh, "triangles"
        ), "Quadric decimation only works for triangular meshes"

        points = mesh.points.clone().cpu().numpy()
        triangles = mesh.triangles.clone().cpu().numpy()

        # Run the quadric decimation algorithm
        # import pyDecimation

        decimated_points, decimated_triangles, collapses = fast_simplification.simplify(
            points=points,
            triangles=triangles,
            target_reduction=self.target_reduction,
            return_collapses=True,
        )

        actual_reduction = 1 - len(decimated_points) / len(points)

        # Save the results
        self.collapses_ = collapses
        self.ref_mesh_ = mesh
        self.actual_reduction_ = actual_reduction

    @typecheck
    def transform(
        self, mesh: PolyData, target_reduction: Optional[float] = 1
    ) -> PolyData:
        """
        Transform a mesh using the decimation algorithm.
        Args:
            mesh (PolyData): The mesh to transform.
            target_reduction (float, optional): The target reduction of the number of points in the low-resolution mesh. Must be between 0 and 1. Defaults to 1.
                                                If the target_reduction is greater than the target_reduction used to fit the decimation object, the decimation
                                                will be applied with the target_reduction used to fit the decimation object. Defaults to 1.
        """
        assert (
            0 <= target_reduction <= 1
        ), "The target reduction must be between 0 and 1"
        if target_reduction > self.target_reduction:
            target_reduction = self.target_reduction

        assert (
            mesh.n_points == self.ref_mesh.n_points
        ), "The number of points of the mesh to decimate must be the same as the reference mesh"
        assert torch.allclose(
            mesh.triangles.cpu(), self.ref_mesh.triangles.cpu()
        ), "The triangles of the mesh to decimate must be the same as the reference mesh"

        device = mesh.device

        # Replay the decimation process on the mesh
        points = mesh.points.clone().cpu().numpy()
        triangles = mesh.triangles.clone().cpu().numpy()

        # Compute the number of collapses to apply
        rate = target_reduction / self.target_reduction
        n_collapses = int(rate * len(self.collapses))

        # Apply the collapses
        points, triangles, indice_mapping = fast_simplification.replay_simplification(
            points=points,
            triangles=triangles,
            collapses=self.collapses[0:n_collapses],
        )

        self.indice_mapping_ = torch.Tensor(indice_mapping).to(int_dtype)

        # If there are landmarks on the mesh, we compute the coordinates of the landmarks in the decimated mesh
        if mesh.landmarks is not None:
            coalesced_landmarks = mesh.landmarks.coalesce()
            l_values = coalesced_landmarks.values()
            l_indices = coalesced_landmarks.indices()
            l_size = coalesced_landmarks.size()
            n_landmarks = l_size[0]

            new_indices = l_indices.clone()
            # the second line of new_indices corresponds to the indices of the points, we need to apply the mapping
            new_indices[1] = torch.from_numpy(indice_mapping[new_indices[1]])

            # If there are landmarks in the decimated mesh, we create a sparse tensor with the landmarks
            landmarks = torch.sparse_coo_tensor(
                values=l_values, indices=new_indices, size=(n_landmarks, len(points))
            )
        else:
            landmarks = None

        return PolyData(
            torch.from_numpy(points).to(float_dtype),
            triangles=torch.from_numpy(triangles).to(int_dtype),
            landmarks=landmarks,
            device=device,
        )

    @typecheck
    def fit_transform(self, mesh: PolyData) -> PolyData:
        """
        Fit and transform a mesh using the decimation algorithm. The behavior depends on the implementation of the algorithm.

        """

        self.fit(mesh)
        return self.transform(mesh)

    @typecheck
    @property
    def collapses(self):
        """
        Returns the collapses of the decimation algorithm.
        """
        if hasattr(self, "collapses_"):
            return self.collapses_
        else:
            raise ValueError("The decimation object has not been fitted yet.")

    @typecheck
    @property
    def indice_mapping(self):
        """
        Returns the indice mapping of the decimation algorithm.
        """
        if hasattr(self, "indice_mapping_"):
            return self.indice_mapping_
        else:
            raise ValueError("The decimation object has not been fitted yet.")

    @typecheck
    @property
    def actual_reduction(self):
        """
        Returns the actual reduction of the decimation algorithm.
        """
        if hasattr(self, "actual_reduction_"):
            return self.actual_reduction_
        else:
            raise ValueError("The decimation object has not been fitted yet.")

    @typecheck
    @property
    def ref_mesh(self):
        """
        Returns the reference mesh of the decimation algorithm.
        """
        if hasattr(self, "ref_mesh_"):
            return self.ref_mesh_
        else:
            raise ValueError("The decimation object has not been fitted yet.")
