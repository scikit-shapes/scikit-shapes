"""Decimation module."""

import numpy as np
from ..data import PolyData
import torch
from ..types import float_dtype, int_dtype, Number
from ..input_validation import typecheck, one_and_only_one, no_more_than_one
from typing import Optional
import fast_simplification


class Decimation:
    """Decimation class.

    This class implements the quadric decimation algorithm. The goal of
    decimation is to reduce the number of points of a triangular mesh while
    preserving the global aspect of the shape.

    Examples
    --------
    ```python
    # assume that pose1 and pose2 are two meshes with the same connectivity:

    pose1, pose2 = sks.PolyData("data/pose1.vtk","data/pose2.vtk")
    decimator = sks.Decimation(target_reduction=0.5)
    decimator.fit(cat1)
    pose1_decimated = decimator.transform(cat1)
    pose2_decimated = decimator.transform(cat2)
    assert torch.allclose(pose1_decimated.triangles, pose2_decimated.triangles)
    # pose1_decimated and pose2_decimated have the same connectivity

    # if landmarks are present in the meshes, they are kept after the
    # decimation
    if pose1.landmarks is not None:
        assert pose1_decimated.landmarks is not None
    if pose2.landmarks is not None:
        assert pose2_decimated.landmarks is not None
    ```
    """

    @one_and_only_one(parameters=["target_reduction", "n_points", "ratio"])
    @typecheck
    def __init__(
        self,
        *,
        target_reduction: Optional[float] = None,
        n_points: Optional[int] = None,
        ratio: Optional[Number] = None,
    ) -> None:
        """Class constructor.

        Initialize the quadric decimation algorithm with a target reduction or
        the desired number for decimated mesh.

        Parameters
        ----------
        target_reduction
            The target rate of triangles to delete during decimation. Must
            be between 0 and 1.
        n_points
            The desired number of points in the decimated mesh.

        Raises
        ------
            ValueError: If both target_reduction and n_points are provided or
            if none of them is provided.
        """
        if target_reduction is not None:
            assert (
                target_reduction > 0 and target_reduction < 1
            ), "The target reduction must be between 0 and 1"
            self.target_reduction = target_reduction

        if n_points is not None:
            self.n_points = n_points

        if ratio is not None:
            assert ratio > 0 and ratio < 1, "The ratio must be between 0 and 1"
            self.target_reduction = 1 - ratio

    @typecheck
    def fit(self, mesh: PolyData) -> None:
        """Fit the decimation algorithm to a mesh.

        Parameters
        ----------
        mesh
            The mesh to fit the decimation object to.

        Raises
        ------
        ValueError
            If the mesh is not triangular.

        """
        if hasattr(self, "n_points"):
            if not (1 <= self.n_points <= mesh.n_points):
                raise ValueError(
                    "The n_points must be positive and lower"
                    + " than the number of points of the mesh"
                    + " to decimate"
                )

            self.target_reduction = 1 - self.n_points / mesh.n_points

        if not mesh.is_triangle_mesh():
            raise ValueError(
                "Quadric decimation only works for triangular" + " meshes"
            )

        points = mesh.points.clone().cpu().numpy().astype(np.float32)
        triangles = mesh.triangles.clone().cpu().numpy().astype(np.int64)

        # Run the quadric decimation algorithm
        # import pyDecimation

        (
            decimated_points,
            decimated_triangles,
            collapses,
        ) = fast_simplification.simplify(
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

    @no_more_than_one(parameters=["target_reduction", "n_points", "ratio"])
    @typecheck
    def transform(
        self,
        mesh: PolyData,
        *,
        target_reduction: Optional[float] = None,
        n_points: Optional[int] = None,
        ratio: Optional[float] = None,
    ) -> PolyData:
        """Transform a mesh using the decimation algorithm.

        The decimation must have been fitted to a mesh before calling this
        method. The mesh to decimate could be:
        - the same mesh as the one used to fit the decimation object
        - a mesh with the same connectivity as the one used to fit the
            decimation object (same number of points and same triangles)

        Parameters
        ----------
        mesh
            The mesh to transform.
        target_reduction
            The target reduction to apply to the mesh.
        n_points
            The targeted number of points.
        ratio
            The ratio of the number of points of the mesh to decimate over the
            number of points of the mesh used to fit the decimation object.

        Raises
        ------
        ValueError
            If the decimation object has not been fitted yet.
            If the number of points or the triangles structure of the mesh to
            decimate is not the same as the mesh using to fit.

        Returns
        -------
        PolyData
            The decimated mesh.
        """
        if target_reduction is None and n_points is None:
            # default, target_reduction is the same as in __init__
            target_reduction = self.target_reduction

        if self.collapses_ is None:
            raise ValueError("The decimation object has not been fitted yet.")

        if target_reduction is not None:
            if not (0 <= target_reduction <= 1):
                raise ValueError(
                    "The target reduction must be between 0 and 1"
                )
        elif ratio is not None:
            if not (0 <= ratio <= 1):
                raise ValueError("The ratio must be between 0 and 1")
            target_reduction = 1 - ratio

        elif n_points is not None:
            if not (1 <= n_points <= self.ref_mesh.n_points):
                raise ValueError(
                    "The n_points must be positive and lower"
                    + " than the number of points of the mesh"
                    + " to decimate"
                )

            target_reduction = 1 - n_points / mesh.n_points

        if target_reduction > self.target_reduction:
            target_reduction = self.target_reduction

        if mesh.n_points != self.ref_mesh.n_points:
            raise ValueError(
                "The number of points of the mesh to decimate must be the same"
                + " as the reference mesh"
            )

        if not torch.allclose(
            mesh.triangles.cpu(), self.ref_mesh.triangles.cpu()
        ):
            raise ValueError(
                "The triangles of the mesh to decimate must be the same as the"
                + " reference mesh"
            )

        device = mesh.device

        # Replay the decimation process on the mesh
        points = mesh.points.clone().cpu().numpy().astype(np.float32)
        triangles = mesh.triangles.clone().cpu().numpy().astype(np.int64)

        # Compute the number of collapses to apply
        rate = target_reduction / self.target_reduction
        n_collapses = int(rate * len(self.collapses))

        # Apply the collapses
        (
            points,
            triangles,
            indice_mapping,
        ) = fast_simplification.replay_simplification(
            points=points,
            triangles=triangles,
            collapses=self.collapses[0:n_collapses],
        )

        self.indice_mapping_ = torch.Tensor(indice_mapping).to(int_dtype)

        # If there are landmarks on the mesh, we compute the coordinates of the
        # landmarks in the decimated mesh
        if mesh.landmarks is not None:
            coalesced_landmarks = mesh.landmarks.coalesce()
            l_values = coalesced_landmarks.values()
            l_indices = coalesced_landmarks.indices()
            l_size = coalesced_landmarks.size()
            n_landmarks = l_size[0]

            new_indices = l_indices.clone()
            # the second line of new_indices corresponds to the indices of the
            # points, we need to apply the mapping
            new_indices[1] = torch.from_numpy(indice_mapping[new_indices[1]])

            # If there are landmarks in the decimated mesh, we create a sparse
            # tensor with the landmarks
            landmarks = torch.sparse_coo_tensor(
                values=l_values,
                indices=new_indices,
                size=(n_landmarks, len(points)),
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
        """Decimate and return decimated mesh."""
        self.fit(mesh)
        return self.transform(mesh)

    @typecheck
    @property
    def collapses(self):
        """Returns the collapses of the decimation algorithm."""
        if hasattr(self, "collapses_"):
            return self.collapses_
        else:
            raise ValueError("The decimation object has not been fitted yet.")

    @typecheck
    @property
    def indice_mapping(self):
        """Returns the indice mapping of the decimation algorithm."""
        if hasattr(self, "indice_mapping_"):
            return self.indice_mapping_
        else:
            raise ValueError("The decimation object has not been fitted yet.")

    @typecheck
    @property
    def actual_reduction(self):
        """Returns the actual reduction of the decimation algorithm."""
        if hasattr(self, "actual_reduction_"):
            return self.actual_reduction_
        else:
            raise ValueError("The decimation object has not been fitted yet.")

    @typecheck
    @property
    def ref_mesh(self):
        """Returns the reference mesh of the decimation algorithm."""
        if hasattr(self, "ref_mesh_"):
            return self.ref_mesh_
        else:
            raise ValueError("The decimation object has not been fitted yet.")
