"""Decimation module."""

from __future__ import annotations

import fast_simplification
import numpy as np
import torch

from ..data import PolyData
from ..errors import NotFittedError
from ..input_validation import no_more_than_one, one_and_only_one, typecheck
from ..types import Int1dTensor, Number, float_dtype, int_dtype, polydata_type


class Decimation:
    """Decimation class.

    This class implements the quadric decimation algorithm. The goal of
    decimation is to reduce the number of points of a triangular mesh while
    preserving the global aspect of the shape.

    Parameters
    ----------
    target_reduction
        The target rate of triangles to delete during decimation. Must
        be between 0 and 1.
    n_points
        The desired number of points in the decimated mesh.

    Raises
    ------
    InputStructureError
        If both target_reduction and n_points are provided or
        if none of them is provided.

    Examples
    --------
    Decimate a mesh with a target reduction:
    ```python
    import skshapes as sks

    mesh = sks.Sphere()
    decimator = sks.Decimation(target_reduction=0.5)
    decimated_mesh = decimator.fit_transform(mesh)
    ```

    Decimate two meshes with the same connectivity (same triangles):
    ```python
    # assume that pose1 and pose2 are two meshes with the same connectivity:

    pose1, pose2 = sks.PolyData("data/pose1.vtk", "data/pose2.vtk")
    decimator = sks.Decimation(n_points=50)
    decimator.fit(cat1)
    pose1_decimated = decimator.transform(cat1)
    pose2_decimated = decimator.transform(cat2)

    # pose1_decimated and pose2_decimated have the same connectivity
    assert torch.allclose(pose1_decimated.triangles, pose2_decimated.triangles)

    # if landmarks are present in the meshes, they are kept after the
    # decimation
    if pose1.landmarks is not None:
        assert pose1_decimated.landmarks is not None
    if pose2.landmarks is not None:
        assert pose2_decimated.landmarks is not None
    ```

    Decimation is often used through the Multiscale interface.

    """

    @one_and_only_one(parameters=["target_reduction", "n_points", "ratio"])
    @typecheck
    def __init__(
        self,
        *,
        target_reduction: float | None = None,
        n_points: int | None = None,
        ratio: Number | None = None,
    ) -> None:
        if target_reduction is not None:
            if not (0 < target_reduction < 1):
                msg = "target_reduction must be in the range (0, 1)"
                raise ValueError(msg)
            self.target_reduction = target_reduction

        if n_points is not None:
            if n_points <= 0:
                msg = "n_points must be positive"
                raise ValueError(msg)

            self.n_points = n_points

        if ratio is not None:
            if not (0 < ratio < 1):
                msg = "ratio must be in the range (0, 1)"
                raise ValueError(msg)
            self.target_reduction = 1 - ratio

    @typecheck
    def fit(self, mesh: polydata_type) -> Decimation:
        """Fit the decimation algorithm to a mesh.

        Parameters
        ----------
        mesh
            The mesh to fit the decimation object to.

        Raises
        ------
        ValueError
            If the mesh is not triangular.

        Returns
        -------
        Decimation
            self

        """
        if hasattr(self, "n_points"):
            if not (self.n_points <= mesh.n_points):
                msg = "n_points must be lower than mesh.n_points"
                raise ValueError(msg)

            self.target_reduction = 1 - self.n_points / mesh.n_points

        if not mesh.is_triangle_mesh():
            msg = "Quadric decimation only works on triangle meshes"
            raise ValueError(msg)

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
        return self

    @no_more_than_one(
        parameters=[
            "target_reduction",
            "n_points",
            "ratio",
            "n_points_strict",
        ]
    )
    @typecheck
    def transform(
        self,
        mesh: polydata_type,
        *,
        target_reduction: float | None = None,
        n_points: int | None = None,
        n_points_strict: int | None = None,
        ratio: float | None = None,
        return_indice_mapping: bool = False,
    ) -> polydata_type | tuple[polydata_type, Int1dTensor]:
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
            The targeted number of points. Fast but it is not guaranteed that
            the decimation algorithm will exactly reach this number of points.
            If you want to be sure that the decimation algorithm will reach
            this number of points, use n_points_strict instead.
        n_points_strict
            The targeted number of points. This parameter can lead to a slower
            decimation algorithm because the algorithm will try to reach this
            number of points exactly, and this may require many iterations.
        ratio
            The ratio of the number of points of the mesh to decimate over the
            number of points of the mesh used to fit the decimation object.
        return_indice_mapping
            If True, the indice mapping is returned as well as the decimated
            mesh.

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
        if n_points_strict is not None:
            # We want to reach the target number of points exactly
            n_points = n_points_strict
            done = False
            max_iter = 10
            i = 0
            while not done and i < max_iter:
                coarse_mesh = self.transform(mesh=mesh, n_points=n_points)
                if coarse_mesh.n_points == n_points_strict:
                    # We reached the target number of points
                    done = True
                else:
                    # We did not reach the target number of points
                    # We increase the target number of points
                    # before the next iteration
                    n_points += n_points_strict - coarse_mesh.n_points
                i += 1

            if not done:
                # We did not reach the target number of points after max_iter
                # iterations
                raise ValueError(
                    "The decimation algorithm did not reach the target number"
                    + " of points after "
                    + str(max_iter)
                    + " iterations."
                )

            return coarse_mesh

        if target_reduction is None and n_points is None and ratio is None:
            # default, target_reduction is the same as in __init__
            target_reduction = self.target_reduction

        if not hasattr(self, "collapses_") or self.collapses_ is None:
            msg = "The decimation object has not been fitted yet."
            raise NotFittedError(msg)

        if target_reduction is not None:
            if not (0 <= target_reduction <= 1):
                msg = "target_reduction must be in the range (0, 1)"
                raise ValueError(msg)
            ratio = 1 - target_reduction
            n_target_points = int(ratio * self.ref_mesh.n_points)

        elif ratio is not None:
            if not (0 <= ratio <= 1):
                msg = "ratio must be in the range (0, 1)"
                raise ValueError(msg)
            n_target_points = int(ratio * self.ref_mesh.n_points)

        elif n_points is not None:
            if n_points <= 0:
                msg = "n_points must be positive"
                raise ValueError(msg)
            if n_points > self.ref_mesh.n_points:
                msg = "n_points must be lower than mesh.n_points"
                raise ValueError(msg)

            n_target_points = n_points
        else:
            # No target reduction, no n_points, no ratio
            # We use the same target reduction as in __init__
            n_target_points = self.ref_mesh.n_points - len(self.collapses_)

        if mesh.n_points != self.ref_mesh.n_points or not torch.allclose(
            mesh.triangles.cpu(), self.ref_mesh.triangles.cpu()
        ):
            msg = (
                "mesh.n_points and mesh.triangles must be the same as the"
                " the mesh used in fit"
            )
            raise ValueError(msg)

        device = mesh.device

        # Replay the decimation process on the mesh
        points = mesh.points.clone().cpu().numpy().astype(np.float32)
        triangles = mesh.triangles.clone().cpu().numpy().astype(np.int64)

        # Compute the number of collapses to apply
        n_collapses = self.ref_mesh.n_points - n_target_points

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

        # Convert the indice_mapping numpy array to torch tensor
        indice_mapping = torch.Tensor(indice_mapping).to(int_dtype)

        decimated_mesh = PolyData(
            torch.from_numpy(points).to(float_dtype),
            triangles=torch.from_numpy(triangles).to(int_dtype),
            landmarks=landmarks,
            device=device,
        )

        if not return_indice_mapping:
            return decimated_mesh
        else:
            return decimated_mesh, indice_mapping

    @no_more_than_one(
        parameters=["target_reduction", "n_points", "ratio", "n_points_strict"]
    )
    @typecheck
    def fit_transform(
        self,
        mesh: polydata_type,
        *,
        target_reduction: float | None = None,
        n_points: int | None = None,
        n_points_strict: int | None = None,
        ratio: float | None = None,
        return_indice_mapping: bool = False,
    ) -> polydata_type | tuple[polydata_type, Int1dTensor]:
        """Decimate and return decimated mesh."""
        self.fit(mesh)
        kwargs = {
            "target_reduction": target_reduction,
            "n_points": n_points,
            "n_points_strict": n_points_strict,
            "ratio": ratio,
            "return_indice_mapping": return_indice_mapping,
        }
        return self.transform(mesh, **kwargs)

    @typecheck
    @property
    def collapses(self):
        """Returns the collapses of the decimation algorithm."""
        if hasattr(self, "collapses_"):
            return self.collapses_
        else:
            msg = "The decimation object has not been fitted yet."
            raise NotFittedError(msg)

    @typecheck
    @property
    def actual_reduction(self):
        """Returns the actual reduction of the decimation algorithm."""
        if hasattr(self, "actual_reduction_"):
            return self.actual_reduction_
        else:
            msg = "The decimation object has not been fitted yet."
            raise NotFittedError(msg)

    @typecheck
    @property
    def ref_mesh(self):
        """Returns the reference mesh of the decimation algorithm."""
        if hasattr(self, "ref_mesh_"):
            return self.ref_mesh_
        else:
            msg = "The decimation object has not been fitted yet."
            raise NotFittedError(msg)
