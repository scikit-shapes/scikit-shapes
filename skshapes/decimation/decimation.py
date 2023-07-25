import numpy as np
import pyvista

# from gudhi.point_cloud.knn import KNearestNeighbors
import numba

from .utils import decimation
from ..data import PolyData
import torch


# from pyvista.core.filters import _get_output
# import vtk
# from vtk.util.numpy_support import vtk_to_numpy


# def _do_decimation(mesh, target_reduction):
#     """Decimate a mesh using vtkQuadricDecimation. and return the decimated mesh,
#     the collapses history and the newpoints
#     we assume that the mesh is a Pyvista PolyData or vtkPolyData

#     Args:
#         mesh (vtkPolyData or pyVista PolyData): the mesh to decimate
#         target_reduction (float): the target reduction

#     Returns:
#         vtkPolyData: the decimated mesh
#         np.ndarray: the collapses history
#         np.ndarray: the newpoints
#     """
#     # Test wether the mesh is a vtkPolyData
#     if not isinstance(mesh, vtk.vtkPolyData):
#         raise TypeError("Input mesh must be a vtkPolyData or pyvista.PolyData")

#     alg = vtk.vtkQuadricDecimation()
#     alg.SetInputData(mesh)
#     alg.SetTargetReduction(target_reduction)
#     alg.Update()

#     output_mesh = _get_output(alg)
#     collapses = vtk_to_numpy(alg.GetSuccessiveCollapses())
#     newpoints = vtk_to_numpy(alg.GetNewPoints())

#     return output_mesh, collapses, newpoints


# @numba.jit(nopython=True, fastmath=True)
def _decimate(points, alphas, collapses_history):
    """
    This function applies the decimation to a mesh that is in correspondence with the reference mesh given the information about successive collapses.

    Args:
        points (np.ndarray): the points of the mesh to decimate.
        alphas (np.ndarray): the list of alpha coefficients such that when (e0, e1) collapses : e0 <- alpha * e0 + (1-alpha) * e1
        collapses_history (np.ndarray): the history of collapses, a list of edges (e0, e1) that have been collapsed. The convention is that e0 is
        the point that remains and e1 is the point that is removed.

    Returns:
        points (np.ndarray): the decimated points.

    """

    for i in range(len(collapses_history)):
        e0, e1 = collapses_history[i]
        points[e0] = alphas[i] * points[e0] + (1 - alphas[i]) * points[e1]

    return points


@numba.jit(nopython=True, fastmath=True)
def _compute_alphas(points, collapses_history, newpoints_history):
    """ """

    alphas = np.zeros(len(collapses_history))

    for i in range(len(collapses_history)):
        e0, e1 = collapses_history[i]

        alpha = np.linalg.norm(newpoints_history[i] - points[e1]) / np.linalg.norm(
            points[e0] - points[e1]
        )
        points[e0] = alpha * points[e0] + (1 - alpha) * points[e1]
        alphas[i] = alpha

    return alphas


class QuadricDecimation:
    """
    This class implements the quadric decimation algorithm and make it possible to run it in parallel.
    """

    def __init__(self, target_reduction=0.5):
        """
        Initialize the quadric decimation algorithm with a target reduction.

        Args:
            target_reduction (float): the target reduction of the mesh.
            use_numba (bool): whether to use numba to speed up the computation when replaying decimation process.
        """
        self.target_reduction = target_reduction

    def fit(self, mesh):
        """
        Fit the quadric decimation algorithm on a reference mesh.
        First, the mesh is decimated with the target reduction (using vtk QuadricDecimation through pyvista wrapper).
        Then, the history of collapses and new points is read from files and the mesh is reconstructed.
        In addition, the information about the decimation process is stored in the object in order to be able to apply the
        same decimation on other meshes that are in correspondence.

        More precisely, the following information are stored:
        - self.collapses_history : the history of collapses, a list of edges (e0, e1) that have been collapsed. The convention is that e0 is
        the point that remains and e1 is the point that is removed.
        - self.alpha the list of alpha coefficients such that when (e0, e1) collapses : e0 <- alpha * e0 + (1-alpha) * e1
        - self.faces : the faces of the decimated mesh (in padding mode)

        Args:
            mesh (pyvista.PolyData): the reference mesh.
        """

        # decimated_mesh = mesh.decimate(
        #     target_reduction=self.target_reduction,
        #     volume_preservation=True,
        #     attribute_error=True,
        # )

        decimated_points, collapses_history, newpoints_history = decimation(
            shape=mesh, target_reduction=self.target_reduction
        )

        alphas = _compute_alphas(
            np.array(mesh.points), collapses_history, newpoints_history
        )

        self.alphas = alphas
        self.collapses_history = collapses_history

    def transform(self, mesh):
        """
        This function applies the decimation to a mesh that is in correspondence with the reference mesh.

        Args:
            mesh (pyvista.PolyData): the mesh to decimate (must be in corresppondance with the fitted one).

        Returns:
            pyvista.PolyData: the decimated mesh.
        """

        points = np.array(mesh.points.clone())
        points = _decimate(
            points, collapses_history=self.collapses_history, alphas=self.alphas
        )

        keep = np.setdiff1d(np.arange(len(points)), self.collapses_history[:, 1])

        # Compute the mapping from original indices to new indices

        # start with identity mapping
        indice_mapping = np.arange(len(points), dtype=int)
        # First round of mapping
        origin_indices = self.collapses_history[:, 1]
        indice_mapping[origin_indices] = self.collapses_history[:, 0]

        previous = np.zeros(len(indice_mapping))
        while not np.array_equal(previous, indice_mapping):
            previous = indice_mapping.copy()
            indice_mapping[origin_indices] = indice_mapping[
                indice_mapping[origin_indices]
            ]

        points = points[keep]

        application = dict([keep[i], i] for i in range(len(keep)))
        indice_mapping = np.array([application[i] for i in indice_mapping])

        # Ugly way to compute new triangles
        if hasattr(mesh, "triangles"):
            triangles_copy = mesh.triangles.clone()

            for c in self.collapses_history:
                origin = c[1]
                destination = c[0]
                triangles_copy[triangles_copy == origin] = destination

            keep_triangle = (
                (triangles_copy[0] != triangles_copy[1])
                * (triangles_copy[1] != triangles_copy[2])
                * (triangles_copy[0] != triangles_copy[2])
            )
            new_triangles = triangles_copy[:, keep_triangle]
            new_triangles = torch.from_numpy(indice_mapping[new_triangles])

        return PolyData(torch.from_numpy(points), triangles=new_triangles)

    # def partial_transform(self, mesh, n_collapses):
    #     """
    #     Apply a partial decimation to a mesh that is in correspondence with the reference mesh. The number of collapses is given by n_collapses.
    #     It returns the decimated mesh as a PolyData with only points/edges structure. In addition, it returns the mapping from the indices of the
    #     point in the original mesh to the indices of the decimated mesh (useful to propagate landmarks).

    #     Args:
    #         mesh (pyvista.PolyData): the mesh to decimate (must be in corresppondance with the fitted one).
    #         n_collapses (int): the number of collapses to apply.

    #     Returns:
    #         pyvista.PolyData: the decimated mesh.
    #         np.ndarray: the mapping from the indices of the point in the original mesh to the indices of the decimated mesh.
    #     """

    #     if n_collapses > len(self.collapses_history):
    #         n_collapses = len(self.collapses_history)

    #     skeleton = mesh.extract_all_edges()  # Extract the skeleton of the mesh

    #     # extract_all_edges() returns a PolyData object with points and lines
    #     # but points are not aligned with the points of the mesh
    #     # we need to align them

    #     # As extract_all_edges() shuffles the points, we need to reorder the edges
    #     knn = KNearestNeighbors(
    #         k=1,
    #         return_distance=False,
    #         return_index=True,
    #         implementation="ckdtree",
    #         n_jobs=-1,
    #     ).fit(skeleton.points)

    #     indices_map = knn.transform(mesh.points)[:, 0]
    #     inverse_map = np.argsort(indices_map)

    #     lines = skeleton.lines.reshape(-1, 3)
    #     edges = inverse_map[lines[:, 1:]]

    #     # Start with the points of the reference mesh
    #     points = mesh.points.copy()

    #     # At this point, the couple (points, edges) is the skeleton of the reference mesh
    #     # with the right order of points to start the decimation

    #     if not self.use_numba:
    #         for i in range(n_collapses):
    #             e0, e1 = self.collapses_history[i]
    #             points[e0] = (
    #                 self.alphas[i] * points[e0] + (1 - self.alphas[i]) * points[e1]
    #             )
    #     else:
    #         points = _decimate(
    #             points,
    #             collapses_history=self.collapses_history[0:n_collapses],
    #             alphas=self.alphas[0:n_collapses],
    #         )

    #     dont_keep = self.collapses_history[0:n_collapses, 1]
    #     keep = np.setdiff1d(np.arange(len(points)), dont_keep)

    #     indice_mapping = np.arange(len(points), dtype=int)
    #     indice_mapping[dont_keep] = self.collapses_history[0:n_collapses, 0]

    #     previous = np.zeros(len(indice_mapping))
    #     while not np.array_equal(previous, indice_mapping):
    #         previous = indice_mapping.copy()
    #         indice_mapping[dont_keep] = indice_mapping[indice_mapping[dont_keep]]

    #     points = points[keep]

    #     tmp = {keep[i]: i for i in range(len(keep))}
    #     f = lambda x: tmp[x]
    #     indice_mapping = np.vectorize(f)(indice_mapping)

    #     # Apply the mapping to the edges id
    #     updated_edges = indice_mapping[edges]

    #     # Remove the (i, i) edges
    #     edges_to_remove_id = np.argwhere(
    #         (updated_edges[:, 0] - updated_edges[:, 1]) == 0
    #     ).T[0]
    #     edges_to_remove_id
    #     updated_edges = np.delete(updated_edges, edges_to_remove_id, axis=0)

    #     # Convert to pyVista lines format
    #     lines = (
    #         np.hstack((np.ones((len(updated_edges), 1)) * 2, updated_edges))
    #         .astype(int)
    #         .reshape(-1)
    #     )

    #     return pyvista.PolyData(points, lines=lines), indice_mapping


if __name__ == "__main__":
    import os
    from time import time

    # Test the parallel quadric decimation
    datafolder = "/home/louis/Environnements/singularity_homes/keops-full/GitHub/scikit-shapes-draft/data/SCAPE/scapecomp/"
    filenames = [
        f
        for f in os.listdir(datafolder)
        if f.endswith(".ply") and "shaked" not in f and "mesh" in f
    ]
    meshes = [pyvista.read(datafolder + "/" + f) for f in filenames]
    print("Number of meshes: ", len(meshes))
    reference = meshes[0]
    d = QuadricDecimation(target_reduction=0.99, use_numba=True)

    start = time()
    d.fit(reference)
    print("Fitting time: ", time() - start)

    target_folder = "/home/louis/data/SCAPE/low_resolution/"

    transform_time = 0.0
    save_time = 0.0
    for i in range(len(meshes)):
        if meshes[i].n_points == 12500:
            start = time()
            decimated_mesh = d.transform(meshes[i])
            transform_time += time() - start
            start = time()
            decimated_mesh.save(target_folder + "/" + filenames[i])
            save_time += time() - start

    print("Transform time: ", transform_time)
    print("Save time: ", save_time)

    decimated_mesh.plot()
