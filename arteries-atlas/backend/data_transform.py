import numpy as np
import pyvista as pv
import scipy


def add_features(polydata: pv.PolyData, features: dict) -> None:
    """
    Add data to a polydata object from the values specified in the dictionary input.

    Parameters
    ----------
    polydata: pv.PolyData
        Polydata structure to add features to, containing N points.

    features: dict
        Dictionary of (N) np.ndarray containing the features to add to the Polydata.

    """
    for ft in features:
        polydata[ft] = features[ft]  # TODO harmoniser shape


def graph_polyline(adjmatrix: scipy.sparse.csr_array, pos: np.ndarray, features: dict | None = None) -> pv.PolyData:
    """
    Convert a graph to a pv.Polydata object, with features given by the provided dictionary.

    Parameters
    ----------
    adjmatrix: (N, N) scipy.sparse.csr_matrix of bool
        Adjacency matrix of the graph.

    pos: (N, D) np.ndarray of np.float32
        Position of the graph nodes. If D==2, the last coordinate will be set to 0 in order to obtain a 3D point set.

    features: dict
        Dictionary of (N) np.ndarray containing features to add to the Polydata.

    Returns
    -------
    polyline: pv.PolyData
        Polydata structure containing N points.

    """
    size, dim = pos.shape

    adjmatrix_coo = scipy.sparse.tril(adjmatrix + adjmatrix.T).tocoo()

    edges = np.stack([adjmatrix_coo.col, adjmatrix_coo.row], axis=1)
    edges = np.ravel([np.full(shape=(edges.shape[0]), fill_value=2, dtype=int), edges[:, 0], edges[:, 1]], order='F')

    polyline = pv.PolyData()
    polyline.points = np.hstack([pos, np.zeros((size, 1))]) if dim == 2 else pos
    polyline.lines = edges

    if features is not None:
        add_features(polyline, features)

    return polyline


def tree_polyline(branches, pos, features=None):  # TODO ajouter docstring
    size, dim = pos.shape

    edges = np.concatenate([[br[2] - br[1] + 1, br[0], *list(range(br[1], br[2]))] for br in branches])

    polyline = pv.PolyData()
    polyline.points = np.hstack([pos, np.zeros((size, 1))]) if dim == 2 else pos
    polyline.lines = edges

    if features is not None:
        add_features(polyline, features)

    return polyline


def polyline_tube(polyline, radius_ft=None, width=0.5):  # TODO ajouter docstring
    if radius_ft is not None:
        tube = polyline.tube(scalars=radius_ft, absolute=True)
        tube["silhouette_width"] = width * np.median(tube[radius_ft]) * np.ones(tube.n_points)
    else:
        tube = polyline.tube(radius=1, absolute=True)
        tube["silhouette_width"] = width * np.ones(tube.n_points)
    return tube
