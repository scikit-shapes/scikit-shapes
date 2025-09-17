import numpy as np
import pyvista as pv
import scipy

from .data_transform import graph_polyline, polyline_tube, tree_polyline


def plot_polydata(polydata: pv.PolyData, pl: pv.Plotter = None, **kwargs) -> pv.Plotter:
    """
    Return a pv.Plotter object where the input polydata has been added as a volume.

    Parameters
    ----------
    polydata: pv.PolyData
        PolyData to plot.

    pl: pv.Plotter, optional
        Plotter on which to add the image.

    **kwargs
        Extra arguments to provide to the `pl.add_mesh` method.

    Returns
    -------
    pl: pv.Plotter
        Plotter where the polydata has been plotted.

    """
    if pl is None:
        pl = pv.Plotter(border=True, border_width=5)

    pl.add_mesh(polydata, **kwargs)

    return pl


def plot_contoured_polyline(polyline: pv.PolyData, dim: int = 3, pl: pv.Plotter = None,
                            ambient=0.8, diffuse=1, **kwargs) -> pv.Plotter:
    """
    Return a pv.Plotter object where a tube representing the input 2D polydata has been added.

    Parameters
    ----------
    polyline: pv.PolyData
        PolyData to plot, representing a 1D shape in a 3D space.

    dim: int, default=3
        Original dimension of the represented shape. If dim==2, the camera will be adjusted to plot only the dimensions
        'x' and 'y' of the space.

    pl: pv.Plotter, optional
        Plotter on which to add the image.

    **kwargs
        Extra arguments to provide to the `plot_polydata` function.

    Returns
    -------
    pl: pv.Plotter
        Plotter where a tube representing the polydata has been plotted.

    """
    tube = polyline_tube(polyline, 'radius') if "radius" in polyline.array_names else polyline_tube(polyline)

    tube_inflated = tube.warp_by_scalar(scalars="silhouette_width")

    scalars = "color" if "color" in polyline.array_names else None

    pl = plot_polydata(tube,
                       pl=pl,
                       scalars=scalars,
                       color='grey',
                       interpolation="gouraud",
                       roughness=1,
                       ambient=ambient,
                       diffuse=diffuse,
                       smooth_shading=True,
                       **kwargs
                       )

    pl = plot_polydata(tube_inflated,
                       pl=pl,
                       color="black",
                       culling="front",
                       interpolation="pbr",
                       roughness=1)

    pl.enable_shadows()
    pl.enable_anti_aliasing("ssaa", all_renderers=False)

    if dim == 2:
        pl.camera_position = "xy"
        pl.camera.azimuth = 0
        pl.camera.elevation = 0

    pl.camera.zoom(1.4)

    return pl


def plot_graph(adjmatrix: scipy.sparse.csr_matrix, pos: np.ndarray, radius: np.ndarray = None,
               colors: np.ndarray = None, pl: pv.Plotter = None, **kwargs) -> pv.Plotter:
    """
    Return a pv.Plotter object representing a graph.

    Parameters
    ----------
    adjmatrix: (N, N) scipy.sparse.csr_matrix of bool
        Adjacency matrix of the graph.

    pos: (N, D) np.ndarray of np.float32
        Position of the graph nodes. If D==2, the last coordinate will be set to 0 in order to obtain a 3D point set.

    radius: (N) np.ndarray of np.float32, optional # TODO harmoniser shape
        Radius of each node of the graph.

    colors: (N) np.ndarray of np.float32, optional # TODO harmoniser shape
        Value used to define the colors of the nodes.

    pl: pv.Plotter, optional
        Plotter on which to add the image.

    **kwargs
        Extra arguments to provide to the plot.

    Returns
    -------
    pl: pv.Plotter
        Plotter where a tube representing the polydata has been plotted.

    """
    features = {}
    if radius is not None:
        features['radius'] = radius

    if colors is not None:
        features['color'] = colors

    polyline = graph_polyline(adjmatrix, pos, features)
    return plot_contoured_polyline(polyline, dim=pos.shape[1], pl=pl, **kwargs)


def plot_branches(branches, pos, radius=None, colors=None, pl=None, **kwargs):  # TODO docstring
    features = {}
    if radius is not None:
        features['radius'] = radius

    if colors is not None:
        features['color'] = colors

    polyline = tree_polyline(branches, pos, features)

    return plot_contoured_polyline(polyline, dim=pos.shape[1], pl=pl, **kwargs)
