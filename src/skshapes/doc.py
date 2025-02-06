"""Utility functions for the documentation."""

import numpy as np
import pyvista as pv
import torch
from matplotlib.collections import LineCollection

from ._data import PolyData


def display(
    shape=None,
    plotter=None,
    title=None,
    color="ivory",
    show_edges=False,
    smooth=False,
    scalars=None,
    cmap="RdBu_r",
    silhouette=True,
    vectors=None,
    vectors_color="skyblue",
    opacity=None,
    point_size=60,
):
    """Uses PyVista to display a 3D shape according to our style guide.

    .. warning::

        This function is used in the documentation and may change
        without notice. It is not meant to be used in production code.
        If you like the style, please feel free to copy the code of this utility
        function and adapt it to your needs.


    .. testcode::

        import torch
        import pyvista as pv
        import skshapes as sks


    .. testcode::
        :hide:

        pv.OFF_SCREEN = True


    .. testcode::

        shape = sks.Sphere()

        # Display the shape as a raw, lowpoly mesh in a new window
        sks.doc.display(shape, title="A sphere", show_edges=True)

        # Display the shape as a smooth surface mesh in a subplot
        pl = pv.Plotter(shape=(1, 2))
        pl.subplot(0, 0)
        shape.point_data["scalars"] = torch.arange(shape.n_points)
        sks.doc.display(shape, plotter=pl, scalars="scalars", smooth=True)
        pl.subplot(0, 1)
        sks.doc.display(shape, plotter=pl, scalars="scalars", smooth=True, cmap="viridis")
        pl.show()

        # Display the shape as a red point cloud
        sks.doc.display(shape.points, color="red")
        print("Done!")

    .. testoutput::

        Done!


    """

    pl = pv.Plotter(window_size=[800, 800]) if plotter is None else plotter

    # If shape is a numpy array or a torch Tensor, convert it to PolyData
    if isinstance(shape, np.ndarray | torch.Tensor):
        shape = PolyData(points=shape)

    if shape.is_triangle_mesh:
        style = "surface"
    elif shape.is_wireframe:
        style = "wireframe"
    else:
        style = "points"

    mesh = shape.to_pyvista()

    # Convert the shape to a PyVista with normals for smooth rendering and silhouette
    if style == "surface":
        mesh = mesh.compute_normals()

        # Simple heuristic to check if the normals point outwards.
        n = mesh.point_normals
        x = mesh.points
        p = x - x.mean(axis=0)
        if np.mean(np.sum(p * n, axis=1)) < 0:
            # If the normals are not correctly oriented, flip them
            mesh = mesh.compute_normals(flip_normals=True)

    if style == "surface":
        if smooth:
            mesh = mesh.subdivide(subfilter="loop", nsub=4)
            material = dict(
                pbr=True,
                metallic=0.0,
                roughness=0.8,
            )
        else:
            material = dict(
                smooth_shading=False,
            )
    elif style == "points":
        material = dict(
            point_size=point_size,
            render_points_as_spheres=True,
        )

    line_width = 0.005 * mesh.length if show_edges else 0

    pl.add_mesh(
        mesh,
        style=style,
        show_edges=show_edges,
        line_width=line_width,
        color=color,
        scalars=scalars,
        cmap=cmap,
        opacity=opacity,
        **material,
    )

    if style == "surface" and opacity is None:
        silhouette_width = pl.shape[0] * 0.0025 * mesh.length
        mesh["silhouette_width"] = silhouette_width * np.ones(mesh.n_points)
        # Now use those normals to warp the surface
        silhouette = mesh.warp_by_scalar(scalars="silhouette_width")

        pl.add_mesh(
            silhouette,
            color="black",
            culling="front",
            interpolation="pbr",
            roughness=1,
        )

    if vectors is not None:
        if not isinstance(vectors_color, str):
            mesh.point_data["vectors_rgba"] = vectors_color.cpu().numpy()

        mesh.point_data["vectors"] = vectors
        arrows = mesh.glyph(
            scale="vectors",
            orient="vectors",
        )

        if isinstance(vectors_color, str):
            pl.add_mesh(arrows, color=vectors_color)
        else:
            pl.add_mesh(arrows, scalars="vectors_rgba")

        if False:
            pl.add_arrows(
                mesh.points,
                vectors.cpu().numpy(),
                mag=1,
                color=vectors_color,
            )

    # elev = 0, azim = 0 is the +x direction
    # elev = 0, azim = 90 is the +y direction
    # elev = 90, azim = 0 is the +z direction
    if True:  # pl.shape != (1, 1):
        pl.remove_all_lights()
        light_intensity = 0.6
        light_elev = 40
        light_azim = 90
        headlight_intensity = 0.5

        if "pbr" in material:
            light_intensity = 2.5
            headlight_intensity = 2.0

        n_lights = np.ceil(light_intensity).astype(int)

        light = pv.Light(
            light_type="camera light",
            intensity=light_intensity / n_lights,
        )
        light.set_direction_angle(light_elev, light_azim)
        for _ in range(n_lights):
            pl.add_light(light)

        n_headlights = np.ceil(headlight_intensity).astype(int)
        light = pv.Light(
            light_type="headlight",
            intensity=headlight_intensity / n_headlights,
        )
        for _ in range(n_headlights):
            pl.add_light(light)

    if title is not None:
        pl.add_text(
            title,
            font_size=12,
            color="black",
            font="times",
            position="upper_edge",
        )
    if scalars is not None:
        pl.remove_scalar_bar()

    # Unfortunately, SSAO is buggy with subplots, as discussed in this open issue:
    # https://gitlab.kitware.com/vtk/vtk/-/issues/18849
    if pl.shape == (1, 1):
        pl.enable_ssao(radius=mesh.length / 16)
        pl.enable_anti_aliasing("ssaa", multi_samples=32)
        # pl.enable_shadows()

    pl.camera_position = "xy"
    pl.enable_parallel_projection()
    pl.camera.zoom(1.2)

    if plotter is None:
        pl.show()


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[
        :, np.newaxis, :
    ]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)
