"""Utility functions for the documentation."""

import numpy as np
import pyvista as pv
import torch

from .data import PolyData


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
    style="surface",
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
            point_size=20,
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
        **material,
    )

    if style == "surface":
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
            light_intensity = 3.0
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
        pl.enable_ssao(radius=1)
        # pl.enable_shadows()

    pl.camera_position = "xy"
    pl.enable_parallel_projection()
    pl.camera.zoom(1.2)

    if plotter is None:
        pl.show()
