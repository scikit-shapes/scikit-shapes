"""
.. _multiscale_landmarks_example:

Multiscaling and landmarks
==========================

The :class:`skshapes.Multiscale<skshapes.multiscaling.multiscale.Multiscale>` class
preserves landmarks across scales.
"""

import pyvista as pv

import skshapes as sks

###############################################################################
# To demonstrate this, we first load a mesh with a collection of key points.

mesh = sks.PolyData(pv.examples.download_louis_louvre().clean())

landmarks = [151807, 21294, 23344, 25789, 131262, 33852, 171465, 191680]
landmarks += [172653, 130895, 9743, 19185, 143397, 200885]

mesh.landmark_indices = landmarks

###############################################################################
# Then, we compute a multiscale representation of our mesh
# using 10%, 1% and 0.1% of the original point count.

ratios = [1, 0.1, 0.01, 0.001]
multiscale = sks.Multiscale(shape=mesh, ratios=ratios)

###############################################################################
# The ``landmark_points`` and ``landmark_indices`` attributes of the shape
# are transported consistently between the different scales.

pl = pv.Plotter()

pl.open_gif("animation.gif", fps=1)
for ratio in ratios:
    mesh_i = multiscale.at(ratio=ratio)
    print(f"with {mesh_i.n_points:,d} points, landmarks = ", mesh_i.landmark_indices)

    pl.clear_actors()
    sks.doc.display(plotter=pl, shape=mesh_i)
    sks.doc.display(plotter=pl, shape=mesh_i.landmark_points, color="red")
    pl.camera_position = "xz"
    pl.camera.zoom(1.4)
    pl.write_frame()

pl.show()
