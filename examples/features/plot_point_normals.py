"""
.. _point_normals_example:

How to compute point normals
==================================

We use the :meth:`~skshapes.data.polydata.PolyData.point_normals` method to compute normal vectors at each vertex of a triangle mesh. This is useful for estimating the curvature of the surface.
"""

###############################################################################
# First, we load the Stanford bunny as a triangle mesh.

import pyvista as pv

import skshapes as sks

mesh = sks.PolyData(pv.examples.download_bunny()).resample(n_points=1000)

###############################################################################
# Then, we compute the point normals.

normals = mesh.point_normals()

sks.doc.display(shape=mesh, vectors=0.01 * normals, title="Surface normals")
