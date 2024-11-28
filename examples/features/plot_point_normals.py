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

bunny = sks.PolyData(pv.examples.download_bunny())

###############################################################################
# Then, we compute the point normals.

normals = bunny.point_normals()
print(normals)
