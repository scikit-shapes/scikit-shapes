"""
Transfer meshes between scikit-shapes and PyVista
=================================================

[PyVista](https://docs.pyvista.org/version/stable/) is a Python library for
3D visualization and mesh manipulation. It is a python wrapper of the
[VTK](https://vtk.org/) library. PyVista is used in scikit-shapes to load and
save meshes in various formats.

Scikit-shapes [PolyData](skshsapes.data.polydata.PolyData) can be exported to
PyVista PolyData, and vice versa. This is useful to use the PyVista plotting
capabilities.

In this example, we show how to transfer data from scikit-shapes to PyVista and
back. We also show that point data and landmarks are preserved during the
conversion.
"""

# %%
# ## Load a mesh from PyVista
#
# We load a quadrangulated cow mesh from PyVista. As scikit-shapes meshes are
# triangular, an automatic triangulation is performed when importing the mesh.

import pyvista as pv
import torch
from pyvista import examples

import skshapes as sks

mesh_pyvista = examples.download_cow()
mesh_sks = sks.PolyData(mesh_pyvista)

assert mesh_pyvista.n_points == mesh_sks.n_points

# %%
# ## Add point data to the mesh

mesh_sks["signal"] = mesh_sks.points[:, 0]
# Now, export the mesh to PyVista
mesh_pv2 = mesh_sks.to_pyvista()
# The signal is transferred to the PyVista mesh as a point data array
# with the same name and can be plotted
mesh_pv2.plot(scalars="signal", cpos="xy")

# Back to scikit-shapes, the signal is preserved
mesh_sks_back = sks.PolyData(mesh_pv2)
assert torch.allclose(mesh_sks_back["signal"], mesh_sks["signal"])

# %%
# ## Landmarks are also preserved
#

# Set some landmarks
landmarks_indices = [0, 10, 154, 125, 1544, 187, 32, 252, 1214]
mesh_sks.landmark_indices = landmarks_indices

# Export to PyVista
mesh_pv3 = mesh_sks.to_pyvista()

# The landmarks are stored in the field data of the PyVista mesh
# there can ba accessed as a 3D point cloud with the name "landmark_points"
plotter = pv.Plotter()
plotter.add_mesh(mesh_pv3, color="w")
plotter.add_points(
    mesh_pv3.field_data["landmark_points"],
    color="r",
    point_size=10,
    label="landmarks",
)
plotter.add_legend()
plotter.view_xy()
plotter.show()

# Back to scikit-shapes, the landmarks are preserved
mesh_sks_back = sks.PolyData(mesh_pv3)
assert torch.allclose(
    mesh_sks_back.landmark_indices, mesh_sks.landmark_indices
)
