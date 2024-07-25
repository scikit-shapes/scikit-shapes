"""
PolyData conversion from and to PyVista or Vedo
===============================================

`PyVista <https://docs.pyvista.org/version/stable/>`_ is a Python library for
3D visualization and mesh manipulation. It is a python wrapper of the
`VTK <https://vtk.org/>`_ library. PyVista is used in scikit-shapes as a backend
to load and save meshes in various formats.

`Vedo <https://vedo.embl.es/>`_ is another Python library for 3D visualization.
It is also a python wrapper of the VTK library. It is used in scikit-shapes for
interactive visualization.

Scikit-shapes [PolyData](skshsapes.data.polydata.PolyData) can be exported to
PyVista PolyData or Vedo Mesh, and vice versa. This allows to leverage the
visualization and mesh manipulation capabilities of both libraries.

In addition to the mesh geometry, the signals and landmarks are encoded in the
PyVista or Vedo mesh, and are preserved when the mesh is loaded back from PyVista.
"""

###############################################################################
# Load a mesh from PyVista's examples
# -----------------------------------
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

###############################################################################
# Preservation of signals and landmarks
# -------------------------------------
#
# The methods `.to_pyvista()` and `.to_vedo()` allow to convert a `PolyData` object
# to a PyVista or Vedo mesh. The point data and landmarks are preserved during
# the conversion: if the mesh is loaded back from PyVista or Vedo, the signals
# and landmarks are still there.
#
# The following illustrates the preservation of signals and landmarks when
# converting a `PolyData` object to PyVista and back.


mesh_sks.point_data["signal"] = mesh_sks.points[:, 0]
# Now, export the mesh to PyVista
mesh_pv2 = mesh_sks.to_pyvista()
# The signal is transferred to the PyVista mesh as a point data array
# with the same name and can be plotted
mesh_pv2.plot(scalars="signal", cpos="xy")

# Back to scikit-shapes, the signal is preserved
mesh_sks_back = sks.PolyData(mesh_pv2)
assert torch.allclose(
    mesh_sks_back.point_data["signal"],
    mesh_sks.point_data["signal"]
    )


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
