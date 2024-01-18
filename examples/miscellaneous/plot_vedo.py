"""
Transfer meshes between scikit-shapes and Vedo
==============================================

[Vedo](https://vedo.embl.es/docs/vedo.html) is a Python library for
3D visualization, interactive applications creation and mesh manipulation.

Scikit-shapes [PolyData](skshsapes.data.polydata.PolyData) can be exported to
Vedo, and vice versa.

In this example, we show how to transfer data from scikit-shapes to Vedo and
back. We also show that point data and landmarks are preserved during the
conversion.
"""

# %%
# ## Load a mesh from Vedo
# 
# We load a quadrangulated cow mesh from Vedo. As scikit-shapes meshes are
# triangular, an automatic triangulation is performed when importing the mesh.

import vedo
import skshapes as sks
import torch

mesh_vedo = vedo.Mesh(vedo.dataurl+"spider.ply")
mesh_sks = sks.PolyData(mesh_vedo)

assert mesh_vedo.nvertices == mesh_sks.n_points

# %%
# ## Add point data to the mesh

mesh_sks["signal"] = mesh_sks.points[:, 0]
# Now, export the mesh to Vedo
mesh_vedo2 = mesh_sks.to_vedo()
# The signal is transfered to the Vedo mesh as a point data array
# with the same name and can be plotted
mesh_vedo2.cmap("jet", mesh_vedo2.pointdata["signal"])
vedo.show(mesh_vedo2, offscreen=True)

# Back to scikit-shapes, the signal is preserved
mesh_sks_back = sks.PolyData(mesh_vedo2)
assert torch.allclose(mesh_sks_back["signal"], mesh_sks["signal"])

# %%
# ## Landmarks are also preserved
#

# Set some landmarks
landmarks_indices = [0, 1500, 2400, 685, 4669, 3100, 2000]
mesh_sks.landmark_indices = landmarks_indices

# Export to Vedo
mesh_vedo3 = mesh_sks.to_vedo()

# The landmarks are stored in the field data of the Vedo mesh
# there can ba accessed as a 3D point cloud with the name "landmark_points"
landmark_points = mesh_vedo3.metadata["landmark_points"]
vedo.show(
    [mesh_vedo3.c('w', alpha=0.8),
    vedo.Points(landmark_points, c='r', r=20)],
    offscreen=True
    )

# Back to scikit-shapes, the landmarks are preserved
mesh_sks_back = sks.PolyData(mesh_vedo3)
assert torch.allclose(
    mesh_sks_back.landmark_indices,
    mesh_sks.landmark_indices
)
