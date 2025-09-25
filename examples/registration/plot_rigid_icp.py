"""
The `RigidDeformation` class: rigid deformation model
================================================================

This notebook describes the :class:`RigidDeformation` class,
the common class for using rigid deformation model (rigid ICP).

- Initialize a ClosestPointCoupling and a RigidDeformation object
- Fit a rigid transformation to a source and target shapes
- Transform the source shape with the fitted transformation

"""

###############################################################################
# Load data
# ----------
#
# We load the dragon mesh from the `pyvista` example datasets. This is the source shape.
# We then use reconstruction to create a target shape from the source shape points but with a different tiangulation than the source shape.
# We then apply a rotation and a translation to the target shape to simulate a misalignment between the source and target shapes.

import pyvista as pv
import torch
from pyvista import examples

from skshapes.morphing.rigid_icp import ClosestPointCoupling, RigidDeformation

dragon_source = examples.download_dragon()
dragon_source = dragon_source.decimate_pro(0.9) # Decimate the mesh to reduce the number of points for faster processing.
dragon_target_points = pv.PolyData(dragon_source.points.copy())
dragon_target = dragon_target_points.reconstruct_surface(nbr_sz=20)
dragon_target = dragon_target.rotate_z(40).rotate_x(-20).rotate_y(100).translate([0.05, 0.1, 0.1])

###############################################################################
# Plot the data
# ----------------------

plotter = pv.Plotter()
plotter.add_mesh(dragon_source, color="green", label="Source", show_edges=True)
plotter.add_mesh(dragon_target, color="red", label="Target", show_edges=True)
plotter.add_legend()
plotter.show()

###############################################################################
# Initialize a RigidDeformation object
# --------------------------------------
# We initialize a `ClosestPointCoupling` object to find the closest points
# between the source and target shapes. We then initialize a `RigidDeformation`
# object with the coupling and some parameters.

coupling = ClosestPointCoupling()
coupling.initialize(target=torch.Tensor(dragon_target.points.copy()), leaf_size=40)

rigid_deformation = RigidDeformation(
    coupling=coupling,
    initialization="fpfh",
    robust_loss="cauchy_mad",
    metric_type="point_to_plane",
    tolerance=1e-4,
    scale=False,
    max_iterations=50
)

rigid_deformation.initialize(dragon_source, dragon_target)

###############################################################################
# Fit a rigid transformation to a source and target shapes
# -----------------------------------------------------------

result = rigid_deformation.fit()
transformed_shape = rigid_deformation.transform()
print("Estimated Transformation Matrix:", rigid_deformation.transformation_matrix)

###############################################################################
# Plot the result
# ----------------------

plotter = pv.Plotter()
plotter.add_mesh(dragon_target, color="red", opacity=0.5, label="Target")
plotter.add_mesh(transformed_shape, color="blue", opacity=0.5, label="Transformed")
plotter.add_legend()
plotter.show()
