"""
Rigid registration of 3D meshes
===============================

In this example, we demonstrate how to perform rigid registration (also known
alignment) of two 3D meshes. We also show how to improve the registration by
adding landmarks.
"""

# %%
# ## Load data
#
# We load two meshes from the `pyvista` example datasets. We then rescale them
# to lie in the unit cube to avoid dealing with scale issues.

import pyvista as pv
import skshapes as sks
import torch
from pyvista import examples

shape1 = sks.PolyData(examples.download_human())
shape2 = sks.PolyData(examples.download_doorman())
shape1.point_data.clear()
shape2.point_data.clear()

bounds = (
    lambda shape: torch.max(shape.points, dim=0).values
    - torch.min(shape.points, dim=0).values
)

lims1 = bounds(shape1)
lims2 = bounds(shape2)
rescale1 = torch.max(lims1)
shape1.points -= torch.min(shape1.points, dim=0).values
shape1.points /= rescale1

rescale2 = torch.max(lims2)
shape2.points -= torch.min(shape2.points, dim=0).values
shape2.points /= rescale2

# %%
# ## Plot the data
#
# Let us have a look at the two shapes we want to align. Clearly, they are not
# aligned and a rigid transformation is needed.

plotter = pv.Plotter()
plotter.add_mesh(shape1.to_pyvista())
plotter.add_mesh(shape2.to_pyvista())
plotter.show()

# %%
# ## Apply the registration
#
# We now apply the registration. The meshes points are not in correspondence
# and we need to use a loss function that can handle this. We use the nearest
# neighbors loss.

from skshapes.loss import NearestNeighborsLoss
from skshapes.morphing import RigidMotion
from skshapes.tasks import Registration

loss = NearestNeighborsLoss()
# The parameter n_steps is the number of steps for the motion. For a rigid
# motion, it has no impact on the result as the motion is fully determined by
# a rotation matrix and a translation vector. It is however useful for
# creating a smooth animation of the registration.
model = RigidMotion(n_steps=5)

registration = Registration(
    model=model,
    loss=loss,
    n_iter=2,
    verbose=True,
)

registration.fit(source=shape2, target=shape1)

# %%
# ## Plot an animation of the registration
#
# We can now plot an animation of the registration. We observe that the
# registration is not perfect. This is due to the fact that the nearest
# neighbors loss tries to match the points of the two shapes. However, the
# shapes are not in correspondence and our model gather no information about
# the correspondence. We can improve the registration by adding
# a few landmarks.

path = registration.path_
plotter = pv.Plotter()
plotter.add_mesh(shape1.to_pyvista())
actor = plotter.add_mesh(shape2.to_pyvista())
plotter.open_gif("rigid_registration.gif", fps=3)
for i, shape in enumerate(path):
    plotter.remove_actor(actor)
    actor = plotter.add_mesh(shape.to_pyvista())
    plotter.write_frame()

plotter.show()

# %%
# ## Add landmarks
#
# We now add landmarks to the two shapes. Three landmarks (head, left hand,
# right hand) are enough to greatly improve the registration.

landmarks1 = [5199, 2278, 10013]
landmarks2 = [325, 786, 509]

colors = ["red", "green", "blue"]
plotter = pv.Plotter()
plotter.add_mesh(shape1.to_pyvista())
for i in range(3):
    plotter.add_points(
        shape1.points[landmarks1[i]].numpy(),
        color=colors[i],
        render_points_as_spheres=True,
        point_size=25,
    )
plotter.add_mesh(shape2.to_pyvista())
for i in range(3):
    plotter.add_points(
        shape2.points[landmarks2[i]].numpy(),
        color=colors[i],
        render_points_as_spheres=True,
        point_size=25,
    )
plotter.show()

# %%
# ## Apply the registration with landmarks
#
# We now apply the registration. We use the landmarks to define a landmark
# loss. We use the nearest neighbors loss to match the rest of the points.
# We use the same rigid motion model.

shape1.landmark_indices = landmarks1
shape2.landmark_indices = landmarks2

loss_landmarks = NearestNeighborsLoss() + sks.LandmarkLoss()

registration = Registration(
    model=model,
    loss=loss_landmarks,
    n_iter=2,
    verbose=True,
)

registration.fit(source=shape2, target=shape1)


# %%
# ## Animation of the registration with landmarks
#

path = registration.path_
plotter = pv.Plotter()
plotter.add_mesh(shape1.to_pyvista())
actor = plotter.add_mesh(shape2.to_pyvista())
plotter.open_gif("rigid_registration.gif", fps=3)
for i, shape in enumerate(path):
    plotter.remove_actor(actor)
    actor = plotter.add_mesh(shape.to_pyvista())
    plotter.write_frame()

plotter.show()
