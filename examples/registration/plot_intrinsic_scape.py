"""
Example of elastic registration
===============================
This examples shows the registration of two triangle meshes using an intrinsic
registration model.
"""

# %%
# Load useful packages

import sys

import pykeops
import pyvista as pv

import skshapes as sks

sys.path.append(pykeops.get_build_folder())

# %%
# ## Load and align data


source = sks.Sphere()
target = sks.Sphere()

source.landmark_indices = [0, 1, 2, 3]
target.landmark_indices = [0, 1, 2, 3]

# source = sks.PolyData("../../data/scape/mesh044.ply")
# target = sks.PolyData("../../data/scape/mesh049.ply")
# source.landmark_indices = [6613, 6657, 9346, 9187]
# target.landmark_indices = [6613, 6657, 9346, 9187]

model = sks.RigidMotion()
loss = sks.LandmarkLoss()
optimizer = sks.LBFGS()

rigid_registration = sks.Registration(
    model=model,
    loss=loss,
    optimizer=optimizer,
    verbose=True,
    n_iter=3,
)

source = rigid_registration.fit_transform(source=source, target=target)


plotter = pv.Plotter()
plotter.add_mesh(source.to_pyvista(), color="tan")
plotter.add_mesh(target.to_pyvista(), color="purple", opacity=0.5)
plotter.show()

# %%
# ## Decimate

decimation_module = sks.Decimation(n_points=200)
decimation_module.fit(source)

source = decimation_module.transform(source)
target = decimation_module.transform(target)

plotter = pv.Plotter()
plotter.add_mesh(source.to_pyvista(), color="tan")
plotter.add_mesh(target.to_pyvista(), color="purple", opacity=0.5)
plotter.show()

# %%
# ## Intrinsic registration

model = sks.IntrinsicDeformation(
    n_steps=20,
    metric=sks.AsIsometricAsPossible(),
)

loss = sks.L2Loss()
optimizer = sks.LBFGS()

intrinsic_registration = sks.Registration(
    model=model,
    loss=loss,
    optimizer=optimizer,
    verbose=True,
    n_iter=5,
    regularization_weight=0,
)

intrinsic_registration.fit(source=source, target=target)
path = intrinsic_registration.path_

# plotter = pv.Plotter(shape=(1, 3))
# plotter.subplot(0, 0)
# plotter.add_mesh(target.to_pyvista(), color="purple", opacity=0.5)
# plotter.add_mesh(source.to_pyvista(), color="tan")
# plotter.subplot(0, 1)
# plotter.add_mesh(target.to_pyvista(), color="purple", opacity=0.5)
# plotter.add_mesh(path[14].to_pyvista(), color="tan")
# plotter.subplot(0, 2)
# plotter.add_mesh(target.to_pyvista(), color="purple", opacity=0.5)
# plotter.add_mesh(path[-1].to_pyvista(), color="tan")
# plotter.show()

# plotter = pv.Plotter()
# plotter.add_mesh(target.to_pyvista(), color="purple", opacity=0.5)
# plotter.add_mesh(path[14].to_pyvista(), color="tan")
# plotter.show()

# %%
# ## Add regularization

intrinsic_registration = sks.Registration(
    model=model,
    loss=loss,
    optimizer=optimizer,
    verbose=True,
    n_iter=5,
    regularization_weight=100,
)

intrinsic_registration.fit(source=source, target=target)
path = intrinsic_registration.path_

# plotter = pv.Plotter(shape=(1, 3))
# plotter.subplot(0, 0)
# plotter.add_mesh(target.to_pyvista(), color="purple", opacity=0.5)
# plotter.add_mesh(source.to_pyvista(), color="tan")
# plotter.subplot(0, 1)
# plotter.add_mesh(target.to_pyvista(), color="purple", opacity=0.5)
# plotter.add_mesh(path[14].to_pyvista(), color="tan")
# plotter.subplot(0, 2)
# plotter.add_mesh(target.to_pyvista(), color="purple", opacity=0.5)
# plotter.add_mesh(path[-1].to_pyvista(), color="tan")
# plotter.show()

plotter = pv.Plotter()
plotter.add_mesh(target.to_pyvista(), color="purple", opacity=0.5)
plotter.add_mesh(path[14].to_pyvista(), color="tan")
plotter.show()
