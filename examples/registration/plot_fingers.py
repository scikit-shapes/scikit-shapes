"""
Example of registration with LDDMM
==================================
This examples shows the registration of two triangle meshes using an extrinsic
LDDMM registration model.
"""

# %%
# Load useful packages

import sys

import pykeops
import pyvista as pv
import skshapes as sks

sys.path.append(pykeops.get_build_folder())

# %%
# Prepare the data

source = sks.PolyData("data/fingers/finger0.ply")
target = sks.PolyData("data/fingers/finger1.ply")

source.control_points = source.bounding_grid(N=15, offset=0.15)

# %%
# Prepare the model

gpu = False
n_steps = 3
times_sks = []
times_torchdiffeq = []

loss = sks.L2Loss()
optimizer = sks.LBFGS()

model = sks.ExtrinsicDeformation(
    n_steps=n_steps,
    kernel=sks.GaussianKernel(sigma=0.4),
    control_points=True,
    backend="torchdiffeq",
    solver="rk4",
)

registration = sks.Registration(
    model=model,
    loss=loss,
    optimizer=optimizer,
    verbose=True,
    gpu=gpu,
    n_iter=3,
    regularization=1,
)

# %%
# Fit the registration model

registration.fit(source=source, target=target)
path = registration.path_

# %%
# Visualize the result

cp = source.control_points.copy()
morphed_cp = model.morph(
    cp, parameter=registration.parameter_, return_path=False
).morphed_shape

cpos = [
    (1.0991653431118906, -17.125557106885555, 1.2467166624933634),
    (2.4676010608673096, 2.751017451286316, 1.7395156361162663),
    (0.04440371592867984, -0.027815943269483327, 0.9986263481962382),
]

plotter = pv.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_mesh(target.to_pyvista(), color="purple", opacity=0.5)
plotter.add_mesh(source.to_pyvista(), color="tan")
plotter.add_mesh(cp.to_pyvista(), color="red", show_edges=True, opacity=0.5)
plotter.camera_position = cpos
plotter.subplot(0, 1)
plotter.add_mesh(target.to_pyvista(), color="purple", opacity=0.5)
plotter.add_mesh(path[-1].to_pyvista(), color="tan")
plotter.add_mesh(
    morphed_cp.to_pyvista(), color="red", show_edges=True, opacity=0.5
)
plotter.camera_position = cpos
plotter.show()
