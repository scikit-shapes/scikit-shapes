"""
LDDMM with normalized kernel
============================

This notebook illustrates the interest of normalizing the cometric in the LDDMM
model. We consider the registration of two spheres that differ by a translation
first, and then a more complex registration task with a translation and a deformation.

Without normalization, the carpooling artifact occurs: the sphere
is contracted, then translated and finally expanded. in this situation, event if
the morphed shape matches the target, the intermediate shapes are not meaningful
and the extrapolation is not reliable.

Normalizing the kernel adds regularization to the morphing, leanding to prevention
of the carpooling artifact and improvement of the extrapolation to some extent.

Options for normalization are:

* "rows": normalize the rows of the kernel
* "columns": normalize the columns of the kernel
* "both": normalize both the rows and the columns of the kernel (for quare kernels, algorithm 5.7 in https://www.jeanfeydy.com/geometric_data_analysis.pdf)

Further explanation can be found in the p177 and onwards of https://www.jeanfeydy.com/geometric_data_analysis.pdf.

"""

from time import time

import pyvista as pv
import torch

import skshapes as sks

# sphinx_gallery_thumbnail_number = 9

###############################################################################
# Load data

plot_kwargs = {
    "smooth_shading": True,
    "pbr": True,
    "metallic": 0.7,
    "roughness": 0.6,
}

cpos = [
    (1.6256104086078755, -9.701422233882411, 1.3012755902068773),
    (1.191160019984921, 0.01901107976782581, -0.0052552929581526076),
    (0.006053690112347382, 0.13347614338229413, 0.9910335372649167),
]

source = sks.Sphere()
target = sks.Sphere()

decimation = sks.Decimation(n_points=200)
source = decimation.fit_transform(source)
target = decimation.transform(target)

target.points = target.points + torch.tensor([2, 0.0, 0.0])

plotter = pv.Plotter()
plotter.add_mesh(source.to_pyvista(), color="teal", opacity=0.8, **plot_kwargs)
plotter.add_mesh(target.to_pyvista(), color="red", opacity=0.8, **plot_kwargs)
plotter.camera_position = cpos
plotter.show()


###############################################################################
# LDDM without normalization
# --------------------------

model = sks.ExtrinsicDeformation(
    n_steps=4,
    kernel="gaussian",
    scale=0.3,
)

loss = sks.L2Loss()

task = sks.Registration(
    model=model,
    loss=loss,
    optimizer=sks.LBFGS(),
    n_iter=3,
    regularization_weight=1e-1,
    verbose=True,
)

start = time()
task.fit(source=source, target=target)
print("Elapsed time: ", time() - start)
path = task.path_

plotter = pv.Plotter()
plotter.add_mesh(source.to_pyvista(), color="teal", opacity=0.2, **plot_kwargs)
plotter.add_mesh(target.to_pyvista(), color="red", opacity=0.2, **plot_kwargs)
for i in range(len(path)):
    plotter.add_mesh(
        path[i].to_pyvista(), color="tan", opacity=0.8, **plot_kwargs
    )
plotter.camera_position = cpos
plotter.show()


###############################################################################
# Extrapolation

back = model.morph(
    shape=source,
    parameter=task.parameter_,
    final_time=-1.0,
    return_path=True,
).path

model.n_steps = 8
forward = model.morph(
    shape=source,
    parameter=task.parameter_,
    final_time=2.0,
    return_path=True,
).path

path = back[::-1] + forward[1:]

plotter = pv.Plotter()
plotter.open_gif("lddmm_no_normalization.gif", fps=4)
for i in range(len(path)):
    plotter.clear_actors()
    plotter.add_mesh(
        source.to_pyvista(), color="teal", opacity=0.2, **plot_kwargs
    )
    plotter.add_mesh(
        target.to_pyvista(), color="red", opacity=0.2, **plot_kwargs
    )
    plotter.add_mesh(
        path[i].to_pyvista(), color="tan", opacity=0.8, **plot_kwargs
    )
    plotter.camera_position = cpos
    plotter.write_frame()
plotter.close()

###############################################################################
# Normalizing the rows of the kernel
# ----------------------------------

model_norm = sks.ExtrinsicDeformation(
    n_steps=4,
    kernel="gaussian",
    scale=0.3,
    normalization="rows",
)

task_norm = sks.Registration(
    model=model_norm,
    loss=loss,
    optimizer=sks.LBFGS(),
    n_iter=1,
    regularization_weight=0.0,
    verbose=True,
)

start = time()
task_norm.fit(source=source, target=target)
print("Elapsed time: ", time() - start)
path = task_norm.path_

plotter = pv.Plotter()
plotter.add_mesh(source.to_pyvista(), color="teal", opacity=0.2, **plot_kwargs)
plotter.add_mesh(target.to_pyvista(), color="red", opacity=0.2, **plot_kwargs)
for i in range(len(path)):
    plotter.add_mesh(
        path[i].to_pyvista(), color="tan", opacity=0.8, **plot_kwargs
    )
plotter.camera_position = cpos
plotter.show()

###############################################################################
# Extrapolation

back = model_norm.morph(
    shape=source,
    parameter=task_norm.parameter_,
    final_time=-1.0,
    return_path=True,
).path

model_norm.n_steps = 8
forward = model_norm.morph(
    shape=source,
    parameter=task_norm.parameter_,
    final_time=2.0,
    return_path=True,
).path

path = back[::-1] + forward[1:]

plotter = pv.Plotter()
plotter.open_gif("lddmm_normalization.gif", fps=4)
for i in range(len(path)):
    plotter.clear_actors()
    plotter.add_mesh(
        source.to_pyvista(), color="teal", opacity=0.2, **plot_kwargs
    )
    plotter.add_mesh(
        target.to_pyvista(), color="red", opacity=0.2, **plot_kwargs
    )
    plotter.add_mesh(
        path[i].to_pyvista(), color="tan", opacity=0.8, **plot_kwargs
    )
    plotter.camera_position = cpos
    plotter.write_frame()
plotter.close()

###############################################################################
# Normalizing both rows and columns of the kernel
# -----------------------------------------------

model_norm = sks.ExtrinsicDeformation(
    n_steps=4,
    kernel="gaussian",
    scale=0.3,
    normalization="both",
)

task_norm = sks.Registration(
    model=model_norm,
    loss=loss,
    optimizer=sks.LBFGS(),
    n_iter=1,
    regularization_weight=0.0,
    verbose=True,
)

start = time()
task_norm.fit(source=source, target=target)
elapsed_time = time() - start
path = task_norm.path_

plotter = pv.Plotter()
plotter.add_mesh(source.to_pyvista(), color="teal", opacity=0.2, **plot_kwargs)
plotter.add_mesh(target.to_pyvista(), color="red", opacity=0.2, **plot_kwargs)
for i in range(len(path)):
    plotter.add_mesh(
        path[i].to_pyvista(), color="tan", opacity=0.8, **plot_kwargs
    )
plotter.camera_position = cpos
plotter.show()

###############################################################################
# Extrapolation

back = model_norm.morph(
    shape=source,
    parameter=task_norm.parameter_,
    final_time=-1.0,
    return_path=True,
).path

model_norm.n_steps = 8
forward = model_norm.morph(
    shape=source,
    parameter=task_norm.parameter_,
    final_time=2.0,
    return_path=True,
).path

path = back[::-1] + forward[1:]

plotter = pv.Plotter()
plotter.open_gif("lddmm_normalization.gif", fps=4)
for i in range(len(path)):
    plotter.clear_actors()
    plotter.add_mesh(
        source.to_pyvista(), color="teal", opacity=0.2, **plot_kwargs
    )
    plotter.add_mesh(
        target.to_pyvista(), color="red", opacity=0.2, **plot_kwargs
    )
    plotter.add_mesh(
        path[i].to_pyvista(), color="tan", opacity=0.8, **plot_kwargs
    )
    plotter.camera_position = cpos
    plotter.write_frame()
plotter.close()

###############################################################################
# Example with a more complex shape
# ---------------------------------

n_steps = 3

plot_kwargs = {
    "smooth_shading": True,
    "pbr": True,
    "metallic": 0.7,
    "roughness": 0.6,
}

cpos = [
    (3.6401575998373183, -1.183408993703478, 1.0915912440258628),
    (0.7463583722710609, 0.762569822371006, 0.48035204596817493),
    (-0.1745415166347431, 0.04933887578777028, 0.9834129012306287),
]

# 5 - 8
source = sks.PolyData("../test_data/cactus/cactus3.ply")
target = sks.PolyData("../test_data/cactus/cactus11.ply")
target.points += torch.Tensor([0.5, 0.5, 0])

decimation = sks.Decimation(n_points=500)
source = decimation.fit_transform(source)
target = decimation.transform(target)


model = sks.ExtrinsicDeformation(
    n_steps=n_steps,
    kernel="gaussian",
    scale=0.1,
    normalization="both",
)

loss = sks.L2Loss()

###############################################################################
# Interpolation
# -------------

task = sks.Registration(
    model=model,
    loss=loss,
    optimizer=sks.LBFGS(),
    n_iter=1,
    verbose=True,
    regularization_weight=0.001,
)

start = time()
task.fit(source=source, target=target)
print("Elapsed time: ", time() - start)

path = task.path_

plotter = pv.Plotter()
for frame in path:
    plotter.add_mesh(
        frame.to_pyvista(), color="tan", opacity=0.3, **plot_kwargs
    )
plotter.add_mesh(source.to_pyvista(), color="teal", opacity=0.5, **plot_kwargs)
plotter.add_mesh(target.to_pyvista(), color="red", opacity=0.5, **plot_kwargs)
plotter.camera_position = cpos
plotter.show()

###############################################################################
# Extrapolation
# -------------

back = model.morph(
    shape=source,
    parameter=task.parameter_,
    return_path=True,
    return_regularization=True,
    final_time=-1.0,
).path

model.n_steps = 2 * n_steps

forward = model.morph(
    shape=source,
    parameter=task.parameter_,
    return_path=True,
    return_regularization=True,
    final_time=2.0,
).path

path = back[::-1] + forward[1:]

plotter = pv.Plotter()
for frame in path:
    plotter.add_mesh(
        frame.to_pyvista(), color="tan", opacity=0.3, **plot_kwargs
    )
plotter.add_mesh(source.to_pyvista(), color="teal", opacity=0.5, **plot_kwargs)
plotter.add_mesh(target.to_pyvista(), color="red", opacity=0.5, **plot_kwargs)
plotter.camera_position = cpos
plotter.show()

print(plotter.camera_position)
