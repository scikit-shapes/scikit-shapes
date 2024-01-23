"""
Intrinsic vs Extrinsic deformation
==================================

Intrisic and extrinsic deformations are two different ways to deform a shape.
They both are non-rigid deformations and then offer flexibility to the
registration process.

This example shows how both approaches differ on a simple 2D example.
"""

# %%
# Useful imports

import sys

import pykeops
import skshapes as sks
from utils_karateka import (
    load_data,
    plot_extrinsic_deformation,
    plot_intrinsic_deformation,
    plot_karatekas,
)

sys.path.append(pykeops.get_build_folder())

kwargs = {
    "loss": sks.L2Loss(),
    "optimizer": sks.LBFGS(),
    "n_iter": 5,
    "gpu": False,
    "verbose": True,
}

# %%
# Load the data

source, target = load_data()
plot_karatekas()

# %%
# ## Extrinsic deformation
#
# Extrinsic deformation is a kind of deformation that is transferable from one
# shape to another. A typical application is to deform a grid of control points
# around the shape and then deform the shape using an interpolation.
#
# Intituively, you can think that you are twisting a sheet of paper on which
# the shape is drawn.
print(sys.path)
source.control_points = source.bounding_grid(N=15, offset=0.05)

model = sks.ExtrinsicDeformation(
    n_steps=6,
    kernel=sks.GaussianKernel(sigma=1.0),
    control_points=True,
)

registration = sks.Registration(model=model, regularization=0.1, **kwargs)

registration.fit(source=source, target=target)

# %%
# Visualize the deformation

plot_extrinsic_deformation(
    source=source, target=target, registration=registration, animation=True
)

# %%
# ## Intrinsic deformation
#
# An intrinsic deformation is a deformation that is not transferable from one
# shape to another. It is defined as a sequence of small displacements of the
# points of the shape. In this setting, you can think of the shape as a puppet
# that you can deform thanks to a set of strings attached to its vertices.

model = sks.IntrinsicDeformation(
    n_steps=8,
)

registration = sks.Registration(model=model, regularization=500, **kwargs)

registration.fit(source=source, target=target)


# %%
# Visualize the deformation

plot_intrinsic_deformation(
    source=source, target=target, registration=registration, animation=True
)
