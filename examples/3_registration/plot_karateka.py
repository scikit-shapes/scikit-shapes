"""
Intrinsic vs Extrinsic deformation
==================================

Intrisic and extrinsic deformations are two different approaches to non-rigid
deformation.

This example shows how both approaches differ on a simple 2D example.
"""

import sys

import pykeops
from utils_karateka import (
    load_data,
    plot_extrinsic_deformation,
    plot_intrinsic_deformation,
    plot_karatekas,
)

import skshapes as sks

sys.path.append(pykeops.get_build_folder())

kwargs = {
    "loss": sks.L2Loss(),
    "optimizer": sks.LBFGS(),
    "n_iter": 5,
    "gpu": False,
    "verbose": True,
}

# %% [markdown]
# Load the data

# %%
source, target = load_data()
plot_karatekas()

###############################################################################
# Extrinsic deformation
# ---------------------
#
# Extrinsic deformation is a kind of deformation that relies on the deformation
# of the ambient space, which is transferred to the shape.
# Intituively, you can think that you are twisting a sheet of paper on which
# the shape is drawn.

source.control_points = source.bounding_grid(N=10, offset=0.05)

model = sks.ExtrinsicDeformation(
    n_steps=6,
    kernel="gaussian",
    scale=1.0,
    control_points=True,
)

registration = sks.Registration(
    model=model, regularization_weight=0.1, **kwargs
)

registration.fit(source=source, target=target)

###############################################################################
# Visualize the deformation

# %%
plot_extrinsic_deformation(
    source=source, target=target, registration=registration, animation=True
)

###############################################################################
# Intrinsic deformation
# ---------------------
#
# An intrinsic deformation is a deformation that is not transferable from one
# shape to another. It is defined as a sequence of small displacements of the
# points of the shape. In this setting, you can think of the shape as a puppet
# that you can deform thanks to a set of strings attached to its vertices.
#
# In this example, we use the "as isometric as possible" metric, which tries to
# preserve the lengths of the edges of the shape.


model = sks.IntrinsicDeformation(
    n_steps=8,
    metric="as_isometric_as_possible",
)

registration = sks.Registration(
    model=model, regularization_weight=500, **kwargs
)

registration.fit(source=source, target=target)


###############################################################################
# Visualize the deformation

plot_intrinsic_deformation(
    source=source, target=target, registration=registration, animation=True
)
