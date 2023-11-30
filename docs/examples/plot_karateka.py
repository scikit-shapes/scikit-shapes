"""
Make the karateka moves
=======================
"""

# %%
# Useful imports

import sys
print(sys.path)
import skshapes as sks
from utils_karateka import *

kwargs = {
    "loss": sks.L2Loss(),
    "optimizer": sks.LBFGS(),
    "n_iter": 5,
    "gpu": False,
    "verbose": True,
}
import pykeops
print(pykeops.config.sys.path)

# pykeops_nvrtc = importlib.import_module("pykeops_nvrtc")
# https://github.com/Louis-Pujol/mkdocs-gallery
# See https://github.com/smarie/mkdocs-gallery/pull/82

# %%
# Load the data

source, target = load_data()
plot_karatekas()
print(sys.path)

# %%
# Register with an extrinsic deformation
print(sys.path)
source.control_points = source.bounding_grid(N=25, offset=0.15)

model = sks.ExtrinsicDeformation(
    kernel=sks.GaussianKernel(sigma=1.0), control_points=True
)

registration = sks.Registration(model=model, regularization=0.1, **kwargs)

registration.fit(source=source, target=target)

# %%
# Visualize the deformation

plot_extrinsic_deformation(
    source=source, target=target, registration=registration
)

# %%
# Register with en intrinsic deformation (small regularization)

model = sks.IntrinsicDeformation(
    n_steps=8,
)

registration = sks.Registration(model=model, regularization=0.0001, **kwargs)

registration.fit(source=source, target=target)


# %%
# Visualize the deformation

plot_intrinsic_deformation(
    source=source, target=target, registration=registration
)


# %%
# Register with en intrinsic deformation (with regularization)

model = sks.IntrinsicDeformation(
    n_steps=8,
)

registration = sks.Registration(model=model, regularization=100, **kwargs)

registration.fit(source=source, target=target)


# %%
# Visualize the deformation

plot_intrinsic_deformation(
    source=source, target=target, registration=registration
)
