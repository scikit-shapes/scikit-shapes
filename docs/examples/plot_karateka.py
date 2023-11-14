"""
Make the karateka moves
=======================
"""

# %%
# Useful imports

import importlib
import skshapes as sks
from utils_karateka import load_data, plot_karatekas, plot_path

# pykeops_nvrtc = importlib.import_module("pykeops_nvrtc")
# https://github.com/Louis-Pujol/mkdocs-gallery
# See https://github.com/smarie/mkdocs-gallery/pull/82

# %%
# Load the data

source, target = load_data()
plot_karatekas()

# %%
# Register with no regularization

n_steps = 10
loss = sks.L2Loss()
model = sks.VectorFieldDeformation(n_steps=n_steps)
optimizer = sks.LBFGS()
n_iter = 5
regularization = 0

registration = sks.Registration(
    model=model,
    loss=loss,
    optimizer=optimizer,
    gpu=False,
    n_iter=n_iter,
    verbose=True,
    regularization=regularization,
)

registration.fit(source=source, target=target)

# %%
# Visualize the registration path

path = registration.path_
plot_path(path=path)

# %%
# Register with regularization = 100

n_steps = 10
loss = sks.L2Loss()
model = sks.VectorFieldDeformation(n_steps=n_steps)
optimizer = sks.LBFGS()
n_iter = 5
regularization = 100

registration = sks.Registration(
    model=model,
    loss=loss,
    optimizer=optimizer,
    gpu=False,
    n_iter=n_iter,
    verbose=True,
    regularization=regularization,
)

registration.fit(source=source, target=target)

# %%
# Visualize the registration path

path = registration.path_
plot_path(path=path)

# %%
# Register with regularization = 1000

n_steps = 10
loss = sks.L2Loss()
model = sks.VectorFieldDeformation(n_steps=n_steps)
optimizer = sks.LBFGS()
n_iter = 5
regularization = 1000

registration = sks.Registration(
    model=model,
    loss=loss,
    optimizer=optimizer,
    gpu=False,
    n_iter=n_iter,
    verbose=True,
    regularization=regularization,
)

registration.fit(source=source, target=target)

# %%
# Visualize the registration path

path = registration.path_
plot_path(path=path)

# %%
# Register with regularization = 1000

n_steps = 2
loss = sks.L2Loss()
model = sks.KernelDeformation(
    n_steps=n_steps,
    kernel=sks.GaussianKernel(sigma=1),
    control_points="grid",
    n_grid=5,
)
optimizer = sks.LBFGS()
n_iter = 5
regularization = 0.1

registration = sks.Registration(
    model=model,
    loss=loss,
    optimizer=optimizer,
    gpu=False,
    n_iter=n_iter,
    verbose=True,
    regularization=regularization,
)

registration.fit(source=source, target=target)

# %%
# Visualize the registration path

path = registration.path_
plot_path(path=path)
