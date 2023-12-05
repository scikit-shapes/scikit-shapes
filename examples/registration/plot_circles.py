"""
Example of rigid motion registration
====================================
This is an example of rigid motion registration. A rigid motion is a
transformation that preserves distances between points, it is defined by a
rotation and a translation.

In this example, we show how to register a circle to a circle using a rigid
motion."""

# %%
# Load useful packages
# --------------------
#
# Note : code for generating the images was moved in a separate file
# to improve this page's readability.
# See docs/mkdocs_examples/utils_plot_circles.py


import skshapes as sks
import torch
from math import pi
from utils_plot_circles import plot1, plot2


# %%
# Prepare the data
# ----------------

# Define a circle and remove the first edge
circle = sks.Circle(n_points=20)
circle.edges = circle.edges[1:]

# Define a random rigid motion
torch.manual_seed(1)
parameter = torch.rand(3) - 0.5
parameter[0] *= 2 * pi

# Apply the rigid motion to the circle
# to obtain the target
source = circle
morphing = sks.morphing.RigidMotion()
target = morphing.morph(shape=source, parameter=parameter).morphed_shape

#  Visualize source and target
plot1(source, target)


# %%
# Rigid registration
# ------------------
# Define the registration

n_steps = 1
loss = sks.L2Loss()
model = sks.RigidMotion(n_steps=n_steps)
optimizer = sks.LBFGS()

registration = sks.Registration(
    model=model,
    loss=loss,
    optimizer=optimizer,
    gpu=False,
    n_iter=3,
    verbose=True,
)

# %%
# Fit the registration model

registration.fit(source=source, target=target)
morphed_circle = registration.transform(source=source)

# %%
# Visualize the registration path

path = [source, morphed_circle]
plot2(path, target)
