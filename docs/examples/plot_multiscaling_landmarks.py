"""
Multiscaling and landmarks
==========================

Landmarks are preserved during the multiscale representation.
"""

# %%
# Imports

from pyvista import examples
import pyvista
import skshapes as sks
import numpy as np
import time
import matplotlib.pyplot as plt

# %%
# Load a mesh and define landmarks

mesh = sks.PolyData(examples.download_louis_louvre().clean())

landmarks = [
    151807,
    21294,
    23344,
    25789,
    131262,
    33852,
    171465,
    191680,
    172653,
    130895,
    9743,
    19185,
    143397,
    200885,
]

mesh.landmark_indices = landmarks

# %%
# Decimate

rates = [0.9, 0.99, 0.999]

d = sks.Decimation(target_reduction=np.max(rates))
start = time.time()
d.fit(mesh)
print(time.time() - start)

low_resolutions = dict()

for rate in rates:
    print(rate)
    start = time.time()
    low_resolutions[rate] = d.transform(mesh, target_reduction=rate)
    print(time.time() - start)

# %%
# Visualize

cpos = [
    (-12.213091804300387, -47.405946354834505, 4.9647045737122575),
    (1.606399655342102, 2.120710074901581, 8.925199747085571),
    (-0.06490518888405027, -0.0615358717058527, 0.9959922956274946),
]

p = pyvista.Plotter(shape=(1, len(rates) + 1))
p.subplot(0, 0)
p.add_mesh(mesh.to_pyvista(), color="tan")
p.add_points(mesh.landmark_points.numpy(), color="red")
p.camera_position = cpos
for i, rate in enumerate(rates):
    p.subplot(0, i + 1)
    p.add_mesh(low_resolutions[rate].to_pyvista(), color="tan")
    p.add_points(low_resolutions[rate].landmark_points.numpy(), color="red")
    p.camera_position = cpos
p.show()
plt.imshow(p.image)
plt.axis("off")
plt.show()
