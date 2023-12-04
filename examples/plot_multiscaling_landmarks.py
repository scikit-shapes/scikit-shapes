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

cpos = [(11.548847281353684, -19.784217817604652, 8.581378858601008),
 (1.606399655342102, 2.120710074901581, 8.925199747085571),
 (-0.021305501811855157, -0.025357814432988603, 0.9994513779267742)]

p = pyvista.Plotter()
p.add_mesh(mesh.to_pyvista(), color="tan", opacity=0.9)
p.add_points(
    mesh.landmark_points.numpy(),
    color="red",
    point_size=20,
    render_points_as_spheres=True,
    )
p.camera_position = cpos
p.show()



# p = pyvista.Plotter(shape=(1, len(rates) + 1))
# p.subplot(0, 0)
# p.add_mesh(mesh.to_pyvista(), color="tan")
# p.add_points(mesh.landmark_points.numpy(), color="red")
# p.camera_position = cpos
# for i, rate in enumerate(rates):
#     p.subplot(0, i + 1)
#     p.add_mesh(low_resolutions[rate].to_pyvista(), color="tan")
#     p.add_points(low_resolutions[rate].landmark_points.numpy(), color="red")
#     p.camera_position = cpos
# p.show()

