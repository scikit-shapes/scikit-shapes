from pyvista import examples
import pyvista
import skshapes as sks
import numpy as np
import torch
import time


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

mesh.landmarks = landmarks


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


from vedo.applications import Browser
import vedo

b = Browser(
    [
        mesh.to_vedo()
        + vedo.pointcloud.Points(mesh.landmark_points.cpu().numpy())
        .c("red")
        .ps(10)
        .render_points_as_spheres()
    ]
    + [
        low_resolutions[rate].to_vedo()
        + vedo.pointcloud.Points(
            low_resolutions[rate].landmark_points.cpu().numpy()
        )
        .c("red")
        .ps(10)
        .render_points_as_spheres()
        for rate in rates
    ]
)
b.show()
