from pyvista import examples
import fast_simplification
import pyvista
import numpy as np
from time import time

import torch

import skshapes as sks


from pyvista import examples

mesh = sks.PolyData(examples.download_louis_louvre())

start = time()
mesh.edges
print(f"compute edges : {time() - start}")

start = time()
mesh.edge_lengths
print(f"compute edge lengths : {time() - start}")

mesh_pv = mesh.to_pyvista()
start = time()
mesh_pv.extract_all_edges()
print(f"Extract edges : {time() - start}")

mesh.point_data.clear()


scale1, scale2 = 0.01, 0.001
M = sks.Multiscale(mesh, scales=[scale1, scale2])

print(sks.multiscaling.edge_smoothing)


import pyvista

print(M.at(1).point_data.keys())

# Signal manipulation

import numpy as np

signal_h = mesh.points[:, 0].clone()

# p = pyvista.Plotter()
# p.add_mesh(mesh.to_pyvista(), scalars=signal_h)
# p.show()

cpos = [
    (-3.7323781150895634, -27.875231208426484, 9.019381348367414),
    (1.6171762943267822, 2.0937089920043945, 8.911353826522827),
    (0.022876809290522153, -0.00047988011019209657, 0.9997381763800786),
]

p = pyvista.Plotter(shape=(2, 3))
p.subplot(0, 0)
p.add_mesh(mesh.to_pyvista(), show_edges=False, scalars=signal_h)
# p.add_points(
#     np.array(mesh.points),
#     render_points_as_spheres=True,
#     scalars=signal_h,
#     point_size=10,
# )
p.camera_position = cpos
p.subplot(0, 1)
scalars = M.signal_from_high_to_low_res(
    signal_h, high_res=1, low_res=scale1, reduce="mean"
)
p.add_mesh(M.at(scale1).to_pyvista(), show_edges=False, scalars=scalars)
# p.add_points(
#     np.array(M.at(scale1).points),
#     render_points_as_spheres=True,
#     scalars=scalars,
#     point_size=10,
# )
p.camera_position = cpos
p.subplot(0, 2)
scalars = M.signal_from_high_to_low_res(
    signal_h, high_res=1, low_res=scale2, reduce="mean"
)
p.add_mesh(M.at(scale2).to_pyvista(), show_edges=False, scalars=scalars)
# p.add_points(
#     np.array(M.at(scale2).points),
#     render_points_as_spheres=True,
#     scalars=scalars,
#     point_size=10,
# )
p.camera_position = cpos
signal_l = scalars.clone()


p.subplot(1, 0)
scalars = M.signal_from_low_to_high_res(signal_l, low_res=scale2, high_res=1)
p.add_mesh(mesh.to_pyvista(), show_edges=False, scalars=scalars)
# p.add_points(
#     np.array(mesh.points), render_points_as_spheres=True, scalars=scalars, point_size=10
# )
p.camera_position = cpos
p.subplot(1, 1)
scalars = M.signal_from_low_to_high_res(signal_l, low_res=scale2, high_res=scale1)
p.add_mesh(M.at(scale1).to_pyvista(), show_edges=False, scalars=scalars)
# p.add_points(
#     np.array(M.at(scale1).points),
#     render_points_as_spheres=True,
#     scalars=scalars,
#     point_size=10,
# )
p.camera_position = cpos
signal_inter = scalars.clone()

p.subplot(1, 2)
scalars = signal_l
p.add_mesh(M.at(scale2).to_pyvista(), show_edges=False, scalars=scalars)
# p.add_points(
#     np.array(M.at(scale2).points),
#     render_points_as_spheres=True,
#     scalars=signal_l,
#     point_size=10,
# )
p.camera_position = cpos


# signal_inter = M.signal_from_low_to_high_res(signal_l, low_res=scale2, high_res=scale1)
# signal_inter = sks.multiscaling.edge_smoothing(signal_inter, M.at(scale1), weight_by_length=True)

# signal_high = M.signal_from_low_to_high_res(signal_inter, low_res=scale1, high_res=1)


# start = time()
# _ = sks.multiscaling.edge_smoothing(signal_high, M.at(1), weight_by_length=True)
# print(f"GPU : {time() - start}")

# start = time()
# signal_high = sks.multiscaling.edge_smoothing(signal_high, M.at(1), weight_by_length=True, gpu=False)
# print(f"CPU : {time() - start}")

# p.subplot(2, 0)
# scalars = signal_high
# p.add_mesh(mesh.to_pyvista(), show_edges=False, scalars=scalars)
# # p.add_points(
# #     np.array(mesh.points), render_points_as_spheres=True, scalars=scalars, point_size=10
# # )
# p.camera_position = cpos
# p.subplot(2, 1)
# # scalars = signal_inter
# p.add_mesh(M.at(scale1).to_pyvista(), show_edges=True, render_points_as_spheres=True)
# # p.add_points(
# #     np.array(M.at(scale1).points),
# #     render_points_as_spheres=True,
# #     scalars=scalars,
# #     point_size=10,
# # )
# p.camera_position = cpos

# signal_inter = scalars.clone()

# p.subplot(2, 2)
# p.add_mesh(M.at(scale2).to_pyvista(), show_edges=True, render_points_as_spheres=True)
# p.add_points(
#     np.array(M.at(scale2).points),
#     render_points_as_spheres=True,
#     scalars=signal_l,
#     point_size=10,
# )
# p.camera_position = cpos
p.show()


additional_scales = [0.05, 0.1]
signals = {}
signals[scale2] = signal_l.clone()

signal_inter = M.signal_from_low_to_high_res(signal_l, low_res=scale2, high_res=scale1)
signal_inter = sks.multiscaling.edge_smoothing(
    signal_inter, M.at(scale1), weight_by_length=True
)
signals[scale1] = signal_inter.clone()

for i, scale in enumerate(additional_scales):
    M.add_scale(scale)
    if i == 0:
        signal_inter = M.signal_from_low_to_high_res(
            signal_inter, low_res=scale1, high_res=scale
        )
    else:
        signal_inter = M.signal_from_low_to_high_res(
            signal_inter, low_res=additional_scales[i - 1], high_res=scale
        )

    signal_inter = sks.multiscaling.edge_smoothing(
        signal_inter, M.at(scale), weight_by_length=True
    )
    signals[scale] = signal_inter.clone()

signal_inter = M.signal_from_low_to_high_res(
    signal_inter, low_res=additional_scales[-1], high_res=1
)
signal_inter = sks.multiscaling.edge_smoothing(
    signal_inter, M.at(1), weight_by_length=True
)
signals[1] = signal_inter.clone()

signals2 = {}
signals2[scale2] = signal_l.clone()
signal_inter = M.signal_from_low_to_high_res(signal_l, low_res=scale2, high_res=scale1)
signal_inter = sks.multiscaling.edge_smoothing(
    signal_inter, M.at(scale1), weight_by_length=True
)
signals2[scale1] = signal_inter.clone()

signal_inter = M.signal_from_low_to_high_res(signal_inter, low_res=scale1, high_res=1)
signal_inter = sks.multiscaling.edge_smoothing(
    signal_inter, M.at(1), weight_by_length=True
)
signals2[1] = signal_inter.clone()


p = pyvista.Plotter(shape=(2, 5))
p.subplot(0, 0)
scalars = signals2[1]
p.add_mesh(M.at(1).to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos
for i in range(2):
    p.subplot(0, i + 1)
    p.add_mesh(M.at(additional_scales[i - 1]).to_pyvista(), show_edges=False)
    p.camera_position = cpos
p.subplot(0, 3)
scalars = signals2[scale1]
p.add_mesh(M.at(scale1).to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos
p.subplot(0, 4)
scalars = signals2[scale2]
p.add_mesh(M.at(scale2).to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos

p.subplot(1, 0)
scalars = signals[1]
p.add_mesh(M.at(1).to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos
for i in range(2):
    p.subplot(1, i + 1)
    scalars = signals[additional_scales[i - 1]]
    p.add_mesh(
        M.at(additional_scales[i - 1]).to_pyvista(), show_edges=False, scalars=scalars
    )
    p.camera_position = cpos
p.subplot(1, 3)
scalars = signals[scale1]
p.add_mesh(M.at(scale1).to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos
p.subplot(1, 4)
scalars = signals[scale2]
p.add_mesh(M.at(scale2).to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos

p.show()
