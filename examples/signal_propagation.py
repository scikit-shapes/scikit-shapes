from pyvista import examples
import fast_simplification
import pyvista
import numpy as np
from time import time

import torch

import skshapes as sks


from pyvista import examples

mesh = sks.PolyData(examples.download_louis_louvre())

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

start = time()
mesh.edges
print(f"compute edges : {time() - start}")

start = time()
mesh.edge_lengths
print(f"compute edge lengths : {time() - start}")

mesh.point_data.clear()


start_create_ratio = time()
ratio1, ratio2 = 0.01, 0.001
M = sks.Multiscale(mesh, ratios=[ratio1, ratio2])


import pyvista

# print(M.at(1).point_data.keys())

# Signal manipulation

import numpy as np

indice = 3450

indice = torch.argmax(mesh.points[:, 2])
indice = int()

try:
    import potpourri3d as pp3d

    V = mesh.points.cpu().numpy()
    F = mesh.triangles.cpu().numpy().T

    solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
    signal_h = torch.from_numpy(solver.compute_distance_multisource(landmarks)).to(
        torch.float32
    )


except ImportError:
    signal_h = mesh.points[:, 0].clone()

cpos = [
    (-3.7323781150895634, -27.875231208426484, 9.019381348367414),
    (1.6171762943267822, 2.0937089920043945, 8.911353826522827),
    (0.022876809290522153, -0.00047988011019209657, 0.9997381763800786),
]

p = pyvista.Plotter(shape=(2, 3))
p.subplot(0, 0)
p.add_mesh(mesh.to_pyvista(), show_edges=False, scalars=signal_h)
p.camera_position = cpos

p.subplot(0, 1)
start_propagate = time()
scalars = M.signal_from_high_to_low_res(
    signal_h, high_res=1, low_res=ratio1, reduce="mean"
)
print(f"propagate : {time() - start_propagate}")
p.add_mesh(M.at(ratio1).to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos

p.subplot(0, 2)
scalars = M.signal_from_high_to_low_res(
    signal_h, high_res=1, low_res=ratio2, reduce="mean"
)
p.add_mesh(M.at(ratio2).to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos
signal_l = scalars.clone()

p.subplot(1, 0)
scalars = M.signal_from_low_to_high_res(signal_l, low_res=ratio2, high_res=1)
p.add_mesh(mesh.to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos

print(torch.norm(signal_h - scalars))

p.subplot(1, 1)
scalars = M.signal_from_low_to_high_res(signal_l, low_res=ratio2, high_res=ratio1)
p.add_mesh(M.at(ratio1).to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos
signal_inter = scalars.clone()

p.subplot(1, 2)
scalars = signal_l
p.add_mesh(M.at(ratio2).to_pyvista(), show_edges=False, scalars=scalars)

p.camera_position = cpos


print(f"Part 1 : {time() - start}")
p.show()
start = time()

additional_ratios = [0.05, 0.1]
signals = {}
signals[ratio2] = signal_l.clone()

signal_inter = M.signal_from_low_to_high_res(signal_l, low_res=ratio2, high_res=ratio1)
signal_inter = sks.multiscaling.edge_smoothing(
    signal_inter, M.at(ratio1), weight_by_length=True
)
signals[ratio1] = signal_inter.clone()

for i, ratio in enumerate(additional_ratios):
    M.add_ratio(ratio)
    if i == 0:
        signal_inter = M.signal_from_low_to_high_res(
            signal_inter, low_res=ratio1, high_res=ratio
        )
    else:
        signal_inter = M.signal_from_low_to_high_res(
            signal_inter, low_res=additional_ratios[i - 1], high_res=ratio
        )

    signal_inter = sks.multiscaling.edge_smoothing(
        signal_inter, M.at(ratio), weight_by_length=True
    )
    signals[ratio] = signal_inter.clone()

signal_inter = M.signal_from_low_to_high_res(
    signal_inter, low_res=additional_ratios[-1], high_res=1
)
signal_inter = sks.multiscaling.edge_smoothing(
    signal_inter, M.at(1), weight_by_length=True
)
signals[1] = signal_inter.clone()

signals2 = {}
signals2[ratio2] = signal_l.clone()
signal_inter = M.signal_from_low_to_high_res(signal_l, low_res=ratio2, high_res=ratio1)
signal_inter = sks.multiscaling.edge_smoothing(
    signal_inter, M.at(ratio1), weight_by_length=True
)
signals2[ratio1] = signal_inter.clone()

signal_inter = M.signal_from_low_to_high_res(signal_inter, low_res=ratio1, high_res=1)
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
    p.add_mesh(M.at(additional_ratios[i - 1]).to_pyvista(), show_edges=False)
    p.camera_position = cpos
p.subplot(0, 3)
scalars = signals2[ratio1]
p.add_mesh(M.at(ratio1).to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos
p.subplot(0, 4)
scalars = signals2[ratio2]
p.add_mesh(M.at(ratio2).to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos

p.subplot(1, 0)
scalars = signals[1]
p.add_mesh(M.at(1).to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos
for i in range(2):
    p.subplot(1, i + 1)
    scalars = signals[additional_ratios[i - 1]]
    p.add_mesh(
        M.at(additional_ratios[i - 1]).to_pyvista(), show_edges=False, scalars=scalars
    )
    p.camera_position = cpos
p.subplot(1, 3)
scalars = signals[ratio1]
p.add_mesh(M.at(ratio1).to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos
p.subplot(1, 4)
scalars = signals[ratio2]
p.add_mesh(M.at(ratio2).to_pyvista(), show_edges=False, scalars=scalars)
p.camera_position = cpos

print(f"Part 2 : {time() - start}")

print(torch.norm(signals2[1] - signal_h))
print(torch.norm(signals[1] - signal_h))

p.show()
