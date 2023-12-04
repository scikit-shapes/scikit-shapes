"""
Multiscaling and signal propagation
===================================

"""


import skshapes as sks
import pyvista as pv
from pyvista import examples
import torch
import matplotlib.pyplot as plt


def stripify(signal, n_strips=5):
    signal = (signal - torch.min(signal)) / (
        torch.max(signal) - torch.min(signal)
    )
    signal *= n_strips
    signal = signal.floor() % 2
    return signal


def show(M, signal_name, revert=False, cpos=None):
    p = pv.Plotter(shape=(1, len(ratios)))
    for i, r in enumerate(ratios):
        if revert:
            p.subplot(0, len(ratios) - i - 1)
        else:
            p.subplot(0, i)
        mesh_pv = M.at(ratio=r).to_pyvista()
        p.add_mesh(mesh_pv, scalars=signal_name)
        if cpos is not None:
            p.camera_position = cpos
    p.show()
    print(p.camera_position)


mesh = sks.PolyData(examples.download_louis_louvre())
mesh["signal"] = stripify(mesh.points[:, 0])

cpos = [
    (31.380849347041348, 31.895159766600827, 38.69964943878482),
    (1.606399655342102, 2.120710074901581, 8.925199747085571),
    (0.0, 0.0, 1.0),
]


mesh = sks.PolyData(examples.download_louis_louvre())  # Load a mesh
ratios = [
    1,
    0.5,
    0.1,
    0.05,
    0.01,
]  # Define the ratios at which the mesh will be scaled
fine_to_coarse_policy = sks.FineToCoarsePolicy(reduce="mean")


M = sks.Multiscale(mesh, ratios=ratios)  # Create the multiscale object

mesh["signal"] = stripify(
    mesh.points[:, 0], n_strips=8
)  # Define a signal on the mesh
M.propagate(
    signal_name="signal",
    from_ratio=1,
    fine_to_coarse_policy=fine_to_coarse_policy,
)

assert M.at(ratio=0.05)["signal"].shape[0] == M.at(ratio=0.05).n_points

coarse_signal = M.at(ratio=0.01)["signal"]  # Get the coarse signal
M.at(ratio=0.01)[
    "signal2"
] = coarse_signal  # Define a new signal on the coarse mesh

M.propagate(
    signal_name="signal2",
    from_ratio=0.01,
    coarse_to_fine_policy=sks.CoarseToFinePolicy(smoothing="constant"),
)

M.at(ratio=0.01)[
    "signal3"
] = coarse_signal  # Define a new signal on the coarse mesh
M.propagate(
    signal_name="signal3",
    from_ratio=0.01,
    coarse_to_fine_policy=sks.CoarseToFinePolicy(smoothing="mesh_convolution"),
)


cpos = [
    (-4.840495101326498, -20.983840666682138, 7.077599519421893),
    (1.606399655342102, 2.120710074901581, 8.925199747085571),
    (-0.06490518888405025, -0.06153587170585269, 0.9959922956274944),
]

p = pv.Plotter(shape=(3, len(ratios)))
for i, r in enumerate(ratios):
    p.subplot(0, i)
    mesh_pv = M.at(ratio=r).to_pyvista()
    p.add_mesh(mesh_pv, scalars="signal")
    p.camera_position = cpos
    p.subplot(1, i)
    mesh_pv = M.at(ratio=r).to_pyvista()
    p.add_mesh(mesh_pv, scalars="signal2")
    p.camera_position = cpos
    p.subplot(2, i)
    mesh_pv = M.at(ratio=r).to_pyvista()
    p.add_mesh(mesh_pv, scalars="signal3")
    p.camera_position = cpos
p.show()
