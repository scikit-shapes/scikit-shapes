from re import M
import skshapes as sks
import pyvista as pv
from pyvista import examples
import numpy as np


def show(M, signal_name):
    p = pv.Plotter(shape=(1, len(ratios)))
    for i, r in enumerate(ratios):
        p.subplot(0, i)
        mesh_pv = M.at(r).to_pyvista()
        p.add_mesh(mesh_pv, scalars=signal_name)
    p.show()


mesh = sks.PolyData(examples.download_louis_louvre())
mesh["signal"] = mesh.points[:, 0]

ratios = [1, 0.5, 0.1, 0.05, 0.001]

downscale_policy = {
    "reduce": "mean",
    "pass_through": False,
}

upscale_policy = {
    "smoothing": "mesh_convolution",
    "n_smoothing_steps": 1,
    "pass_through": False,
}

M = sks.Multiscale(
    mesh,
    ratios=ratios,
    downscale_policy=downscale_policy,
    upscale_policy=upscale_policy,
)

show(M, "signal")

M[0.001]["signal2"] = M[0.001].points[:, 0]
show(M, "signal2")
