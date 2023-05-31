from skshapes.data import read
from skshapes.loss import LandmarkLoss
from skshapes.morphing import ElasticMetric
from skshapes.tasks import Registration
from skshapes.optimization import LBFGS

from vedo.applications import Browser
import vedo

import torch

# Load some shapes
import os

datafolder = "data/SCAPE_low_resolution"

source = read(datafolder + "/" + "mesh004.ply")
target = read(datafolder + "/" + "mesh038.ply")

source.landmarks = torch.arange(source.points.shape[0], dtype=torch.int64)
target.landmarks = torch.arange(target.points.shape[0], dtype=torch.int64)


def foo(regularization):

    n_steps = 5

    r = Registration(
        model=ElasticMetric(n_steps=5),
        loss=LandmarkLoss(),
        optimizer=LBFGS(),
        verbose=1,
        n_iter=5,
        regularization=regularization,
        device="cpu",
    )

    newshape = r.fit_transform(source=source, target=target)

    parameter = r.parameter.detach().cpu().clone()
    n_frames = parameter.shape[0] + 1

    meshes = [source.copy(device="cpu") for _ in range(n_frames)]
    for i in range(n_frames - 1):
        meshes[i + 1].points = meshes[i].points + parameter[i]

    meshes = [vedo.Mesh(mesh.to_pyvista()) for mesh in meshes] + [
        vedo.Mesh(target.to_pyvista())
    ]

    plt = Browser(meshes, resetcam=0, axes=0)  # a vedo.Plotter
    plt.show().close()


foo(regularization=0)
foo(regularization=1e3)
