from skshapes.morphing import ElasticMetric, RigidMotion, ElasticMetric2
from skshapes.loss import OptimalTransportLoss, LandmarkLoss, NearestNeighborsLoss
from skshapes.data import read
from skshapes.tasks import DistanceMatrix, Registration, Registration2
import torch

# Load some shapes
import os

datafolder = "data/SCAPE_low_resolution"

source = read(datafolder + "/" + "mesh004.ply")
target = read(datafolder + "/" + "mesh038.ply")

# source = read(datafolder + "/" + "mesh014.ply")
# target = read(datafolder + "/" + "mesh041.ply")

source.landmarks = torch.arange(source.points.shape[0], dtype=torch.int64)
target.landmarks = torch.arange(target.points.shape[0], dtype=torch.int64)


def foo(regularization):

    n_steps = 10

    r = Registration2(
        model=ElasticMetric2(n_steps=n_steps),
        loss=LandmarkLoss(),
        verbose=1,
        n_iter=15,
        regularization=regularization,
        device="cpu",
    )

    newshape = r.fit_transform(source=source, target=target)
    newshape.to_pyvista().save("data/newshape.vtk")

    from vedo.applications import Browser
    import vedo

    # vedo.settings.default_backend = "vtk"

    parameter = r.parameter.detach().cpu().clone()
    n_frames = parameter.shape[0] + 1

    alpha = r.parameter

    model = ElasticMetric2(n_steps=n_steps)
    meshes = model.morph(shape=source, parameter=alpha, return_path=True)

    # meshes = [source.copy(device='cpu') for _ in range(n_frames)]
    # for i in range(n_frames - 1):
    #     meshes[i + 1].points = meshes[i].points + parameter[i]

    meshes = [vedo.Mesh(mesh.to_pyvista()) for mesh in meshes] + [
        vedo.Mesh(target.to_pyvista())
    ]
    # print(meshes[0])

    plt = Browser(meshes, resetcam=0, axes=0)  # a vedo.Plotter
    plt.show().close()


foo(regularization=1e4)
foo(regularization=0)
