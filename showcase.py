# Load data
from skshapes.data import read
import pyvista
from skshapes.morphing import ElasticMetric, RigidMotion
from skshapes.loss import (
    NearestNeighborsLoss,
    OptimalTransportLoss,
    LandmarkLoss,
    L2Loss,
)
from skshapes.optimization import LBFGS
from skshapes.tasks import Registration

import vedo
from vedo.applications import Browser

source = read("data/SCAPE_low_resolution/mesh001.ply")
target = read("data/SCAPE_low_resolution/mesh041.ply")


# ElasticMetric can only be used with polydata
# RigidMotion can be used with all type of shapes


def foo(model, loss, optimizer, regularization):

    r = Registration(
        model=model,
        loss=loss,
        optimizer=optimizer,
        n_iter=5,
        device="cpu",
        regularization=regularization,
        verbose=1,
    )

    newshape = r.fit_transform(source=source, target=target)

    parameter = r.parameter.detach().cpu().clone()

    a = model.morph(shape=source, parameter=parameter, return_path=True)

    meshes = a.path
    meshes.append(target)

    plt = Browser(
        [vedo.Mesh(m.to_pyvista()) for m in meshes], resetcam=0, axes=0
    )  # a vedo.Plotter
    plt.show().close()


model = RigidMotion()
loss = OptimalTransportLoss()
loss = L2Loss()
optimizer = LBFGS()


vedo.show([vedo.Mesh(source.to_pyvista()), vedo.Mesh(target.to_pyvista())])

foo(model, loss, optimizer, regularization=100)
