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
from skshapes.applications import LandmarkSetter

import vedo
from vedo.applications import Browser


def foo(model, loss, optimizer, regularization):
    r = Registration(
        model=model,
        loss=loss,
        optimizer=optimizer,
        n_iter=5,
        device="cuda",
        regularization=regularization,
        verbose=1,
    )

    newshape = r.fit_transform(source=source, target=target)

    parameter = r.parameter.detach().clone()
    meshes = model.morph(shape=source, parameter=parameter, return_path=True).path

    meshes.append(target)

    plt = Browser(
        [vedo.Mesh(m.to_pyvista()) for m in meshes], resetcam=0, axes=0
    )  # a vedo.Plotter
    plt.show().close()


source = read("data/SCAPE/mesh014.ply")
target = read("data/SCAPE/mesh016.ply")

# Align the two meshes
r = Registration(
    model=RigidMotion(),
    loss=L2Loss(),
    optimizer=LBFGS(),
    n_iter=3,
    device="cuda",
    regularization=0,
)

source = r.fit_transform(source=source, target=target)

vedo.show([vedo.Mesh(source.to_pyvista()), vedo.Mesh(target.to_pyvista())])

model = ElasticMetric(n_steps=5)
loss = L2Loss()
optimizer = LBFGS()
regularization = 0
foo(model, loss, optimizer, regularization)


model = ElasticMetric(n_steps=5)
loss = L2Loss()
optimizer = LBFGS()
regularization = 1e3
foo(model, loss, optimizer, regularization)


model = ElasticMetric(n_steps=5)
loss = NearestNeighborsLoss()
optimizer = LBFGS()
regularization = 1e3
foo(model, loss, optimizer, regularization)

app = LandmarkSetter(meshes=[source, target])
app.start()


model = ElasticMetric(n_steps=5)
loss = LandmarkLoss()
optimizer = LBFGS()
regularization = 0
foo(model, loss, optimizer, regularization)

model = ElasticMetric(n_steps=5)
loss = NearestNeighborsLoss() + LandmarkLoss()
optimizer = LBFGS()
regularization = 1e3
foo(model, loss, optimizer, regularization)
