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


def foo(model, loss, optimizer, regularization, nframes=5):
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

    parameter = r.parameter.detach().cpu().clone()

    meshes = []
    meshes.append(source)
    for i in range(nframes):
        meshes.append(
            model.morph(
                shape=source, parameter=(i + 1) * parameter / nframes, return_path=True
            ).morphed_shape
        )

    meshes.append(target)

    plt = Browser(
        [vedo.Mesh(m.to_pyvista()) for m in meshes], resetcam=0, axes=0
    )  # a vedo.Plotter
    plt.show().close()


source = read("data/SCAPE_low_resolution/mesh001.ply")
target = read("data/SCAPE_low_resolution/mesh041.ply")

source = read("../scikit-shapes-draft/data/SCAPE/scapecomp/mesh001.ply")
target = read("../scikit-shapes-draft/data/SCAPE/scapecomp/mesh041.ply")

print("Two meshes to be registered:")
vedo.show([vedo.Mesh(source.to_pyvista()), vedo.Mesh(target.to_pyvista())])


print("L2 Loss (assume the correspondence is known)")
model = RigidMotion()
loss = L2Loss()
optimizer = LBFGS()
foo(model, loss, optimizer, regularization=0)

print("ICP (np correspondence)")
model = RigidMotion()
loss = NearestNeighborsLoss()
optimizer = LBFGS()
foo(model, loss, optimizer, regularization=0)


app = LandmarkSetter(meshes=[source, target])
app.start()

print("Landmark Loss (user correspondence)")
model = RigidMotion()
loss = LandmarkLoss()
optimizer = LBFGS()
foo(model, loss, optimizer, regularization=0)


print("Landmark Loss (user correspondence)")
model = RigidMotion()
loss = LandmarkLoss() + 1.0 * NearestNeighborsLoss()
optimizer = LBFGS()
foo(model, loss, optimizer, regularization=0)
