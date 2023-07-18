import sys

sys.path.append(sys.path[0][:-4])

from skshapes.tasks import Registration
from skshapes.data import read
import torch
import skshapes

from skshapes.loss import NearestNeighborsLoss, OptimalTransportLoss, L2Loss
from skshapes.morphing import ElasticMetric, RigidMotion
from skshapes.optimization import LBFGS


def test_registration():
    # Load two meshes
    datafolder = "data/SCAPE_low_resolution"
    import os

    os.chdir(sys.path[-1])
    print(os.getcwd())
    print(datafolder + "/" + "mesh001.ply")
    source = read(datafolder + "/" + "mesh001.ply")
    target = read(datafolder + "/" + "mesh041.ply")

    # Few type checks
    assert isinstance(source, skshapes.ShapeType)
    assert isinstance(target, skshapes.PolyDataType)

    assert source.landmarks is None
    # Add landmarks
    # source.landmarks = source.points
    # target.landmarks = target.points
    # assert source.landmarks is not None

    # Try different combinations of loss and model for registration
    # and check that no error is raised
    losses = [
        NearestNeighborsLoss(),
        OptimalTransportLoss(),
        L2Loss(),
        0.8 * L2Loss() + 2 * OptimalTransportLoss(),
    ]
    models = [ElasticMetric(), RigidMotion()]
    for loss in losses:
        for model in models:
            r = Registration(
                model=model,
                loss=loss,
                optimizer=LBFGS(),
                n_iter=1,
                device="cpu",
                regularization=1,
            )
            print(loss, model)
            r.fit_transform(source=source, target=target)
