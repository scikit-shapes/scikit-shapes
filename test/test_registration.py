import skshapes as sks
import pyvista
import torch
import pytest


def test_registration():
    # Load two meshes
    source = sks.PolyData(pyvista.Sphere())
    target = sks.PolyData(pyvista.Sphere())

    # Few type checks
    assert isinstance(source, sks.data.baseshape.BaseShape)
    assert isinstance(target, sks.PolyData)
    assert source.landmarks is None

    losses = [
        sks.loss.NearestNeighborsLoss(),
        sks.loss.OptimalTransportLoss(),
        sks.loss.L2Loss(),
        0.8 * sks.loss.L2Loss() + 2 * sks.loss.OptimalTransportLoss(),
    ]
    models = [sks.morphing.ElasticMetric(), sks.morphing.RigidMotion()]
    for loss in losses:
        for model in models:
            r = sks.tasks.Registration(
                model=model,
                loss=loss,
                optimizer=sks.LBFGS(),
                n_iter=1,
                gpu=False,
                regularization=1,
            )
            print(loss, model)
            r.fit_transform(source=source, target=target)


import skshapes as sks
import pyvista.examples
from typing import get_args

shape1 = sks.PolyData(pyvista.Sphere())
shape2 = sks.PolyData(pyvista.Sphere()).decimate(target_reduction=0.5)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda is required for this test"
)
def test_registration_device():
    """This test ensure the behavior of the registration task with respect to the devices of the source, target and the gpu argument.
    Expected behavior:
        - If the gpu argument is True, the optimization should occurs on the gpu
        - If the gpu argument is False, the optimization should occurs on the gpu
        - The device of the output of .transform() and of .parameter_ should be the same as the device of the source and the target
        - If source.device != target.device, an error should be raised
    """

    n_steps = 5
    model = sks.RigidMotion(n_steps=n_steps)
    loss = sks.OptimalTransportLoss()
    optimizer = sks.LBFGS()

    for device in ["cpu", "cuda"]:
        for gpu in [False, True]:
            source = shape1.to(device)
            target = shape2.to(device)

            task = sks.Registration(
                model=model,
                loss=loss,
                optimizer=optimizer,
                n_iter=3,
                gpu=gpu,
                regularization=10,
                verbose=1,
            )

            newshape = task.fit_transform(source=source, target=target)

            # Check that the device on which the optimization is performed corresponds to the gpu argument
            if gpu:
                assert task.internal_parameter_device_type == "cuda"
            else:
                assert task.internal_parameter_device_type == "cpu"

            # Check that the device of the output is the same as the input's shapes
            assert task.parameter_.device.type == source.device.type
            assert newshape.device.type == target.device.type

            # Check that the length of task.path_ is equal to n_steps
            assert len(task.path_) == n_steps + 1

    # Check that if source and target are on different devices, an error is raised
    source = shape1.to("cpu")
    target = shape2.to("cuda")
    try:
        task.fit(source=source, target=target)
    except:
        pass
    else:
        raise AssertionError("Should have raised an error")
