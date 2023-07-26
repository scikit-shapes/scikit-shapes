import skshapes as sks
import pyvista.examples
from typing import get_args
import pytest
import torch

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
    model = sks.RigidMotion()
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

    # Check that if source and target are on different devices, an error is raised
    source = shape1.to("cpu")
    target = shape2.to("cuda")
    try:
        task.fit(source=source, target=target)
    except:
        pass
    else:
        raise AssertionError("Should have raised an error")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda is required for this test"
)
def test_loss_device():
    source = shape1.to("cpu")
    target = shape2.to("cuda")

    list_of_losses = [
        i
        for i in sks.loss.__dict__.values()
        if isinstance(i, type) and i in get_args(sks.Loss)
    ]
    for loss in list_of_losses:
        l = loss()

        try:
            l(source=source, target=target)
        except:
            pass
        else:
            l(source=source, target=target)
            print(loss)
            raise AssertionError("Should have raised an error")
