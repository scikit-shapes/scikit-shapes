import skshapes as sks
import pyvista
import torch
import pytest

from hypothesis import given, settings
from hypothesis import strategies as st

list_models = [
    sks.RigidMotion(),
    sks.KernelDeformation(),
    sks.VectorFieldDeformation(),
]
list_losses = [
    sks.L2Loss(),
    sks.LpLoss(p=1),
    sks.OptimalTransportLoss(),
    sks.NearestNeighborsLoss(),
    sks.LandmarkLoss(),
]

list_optimizers = [
    sks.LBFGS(),
    sks.SGD(),
    sks.Adam(),
    sks.Adagrad(),
]


@given(
    model=st.sampled_from(list_models),
    loss=st.sampled_from(list_losses),
    optimizer=st.sampled_from(list_optimizers),
    n_iter=st.integers(min_value=1, max_value=3),
    regularization=st.floats(min_value=0, max_value=1),
    gpu=st.booleans(),
    verbose=st.booleans(),
    provide_initial_parameter=st.booleans(),
)
@settings(deadline=None, max_examples=5)
def test_registration_hypothesis(
    model,
    loss,
    optimizer,
    n_iter,
    regularization,
    gpu,
    verbose,
    provide_initial_parameter,
):
    if (
        gpu
        and torch.cuda.is_available()
        and isinstance(loss, sks.OptimalTransportLoss)
    ):
        return None

    # Load two meshes

    source = sks.PolyData(pyvista.Sphere()).decimate(target_reduction=0.95)
    target = sks.PolyData(pyvista.Sphere()).decimate(target_reduction=0.95)

    source.landmark_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    target.landmark_indices = [9, 10, 11, 12, 13, 14, 15, 16, 17]

    # Few type checks
    assert isinstance(source, sks.data.baseshape.BaseShape)
    assert isinstance(target, sks.PolyData)

    # Initialize the registration object
    r = sks.Registration(
        model=model,
        loss=loss,
        optimizer=optimizer,
        n_iter=n_iter,
        gpu=gpu,
        regularization=regularization,
        verbose=verbose,
    )

    # Check that the registration object is correctly initialized
    assert r.model == model
    assert r.loss == loss
    assert r.optimizer == optimizer
    assert r.n_iter == n_iter
    assert r.regularization == regularization

    # If applicable, provide an initial parameter
    if provide_initial_parameter:
        initial_parameter = torch.rand(r.model.parameter_shape(shape=source))
    else:
        initial_parameter = None

    # Try to transform the source shape without fitting first
    try:
        r.transform(source=source)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Should have raised an error, fit must be called"
            + "before transform"
        )

    # Fit
    r.fit(source=source, target=target, initial_parameter=initial_parameter)

    # Transform
    output1 = r.transform(source=source)

    # Fit_transform
    output2 = r.fit_transform(
        source=source, target=target, initial_parameter=initial_parameter
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda is required for this test"
)
def test_registration_device():
    """This test ensure the behavior of the registration task with respect to
        the devices of the source, target and the gpu argument.
    Expected behavior:
        - If the gpu argument is True, the optimization should occurs on the
            gpu
        - If the gpu argument is False, the optimization should occurs on the
            gpu
        - The device of the output of .transform() and of .parameter_ should be
            the same as the device of the source and the target
        - If source.device != target.device, an error should be raised
    """
    shape1 = sks.PolyData(pyvista.Sphere()).decimate(target_reduction=0.9)
    shape2 = sks.PolyData(pyvista.Sphere()).decimate(target_reduction=0.95)

    n_steps = 2
    models = [
        sks.RigidMotion(n_steps=n_steps),
        sks.KernelDeformation(n_steps=n_steps),
        sks.VectorFieldDeformation(n_steps=n_steps),
    ]
    loss = sks.OptimalTransportLoss()
    optimizer = sks.LBFGS()

    for model in models:
        for device in ["cpu", "cuda"]:
            for gpu in [False, True]:
                source = shape1.to(device)
                target = shape2.to(device)

                task = sks.Registration(
                    model=model,
                    loss=loss,
                    optimizer=optimizer,
                    n_iter=1,
                    gpu=gpu,
                    regularization=0.1,
                    verbose=1,
                )

                newshape = task.fit_transform(source=source, target=target)

                # Check that the device on which the optimization is performed
                # corresponds to the gpu argument
                if gpu:
                    assert task.internal_parameter_device_type == "cuda"
                else:
                    assert task.internal_parameter_device_type == "cpu"

                # Check that the device of the output is the same as the
                # input's shapes
                assert task.parameter_.device.type == source.device.type
                assert newshape.device.type == target.device.type

                # Check that the length of task.path_ is equal to n_steps
                assert len(task.path_) == n_steps + 1

        # Check that if source and target are on different devices, an error is
        #  raised
        source = shape1.to("cpu")
        target = shape2.to("cuda")
        try:
            task.fit(source=source, target=target)
        except ValueError:
            pass
        else:
            raise AssertionError("Should have raised an error")
