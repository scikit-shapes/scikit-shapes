"""Test the registration task."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import skshapes as sks
from skshapes.errors import DeviceError, NotFittedError

list_models = [
    sks.RigidMotion,
    sks.ExtrinsicDeformation,
    sks.IntrinsicDeformation,
]
list_losses = [
    sks.L2Loss(),
    sks.LpLoss(p=1),
    sks.OptimalTransportLoss(),
    sks.NearestNeighborsLoss(),
    sks.LandmarkLoss(),
]

list_optimizers = [
    sks.LBFGS(max_iter=1, max_eval=1),
    sks.SGD(),
    sks.Adam(),
    sks.Adagrad(),
]

mesh_3d = sks.Sphere().resample(n_points=5)
mesh_2d = sks.Circle(n_points=7)


@given(
    model=st.sampled_from(list_models),
    loss=st.sampled_from(list_losses),
    optimizer=st.sampled_from(list_optimizers),
    n_iter=st.integers(min_value=0, max_value=1),
    regularization_weight=st.sampled_from([0, 0.1]),
    provide_initial_parameter=st.booleans(),
    dim=st.integers(min_value=2, max_value=3),
)
@settings(deadline=None)
def test_registration_hypothesis(
    model,
    loss,
    optimizer,
    n_iter,
    regularization_weight,
    provide_initial_parameter,
    dim,
):
    """Test that the registration task not failed with random params."""
    # Load two meshes
    n_steps = 1

    if dim == 3:
        source = mesh_3d
        target = mesh_3d
    else:
        source = mesh_2d
        target = mesh_2d

    assert source.dim == target.dim
    assert source.dim == dim

    source.landmark_indices = [0, 1, 2, 3, 4]
    target.landmark_indices = [3, 2, 1, 0, 4]

    # Few type checks
    assert isinstance(source, sks.PolyData)
    assert isinstance(target, sks.PolyData)

    # Initialize the registration object
    model = model(n_steps=n_steps)
    r = sks.Registration(
        model=model,
        loss=loss,
        optimizer=optimizer,
        n_iter=n_iter,
        gpu=False,
        regularization_weight=regularization_weight,
        verbose=False,
    )

    # Check that the registration object is correctly initialized
    assert r.model == model
    assert r.loss == loss
    assert r.optimizer == optimizer
    assert r.n_iter == n_iter
    assert r.regularization_weight == regularization_weight

    # If applicable, provide an initial parameter
    if provide_initial_parameter:
        initial_parameter = torch.rand(r.model.parameter_shape(shape=source))
    else:
        initial_parameter = None

    # Try to transform the source shape without fitting first
    with pytest.raises(NotFittedError):
        r.transform(source=source)
    # Fit
    r.fit(source=source, target=target, initial_parameter=initial_parameter)

    # Transform
    r.transform(source=source)

    # Fit_transform
    r.fit_transform(
        source=source, target=target, initial_parameter=initial_parameter
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda is required for this test"
)
def test_registration_device():
    """Test the behavior of the registration task with respect to the device.

    This test ensure the behavior of the registration task with respect to
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
    shape1 = mesh_3d
    shape2 = mesh_3d

    n_steps = 1
    models = [
        sks.RigidMotion(n_steps=n_steps),
        sks.ExtrinsicDeformation(n_steps=n_steps),
        sks.IntrinsicDeformation(n_steps=n_steps),
    ]
    loss = sks.L2Loss()
    optimizer = sks.SGD()

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
                    regularization_weight=0,
                    verbose=1,
                ).fit(source=source, target=target)

                newshape = task.transform(source=source)

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
        with pytest.raises(DeviceError):
            task.fit(source=source, target=target)


@pytest.mark.parametrize(
    ("kernel", "control_points", "normalization", "n_steps"),
    [
        ("gaussian", False, "both", 1),
        ("gaussian", True, "rows", 2),
        ("uniform", False, "columns", 2),
    ],
)
def test_extrinsic_control_points(
    kernel, control_points, normalization, n_steps
):
    """Test to run extrinsic deformation with control points."""
    mesh1 = mesh_3d
    mesh2 = mesh_3d

    mesh1.control_points = mesh1.bounding_grid(N=2, offset=0.25)

    # Define the model
    model = sks.ExtrinsicDeformation(
        n_steps=n_steps,
        kernel=kernel,
        scale=0.1,
        normalization=normalization,
        control_points=control_points,
    )

    loss = sks.OptimalTransportLoss() + 2 * sks.EmptyLoss()
    optimizer = sks.SGD()

    registration = sks.Registration(
        model=model,
        loss=loss,
        optimizer=optimizer,
        n_iter=1,
        regularization_weight=0.1,
        gpu=False,
        verbose=True,
    )

    registration.fit(source=mesh1, target=mesh2)
    if control_points:
        assert (
            registration.parameter_.shape == mesh1.control_points.points.shape
        )
    else:
        assert registration.parameter_.shape == mesh1.points.shape


def test_intrinsic_deformation_fixed_endpoints():
    """Test to run intrinsic deformation with fixed endpoints."""
    mesh = mesh_3d

    # Define the endpoints by adding a small offset to the points
    endpoints = mesh.points.clone() + 0.1

    # Define the model with fixed endpoints
    model = sks.IntrinsicDeformation(
        n_steps=2, endpoints=endpoints, metric="as_isometric_as_possible"
    )

    registration = sks.Registration(
        model=model,
        loss=sks.L2Loss(),
        optimizer=sks.LBFGS(),
        n_iter=1,
        regularization_weight=10,
    )

    # Fit the registration (same mesh as source and target)
    registration.fit(source=mesh, target=mesh)

    # Check that the endpoints are fixed
    assert torch.allclose(
        registration.transform(source=mesh).points, endpoints
    )


def test_debug():
    """Test the debug mode for registration."""

    # Define the source and target shapes
    source = mesh_3d
    target = mesh_3d
    target.points += 0.1

    # Define the model
    model = sks.RigidMotion()

    n_iter = 5  # Number of iterations

    # Define and fit the registration with debug=True
    registration = sks.Registration(
        model=model,
        loss=sks.L2Loss(),
        optimizer=sks.SGD(),
        n_iter=n_iter,
        regularization_weight=0,
        debug=True,
    )
    registration.fit(source=source, target=target)

    # Extract the gradients and parameters history
    gradients_list = registration.gradients_list_
    parameters_list = registration.parameters_list_

    expected_shape = model.parameter_shape(shape=source)

    # gradient_list and parameters_list should have the same length as n_iter
    assert len(gradients_list) == n_iter
    assert len(parameters_list) == n_iter

    for i in range(n_iter):
        # With SGD optimizer, there is only one call to the closure per iteration
        assert len(gradients_list[i]) == 1
        assert len(parameters_list[i]) == 1

        # Check the shape of the gradients and parameters
        assert gradients_list[i][0].shape == expected_shape
        assert parameters_list[i][0].shape == expected_shape
