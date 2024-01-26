"""Tests for the convolution module."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import skshapes as sks


def test_mesh_convolution_point_cloud():
    """Test error handling for mesh_convolution on a point cloud."""
    points = torch.rand(10, 3)
    pc = sks.PolyData(points=points)
    with pytest.raises(
        AttributeError,
        match="Mesh convolution is only defined on"
        + " triangle meshes or wireframe PolyData",
    ):
        pc.mesh_convolution()


def test_squared_distance():
    """Test the squared distance function."""
    N, M = 2000, 1000

    # Define two point clouds
    X = torch.randn(N, 3, dtype=sks.float_dtype)
    Y = torch.randn(M, 3, dtype=sks.float_dtype)

    # Define associated PolyData objects
    polydata_x = sks.PolyData(X)
    polydata_y = sks.PolyData(Y)

    # Define a signal on the first point cloud
    signal = torch.randn(N, dtype=sks.float_dtype)

    scale = 0.01
    sqrt_2 = 1.41421356237

    from skshapes.convolutions.squared_distances import squared_distances

    K_ij = squared_distances(
        points=X / (sqrt_2 * scale),
        kernel=lambda d2: (-d2).exp(),
        target_points=Y / (sqrt_2 * scale),
    )

    assert K_ij.shape == (M, N)
    signal_out1 = K_ij @ signal
    assert signal_out1.shape == (M,)

    # Define a linear operator
    Lo = sks.convolutions.LinearOperator(K_ij)
    assert Lo.shape == (M, N)

    Lo2 = polydata_x.point_convolution(
        kernel="gaussian",
        scale=scale,
        normalize=False,
        target=polydata_y,
    )

    assert Lo2.shape == (M, N)
    signal_out2 = Lo2 @ signal
    assert torch.allclose(signal_out1, signal_out2)


def test_convolution_simple():
    """Test the convolution results by comparing with direct computations."""
    X = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=sks.float_dtype
    )

    squared_distances = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 1], [2, 1, 1, 0]],
        dtype=sks.float_dtype,
    )

    scale = 0.01
    gaussian_kernel_torch = (-squared_distances / (2 * scale**2)).exp()

    gaussian_kernel_sks = sks.PolyData(X).point_convolution(
        kernel="gaussian",
        scale=scale,
    )

    a = torch.rand(4).to(sks.float_dtype)
    assert torch.allclose(gaussian_kernel_torch @ a, gaussian_kernel_sks @ a)

    #
    Y = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=sks.float_dtype)

    squared_distances = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 1]], dtype=sks.float_dtype
    )

    gaussian_kernel_torch = (-squared_distances / (2 * scale**2)).exp()

    gaussian_kernel_sks = sks.PolyData(X).point_convolution(
        kernel="gaussian", scale=scale, target=sks.PolyData(Y)
    )

    a = torch.rand(4).to(sks.float_dtype)
    assert torch.allclose(gaussian_kernel_torch @ a, gaussian_kernel_sks @ a)


@given(
    N=st.integers(min_value=2, max_value=500),
    M=st.integers(min_value=2, max_value=500),
    scale=st.floats(min_value=0.01, max_value=100),
    kernel=st.sampled_from(["gaussian", "uniform"]),
    normalize=st.booleans(),
    dim=st.integers(min_value=1, max_value=2),
)
@settings(deadline=None)
def test_convolution_functional(
    N: int,
    M: int,
    scale: float,
    kernel: str,
    normalize: bool,
    dim: int,
):
    """Test the convolution results by comparing with direct computations."""
    # Sample two point clouds
    X = torch.rand(N, 3).to(sks.float_dtype)
    Y = torch.rand(M, 3).to(sks.float_dtype)

    # Create two PolyData objects from the point clouds
    polydata_x = sks.PolyData(X)
    polydata_y = sks.PolyData(Y)

    weights_j = polydata_x.point_weights

    assert weights_j.dtype == sks.float_dtype
    assert polydata_x.points.dtype == sks.float_dtype
    assert polydata_y.points.dtype == sks.float_dtype

    # Compute the squared distances between the two point clouds
    yi = Y.view(M, 1, 3) / scale
    xj = X.view(1, N, 3) / scale
    squared_distances = ((yi - xj) ** 2).sum(-1)

    if kernel == "gaussian":
        kernel_torch = (-squared_distances / 2).exp()
    elif kernel == "uniform":
        kernel_torch = 1.0 * ((squared_distances) <= 1)

    kernel_torch = kernel_torch.to(sks.float_dtype)
    # assert kernel_torch.dtype == sks.float_dtype

    if normalize:
        total_weights_i = (kernel_torch @ weights_j).clip(min=1e-6)
        norm_i = 1.0 / total_weights_i
        o_s = norm_i if dim == 1 else norm_i.view(-1, 1)
        i_s = weights_j if dim == 1 else weights_j.view(-1, 1)

        assert i_s.dtype == sks.float_dtype
        assert o_s.dtype == sks.float_dtype
    else:
        o_s = 1
        i_s = 1

    kernel_sks = polydata_x.point_convolution(
        kernel=kernel,
        scale=scale,
        normalize=normalize,
        target=polydata_y,
    )

    if dim == 1:
        a = torch.rand(N, dtype=sks.float_dtype)
    elif dim == 2:
        D = torch.randint(low=2, high=4, size=(1,))[0]
        a = torch.rand(N, D, dtype=sks.float_dtype)

    assert a.dtype == sks.float_dtype
    assert kernel_torch.dtype == sks.float_dtype
    assert (kernel_sks @ a).dtype == sks.float_dtype

    A = o_s * (kernel_torch @ (i_s * a))
    B = kernel_sks @ a

    assert torch.allclose(A, B, atol=1e-4)


def test_mesh_convolution_constant_signal():
    """Test that a constant signal is kept unchanged."""
    mesh = sks.Sphere()
    # define a constant signal
    signal = torch.rand(1) * torch.ones(mesh.n_points, dtype=sks.float_dtype)

    # assert that the signal is unchanged by the convolution
    for weight_by_length in [True, False]:
        kernel = mesh.mesh_convolution(weight_by_length=weight_by_length)
        assert kernel.shape == (mesh.n_points, mesh.n_points)
        assert torch.allclose(kernel @ signal, signal)


def test_multidimensional_matrix_multiplication():
    """Test the LinearOperator class for multidimensional matrix product.

    More precisely, we test that if M is a Linearoperator of shape (n, m) and
    A a tensor of shape (m, *t), then M @ A is well defined and results in a
    (n, *t) tensor, t being a tuple of integers.
    """

    # Define a LinearOperator of shape (n, m)
    def randint(up, low=1):
        return torch.randint(low, up, (1,))[0] if up > low else low

    n, m = randint(10, low=2), randint(10, low=2)
    a, b, c = randint(10, low=2), randint(10, low=2), randint(10, low=2)

    matrix = torch.rand(n, m).to(sks.float_dtype)
    M = sks.convolutions.LinearOperator(matrix)
    A = torch.rand(m, a, b, c).to(sks.float_dtype)

    # assert that the @ operator is well defined and that the output has the
    # right shape
    result = M @ A
    assert result.shape == (n, a, b, c)

    # take a random index (i, j, k, l) and assert that the output is correct
    # at this index
    i, j, k, l = randint(n), randint(a), randint(b), randint(c)  # noqa: E741
    assert torch.isclose(
        result[i, j, k, l],
        sum([matrix[i, ii] * A[ii, j, k, l] for ii in range(m)]).to(
            sks.float_dtype
        ),
    )


def test_multidimensional_signal_convolution():
    """Test well defineness for multidimensional signals.

    The mesh convolution operator is defined as a matrix of shape
    (n_points, n_points) where n_points is the number of points of the mesh. In
    this test, we define a signal on the mesh, where each point is associated a
    tensor of shape (a, b, c) where a, b, c are random integers between 1 and
    10. The mesh convolution operator is then applied to this signal, and the
    output signal is checked to have the right shape after matrix
    multiplication.
    """
    mesh = sks.Sphere()
    n_points = mesh.n_points

    # Set a multidimensional signal on the mesh, to each point is associated a
    # tensor of shape (a, b, c) where a, b, c are random integers between 1 and
    # 10
    def randtriplet(n_max):
        return (int(i) for i in torch.randint(1, n_max, (3,)))

    a, b, c = randtriplet(10)
    data = torch.rand(n_points, a, b, c).to(sks.float_dtype)
    mesh["signal"] = data

    # Compute the mesh convolution operator
    M = mesh.mesh_convolution()
    # assert that the @ operator is well defined
    # and that the output signal has the expected shape
    assert M.shape == (n_points, n_points)
    assert mesh["signal"].shape == (n_points, a, b, c)
    assert (M @ data).shape == (n_points, a, b, c)
