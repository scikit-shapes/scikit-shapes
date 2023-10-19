import skshapes as sks
import torch
from pykeops.torch import LazyTensor
from typing import Optional


def test_squared_distance():
    N, M = 2000, 1000

    # Define two point clouds
    X = torch.randn(N, 3)
    Y = torch.randn(M, 3)

    # Define associated PolyData objects
    polydata_x = sks.PolyData(X)
    polydata_y = sks.PolyData(Y)

    # Define a signal on the first point cloud
    signal = torch.randn(N)

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


def test_convolution_trivial():
    #
    X = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=torch.float32)

    squared_distances = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 1], [2, 1, 1, 0]], dtype=torch.float32
    )

    scale = 0.01
    gaussian_kernel_torch = (-squared_distances / (2 * scale**2)).exp()

    gaussian_kernel_sks = sks.PolyData(X).point_convolution(
        kernel="gaussian",
        scale=scale,
    )

    a = torch.rand(4)
    print(gaussian_kernel_torch @ a)
    print(gaussian_kernel_sks @ a)
    assert torch.allclose(gaussian_kernel_torch @ a, gaussian_kernel_sks @ a)

    #
    Y = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32)

    squared_distances = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 1]], dtype=torch.float32
    )

    gaussian_kernel_torch = (-squared_distances / (2 * scale**2)).exp()

    gaussian_kernel_sks = sks.PolyData(X).point_convolution(
        kernel="gaussian", scale=scale, target=sks.PolyData(Y)
    )

    a = torch.rand(4)
    assert torch.allclose(gaussian_kernel_torch @ a, gaussian_kernel_sks @ a)


from hypothesis import given, settings
from hypothesis import strategies as st

from math import isnan

from .utils import create_point_cloud, create_shape


@given(
    N=st.integers(min_value=2, max_value=500),
    M=st.integers(min_value=2, max_value=500),
    scale=st.one_of(st.floats(min_value=0.01, max_value=100), st.none()),
    kernel=st.sampled_from(["gaussian", "uniform"]),
    normalize=st.booleans(),
    dim=st.integers(min_value=1, max_value=2),
)
@settings(deadline=None)
def test_convolution_functional(
    N: int,
    M: int,
    scale: Optional[float],
    kernel: str,
    normalize: bool,
    dim: int,
):
    X = torch.rand(N, 3).to(torch.float32)
    Y = torch.rand(M, 3).to(torch.float32)

    polydata_x = sks.PolyData(X)
    polydata_y = sks.PolyData(Y)
    weights_j = polydata_x.point_weights

    yi = Y.view(M, 1, 3)
    xj = X.view(1, N, 3)

    squared_distances = ((yi - xj) ** 2).sum(-1)

    try:
        if kernel == "gaussian":
            kernel_torch = (
                (-squared_distances / (2 * scale**2)).exp().clip(min=cutoff)
            )
        elif kernel == "uniform":
            kernel_torch = 1.0 * ((squared_distances / (scale**2)) <= 1)
    except:
        scale = None
        kernel_torch = torch.ones_like(squared_distances, dtype=torch.float32)

    if normalize:
        total_weights_i = (kernel_torch @ weights_j).clip(min=1e-6)
        norm_i = 1.0 / total_weights_i
        o_s = norm_i if dim == 1 else norm_i.view(-1, 1)
        i_s = weights_j if dim == 1 else weights_j.view(-1, 1)
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
        a = torch.rand(N)
    elif dim == 2:
        D = torch.randint(low=2, high=4, size=(1,))[0]
        a = torch.rand(N, D)

    A = o_s * (kernel_torch @ (i_s * a))
    B = kernel_sks @ a

    assert torch.allclose(o_s * (kernel_torch @ (i_s * a)), kernel_sks @ a, atol=1e-5)


def test_mesh_convolution():
    mesh = sks.Sphere()
    # define a constant signal
    signal = torch.rand(1) * torch.ones(mesh.n_points, dtype=torch.float32)

    # assert that the signal is unchanged by the convolution
    for weight_by_length in [True, False]:
        kernel = mesh.mesh_convolution(weight_by_length=weight_by_length)
        assert kernel.shape == (mesh.n_points, mesh.n_points)
        assert torch.allclose(kernel @ signal, signal)

def test_multidimensional_matrix_multiplication():
    """test the LinearOperator class for multidimensional matrix multiplication

    More precisely, we test that if M is a Linearoperator of shape (n, m) and A a
    tensor of shape (m, *t), then M @ A is well defined and results in a (n, *t)
    ensor, t being a tuple of integers.
    """

    # Define a LinearOperator of shape (n, m)
    randint = lambda x : torch.randint(1, x, (1,))[0]

    n, m = randint(10), randint(10)
    a, b, c = randint(10), randint(10), randint(10)

    matrix = torch.rand(n, m)
    M = sks.convolutions.LinearOperator(matrix)
    A = torch.rand(m, a, b, c)

    # assert that the @ operator is well defined and that the output has the right shape
    result = M @ A
    assert result.shape == (n, a, b, c)

    # take a random index (i, j, k, l) and assert that the output is correct
    # at this index
    i, j, k, l = randint(n), randint(a), randint(b), randint(c)
    assert result[i, j, k, l] == sum([matrix[i, ii] * A[ii, j, k, l] for ii in range(m)])



def test_multidimensional_signal_convolution():
    """Test that the mesh convolution operator is well defined for multidimensional signals
    
    The mesh convolution operator is defined as a matrix of shape (n_points, n_points)
    where n_points is the number of points of the mesh. In this test, we define a signal
    on the mesh, where each point is associated a tensor of shape (a, b, c) where a, b, c
    are random integers between 1 and 10. The mesh convolution operator is then applied
    to this signal, and the output signal is checked to have the right shape after matrix
    multiplication.
    """

    mesh = sks.Sphere()
    n_points = mesh.n_points

    # Set a multidimensional signal on the mesh, to each point is associated a tensor of shape (a, b, c)
    # where a, b, c are random integers between 1 and 10
    randtriplet = lambda n_max : (int(i) for i in torch.randint(1, n_max, (3,)))
    a, b, c = randtriplet(10)
    data = torch.rand(n_points, a, b, c)
    mesh["signal"] = data

    # Compute the mesh convolution operator
    M = mesh.mesh_convolution()
    # assert that the @ operator is well defined
    # and that the output signal has the expected shape
    assert M.shape == (n_points, n_points)
    assert mesh["signal"].shape == (n_points, a, b, c)
    assert (M @ data).shape == (n_points, a, b, c)
