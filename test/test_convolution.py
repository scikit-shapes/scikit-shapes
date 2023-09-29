import skshapes as sks
import torch
from pykeops.torch import LazyTensor


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
