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

    # Compute the linear operator from the PolyData objects
    Lo2 = sks.convolutions.convolution(
        source=polydata_x,
        target=polydata_y,
        kernel="gaussian",
        scale=scale,
        normalize=False,
    )

    signal_out2 = Lo2 @ signal
    assert signal_out2.shape == (M,)

    assert torch.allclose(signal_out1, signal_out2)

    Lo3 = polydata_x.point_convolution(
        kernel="gaussian",
        scale=scale,
        normalize=False,
        target=polydata_y,
    )

    assert Lo3.shape == (M, N)
    signal_out3 = Lo3 @ signal
    assert torch.allclose(signal_out1, signal_out3)


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

import numpy as np
import vedo as vd

from .utils import create_point_cloud, create_shape


def test_uniform():
    X = torch.rand(100, 3).to(torch.float32)

    K = sks.PolyData(X).point_convolution(
        kernel="uniform",
        scale=0.01,
    )


@given(
    N=st.integers(min_value=2, max_value=500),
    M=st.integers(min_value=2, max_value=500),
    scale=st.floats(min_value=0.01, max_value=100),
    kernel=st.sampled_from(["gaussian", "uniform"]),
)
@settings(deadline=None)
def test_convolution_functional(N: int, M: int, scale: float, kernel: str):
    scale = scale

    X = torch.rand(N, 3).to(torch.float32)
    Y = torch.rand(M, 3).to(torch.float32)

    yi = Y.view(M, 1, 3)
    xj = X.view(1, N, 3)

    squared_distances = ((yi - xj) ** 2).sum(-1)

    if kernel == "gaussian":
        kernel_torch = (-squared_distances / (2 * scale**2)).exp()
    elif kernel == "uniform":
        kernel_torch = 1.0 * ((squared_distances / (scale**2)) <= 1)

    kernel_sks = sks.PolyData(X).point_convolution(
        kernel=kernel, scale=scale, target=sks.PolyData(Y)
    )

    a = torch.rand(N)

    if kernel == "uniform":
        print(torch.max(torch.abs(kernel_torch @ a - kernel_sks @ a)))
    assert torch.allclose(kernel_torch @ a, kernel_sks @ a, atol=1e-6)
