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
