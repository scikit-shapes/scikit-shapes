import skshapes as sks
import torch
import numpy as np

from pyvista import examples
import pyvista


from skshapes.utils import scatter


def test_scatter():
    """A toy example to test the scatter function"""

    src = torch.tensor([1, -1, 0.5], dtype=torch.float32)
    index = torch.tensor([0, 1, 1], dtype=torch.int64)

    assert torch.allclose(
        scatter(src=src, index=index, reduce="mean"),
        torch.tensor([1, -0.25], dtype=torch.float32),
    )
    assert torch.allclose(
        scatter(src=src, index=index, reduce="sum"),
        torch.tensor([1, -0.5], dtype=torch.float32),
    )
    assert torch.allclose(
        scatter(src=src, index=index, reduce="min"),
        torch.tensor([1, -1], dtype=torch.float32),
    )
    assert torch.allclose(
        scatter(src=src, index=index, reduce="max"),
        torch.tensor([1, 0.5], dtype=torch.float32),
    )


def test_multiscale():
    """Test the multiscale class for triangle meshes.

    This test is based on the bunny example from pyvista.examples (~34K points).)

    We initialize a multiscale object with 10 random ratios, pick two random ratios
    and test the signal propagation from high to low resolution and back, specifically
    the composition of two propagations.
    """

    # Load a triangle mesh (~34k points)
    mesh = sks.PolyData(examples.download_bunny())

    # Pick 10 random ratios
    ratios = torch.rand(10)

    # Create the multiscale object
    M = sks.Multiscale(mesh, ratios=ratios)

    # Print the number of points at each ratio
    for ratio in torch.sort(ratios)[0]:
        print(f"ratio : {ratio}, n_points = {M.at(float(ratio)).n_points}")
    print(f"ratio : {1}, n_points = {M.at(1).n_points}")

    # Pick a 2 random ratios
    r = torch.rand(2)
    low_res_ratio, intermediate_ratio = r.min(), r.max()
    low_res_ratio = float(low_res_ratio)
    intermediate_ratio = float(intermediate_ratio)

    # Test indice mapping
    im = M.indice_mapping(high_res=intermediate_ratio, low_res=low_res_ratio)
    # Check that the number of points is correct
    assert im.shape[0] == M.at(intermediate_ratio).n_points
    # Check that the indices are correct
    assert (im.max() + 1) == M.at(low_res_ratio).n_points

    # Test composition of indice mapping
    im_from_intermediate_ratio_to_low_res_ratio = M.indice_mapping(
        high_res=1, low_res=low_res_ratio
    )
    im1 = M.indice_mapping(high_res=1, low_res=intermediate_ratio)
    im2 = M.indice_mapping(high_res=intermediate_ratio, low_res=low_res_ratio)
    assert torch.allclose(im_from_intermediate_ratio_to_low_res_ratio, im2[im1])

    # Test signal propagation from high to low resolution
    signal_intermediate_ratio = torch.rand(M.at(1).n_points)
    reduce_options = ["sum", "min", "max"]
    # Composition with "mean" doesn't work because of weighting
    i = torch.randint(0, len(reduce_options), (1,)).item()
    reduce = reduce_options[i]
    # Direct propagation from ratio 1 to low_res_ratio
    low_resol_signal = M.signal_from_high_to_low_res(
        signal_intermediate_ratio, high_res=1, low_res=low_res_ratio, reduce=reduce
    )
    # Composition of two propagations
    intermediate_signal = M.signal_from_high_to_low_res(
        signal_intermediate_ratio, high_res=1, low_res=intermediate_ratio, reduce=reduce
    )
    b = M.signal_from_high_to_low_res(
        intermediate_signal,
        high_res=intermediate_ratio,
        low_res=low_res_ratio,
        reduce=reduce,
    )
    # Check that the two signals are equal
    assert torch.allclose(low_resol_signal, b)

    # Test signal propagation from low to high resolution with constant smoothing
    tmp = M.signal_from_low_to_high_res(
        low_resol_signal,
        low_res=low_res_ratio,
        high_res=intermediate_ratio,
        smoothing="constant",
    )
    assert tmp.shape[0] == M.at(intermediate_ratio).n_points

    # Propagate again to low resolution and check that we recover the original signal
    # with reduce="min", "max" or "mean"
    back = M.signal_from_high_to_low_res(
        tmp, high_res=intermediate_ratio, low_res=low_res_ratio, reduce="min"
    )
    back2 = M.signal_from_high_to_low_res(
        tmp, high_res=intermediate_ratio, low_res=low_res_ratio, reduce="max"
    )
    back3 = M.signal_from_high_to_low_res(
        tmp, high_res=intermediate_ratio, low_res=low_res_ratio, reduce="mean"
    )
    assert torch.allclose(back, low_resol_signal)
    assert torch.allclose(back2, low_resol_signal)
