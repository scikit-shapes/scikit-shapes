import skshapes as sks
import torch
import numpy as np

from pyvista import examples
import pyvista


from skshapes.utils import scatter

from hypothesis import given, settings
from hypothesis import strategies as st


def test_scattrer_toy():
    """A toy example to test the scatter function"""

    # dimension 1
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


@given(
    n=st.integers(min_value=1, max_value=500),
    n_dim=st.integers(min_value=0, max_value=2),
)
@settings(deadline=None)
def test_scatter_multidim(n, n_dim):
    """Assert that the scatter function works as expected for multidimensional signals"""

    # dimension d
    d = torch.randint(1, 10, (n_dim,))

    src = torch.rand(n, *d)
    print(src.shape)

    index = torch.randint(0, n, (n,)).view(-1)

    # Convert the index to consecutive integers
    index2 = index.clone()
    for i, v in enumerate(torch.unique(index)):
        index2[index == v] = i
    index = index2

    output_multidim = scatter(src=src, index=index, reduce="mean")

    if n_dim == 1:
        i = torch.randint(0, src.shape[1], (1,))
        output = scatter(src=src[:, i].view(-1), index=index, reduce="mean")
        assert torch.allclose(output, output_multidim[:, i].view(-1))

    elif n_dim == 2:
        i = torch.randint(0, src.shape[1], (1,))
        j = torch.randint(0, src.shape[2], (1,))
        output = scatter(src=src[:, i, j].view(-1), index=index, reduce="mean")
        assert torch.allclose(output, output_multidim[:, i, j].view(-1))

    elif n_dim == 0:
        output = scatter(src=src.view(-1), index=index, reduce="mean")
        assert torch.allclose(output, output_multidim.view(-1))


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
    coarse_ratio, intermediate_ratio = r.min(), r.max()
    coarse_ratio = float(coarse_ratio)
    intermediate_ratio = float(intermediate_ratio)

    # Test indice mapping
    im = M.indice_mapping(fine_ratio=intermediate_ratio, coarse_ratio=coarse_ratio)
    # Check that the number of points is correct
    assert im.shape[0] == M.at(intermediate_ratio).n_points
    # Check that the indices are correct
    assert (im.max() + 1) == M.at(coarse_ratio).n_points

    # Test composition of indice mapping
    im_from_intermediate_ratio_to_coarse_ratio = M.indice_mapping(
        fine_ratio=1, coarse_ratio=coarse_ratio
    )
    im1 = M.indice_mapping(fine_ratio=1, coarse_ratio=intermediate_ratio)
    im2 = M.indice_mapping(fine_ratio=intermediate_ratio, coarse_ratio=coarse_ratio)
    assert torch.allclose(im_from_intermediate_ratio_to_coarse_ratio, im2[im1])

    # Test signal propagation from high to low resolution
    signal_intermediate_ratio = torch.rand(M.at(1).n_points)
    reduce_options = ["sum", "min", "max"]
    # Composition with "mean" doesn't work because of weighting
    i = torch.randint(0, len(reduce_options), (1,)).item()
    reduce = reduce_options[i]
    # Direct propagation from ratio 1 to coarse_ratio
    low_resol_signal = M.signal_from_fine_to_coarse(
        signal_intermediate_ratio,
        fine_ratio=1,
        coarse_ratio=coarse_ratio,
        reduce=reduce,
    )
    # Composition of two propagations
    intermediate_signal = M.signal_from_fine_to_coarse(
        signal_intermediate_ratio,
        fine_ratio=1,
        coarse_ratio=intermediate_ratio,
        reduce=reduce,
    )
    b = M.signal_from_fine_to_coarse(
        intermediate_signal,
        fine_ratio=intermediate_ratio,
        coarse_ratio=coarse_ratio,
        reduce=reduce,
    )
    # Check that the two signals are equal
    assert torch.allclose(low_resol_signal, b)

    # Test signal propagation from low to high resolution with constant smoothing
    tmp = M.signal_from_coarse_to_fine(
        low_resol_signal,
        coarse_ratio=coarse_ratio,
        fine_ratio=intermediate_ratio,
        smoothing="constant",
    )
    assert tmp.shape[0] == M.at(intermediate_ratio).n_points

    # Propagate again to low resolution and check that we recover the original signal
    # with reduce="min", "max" or "mean"
    back = M.signal_from_fine_to_coarse(
        tmp, fine_ratio=intermediate_ratio, coarse_ratio=coarse_ratio, reduce="min"
    )
    back2 = M.signal_from_fine_to_coarse(
        tmp, fine_ratio=intermediate_ratio, coarse_ratio=coarse_ratio, reduce="max"
    )
    back3 = M.signal_from_fine_to_coarse(
        tmp, fine_ratio=intermediate_ratio, coarse_ratio=coarse_ratio, reduce="mean"
    )
    assert torch.allclose(back, low_resol_signal)
    assert torch.allclose(back2, low_resol_signal)

    signal = torch.rand(M.at(1).n_points)
    signal_out = M.signal_convolution(
        signal,
        signal_ratio=1,
        target_ratio=coarse_ratio,
        kernel="gaussian",
        scale=0.01,
        normalize=True,
    )

    assert signal_out.shape[0] == M.at(coarse_ratio).n_points


def test_multiscale_signal_api():
    """Test the multiscale signal API"""

    mesh = sks.PolyData(examples.download_bunny())
    ratios = [0.01, 0.1, 0.5, 1]

    coarse_to_fine_policy = {"smoothing": "constant"}
    fine_to_coarse_policy = {"reduce": "mean"}

    M = sks.Multiscale(mesh, ratios=ratios)

    n = M.at(0.1).n_points
    M.at(0.1)["test_signal"] = torch.rand(n)
    assert "test_signal" not in M.at(0.5).point_data.keys()

    M.add_signal(
        key="test_signal",
        at=0.1,
        coarse_to_fine_policy=coarse_to_fine_policy,
        fine_to_coarse_policy=fine_to_coarse_policy,
    )

    M.at(0.5)["test_signal"] = torch.rand(M.at(0.5).n_points)

    assert "test_signal" in M.at(0.5).point_data.keys()

    M.add_ratio(0.2)
    n_points = M.at(0.2).n_points
    assert M.at(0.2)["test_signal"].shape[0] == n_points


if False:
    M = sks.Multiscale(mesh, n_points=(100, 1000, 10000))
    M.at(n_points=1000)["color"] = torch.rand(1000)
    M.propagate("color", from_n_points=1000, to_n_points=100, policy="mean")
    M.propagate("color", from_n_points=100, to_n_points=1000, policy="smooth")

    M.propagate("color", from_n_points=100, to_n_points="all", policy="smooth")
    M.add_ratio(0.5, propagate=None, policy=None)
    M.propagate(...)

    policy = sks.Policy(
        smoothing="mesh_convolution",
        n_smoothing_steps=10,
        pass_through_all_scales=True,
    )
