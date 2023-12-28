import skshapes as sks
import torch
from pyvista import examples
from skshapes.utils import scatter
from hypothesis import given, settings
from hypothesis import strategies as st


def test_multiscale_api():

    mesh = sks.Sphere()

    # Initialize with ratios
    ratios = [0.01, 0.1, 0.5, 1]
    M = sks.Multiscale(mesh, ratios=ratios)

    # Initialize with n_points
    n_points = [10, 100]
    M = sks.Multiscale(mesh, n_points=n_points)

    # Parallel initialization
    mesh1 = sks.Sphere()
    mesh2 = sks.Sphere()
    decimation_module = sks.Decimation(n_points=1).fit(mesh1)

    M1 = sks.Multiscale(
        mesh1, n_points=n_points, decimation_module=decimation_module
    )

    M2 = sks.Multiscale(
        mesh2, n_points=n_points, decimation_module=decimation_module
    )

    ratio = torch.rand(1).item()
    assert torch.allclose(
        M1.at(ratio=ratio).points,
        M2.at(ratio=ratio).points,
    )

    # propagate a signal with default policies
    M1.at(n_points=100)["test_signal"] = torch.rand(
        M1.at(n_points=100).n_points
    )
    M1.propagate(
        signal_name="test_signal",
        from_n_points=100,
    )


def test_scattrer_toy():
    """A toy example to test the scatter function"""

    # dimension 1
    src = torch.tensor([1, -1, 0.5], dtype=sks.float_dtype)
    index = torch.tensor([0, 1, 1], dtype=sks.int_dtype)

    assert torch.allclose(
        scatter(src=src, index=index, reduce="mean"),
        torch.tensor([1, -0.25], dtype=sks.float_dtype),
    )
    assert torch.allclose(
        scatter(src=src, index=index, reduce="sum"),
        torch.tensor([1, -0.5], dtype=sks.float_dtype),
    )
    assert torch.allclose(
        scatter(src=src, index=index, reduce="min"),
        torch.tensor([1, -1], dtype=sks.float_dtype),
    )
    assert torch.allclose(
        scatter(src=src, index=index, reduce="max"),
        torch.tensor([1, 0.5], dtype=sks.float_dtype),
    )

    # Check that scatter is working when the indices are not consecutive

    index = torch.tensor([0, 1, 3], dtype=sks.int_dtype)
    assert torch.allclose(
        scatter(src=src, index=index, reduce="mean"),
        torch.tensor([1, -1, 0, 0.5], dtype=sks.float_dtype),
    )

    # Check the min_length parameter of scatter and blank_value != 0
    index = torch.tensor([0, 1, 1], dtype=sks.int_dtype)
    assert torch.allclose(
        scatter(
            src=src, index=index, reduce="mean", min_length=4, blank_value=2
        ),
        torch.tensor([1, -0.25, 2, 2], dtype=sks.float_dtype),
    )


@given(
    n=st.integers(min_value=1, max_value=500),
    n_dim=st.integers(min_value=0, max_value=2),
)
@settings(deadline=None)
def test_scatter_multidim(n, n_dim):
    """Assert that the scatter function works as expected for multidimensional
    signals"""

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


@given(
    init_type=st.sampled_from(["ratios", "n_points"]),
)
@settings(deadline=None)
def test_multiscale(init_type):
    """Test the multiscale class for triangle meshes.

    This test is based on the bunny example from pyvista.examples (~34K points)

    We initialize a multiscale object with 10 random ratios, pick two random
    ratios and test the signal propagation from high to low resolution and
    back, specifically the composition of two propagations.
    """

    # Load a triangle mesh (~34k points)
    mesh = sks.PolyData(examples.download_bunny())

    if init_type == "ratios":
        # Pick 10 random ratios
        ratios = torch.rand(10)
        # Create the multiscale object
        M = sks.Multiscale(mesh, ratios=ratios)
    elif init_type == "n_points":
        # Pick 10 random n_points
        n_points = torch.randint(1, mesh.n_points, (10,))
        # Create the multiscale object
        M = sks.Multiscale(mesh, n_points=n_points)

    # Pick a 2 random ratios
    r = torch.rand(2)
    coarse_ratio, intermediate_ratio = r.min(), r.max()
    coarse_ratio = float(coarse_ratio)
    intermediate_ratio = float(intermediate_ratio)

    # Test indice mapping
    im = M.indice_mapping(
        fine_ratio=intermediate_ratio, coarse_ratio=coarse_ratio
    )
    # Check that the number of points is correct
    assert im.shape[0] == M.at(ratio=intermediate_ratio).n_points
    # Check that the indices are correct
    assert (im.max() + 1) == M.at(ratio=coarse_ratio).n_points

    # Test composition of indice mapping
    im_from_intermediate_ratio_to_coarse_ratio = M.indice_mapping(
        fine_ratio=1, coarse_ratio=coarse_ratio
    )
    im1 = M.indice_mapping(fine_ratio=1, coarse_ratio=intermediate_ratio)
    im2 = M.indice_mapping(
        fine_ratio=intermediate_ratio, coarse_ratio=coarse_ratio
    )
    assert torch.allclose(im_from_intermediate_ratio_to_coarse_ratio, im2[im1])

    # Default policies
    coarse_to_fine_policy = sks.CoarseToFinePolicy(smoothing="constant")

    # Test signal propagation from high to low resolution
    signal_intermediate_ratio = torch.rand(M.at(ratio=1).n_points)
    reduce_options = ["sum", "min", "max"]
    # Composition with "mean" doesn't work because of weighting
    i = torch.randint(0, len(reduce_options), (1,)).item()
    reduce = reduce_options[i]
    fine_to_coarse_policy = sks.FineToCoarsePolicy(reduce=reduce)
    # Direct propagation from ratio 1 to coarse_ratio
    low_resol_signal = M._signal_from_one_scale_to_another(
        source_signal=signal_intermediate_ratio,
        source_ratio=1,
        target_ratio=coarse_ratio,
        fine_to_coarse_policy=fine_to_coarse_policy,
        coarse_to_fine_policy=coarse_to_fine_policy,
    )

    # Composition of two propagations
    intermediate_signal = M._signal_from_one_scale_to_another(
        source_signal=signal_intermediate_ratio,
        source_ratio=1,
        target_ratio=intermediate_ratio,
        fine_to_coarse_policy=fine_to_coarse_policy,
        coarse_to_fine_policy=coarse_to_fine_policy,
    )
    b = M._signal_from_one_scale_to_another(
        source_signal=intermediate_signal,
        source_ratio=intermediate_ratio,
        target_ratio=coarse_ratio,
        fine_to_coarse_policy=fine_to_coarse_policy,
        coarse_to_fine_policy=coarse_to_fine_policy,
    )

    # Check that the two signals are equal
    assert torch.allclose(low_resol_signal, b)

    # Test signal propagation from low to high resolution with constant
    # smoothing
    tmp = M._signal_from_one_scale_to_another(
        source_signal=low_resol_signal,
        source_ratio=coarse_ratio,
        target_ratio=intermediate_ratio,
        fine_to_coarse_policy=fine_to_coarse_policy,
        coarse_to_fine_policy=sks.CoarseToFinePolicy(smoothing="constant"),
    )

    assert tmp.shape[0] == M.at(ratio=intermediate_ratio).n_points

    # Propagate again to low resolution and check that we recover the original
    # signal
    # with reduce="min", "max" or "mean"
    back = M._signal_from_one_scale_to_another(
        source_signal=tmp,
        source_ratio=intermediate_ratio,
        target_ratio=coarse_ratio,
        fine_to_coarse_policy=sks.FineToCoarsePolicy(reduce="min"),
        coarse_to_fine_policy=coarse_to_fine_policy,
    )

    back2 = M._signal_from_one_scale_to_another(
        source_signal=tmp,
        source_ratio=intermediate_ratio,
        target_ratio=coarse_ratio,
        fine_to_coarse_policy=sks.FineToCoarsePolicy(reduce="max"),
        coarse_to_fine_policy=coarse_to_fine_policy,
    )

    back3 = M._signal_from_one_scale_to_another(
        source_signal=tmp,
        source_ratio=intermediate_ratio,
        target_ratio=coarse_ratio,
        fine_to_coarse_policy=sks.FineToCoarsePolicy(reduce="mean"),
        coarse_to_fine_policy=coarse_to_fine_policy,
    )

    assert torch.allclose(back, low_resol_signal)
    assert torch.allclose(back2, low_resol_signal)
    assert torch.allclose(back3, low_resol_signal)

    return None

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


@given(
    smoothing=st.sampled_from(["mesh_convolution", "constant"]),
)
@settings(deadline=None)
def test_multiscale_signal_api(smoothing):
    """Test the multiscale signal API"""

    mesh = sks.PolyData(examples.download_bunny())
    ratios = [0.01, 0.1, 0.5, 1]

    coarse_to_fine_policy = sks.CoarseToFinePolicy(
        smoothing=smoothing,
    )
    fine_to_coarse_policy = sks.FineToCoarsePolicy(
        reduce="mean",
    )

    M = sks.Multiscale(mesh, ratios=ratios)

    n = M.at(ratio=0.1).n_points
    M.at(ratio=0.1)["test_signal"] = torch.rand(n)
    assert "test_signal" not in M.at(ratio=0.5).point_data.keys()

    M.propagate(
        signal_name="test_signal",
        from_ratio=0.1,
        coarse_to_fine_policy=coarse_to_fine_policy,
        fine_to_coarse_policy=fine_to_coarse_policy,
    )

    M.at(ratio=0.5)["test_signal"] = torch.rand(M.at(ratio=0.5).n_points)

    assert "test_signal" in M.at(ratio=0.5).point_data.keys()

    M.append(ratio=0.2)

    assert "test_signal" not in M.at(ratio=0.2).point_data.keys()
