"""Tests for the moments of a shape."""

import sys

import torch
import vedo as vd
from hypothesis import given, settings
from hypothesis import strategies as st

import skshapes as sks

from .utils import create_shape, vedo_frames

sys.path.append(sys.path[0][:-4])


def moments(X):
    """Compute the first 4 moments of a point cloud."""
    # Switch to Float64 precision for better accuracy
    X = X.double()
    N, D = X.shape
    XX = X.view(N, D, 1) * X.view(N, 1, D)  # (N, D, D)
    XXX = X.view(N, D, 1, 1) * X.view(N, 1, D, 1) * X.view(N, 1, 1, D)
    XXXX = (
        X.view(N, D, 1, 1, 1)
        * X.view(N, 1, D, 1, 1)
        * X.view(N, 1, 1, D, 1)
        * X.view(N, 1, 1, 1, D)
    )
    return X.mean(0), XX.mean(0), XXX.mean(0), XXXX.mean(0)


@given(
    n_points=st.integers(min_value=5, max_value=10),
    scale=st.floats(min_value=0.01, max_value=10),
    offset=st.floats(min_value=0.1, max_value=10),
)
@settings(max_examples=1000, deadline=None)
def test_moments_1(*, n_points: int, scale: float, offset: float):
    """Test the first 4 moments of a random shape."""
    points = scale * torch.randn(n_points, 3) + offset * torch.randn(1)
    shape = sks.PolyData(points=points)

    for central in [False, True]:
        gt = moments(points - points.mean(0)) if central else moments(points)

        for order in [1, 2, 3, 4]:
            mom = shape.point_moments(
                order=order, scale=None, central=central, dtype="double"
            )
            mom = mom[0]
            assert torch.allclose(
                mom.double(), gt[order - 1], atol=1e-4, rtol=1e-2
            )


def display_moments(*, scale=1, **kwargs):
    """Display moments of a shape (not a test)."""
    shape = create_shape(**kwargs)

    local_average = shape.point_moments(order=1, scale=scale)
    local_cov = shape.point_moments(order=2, scale=scale, central=True)

    local_QL = torch.linalg.eigh(local_cov)
    local_nuv = local_QL.eigenvectors  # (N, 3, 3)

    local_frame = (
        local_QL.eigenvectors * local_QL.eigenvalues.view(-1, 1, 3).sqrt()
    )
    local_frame = local_frame / scale

    # Pick 10 points at random:
    if shape.n_points > 100:
        mask = torch.randperm(shape.n_points)[:100]
    else:
        mask = torch.arange(shape.n_points)

    # To color our points, let's visualize the window around point 0:
    density = torch.zeros_like(shape.points[:, 0])
    density[0] = 1
    Conv = shape.point_convolution(scale=scale, normalize=True)
    density = Conv.T @ density
    assert density.shape == (len(shape.points),)

    # Let's also use the ratio of eigenvalues (as a proxy for curvature):
    anisotropy = local_QL.eigenvalues[:, 0] / local_QL.eigenvalues.sum(-1)

    point_frame = shape.point_frames(scale=scale).transpose(1, 2)
    curvedness = shape.point_curvedness(scale=scale)
    shape_index = shape.point_shape_indices(scale=scale).abs()

    # Our surface points:
    if shape.triangles is None:
        spheres = vd.Points(
            shape.points, c=(0.5, 0.5, 0.5), r=60 / (shape.n_points ** (1 / 3))
        )
    else:
        spheres = shape.to_vedo()

    spheres_1 = (
        spheres.clone()
        .alpha(0.5)
        .cmap("viridis", density, vmin=0)
        .add_scalarbar()
    )
    spheres_2 = (
        spheres.clone()
        .cmap("viridis", (3 * anisotropy) ** (1 / 3), vmin=0)
        .add_scalarbar()
    )
    spheres_3 = (
        spheres.clone().cmap("viridis", curvedness, vmin=0).add_scalarbar()
    )
    spheres_4 = (
        spheres.clone()
        .cmap("RdBu_r", shape_index, vmin=-1, vmax=1)
        .add_scalarbar()
    )

    # Vectors to the local average:
    quiver = vd.Arrows(
        shape.points[mask], local_average[mask], c="green", alpha=0.9
    )

    # Local tangent frames:
    s = 1.5 * shape.standard_deviation[0] / len(mask) ** (1 / 2)

    plt = vd.Plotter(shape=(2, 2), axes=1)
    plt.at(0).show(
        spheres_1,
        quiver,
        vd.Text2D("Density around point 0", pos="top-middle"),
    )
    plt.at(1).show(
        spheres_2,
        *vedo_frames(shape.points[mask], s * local_nuv[mask]),
        vd.Text2D("Anisotropy", pos="top-middle"),
    )
    plt.at(2).show(
        spheres_3,
        *vedo_frames(shape.points[mask], s * local_frame[mask]),
        vd.Text2D("Curvedness", pos="top-middle"),
    )
    plt.at(3).show(
        spheres_4,
        *vedo_frames(shape.points[mask], s * point_frame[mask]),
        vd.Text2D("Shape index", pos="top-middle"),
    )
    plt.interactive()


if __name__ == "__main__":
    functions = [
        lambda x, y: 0 * x + 0.3 * y + 0.2,
        lambda x, y: (2 - 0.5 * x**2 - y**2).abs().sqrt() - 1,
        lambda x, y: x**2 - y**2,
    ]
    fnames = [
        "~/data/PN1.stl",
    ]

    if False:
        for f in functions:
            display_moments(function=f, scale=0.3, n_points=15, noise=0.01)
    else:
        for f in fnames:
            display_moments(file_name=f, scale=10.0, n_points=1e4)
