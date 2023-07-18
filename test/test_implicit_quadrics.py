import sys

sys.path.append(sys.path[0][:-4])

import torch
import skshapes as sks

from hypothesis import given, settings
from hypothesis import strategies as st

import numpy as np
import vedo as vd


def create_point_cloud(n_points: int, f: callable):
    """Create a point cloud from a function f: R^3 -> R."""
    x = torch.linspace(-1, 1, n_points)
    y = torch.linspace(-1, 1, n_points)
    x, y = torch.meshgrid(x, y, indexing="ij")
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = f(x, y)

    N = len(x)
    assert N == n_points**2

    points = torch.stack([x, y, z], dim=1).view(N, 3)
    return points


def dim4(*, points, offset=0, scale=1):
    """Rescales and adds a fourth dimension to the point cloud."""
    assert points.shape == (len(points), 3)
    points = points - offset
    points = points / scale
    return torch.cat([points, torch.ones_like(points[:, :1])], dim=1)


def quadratic_function(*, points, quadric, offset=0, scale=1):
    assert quadric.shape == (4, 4)
    X = dim4(points=points, offset=offset, scale=scale)
    return ((X @ quadric) * X).sum(-1)


def quadratic_gradient(*, points, quadric, offset=0, scale=1):
    # Make sure that the quadric is symmetric:
    assert quadric.shape == (4, 4)
    quadric = (quadric + quadric.T) / 2

    X = dim4(points=points, offset=offset, scale=scale)

    return (2 / scale) * (X @ quadric)[:,:3]


@given(n_points=st.integers(min_value=5, max_value=10))
@settings(deadline=1000)
def test_quadratic_function(*, n_points: int):
    """Test on a simple dataset z = f(x, y)."""
    # Create the dataset
    points = create_point_cloud(
        n_points=n_points,
        f=lambda x, y: x**2 + y**2,
    )

    N = len(points)
    assert N == n_points**2

    # Compute the implicit quadrics
    quadrics, mean_point, sigma = sks.implicit_quadrics(points=points, scale=1)

    self_scores = quadratic_function(
        points=points,
        quadric=quadrics[0],
        offset=mean_point,
        scale=sigma,
    )

    print(self_scores)
    print(quadrics[0])
    assert self_scores.abs().max() < 1e-2


def display_quadratic_fit(points, highlight=0, scale=1):
    # Fit a quadratic function to the point cloud
    quadrics, mean_point, sigma = sks.implicit_quadrics(points=points, scale=scale)
    quadric = quadrics[highlight]

    # Our surface points:
    spheres = vd.Spheres(points, r=0.03, c="blue")

    main_sphere = vd.Spheres(points[highlight:highlight+1], r=0.06, c="red")

    # Gradients:
    v = quadratic_gradient(
        points=points,
        quadric=quadric,
        offset=mean_point,
        scale=sigma,
    )
    quiver = vd.Arrows(points, points + v, c="green", alpha=0.9)

    # generate an isosurface the volume for each thresholds
    thresholds = [-.05, 0., .05]

    n = 100
    t = torch.linspace(-1, 1, n)
    X, Y, Z = torch.meshgrid(t, t, t, indexing="ij")
    scalar_field = quadratic_function(
        points = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1),
        quadric = quadric,
        offset = mean_point,
        scale = sigma,
    )
    scalar_field = scalar_field.reshape(X.shape).cpu().numpy()

    vol = vd.Volume(scalar_field, origin=(-1, -1, -1), spacing=(2/n,) * 3)
    surf = vol.isosurface(thresholds).cmap("RdBu_r").alpha(0.5)
    surf.add_scalarbar3d()

    plt = vd.Plotter(axes=2)
    plt.show(spheres, main_sphere, quiver, surf)
    plt.close()




if __name__ == "__main__":
    functions = [
        lambda x, y: (1.5 - .5 * x**2 - y**2).abs().sqrt() - 1,
        lambda x, y: x**2 - y**2,
    ]
    for f in functions:
        points = create_point_cloud(
            n_points=20,
            f=f,
        )
        display_quadratic_fit(points, highlight=55, scale=.3)
