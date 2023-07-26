import sys

sys.path.append(sys.path[0][:-4])

import torch
import skshapes as sks

from hypothesis import given, settings
from hypothesis import strategies as st

import numpy as np
import vedo as vd

from .utils import create_point_cloud, vedo_frames


def display_moments(
    *, file_name: str = None, function: callable = None, scale=1, n_points=20, noise=0
):
    if function is not None:
        points = create_point_cloud(n_points=n_points, f=function)
        shape = sks.PolyData(points=points)
        axes = 2
    else:
        shape = sks.PolyData(file_name).decimate(n_points=n_points)
        axes = 1

    shape.points = shape.points + noise * torch.randn(shape.n_points, 3)

    local_average = shape.point_moments(order=1, scale=scale)
    local_cov = shape.point_moments(order=2, scale=scale, central=True)

    local_QL = torch.linalg.eigh(local_cov)
    local_nuv = local_QL.eigenvectors  # (N, 3, 3)

    local_frame = local_QL.eigenvectors * local_QL.eigenvalues.view(-1, 1, 3).sqrt()
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
    curvature = local_QL.eigenvalues[:, 0] / local_QL.eigenvalues.sum(-1)

    # Our surface points:
    if shape.triangles is None:
        spheres = vd.Points(
            shape.points, c=(0.5, 0.5, 0.5), r=60 / (shape.n_points ** (1 / 3))
        )
    else:
        spheres = shape.to_vedo()

    spheres_1 = (
        spheres.clone().alpha(0.5).cmap("viridis", density, vmin=0).add_scalarbar()
    )
    spheres_2 = (
        spheres.clone()
        .cmap("viridis", (3 * curvature) ** (1 / 3), vmin=0)
        .add_scalarbar()
    )

    # Vectors to the local average:
    quiver = vd.Arrows(shape.points[mask], local_average[mask], c="green", alpha=0.9)

    # Local tangent frames:
    s = 2 / len(shape.points) ** (1 / 2)

    plt = vd.Plotter(shape=(1, 3), axes=axes)
    plt.at(0).show(spheres_1, quiver)
    plt.at(1).show(spheres_2, *vedo_frames(shape.points[mask], s * local_nuv[mask]))
    plt.at(2).show(spheres_2, *vedo_frames(shape.points[mask], s * local_frame[mask]))
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

    if True:
        for f in functions:
            display_moments(function=f, scale=0.5, n_points=15, noise = .05)

    if False:
        for f in fnames:
            display_moments(file_name=f, scale=5.0, n_points=1e4)
