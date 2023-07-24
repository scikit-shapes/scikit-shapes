import sys

sys.path.append(sys.path[0][:-4])

import torch
import skshapes as sks

from hypothesis import given, settings
from hypothesis import strategies as st

import numpy as np
import vedo as vd

from .utils import create_point_cloud


def vedo_frames(points, frames):
    n = vd.Arrows(points, points + frames[:, :, 0], c="red", alpha=0.9)
    u = vd.Arrows(points, points + frames[:, :, 1], c="blue", alpha=0.9)
    v = vd.Arrows(points, points + frames[:, :, 2], c="green", alpha=0.9)
    return [n, u, v]


def display_moments(*, function: callable, scale=1, n_points=20):
    points, _ = create_point_cloud(n_points=n_points, f=function, normals=True)
    points = points + 0.01 * torch.randn(len(points), 3)

    shape = sks.PolyData(points=points)

    local_average = shape.point_moments(order=1, scale=scale)
    local_cov = shape.point_moments(order=2, scale=scale, central=True)

    local_frame = torch.linalg.eigh(local_cov)
    local_nuv = local_frame.eigenvectors  # (N, 3, 3)

    local_frame = (
        local_frame.eigenvectors * local_frame.eigenvalues.view(-1, 1, 3).sqrt()
    )
    local_frame = local_frame / scale

    # Pick 10 points at random:
    mask = torch.randperm(len(shape.points))[:10]

    # To color our points, let's visualize the window around point 0:
    density = torch.zeros_like(shape.points[:, 0])
    density[0] = 1
    Conv = shape.point_convolution(scale=scale, normalize=True)
    density = Conv.T @ density
    assert density.shape == (len(shape.points),)

    # Our surface points:
    spheres = vd.Points(
        shape.points, c=(0.5, 0.5, 0.5), r=60 / (len(shape.points) ** (1 / 3))
    )
    spheres = spheres.cmap("viridis", density, vmin=0).add_scalarbar()

    # Vectors to the local average:
    quiver = vd.Arrows(shape.points, local_average, c="green", alpha=0.9)

    # Local tangent frames:
    s = 2 / n_points

    plt = vd.Plotter(shape=(1, 3), axes=2)
    plt.at(0).show(spheres, quiver)
    plt.at(1).show(spheres, *vedo_frames(shape.points, s * local_nuv))
    plt.at(2).show(spheres, *vedo_frames(shape.points, s * local_frame))
    plt.interactive()


if __name__ == "__main__":
    functions = [
        lambda x, y: 0 * x + 0.3 * y + 0.2,
        lambda x, y: (2 - 0.5 * x**2 - y**2).abs().sqrt() - 1,
        lambda x, y: x**2 - y**2,
    ]

    for f in functions:
        display_moments(function=f, scale=0.2, n_points=15)
