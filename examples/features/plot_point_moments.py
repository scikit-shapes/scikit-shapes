"""
.. _point_moments_example:

How to compute point moments
==================================

We use the :meth:`~skshapes.data.polydata.PolyData.point_moments` method to compute local averages and covariance matrices. This is useful for estimating point normals and curvatures.
"""

###############################################################################
# First, create a simple curve in 2D.
#

import torch
from matplotlib import cm
from matplotlib import pyplot as plt

import skshapes as sks

N, D = 32, 2
s = torch.linspace(-.5, 4.5, N+1)[:-1]
x = torch.FloatTensor([[
    t.cos() + .2 * (6 * t).cos(),
    t.sin() + .1 * (4 * t).cos(),
    ]
    for t in s])

shape = sks.PolyData(x)





def display_covariances(moments):

    masses = moments.masses
    centers = moments.means
    axes = moments.covariance_axes
    assert axes.shape == (N, D, D)

    # Draw ellipses
    T = 65
    t = torch.linspace(0, 2 * torch.pi, T)
    t = torch.cat((t, torch.FloatTensor([float("nan")])))
    circle = torch.stack([torch.cos(t), torch.sin(t)], dim=0)
    assert circle.shape == (D, T + 1)
    ellipses = axes @ circle
    assert ellipses.shape == (N, D, T + 1)
    ellipses = centers[:, :, None] + ellipses

    ellipses = ellipses.permute(0, 2, 1).reshape(N * (T + 1), D)

    sks.doc.colored_line(
        ellipses[:, 0],
        ellipses[:, 1],
        masses.view(N, 1).repeat(1, T + 1).reshape(-1),
        plt.gca(),
        cmap=cm.viridis,
        linewidth=2,
        alpha=0.5,
    )
    #plt.plot(centers[i, 0] + ellipses[i, 0], centers[i, 1] + ellipses[i, 1], c="r", )




plt.figure(figsize=(12, 10))

for i, sigma in enumerate([0.1, 0.2, 0.5, 1.0]):
    moments = shape.point_moments(scale=sigma)

    # Plot the ellipses
    ax = plt.subplot(2, 2, i + 1)
    ax.set_title(f"Moments at scale = {sigma:.1f}")
    display_covariances(moments)
    plt.scatter(x[:, 0], x[:, 1], c=moments.masses, cmap="viridis", s=80,edgecolors='black', zorder=2)
    plt.axis("square")
    plt.axis([-1.2, 1.2, -1.2, 1.2])
    plt.colorbar(fraction=0.046, pad=0.04)

# Tight layout
plt.tight_layout()
