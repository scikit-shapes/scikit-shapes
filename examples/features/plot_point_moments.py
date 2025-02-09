"""
.. _point_moments_example:

How to compute point moments
==================================

We use the :meth:`~skshapes.PolyData.point_moments` method to compute local averages and covariance matrices. This is useful for estimating point normals and curvatures.
"""

###############################################################################
# First, we create a simple point cloud in 2D:
#

import torch
from matplotlib import pyplot as plt

import skshapes as sks

n_points, dim = 3, 2
shape = sks.PolyData(torch.rand(n_points, dim))
print(shape)

###############################################################################
# The method :meth:`~skshapes.PolyData.point_neighborhoods` associates
# to each point :math:`x_i` in the shape a weight
# distribution :math:`\nu_i(x)` over the ambient space.
# Typically, this corresponds to a normalized Gaussian window at scale :math:`\sigma`.
# If :math:`\mu(x)` denotes the uniform distribution on the support of the shape, then:
#
# .. math::
#   \nu_i(x) \propto \exp\left(-\frac{||x-x_i||^2}{2\sigma^2}\right) \mu(x)
#
# The method :meth:`~skshapes.PolyData.point_moments` takes as input the same
# arguments, and returns a :class:`skshapes.features.Moments` object
# that computes the moments of order 0, 1 and 2 of these distributions.

moments = shape.point_moments(scale=0.5)

###############################################################################
# The total mass :math:`m_i` of :math:`\nu_i` is an estimate of the local point density:
print(moments.masses)

###############################################################################
# The local mean :math:`\overline{x}_i = \mathbb{E}_{x\sim \nu_i}[x]`
# is a point in the vicinity of :math:`x_i`:
print(moments.means)

###############################################################################
# The local covariance matrix
# :math:`\Sigma_i = \mathbb{E}_{x\sim \nu_i}[(x-\overline{x}_i)(x-\overline{x}_i)^{\intercal}]`
# is symmetric and positive semi-definite:
print(moments.covariances)

###############################################################################
# Its eigenvalues are non-negative and typically of order :math:`\sigma^2`:

L = moments.covariance_eigenvalues
print(L)

###############################################################################
# Its eigenvectors are orthogonal and represent the principal directions of
# the local distribution of points:

Q = moments.covariance_eigenvectors
print(Q)

###############################################################################
# The eigenvectors are stored column-wise, and sorted by increasing eigenvalue:

LQt = L.view(n_points, dim, 1) * Q.transpose(1, 2)
QLQt = Q @ LQt

print(f"Reconstruction error: {(QLQt - moments.covariances).abs().max():.2e}")

###############################################################################
# .. note::
#   These attributes are computed when required, and cached in memory.
#
# On a 2D curve
# --------------
#
# Going further, let's consider a sampled curve in 2D:

shape = sks.doc.wiggly_curve(n_points=32, dim=2)
print(shape)

###############################################################################
# Intuitively, computing local moments
# :math:`m_i` of order 0,
# :math:`\overline{x}_i` of order 1
# and :math:`\Sigma_i` of order 2 is equivalent to
# fitting a Gaussian distribution to each local neighborhood :math:`\nu_i`.
# In dimension :math:`D=2` or :math:`D=3`, if :math:`\lambda_{\mathbb{R}^D}` denotes
# the Lebesgue measure on the ambient space (i.e. the area or the volume), then:
#
# .. math::
#   \text{d}\nu_i(x) ~&\simeq~ m_i\,\text{d}\mathcal{N}(\overline{x}_i, \Sigma_i)(x)  \\
#   &= ~\frac{m_i}{(2\pi\sigma^2)^{D/2} } \, \exp\left[-\tfrac{1}{2}(x-\overline{x}_i)^{\intercal}\Sigma_i^{-1}(x-\overline{x}_i)\right]\, \text{d}\lambda_{\mathbb{R}^D}(x)~
#
# We visualize these descriptors by drawing ellipses centered at :math:`\overline{x}_i`,
# colored by the local densities :math:`m_i` and oriented in :math:`\mathbb{R}^2` or :math:`\mathbb{R}^3`
# with axes that are aligned with the eigenvectors of :math:`\Sigma_i` and
# whose lengths are proportional to the square root of its eigenvalues:

plt.figure(figsize=(12, 10))

for i, sigma in enumerate([0.1, 0.2, 0.5, 1.0]):
    moments = shape.point_moments(scale=sigma)

    # Plot the ellipses
    ax = plt.subplot(2, 2, i + 1)
    ax.set_title(f"Moments at scale = {sigma:.1f}")
    sks.doc.display_covariances(shape.points, moments)

plt.tight_layout()

###############################################################################
# As evidenced here, :meth:`~skshapes.PolyData.point_moments` describes the local
# shape context at scale :math:`\sigma`. We use it to compute point normals and
# curvatures that are robust to noise and sampling artifacts.
#
# On a 3D surface
# ---------------
#
# We can perform the same analysis on a 3D surface:

import pyvista as pv

shape = (
    sks.PolyData(pv.examples.download_bunny())
    .resample(n_points=5000)
    .normalize()
)
print(shape)

###############################################################################
# As expected, the point moments of order 1 and 2 now refer to 3D vectors and matrices:

moments = shape.point_moments(scale=0.05)
print("moments.masses:     ", moments.masses.shape)
print("moments.means:      ", moments.means.shape)
print("moments.covariances:", moments.covariances.shape)

###############################################################################
# We visualize the ellipsoids on a subset of our full point cloud:

landmark_indices = list(range(0, shape.n_points, 50))

pl = pv.Plotter()
sks.doc.display(plotter=pl, shape=shape, opacity=0.5)
sks.doc.display_covariances(
    shape.points, moments, landmark_indices=landmark_indices, plotter=pl
)
pl.show()


###############################################################################
# Using the covariance eigenvalues :math:`\lambda_1 \leqslant \lambda_2 \leqslant \lambda_3`,
# we can compute simple shape descriptors such as the plateness
# which is equal to 0 if the local ellipsoid is a sphere and 1 if it is a 2D disk:
#
# .. math::
#   \text{plateness} = 1 - \frac{\lambda_1}{\sqrt{\lambda_2\lambda_3}}~.

scales = [0.05, 0.1, 0.15, 0.2]

pl = pv.Plotter(shape=(2, 2))
for i, scale in enumerate(scales):
    moments = shape.point_moments(scale=scale)
    eigs = moments.covariance_eigenvalues
    shape.point_data[f"plateness_{i}"] = (
        1 - eigs[:, 0] / (eigs[:, 1] * eigs[:, 2]).sqrt()
    )

    pl.subplot(i // 2, i % 2)
    sks.doc.display(
        plotter=pl,
        shape=shape,
        scalars=f"plateness_{i}",
        scalar_bar=True,
        smooth=1,
        clim=(0, 1),
        title=f"Plateness at scale {scale:.2f}",
    )
pl.show()


###############################################################################
# Likewise, we can compute the tubeness, which is equal to 0 if the local ellipsoid is
# a sphere or a disk, and 1 if it is a 1D segment:
#
# .. math::
#   \text{tubeness} = 1 - \frac{\lambda_2}{\lambda_3}~.

pl = pv.Plotter(shape=(2, 2))
for i, scale in enumerate(scales):
    moments = shape.point_moments(scale=scale)
    eigs = moments.covariance_eigenvalues
    shape.point_data[f"tubeness_{i}"] = 1 - (eigs[:, 1] / eigs[:, 2])

    pl.subplot(i // 2, i % 2)
    sks.doc.display(
        plotter=pl,
        shape=shape,
        scalars=f"tubeness_{i}",
        scalar_bar=True,
        smooth=1,
        clim=(0, 1),
        title=f"Tubeness at scale {scale:.2f}",
    )
pl.show()
