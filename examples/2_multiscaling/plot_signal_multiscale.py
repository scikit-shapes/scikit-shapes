"""
Multiscaling and signal propagation
===================================

This example shows how a signal a signal can be propagated across scales on
a multiscale representation.

The rules for signal propagation are defined by the `FineToCoarsePolicy <../../api/skshapes.html#skshapes.types.FineToCoarsePolicy>`_ and
`CoarseToFinePolicy <../../api/skshapes.html#skshapes.types.FineToCoarsePolicy>`_ classes.
"""


import pyvista as pv
import pyvista.examples

import skshapes as sks

bunny = sks.PolyData(pyvista.examples.download_bunny())
multiscale_bunny = sks.Multiscale(shape=bunny, ratios=[0.1, 0.01, 0.005])


###############################################################################
# Define a signal on the high resolution mesh

signal = multiscale_bunny.at(ratio=1).points[:, 1]
multiscale_bunny.at(ratio=1).point_data["height"] = signal


###############################################################################
# Propagate the signal from fine to coarse resolutions

# define a fine_to_coarse propagation scheme
fine_to_coarse_policy = sks.FineToCoarsePolicy(
    reduce="mean",
)

# propagate the signal from the high resolution to the lower resolutions
multiscale_bunny.propagate(
    signal_name="height",
    from_ratio=1,
    fine_to_coarse_policy=fine_to_coarse_policy,
)

###############################################################################
# Propagate a signal from the lower resolutions to the higher resolution


signal_low = multiscale_bunny.at(ratio=0.005).points[:, 1]
multiscale_bunny.at(ratio=0.005).point_data["height_low_constant"] = signal_low
multiscale_bunny.at(ratio=0.005).point_data["height_low_smoothing"] = signal_low

# define a coarse_to_fine propagation scheme
coarse_to_fine_policy = sks.CoarseToFinePolicy(
    smoothing="constant",
)

# propagate the signal from the lower resolutions to the higher resolution
multiscale_bunny.propagate(
    signal_name="height_low_constant",
    from_ratio=0.005,
    coarse_to_fine_policy=coarse_to_fine_policy,
)

# define a coarse_to_fine propagation scheme
coarse_to_fine_policy = sks.CoarseToFinePolicy(smoothing="mesh_convolution")

# propagate the signal from the lower resolutions to the higher resolution
multiscale_bunny.propagate(
    signal_name="height_low_smoothing",
    from_ratio=0.005,
    coarse_to_fine_policy=coarse_to_fine_policy,
)


###############################################################################
# Visualize the multiscale representation

plotter = pv.Plotter(shape=(3, 4))
row = 0
plotter.subplot(row, 0)
plotter.add_mesh(multiscale_bunny.at(ratio=1).to_pyvista(), scalars="height")
plotter.view_xy()
plotter.subplot(row, 1)
plotter.add_mesh(multiscale_bunny.at(ratio=0.1).to_pyvista(), scalars="height")
plotter.view_xy()
plotter.subplot(row, 2)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.01).to_pyvista(), scalars="height"
)
plotter.view_xy()
plotter.subplot(row, 3)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.005).to_pyvista(), scalars="height"
)
plotter.view_xy()
row = 1
plotter.subplot(row, 0)
plotter.add_mesh(
    multiscale_bunny.at(ratio=1).to_pyvista(), scalars="height_low_constant"
)
plotter.view_xy()
plotter.subplot(row, 1)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.1).to_pyvista(), scalars="height_low_constant"
)
plotter.view_xy()
plotter.subplot(row, 2)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.01).to_pyvista(), scalars="height_low_constant"
)
plotter.view_xy()
plotter.subplot(row, 3)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.005).to_pyvista(),
    scalars="height_low_constant",
)
plotter.view_xy()
row = 2
plotter.subplot(row, 0)
plotter.add_mesh(
    multiscale_bunny.at(ratio=1).to_pyvista(), scalars="height_low_smoothing"
)
plotter.view_xy()
plotter.subplot(row, 1)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.1).to_pyvista(), scalars="height_low_smoothing"
)
plotter.view_xy()
plotter.subplot(row, 2)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.01).to_pyvista(),
    scalars="height_low_smoothing",
)
plotter.view_xy()
plotter.subplot(row, 3)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.005).to_pyvista(),
    scalars="height_low_smoothing",
)
plotter.view_xy()
plotter.show()
