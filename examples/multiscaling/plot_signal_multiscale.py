"""
Multiscaling and signal propagation
===================================

Learn how to propagate a signal on the multiscale
representation.
"""


import pyvista as pv
import pyvista.examples
import skshapes as sks

bunny = sks.PolyData(pyvista.examples.download_bunny())
multiscale_bunny = sks.Multiscale(shape=bunny, ratios=[0.1, 0.01, 0.005])


# %%
# Define a signal on the high resolution mesh

multiscale_bunny.at(ratio=1)["height"] = multiscale_bunny.at(ratio=1).points[
    :, 1
]


# %%
# Propagate the signal to the lower resolutions

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

multiscale_bunny.at(ratio=0.005)["height_low_constant"] = multiscale_bunny.at(
    ratio=0.005
).points[:, 1]

multiscale_bunny.at(ratio=0.005)["height_low_smoothing"] = multiscale_bunny.at(
    ratio=0.005
).points[:, 1]

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


# Plot the signal on the lower resolution meshes
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
