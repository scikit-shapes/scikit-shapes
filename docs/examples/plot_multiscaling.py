"""
Multiscaling
============

This example show the basic usage of the multiscaling module.
"""

# %%
# Load a mesh and create a multiscale representation

import skshapes as sks
import pyvista.examples

bunny = sks.PolyData(pyvista.examples.download_bunny())

multiscale_bunny = sks.Multiscale(shape=bunny, ratios=[0.1, 0.01, 0.005])


# %%
# Visualize the multiscale representation
import matplotlib.pyplot as plt
import pyvista as pv

plotter = pv.Plotter(shape=(1, 4))
plotter.subplot(0, 0)
plotter.add_mesh(multiscale_bunny.at(ratio=1).to_pyvista())
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=1).n_points}")
plotter.view_xy()
plotter.subplot(0, 1)
plotter.add_mesh(multiscale_bunny.at(ratio=0.1).to_pyvista())
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=0.1).n_points}")
plotter.view_xy()
plotter.subplot(0, 2)
plotter.add_mesh(multiscale_bunny.at(ratio=0.01).to_pyvista())
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=0.01).n_points}")
plotter.view_xy()
plotter.subplot(0, 3)
plotter.add_mesh(multiscale_bunny.at(ratio=0.005).to_pyvista())
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=0.005).n_points}")
plotter.view_xy()
plotter.show()
plt.imshow(plotter.image)
plt.axis("off")
plt.show()


# %%
# Define a signal on the high resolution mesh

multiscale_bunny.at(ratio=1)["height"] = multiscale_bunny.at(ratio=1).points[
    :, 1
]

plotter = pv.Plotter()
plotter.add_mesh(
    multiscale_bunny.at(ratio=1).to_pyvista(),
    scalars="height",
    cmap="viridis",
)
plotter.view_xy()
plotter.show()
plt.imshow(plotter.image)
plt.axis("off")
plt.show()

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

# Plot the signal on the lower resolution meshes
plotter = pv.Plotter(shape=(1, 4))
plotter.subplot(0, 0)
plotter.add_mesh(multiscale_bunny.at(ratio=1).to_pyvista(), scalars="height")
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=1).n_points}")
plotter.view_xy()
plotter.subplot(0, 1)
plotter.add_mesh(multiscale_bunny.at(ratio=0.1).to_pyvista(), scalars="height")
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=0.1).n_points}")
plotter.view_xy()
plotter.subplot(0, 2)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.01).to_pyvista(), scalars="height"
)
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=0.01).n_points}")
plotter.view_xy()
plotter.subplot(0, 3)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.005).to_pyvista(), scalars="height"
)
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=0.005).n_points}")
plotter.view_xy()
plotter.show()
plt.imshow(plotter.image)
plt.axis("off")
plt.show()

# %%
# Propagate the signal back to the higher resolutions

# define a signal on the lower resolution meshes
multiscale_bunny.at(ratio=0.005)["height_low"] = multiscale_bunny.at(
    ratio=0.005
).points[:, 1]

# define a coarse_to_fine propagation scheme
coarse_to_fine_policy = sks.CoarseToFinePolicy(
    smoothing="constant",
)

# propagate the signal from the lower resolutions to the higher resolution
multiscale_bunny.propagate(
    signal_name="height_low",
    from_ratio=0.005,
    coarse_to_fine_policy=coarse_to_fine_policy,
)

# Plot the signal on the higher resolution meshes
plotter = pv.Plotter(shape=(1, 4))
plotter.subplot(0, 0)
plotter.add_mesh(
    multiscale_bunny.at(ratio=1).to_pyvista(), scalars="height_low"
)
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=1).n_points}")
plotter.view_xy()
plotter.subplot(0, 1)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.1).to_pyvista(), scalars="height_low"
)
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=0.1).n_points}")
plotter.view_xy()
plotter.subplot(0, 2)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.01).to_pyvista(), scalars="height_low"
)
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=0.01).n_points}")
plotter.view_xy()
plotter.subplot(0, 3)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.005).to_pyvista(), scalars="height_low"
)
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=0.005).n_points}")
plotter.view_xy()
plotter.show()
plt.imshow(plotter.image)
plt.axis("off")
plt.show()

# %% With `smoothing = "mesh_convolution"`
#

# define a coarse_to_fine propagation scheme
coarse_to_fine_policy = sks.CoarseToFinePolicy(
    smoothing="mesh_convolution",
)

# propagate the signal from the lower resolutions to the higher resolution
multiscale_bunny.propagate(
    signal_name="height_low",
    from_ratio=0.005,
    coarse_to_fine_policy=coarse_to_fine_policy,
)

# Plot the signal on the higher resolution meshes
plotter = pv.Plotter(shape=(1, 4))
plotter.subplot(0, 0)
plotter.add_mesh(
    multiscale_bunny.at(ratio=1).to_pyvista(), scalars="height_low"
)
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=1).n_points}")
plotter.view_xy()
plotter.subplot(0, 1)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.1).to_pyvista(), scalars="height_low"
)
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=0.1).n_points}")
plotter.view_xy()
plotter.subplot(0, 2)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.01).to_pyvista(), scalars="height_low"
)
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=0.01).n_points}")
plotter.view_xy()
plotter.subplot(0, 3)
plotter.add_mesh(
    multiscale_bunny.at(ratio=0.005).to_pyvista(), scalars="height_low"
)
plotter.add_text(f"N_points={multiscale_bunny.at(ratio=0.005).n_points}")
plotter.view_xy()
plotter.show()
plt.imshow(plotter.image)
plt.axis("off")
plt.show()
