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

multiscale_bunny = sks.MultiscaleGeneric(shape=bunny, ratios=[0.1, 0.01])


# %%
# Visualize the multiscale representation
import matplotlib.pyplot as plt
import pyvista as pv

plotter = pv.Plotter(shape=(1, 3))
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
