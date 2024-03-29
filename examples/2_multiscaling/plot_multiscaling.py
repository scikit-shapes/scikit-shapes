"""
Multiscaling (triangle mesh)
============================

The Multiscale class is used to create a multiscale representation of a
triangle mesh using quadric decimation. Quadric decimation reduces the number
of points of a mesh by iteratively collapsing the edge with the smallest
quadric error.

For reference on quadric decimation, see: `Garland and Heckbert 1997 <https://www.cs.cmu.edu/~./garland/Papers/quadrics.pdf>`_.

We use the implementation proposed by `fast-simplification <https://github.com/pyvista/fast-simplification>`_.
"""

###############################################################################
# Load a mesh

import pyvista as pv
import pyvista.examples

import skshapes as sks

bunny = sks.PolyData(pyvista.examples.download_bunny())

###############################################################################
# Create the multiscale representation with different ratios of the original
# number of points

multiscale_bunny = sks.Multiscale(shape=bunny, ratios=[0.1, 0.01, 0.005])


###############################################################################
# Visualize the multiscale representation

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
