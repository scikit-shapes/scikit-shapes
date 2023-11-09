"""utility functions for plotting circles"""

import matplotlib.pyplot as plt
import pyvista

pyvista.OFF_SCREEN = True


def plot1(source, target):
    plotter = pyvista.Plotter(window_size=[900, 450])
    plotter.add_mesh(source.to_pyvista(), color="red", label="source")
    plotter.add_mesh(target.to_pyvista(), color="blue", label="target")
    plotter.add_legend(bcolor="w", face="line")
    plotter.show()
    cpos = plotter.camera_position
    plt.imshow(plotter.image)
    plt.axis("off")
    plt.show()


def plot2(path, target):
    cpos = [
        (2.2252678159220034, 2.2602992984993273, 1.820383403037291),
        (0.4048844128847122, 0.43991589546203613, 0.0),
        (0.0, 0.0, 1.0),
    ]

    n_steps = len(path) - 1
    plotter = pyvista.Plotter(
        shape=(1, n_steps + 1), border=False, window_size=[900, 450]
    )
    for i, m in enumerate(path):
        plotter.subplot(0, i)
        plotter.add_text(f"step {i}")
        plotter.add_mesh(m.to_pyvista(), color="red", label="source")
        plotter.add_mesh(target.to_pyvista(), color="blue", label="target")
        plotter.camera_position = cpos
    plotter.show()
    plt.imshow(plotter.image)
    plt.axis("off")
    plt.show()
