import skshapes as sks
import pyvista
import matplotlib.pyplot as plt
import torch
from math import sqrt

sqrt2 = sqrt(2)
pyvista.OFF_SCREEN = True


def load_data():
    edges = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [1, 4],
            [4, 5],
            [1, 6],
            [6, 7],
            [7, 8],
            [6, 9],
            [9, 10],
        ]
    )

    x1 = torch.tensor(
        [
            [4, 9],
            [4, 7],
            [2, 7],
            [2, 8],
            [6, 7],
            [6, 8],
            [4, 4],
            [4 - sqrt2, 4 - sqrt2],
            [4 - sqrt2, 2 - sqrt2],
            [4 + sqrt2, 4 + sqrt2],
            [6 + sqrt2, 4 + sqrt2],
        ],
        dtype=torch.float32,
    )

    x2 = torch.tensor(
        [
            [4, 9],
            [4, 7],
            [2, 7],
            [2, 6],
            [6, 7],
            [6, 6],
            [4, 4],
            [2, 4],
            [2 - sqrt2, 4 - sqrt2],
            [4 + sqrt2, 4 - sqrt2],
            [4 + sqrt2, 2 - sqrt2],
        ],
        dtype=torch.float32,
    )

    polydata1 = sks.PolyData(x1, edges=edges)
    polydata2 = sks.PolyData(x2, edges=edges)

    return polydata1, polydata2


def plot_karatekas():
    polydata1, polydata2 = load_data()
    plotter = pyvista.Plotter(shape=(1, 2), border=False)
    plotter.subplot(0, 0)
    plotter.add_mesh(
        polydata1.to_pyvista(), show_edges=True, line_width=5, color="k"
    )
    plotter.view_xy()
    plotter.subplot(0, 1)
    plotter.add_mesh(
        polydata2.to_pyvista(), show_edges=True, line_width=5, color="k"
    )
    plotter.view_xy()
    plotter.show()
    plt.imshow(plotter.image)
    plt.axis("off")
    plt.show()


def plot_path(path):
    plotter = pyvista.Plotter(shape=(1, len(path)), border=False)
    for i, m in enumerate(path):
        plotter.subplot(0, i)
        plotter.add_mesh(
            m.to_pyvista(), show_edges=True, line_width=5, color="k"
        )
        plotter.view_xy()
    plotter.show()
    plt.imshow(plotter.image)
    plt.axis("off")
    plt.show()
