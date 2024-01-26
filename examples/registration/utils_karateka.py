from math import sqrt

import numpy as np
import pyvista
import torch

import skshapes as sks

sqrt2 = sqrt(2)
pyvista.OFF_SCREEN = True


cpos1 = [
    (4.707106828689575, 4.79289323091507, 28.44947380182544),
    (4.707106828689575, 4.79289323091507, 0.0),
    (0.0, 1.0, 0.0),
]

cpos2 = [
    (3.2928932309150696, 4.792893171310425, 28.44947390345415),
    (3.2928932309150696, 4.792893171310425, 0.0),
    (0.0, 1.0, 0.0),
]


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
            [1.95, 6],
            [6, 7],
            [6.05, 8],
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
            [1.95, 8],
            [6, 7],
            [6.05, 6],
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
    plotter.camera_position = cpos1
    plotter.add_mesh(
        polydata1.to_pyvista(), show_edges=True, line_width=5, color="k"
    )
    plotter.add_text("Source")

    plotter.subplot(0, 1)
    plotter.camera_position = cpos2
    plotter.add_mesh(
        polydata2.to_pyvista(), show_edges=True, line_width=5, color="k"
    )
    plotter.add_text("Target")

    plotter.show()


def plot_extrinsic_deformation(source, target, registration, animation=False):
    if not animation:
        morphed_shape = registration.transform(source=source)
        control_points = source.control_points
        morphed_cp = registration.transform(source=control_points)

        regularization = registration.regularization_
        loss = registration.loss

        initial_loss = loss(source, target)
        final_loss = loss(morphed_shape, target)

        plotter = pyvista.Plotter(shape=(1, 2), border=False)

        plotter.subplot(0, 0)
        plotter.add_mesh(
            source.to_pyvista(), show_edges=True, line_width=5, color="k"
        )
        plotter.add_mesh(
            control_points.to_pyvista(),
            show_edges=True,
            line_width=2,
            color="r",
        )
        plotter.add_mesh(
            target.to_pyvista(),
            show_edges=True,
            line_width=5,
            color="b",
            opacity=0.2,
        )
        plotter.camera_position = cpos1
        plotter.add_text(f"Initial loss: {initial_loss:.2e}")

        plotter.subplot(0, 1)
        plotter.add_mesh(
            morphed_shape.to_pyvista(),
            show_edges=True,
            line_width=5,
            color="k",
        )
        plotter.add_mesh(
            morphed_cp.to_pyvista(), show_edges=True, line_width=2, color="r"
        )
        plotter.add_mesh(
            target.to_pyvista(),
            show_edges=True,
            line_width=5,
            color="b",
            opacity=0.2,
        )
        plotter.camera_position = cpos2

        plotter.add_text(
            f"Loss: {final_loss:.2e}\nRegularization: {regularization:.2e}"
        )

        plotter.show()

    else:
        path_cp = registration.model.morph(
            shape=source.control_points,
            parameter=registration.parameter_,
            return_path=True,
        ).path
        path = registration.path_
        n_frames = len(path)

        plotter = pyvista.Plotter()
        plotter.open_gif("extrinsic_deformation.gif", fps=3)
        plotter.camera_position = cpos1
        for i in range(n_frames):
            plotter.clear_actors()
            plotter.add_mesh(
                path_cp[i].to_pyvista(),
                show_edges=True,
                line_width=2,
                color="r",
            )
            plotter.add_mesh(
                path[i].to_pyvista(), show_edges=True, line_width=5, color="k"
            )
            plotter.add_mesh(
                target.to_pyvista(),
                show_edges=True,
                line_width=5,
                color="b",
                opacity=0.2,
            )
            plotter.write_frame()

        plotter.show()


def plot_intrinsic_deformation(source, target, registration, animation=False):
    if not animation:
        morphed_shape = registration.transform(source=source)
        path = registration.path_

        velocities = registration.parameter_
        regularization = registration.regularization_
        model = registration.model
        loss = registration.loss
        n_steps = model.n_steps

        initial_loss = loss(source, target)
        final_loss = loss(morphed_shape, target)

        for i in range(n_steps):
            v = velocities[:, i, :]
            source[f"v_{i}"] = v

        source_pv = source.to_pyvista()

        plotter = pyvista.Plotter(shape=(1, 2), border=False)

        plotter.subplot(0, 0)
        plotter.camera_position = cpos1
        plotter.add_mesh(
            source.to_pyvista(), show_edges=True, line_width=5, color="k"
        )
        plotter.add_mesh(
            target.to_pyvista(),
            show_edges=True,
            line_width=5,
            color="b",
            opacity=0.2,
        )
        plotter.add_text(f"Initial loss: {initial_loss:.2e}")

        plotter.subplot(0, 1)
        plotter.camera_position = cpos2
        plotter.add_mesh(
            target.to_pyvista(),
            show_edges=True,
            line_width=5,
            color="b",
            opacity=0.2,
        )
        plotter.add_mesh(
            source.to_pyvista(),
            show_edges=True,
            line_width=5,
            color="k",
            opacity=0.2,
        )
        plotter.add_mesh(
            morphed_shape.to_pyvista(),
            show_edges=True,
            line_width=5,
            color="k",
        )

        for i in range(n_steps):
            mesh = path[i].to_pyvista()
            mesh["v"] = np.concatenate(
                [source_pv[f"v_{i}"], np.zeros(shape=(source.n_points, 1))],
                axis=1,
            )
            mesh.active_vectors_name = "v"
            arrows = mesh.arrows

            plotter.add_mesh(arrows, color="r")
        plotter.add_text(
            f"Loss: {final_loss:.2e}\nRegularization: {regularization:.2e}"
        )
        plotter.show()

    else:
        path = registration.path_
        velocities = registration.parameter_

        plotter = pyvista.Plotter()
        plotter.camera_position = cpos1
        plotter.open_gif("intrinsic_deformation.gif", fps=3)
        for i in range(len(path)):
            plotter.clear_actors()
            plotter.add_mesh(
                path[i].to_pyvista(), show_edges=True, line_width=5, color="k"
            )
            plotter.add_mesh(
                target.to_pyvista(),
                show_edges=True,
                line_width=5,
                color="b",
                opacity=0.2,
            )
            if i < len(path) - 1:
                mesh = path[i].to_pyvista()
                mesh["v"] = np.concatenate(
                    [
                        velocities[:, i, :].detach().cpu().numpy(),
                        np.zeros(shape=(source.n_points, 1)),
                    ],
                    axis=1,
                )
                mesh.active_vectors_name = "v"
                arrows = mesh.arrows

                plotter.add_mesh(arrows, color="r", line_width=5)
            plotter.write_frame()

        plotter.show()
