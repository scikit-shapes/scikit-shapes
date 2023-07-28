import sys

sys.path.append(sys.path[0][:-4])

import torch
import skshapes as sks

from hypothesis import given, settings
from hypothesis import strategies as st

import numpy as np
import vedo as vd

from .utils import create_point_cloud, create_shape


def display_curvatures_old(*, function: callable, scale=1):
    points, normals = create_point_cloud(n_points=20, f=function, normals=True)
    points = points + 0.05 * torch.randn(len(points), 3)

    # Fit a quadratic function to the point cloud
    curvatures = sks.smooth_curvatures_2(points=points, normals=normals, scale=scale)

    # Our surface points:
    r = 500 / np.sqrt(len(points))
    spheres = vd.Points(points, r=r).cmap("RdBu_r", curvatures["mean"])
    spheres = spheres.add_scalarbar()

    quiver = vd.Arrows(points, points + 0.2 * normals, c="green", alpha=0.9)

    plt = vd.Plotter(axes=2)
    plt.show(spheres, quiver)
    plt.close()


def display_curvatures(*, scale=1, highlight=0, **kwargs):
    shape = create_shape(**kwargs)
    scales = [scale, 2 * scale, 5 * scale, 10 * scale]

    plt = vd.Plotter(shape=(2, 2), axes=1)

    for i, s in enumerate(scales):
        curvedness = shape.point_curvedness(scale=s)
        quadratic_fit = shape.point_quadratic_fits(scale=s)[highlight]
        assert quadratic_fit.shape == (3, 3, 3)

        n_samples = 5000
        uv1 = s * torch.randn(n_samples, 3, device=shape.device)
        uv1[:, 2] = 1
        local_fit = (
            quadratic_fit.view(1, 3, 3, 3)
            * uv1.view(-1, 1, 3, 1)
            * uv1.view(-1, 1, 1, 3)
        ).sum(dim=(2, 3))
        assert local_fit.shape == (n_samples, 3)

        reference_point = vd.Points(shape.points[highlight].view(1, 3), c="red", r=10)
        local_fit = vd.Points(local_fit, c=(235, 158, 52), r=5)

        # Our surface points:
        if shape.triangles is None:
            shape_ = vd.Points(
                shape.points, c=(0.5, 0.5, 0.5), r=60 / (shape.n_points ** (1 / 3))
            )
        else:
            shape_ = shape.to_vedo()

        plt.at(i).show(
            shape_.clone()
            .alpha(0.5)
            .cmap("viridis", curvedness, vmin=0)
            .add_scalarbar(),
            reference_point,
            local_fit.clone().alpha(0.5),
            vd.Text2D(
                f"Scale {s:.2f}\ncurvedness and local fit around point 0",
                pos="top-middle",
            ),
        )

    plt.interactive()


if __name__ == "__main__":
    functions = [
        lambda x, y: (2 - 0.5 * x**2 - y**2).abs().sqrt() - 1,
        lambda x, y: x**2 - y**2,
    ]

    fnames = [
        "~/data/PN1.stl",
    ]

    if True:
        for f in functions:
            display_curvatures(function=f, scale=0.05, n_points=15, noise=0.01)
    elif True:
        for f in fnames:
            display_curvatures(file_name=f, scale=2.0, n_points=1e4, highlight=0)

    else:
        from torch.profiler import profile, ProfilerActivity

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        myprof = profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        )

        with myprof as prof:
            points = create_point_cloud(
                n_points=3,
                f=functions[0],
            )
            quadrics, mean_point, sigma = sks.implicit_quadrics(points=points, scale=1)

        # Create an "output/" foler if it doesn't exist
        import os

        if not os.path.exists("output"):
            os.makedirs("output")

        # Export to chrome://tracing
        prof.export_chrome_trace(f"output/trace_implicit_quadrics.json")
        prof.export_stacks(
            f"output/stacks_implicit_quadrics.txt",
            "self_cpu_time_total",  # "self_cuda_time_total",
        )
