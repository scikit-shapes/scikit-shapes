import sys

sys.path.append(sys.path[0][:-4])

import torch
import skshapes as sks

from hypothesis import given, settings
from hypothesis import strategies as st

import numpy as np
import vedo as vd

from .utils import create_point_cloud, dim4, quadratic_function, quadratic_gradient


def display_curvatures(*, function: callable, scale=1):

    points, normals = create_point_cloud(
        n_points=20,
        f=function,
        normals=True
    )
    points = points + .05 * torch.randn(len(points), 3)

    # Fit a quadratic function to the point cloud
    curvatures = sks.smooth_curvatures_2(points=points, normals=normals, scale=scale)

    # Our surface points:
    spheres = vd.Points(points, r=40).cmap("RdBu_r", curvatures["mean"])
    spheres = spheres.add_scalarbar3d()

    quiver = vd.Arrows(points, points + .2 * normals, c="green", alpha=0.9)

    plt = vd.Plotter(axes=2)
    plt.show(spheres, quiver)
    plt.close()




if __name__ == "__main__":
    
    functions = [
        lambda x, y: (2 - .5 * x**2 - y**2).abs().sqrt() - 1,
        lambda x, y: x**2 - y**2,
    ]

    if True:
        for f in functions:
            display_curvatures(function = f, scale=.4)

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
            "self_cpu_time_total", # "self_cuda_time_total",
        )