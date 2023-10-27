import sys

sys.path.append(sys.path[0][:-4])

import torch
import skshapes as sks

from hypothesis import given, settings
from hypothesis import strategies as st

import numpy as np
import vedo as vd

from .utils import (
    create_point_cloud,
    dim4,
    quadratic_function,
    quadratic_gradient,
)


@given(n_points=st.integers(min_value=5, max_value=10))
@settings(deadline=10000)
def test_quadratic_function(*, n_points: int):
    """Test on a simple dataset z = f(x, y)."""
    # Create the dataset
    import skshapes.types as types

    points = create_point_cloud(
        n_points=n_points,
        f=lambda x, y: x**2 + y**2,
        dtype=types.float_dtype,
    )

    N = len(points)
    assert N == n_points**2

    # Compute the implicit quadrics
    quadrics, mean_point, sigma = sks.implicit_quadrics(points=points, scale=1)

    self_scores = quadratic_function(
        points=points,
        quadric=quadrics[0],
        offset=mean_point,
        scale=sigma,
    )

    print(self_scores)
    print(quadrics[0])
    assert self_scores.abs().max() < 1e-2


def display_quadratic_fit(points, highlight=0, scale=1):
    # Fit a quadratic function to the point cloud
    quadrics, mean_point, sigma = sks.implicit_quadrics(points=points, scale=scale)
    quadric = quadrics[highlight]

    # Our surface points:
    spheres = vd.Spheres(points, r=0.03, c="blue")

    main_sphere = vd.Spheres(points[highlight : highlight + 1], r=0.06, c="red")

    # Gradients:
    v = quadratic_gradient(
        points=points,
        quadric=quadric,
        offset=mean_point,
        scale=sigma,
    )
    quiver = vd.Arrows(points, points + v, c="green", alpha=0.9)

    # generate an isosurface the volume for each thresholds
    thresholds = [-0.05, 0.0, 0.05]

    n = 100
    t = torch.linspace(-1, 1, n)
    X, Y, Z = torch.meshgrid(t, t, t, indexing="ij")
    scalar_field = quadratic_function(
        points=torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1),
        quadric=quadric,
        offset=mean_point,
        scale=sigma,
    )
    scalar_field = scalar_field.reshape(X.shape).cpu().numpy()

    vol = vd.Volume(scalar_field, origin=(-1, -1, -1), spacing=(2 / n,) * 3)
    surf = vol.isosurface(thresholds).cmap("RdBu_r").alpha(0.5)
    surf.add_scalarbar3d()

    plt = vd.Plotter(axes=2)
    plt.show(spheres, main_sphere, quiver, surf)
    plt.close()


if __name__ == "__main__":
    functions = [
        lambda x, y: (1.5 - 0.5 * x**2 - y**2).abs().sqrt() - 1,
        lambda x, y: x**2 - y**2,
    ]

    if True:
        for f in functions:
            points = create_point_cloud(
                n_points=20,
                f=f,
            )
            display_quadratic_fit(points, highlight=55, scale=0.3)

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
