import sys

sys.path.append(sys.path[0][:-4])

import numpy as np
import torch
import skshapes as sks

from pytest import approx
from hypothesis import given
from hypothesis import strategies as st

import numpy as np
import vedo as vd

from .utils import create_point_cloud, create_shape


@given(
    n_points=st.integers(min_value=400, max_value=500),
    a=st.floats(min_value=-1, max_value=1),
    b=st.floats(min_value=-1, max_value=1),
    c=st.floats(min_value=-1, max_value=1),
    d=st.floats(min_value=-1, max_value=1),
    e=st.floats(min_value=-1, max_value=1),
    f=st.floats(min_value=-1, max_value=1),
)
def test_curvatures_quadratic(
    *, n_points: int, a: float, b: float, c: float, d: float, e: float, f: float
):
    # Our current estimation method relies on the estimation of the tangent plane,
    # and does not give perfect results for quadratic functions "off center".
    # See e.g. the function f(x,y) = y**2 + y.
    #
    # Another issue is a fairly large variance for very large values of the coefficients,
    # e.g. f(x,y) = 2 * x * y
    d = 0 * d
    e = 0 * e

    def poly(x, y):
        return 0.5 * a * x**2 + b * x * y + 0.5 * c * y**2 + d * x + e * y + f

    # See Example 4.2 in Curvature formulas for implicit curves and surfaces,
    # Goldman, 2005, for reference on those formulas, keeping in mind that
    # Grad(f) = (d, e) and H(f) = [[a, b], [b, c]].
    denom = 1 + d**2 + e**2  # 1 + ||Grad(f)||^2
    gauss = a * c - b * b  # det(H(f))
    gauss = gauss / denom**2

    # Term 1: Grad(f)^T . H(f) . Grad(f)
    mean = d * d * a + 2 * d * e * b + e * e * c
    # Term 2: - (1 + ||Grad(f)||^2) * trace(H(f))
    mean = mean - denom * (a + c)
    mean = mean / (2 * denom ** (1.5))
    # Our convention for unoriented point clouds is that the mean curvature is >= 0:
    mean = np.abs(mean)

    # Create a point clouds around [0, 0] in the (x,y) plane and compute z = f(x, y).
    # Point shape.points[0] = [0, 0, f(0, 0)]
    shape = create_shape(shape="unit patch", n_points=n_points, function=poly)

    scales = [0.8, 1]

    for scale in scales:
        kmax, kmin = shape.point_principal_curvatures(scale=scale)
        kmax = kmax[0].item()
        kmin = kmin[0].item()
        assert kmax * kmin == approx(gauss, abs=5e-1, rel=2e-1)
        assert (kmax + kmin) / 2 == approx(mean, abs=5e-1, rel=2e-1)


@given(
    n_points=st.integers(min_value=500, max_value=1000),
    radius=st.floats(min_value=0.1, max_value=10),
    relative_scale=st.floats(min_value=0.1, max_value=0.12),
)
def test_curvatures_sphere(*, n_points: int, radius: float, relative_scale: float):
    # Create a sphere with the correct radius and an arbitrary center:
    shape = create_shape(shape="sphere", n_points=n_points, radius=radius)

    ones = torch.ones_like(shape.points[:, 0])

    scale = relative_scale * radius
    kmax, kmin = shape.point_principal_curvatures(scale=scale)
    assert torch.allclose(kmax, ones / radius, atol=1e-1, rtol=1e-1)
    assert torch.allclose(kmin, ones / radius, atol=1e-1, rtol=1e-1)


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
        Xm = shape.point_moments(order=1, scale=s)
        curvedness = shape.point_curvedness(scale=s)
        kmax, kmin = shape.point_principal_curvatures(scale=s)
        print(f"Kmax: {kmax[highlight]}, Kmin: {kmin[highlight]}")
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
        local_average = vd.Points(Xm[highlight].view(1, 3), c="blue", r=10)
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
            local_average,
            local_fit.clone().alpha(0.5),
            vd.Text2D(
                f"Scale {s:.2f}\ncurvedness and local fit around point 0",
                pos="top-middle",
            ),
        )
    plt.interactive()


if __name__ == "__main__":
    shapes = [
        dict(
            shape="sphere",
            radius=1,
            scale=0.05,
            n_points=500,
        ),
        dict(
            shape="sphere",
            radius=0.1,
            scale=0.005,
            n_points=500,
        ),
        dict(
            shape="sphere",
            radius=10,
            scale=0.5,
            n_points=500,
        ),
        dict(
            function=lambda x, y: 0.5 * x**2 + 0.5 * y**2,
            scale=0.05,
            n_points=31,
            noise=0.0001,
            highlight=0 if False else int(31**2 / 2),
        ),
        dict(
            function=lambda x, y: (2 - 0.5 * x**2 - y**2).abs().sqrt() - 1,
            scale=0.05,
            n_points=15,
            noise=0.01,
        ),
        dict(
            function=lambda x, y: 0.5 * x**2 - 0.5 * y**2,
            scale=0.05,
            n_points=31,
            noise=0.0001,
            highlight=0 if False else int(31**2 / 2),
        ),
        dict(
            shape="unit patch",
            function=lambda x, y: 2 * x * y,
            scale=0.08,
            n_points=1000,
        ),
        dict(
            function=lambda x, y: 0.5 * x * x / 3 - x * y - 0.5 * 1.5 * y * y + 2,
            scale=0.05,
            n_points=15,
            noise=0.0001,
            highlight=0 if False else int(15**2 / 2),
        ),
        dict(
            function=lambda x, y: y**2 + y,
            scale=0.05,
            n_points=31,
            noise=0.0001,
            highlight=0 if False else int(31**2 / 2),
        ),
        dict(
            file_name="~/data/PN1.stl",
            scale=2.0,
            n_points=1e4,
            highlight=0,
        ),
    ]
    shapes = shapes[:-1]
    mode = ["display", "profile"][1]

    if mode == "display":
        for s in shapes:
            display_curvatures(**s)
            print("")

    elif mode == "profile":
        from .utils import profiler

        myprof = profiler()

        descr = shapes[0]
        with myprof as prof:
            scale = descr.pop("scale")
            highlight = descr.pop("highlight", 0)

            shape = create_shape(**descr)
            kmax, kmin = shape.point_principal_curvatures(scale=scale)
            print(f"Kmax: {kmax[highlight]}, Kmin: {kmin[highlight]}")

        print(shape.point_moments.cache_info())

        # Create an "output/" foler if it doesn't exist
        import os

        if not os.path.exists("output"):
            os.makedirs("output")

        # Export to chrome://tracing
        prof.export_chrome_trace(f"output/trace_curvatures.json")
        prof.export_stacks(
            f"output/stacks_curvatures.txt",
            "self_cpu_time_total",  # "self_cuda_time_total",
        )
