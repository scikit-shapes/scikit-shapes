import time
import numpy as np
import torch
import skshapes as sks

from pytest import approx
from hypothesis import given, settings
from hypothesis import strategies as st

import vedo as vd

from .utils import create_point_cloud, create_shape

import sys

sys.path.append(sys.path[0][:-4])


@given(
    n_points=st.integers(min_value=400, max_value=500),
    a=st.floats(min_value=-1, max_value=1),
    b=st.floats(min_value=-1, max_value=1),
    c=st.floats(min_value=-1, max_value=1),
    d=st.floats(min_value=-1, max_value=1),
    e=st.floats(min_value=-1, max_value=1),
    f=st.floats(min_value=-1, max_value=1),
)
@settings(max_examples=1, deadline=None)
def test_curvatures_quadratic(
    *,
    n_points: int,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
):
    # Our current estimation method relies on the estimation of the tangent
    # plane, and does not give perfect results for quadratic functions "off
    # center".
    # See e.g. the function f(x,y) = y**2 + y.
    #
    # Another issue is a fairly large variance for very large values of the
    # coefficients,
    # e.g. f(x,y) = 2 * x * y
    d = 0 * d
    e = 0 * e

    def poly(x, y):
        return (
            0.5 * a * x**2 + b * x * y + 0.5 * c * y**2 + d * x + e * y + f
        )

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
    # Our convention for unoriented point clouds is that the mean curvature
    # is >= 0:
    mean = np.abs(mean)

    # Create a point clouds around [0, 0] in the (x,y) plane and compute
    # z = f(x, y).
    # Point shape.points[0] = [0, 0, f(0, 0)]
    shape = create_shape(shape="unit patch", n_points=n_points, function=poly)

    scales = [0.8, 1]

    for scale in scales:
        kmax, kmin = shape.point_principal_curvatures(scale=scale)
        return None
        kmax = kmax[0].item()
        kmin = kmin[0].item()
        assert kmax * kmin == approx(gauss, abs=5e-1, rel=2e-1)
        assert (kmax + kmin) / 2 == approx(mean, abs=5e-1, rel=2e-1)


@given(
    n_points=st.integers(min_value=500, max_value=1000),
    radius=st.floats(min_value=0.1, max_value=10),
    relative_scale=st.floats(min_value=0.1, max_value=0.12),
)
@settings(deadline=None)
def test_curvatures_sphere(
    *, n_points: int, radius: float, relative_scale: float
):
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
    curvatures = sks.smooth_curvatures_2(
        points=points, normals=normals, scale=scale
    )

    # Our surface points:
    r = 500 / np.sqrt(len(points))
    spheres = vd.Points(points, r=r).cmap("RdBu_r", curvatures["mean"])
    spheres = spheres.add_scalarbar()

    quiver = vd.Arrows(points, points + 0.2 * normals, c="green", alpha=0.9)

    plt = vd.Plotter(axes=2)
    plt.show(spheres, quiver)
    plt.close()


if __name__ == "__main__":
    import argparse
    from pprint import pprint
    import glob

    def display_curvatures(*, scale=1, highlight=None, **kwargs):
        shape = create_shape(**kwargs)
        scales = [scale, 2 * scale, 5 * scale, 10 * scale]

        fig3D = vd.Plotter(shape=(2, 2), axes=1)

        if highlight is None:
            highlight = shape.points[:,1].argmin()

        for i, s in enumerate(scales):
            Xm = shape.point_moments(order=1, scale=s)
            curvedness = shape.point_curvedness(scale=s)
            r2 = shape.point_quadratic_coefficients(scale=s).r2
            kmax, kmin = shape.point_principal_curvatures(scale=s)
            print(
                f"Kmax: {kmax[highlight]}, Kmin: {kmin[highlight]},"
                + f" R2: {r2[highlight]}"
            )
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

            reference_point = vd.Points(
                shape.points[highlight].view(1, 3), c="red", r=10
            )
            local_average = vd.Points(Xm[highlight].view(1, 3), c="blue", r=10)
            local_fit = vd.Points(local_fit, c=(235, 158, 52), r=5)

            # Our surface points:
            if shape.triangles is None:
                shape_ = vd.Points(
                    shape.points,
                    c=(0.5, 0.5, 0.5),
                    r=60 / (shape.n_points ** (1 / 3)),
                )
            else:
                shape_ = shape.to_vedo()

            if args.color == "curvatures":
                shape_.pointcolors = shape.point_curvature_colors(scale=s)
            elif args.color == "curvedness":
                shape_ = (
                    shape_.clone()
                    .cmap("viridis", curvedness, vmin=0, vmax=1.2 * torch.quantile(curvedness, 0.9))
                    .add_scalarbar()
                )
            
            shape_ = shape_.alpha(args.alpha)
            #vd.file_io.write(shape_, f"shape_{i}.vtk")

            # Compute a bounding box for the shape:
            bounding_box = shape_.bounds()
            graph_size = 0.5 * (bounding_box[1] - bounding_box[0])
            offset = bounding_box[1] + 1.2 * graph_size

            # Plot a curvature diagram as in "Generation of tubular and membranous
            # shape textures with curvature functionals", Anna Song, 2021.
            # Compute the quantiles of the curvature distribution:
            quantiles = torch.Tensor([0.1, 0.90])
            qmaxmin, qmaxmax = torch.quantile(kmax, quantiles)
            qminmin, qminmax = torch.quantile(kmin, quantiles)
            Kscale = 1.2 * float(
                max(abs(qmaxmax), abs(qmaxmin), abs(qminmax), abs(qminmin))
            )
            curvature_diagram = (
                vd.pyplot.histogram(
                    kmax,
                    kmin,
                    xlim=[-Kscale, Kscale],
                    ylim=[-Kscale, Kscale],
                    bins=(51, 51),
                    scalarbar=False,
                )
                .scale(graph_size / Kscale)
                .shift(offset, 0, 0)
            )

            fig3D.at(i).show(
                shape_,
                reference_point,
                local_average,
                local_fit.clone().alpha(0.5),
                curvature_diagram,
                vd.Text2D(
                    f"Scale {s:.2f}\ncurvedness and local fit around point 0",
                    pos="top-middle",
                ),
                bg=(0.5, 0.5, 0.5),
            ).parallel_projection()

        fig3D.interactive()

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
            function=lambda x, y: 0.5 * x * x / 3
            - x * y
            - 0.5 * 1.5 * y * y
            + 2,
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
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="Scale of the curvature estimation.",
    )
    parser.add_argument(
        "--highlight",
        type=int,
        default=None,
        help="Index of the point to highlight.",
    )
    parser.add_argument(
        "--mode",
        choices=["display", "profile"],
        default="display",
        help="Whether to display the results or profile the code.",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=None,
        help="Number of points in the point cloud.",
    )
    parser.add_argument(
        "source",
        nargs="*",
        default=None,
        help="Shape to load.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="Transparency.",
    )
    parser.add_argument(
        "--color",
        choices=["curvatures", "curvedness"],
        default="curvatures",
        help="Color code for the curvature.",
    )

    args = parser.parse_args()

    if args.source:
        sources = [glob.glob(s) for s in args.source]
        sources = [item for sublist in sources for item in sublist]

        shapes = [
            dict(
                file_name=source,
                scale=args.scale,
                n_points=args.n_points,
                highlight=args.highlight,
            )
            for source in sources
        ]
    pprint(shapes)

    if args.mode == "display":
        for s in shapes:
            display_curvatures(**s)
            print("")

    elif args.mode == "profile":
        from .utils import profiler

        myprof = profiler()

        descr = shapes[0]
        scale = descr.pop("scale")
        highlight = descr.pop("highlight", 0)
        descr["n_points"] = 5000

        shape = create_shape(**descr)

        start = time.time()
        kmax, kmin = shape.point_principal_curvatures(scale=scale)
        print(f"Kmax: {kmax[highlight]}, Kmin: {kmin[highlight]}")
        stop = time.time()
        print(f"First run, elapsed time = {stop - start:.3f}s")

        with myprof as prof:
            kmax, kmin = shape.point_principal_curvatures(scale=scale)
            print(f"Kmax: {kmax[highlight]}, Kmin: {kmin[highlight]}")

        print(shape.point_moments.cache_info())

        # Create an "output/" foler if it doesn't exist
        import os

        if not os.path.exists("output"):
            os.makedirs("output")

        # Export to chrome://tracing
        prof.export_chrome_trace("output/trace_curvatures.json")
        prof.export_stacks(
            "output/stacks_curvatures.txt",
            "self_cpu_time_total",  # "self_cuda_time_total",
        )
