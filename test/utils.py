import torch
from torch.profiler import profile, ProfilerActivity
import vedo as vd
import skshapes as sks
from skshapes.types import polydata_type
from typing import Optional, Literal
import sys

sys.path.append(sys.path[0][:-4])


def create_point_cloud(
    n_points: int, f: callable, normals=False, dtype=sks.float_dtype
):
    """Create a point cloud from a function f: R^3 -> R."""
    x = torch.linspace(-1, 1, n_points).to(dtype=dtype)
    y = torch.linspace(-1, 1, n_points).to(dtype=dtype)

    if normals:
        x.requires_grad = True
        y.requires_grad = True

    x, y = torch.meshgrid(x, y, indexing="ij")
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = f(x, y).to(dtype=dtype)

    N = len(x)
    assert N == n_points**2

    points = torch.stack([x, y, z], dim=1).view(N, 3)

    if not normals:
        return points

    grad_x, grad_y = torch.autograd.grad(z.sum(), [x, y])
    assert grad_x.shape == (N,)
    assert grad_y.shape == (N,)

    n = torch.stack([-grad_x, -grad_y, torch.ones_like(grad_x)], dim=1)
    n = torch.nn.functional.normalize(n, p=2, dim=-1)

    return points.detach(), n.detach()


def create_shape(
    *,
    shape: Optional[Literal["sphere"]] = None,
    file_name: str = None,
    function: callable = None,
    n_points=20,
    noise=0,
    radius=1,
    offset=0,
):
    if shape == "sphere":
        points = torch.randn(n_points, 3)
        points = radius * torch.nn.functional.normalize(points, p=2, dim=1)
        shape = sks.PolyData(points=points)

    elif shape == "unit patch":
        points = torch.randn(n_points, 3)
        points[0, :] = 0
        points[:, 2] = function(points[:, 0], points[:, 1])
        shape = sks.PolyData(points=points)

    elif function is not None:
        points = create_point_cloud(n_points=n_points, f=function)
        shape = sks.PolyData(points=points)

    else:
        shape = sks.PolyData(file_name)
        if n_points is not None:
            shape = shape.decimate(n_points=n_points)
        print("Loaded shape with {:,} points".format(shape.n_points))

    shape.points = shape.points + offset * torch.randn(1, 3).to(
        sks.float_dtype
    )
    shape.points = shape.points + noise * torch.randn(shape.n_points, 3).to(
        sks.float_dtype
    )
    return shape


def dim4(*, points, offset=0, scale=1):
    """Rescales and adds a fourth dimension to the point cloud."""
    assert points.shape == (len(points), 3)
    points = points - offset
    points = points / scale
    return torch.cat([points, torch.ones_like(points[:, :1])], dim=1)


def quadratic_function(*, points, quadric, offset=0, scale=1):
    assert quadric.shape == (4, 4)
    X = dim4(points=points, offset=offset, scale=scale)
    return ((X @ quadric) * X).sum(-1)


def quadratic_gradient(*, points, quadric, offset=0, scale=1):
    # Make sure that the quadric is symmetric:
    assert quadric.shape == (4, 4)
    quadric = (quadric + quadric.T) / 2

    X = dim4(points=points, offset=offset, scale=scale)

    return (2 / scale) * (X @ quadric)[:, :3]


def vedo_frames(points, frames):
    n = vd.Arrows(points, points + frames[:, :, 0], c="red", alpha=0.9)
    u = vd.Arrows(points, points + frames[:, :, 1], c="blue", alpha=0.9)
    v = vd.Arrows(points, points + frames[:, :, 2], c="green", alpha=0.9)
    return [n, u, v]


def profiler():
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    myprof = profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(
            verbose=True
        ),
    )
    return myprof


def create_2d_polydata() -> tuple[polydata_type, polydata_type]:
    """Generate two 2D shapes: a circle and a line"""
