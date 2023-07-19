import torch

def create_point_cloud(n_points: int, f: callable, normals=False):
    """Create a point cloud from a function f: R^3 -> R."""
    x = torch.linspace(-1, 1, n_points)
    y = torch.linspace(-1, 1, n_points)

    if normals:
        x.requires_grad = True
        y.requires_grad = True

    x, y = torch.meshgrid(x, y, indexing="ij")
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = f(x, y)

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

    return (2 / scale) * (X @ quadric)[:,:3]
