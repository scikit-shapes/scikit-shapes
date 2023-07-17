import sys

sys.path.append(sys.path[0][:-4])

import torch
import skshapes as sks

from hypothesis import given, settings
from hypothesis import strategies as st


sks.implicit_quadrics(
    points=torch.rand(100, 3),
    scale=1,
)


@given(n_points=st.integers(min_value=5, max_value=10))
@settings(deadline=1000)
def test_quadratic_function(*, n_points: int):
    """Test on a simple dataset z = f(x, y)."""
    # Create the dataset
    x = torch.linspace(-1, 1, n_points)
    y = torch.linspace(-1, 1, n_points)
    x, y = torch.meshgrid(x, y, indexing="ij")
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = x**2 + y**2

    N = len(x)
    assert N == n_points**2

    # Compute the implicit quadrics
    points = torch.stack([x, y, z], dim=1).view(N, 3)
    quadrics, mean_point, sigma = sks.implicit_quadrics(points=points, scale=1)

    # Check that the quadrics are correct
    assert quadrics.shape == (N, 4, 4)

    X = torch.stack([x, y, z], dim=1)
    assert X.shape == (N, 3)

    X -= mean_point
    X /= sigma

    X = torch.cat([X, torch.ones_like(X[:,:1])], dim=1)
    assert X.shape == (N, 4)

    XXt = X.view(N, 4, 1) * X.view(N, 1, 4)

    self_scores = (XXt * quadrics[0]).sum(dim=(1, 2))

    print(self_scores)
    print(quadrics[0])
    assert False
