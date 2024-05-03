"""Tests for the loss functions."""

import torch
from hypothesis import given
from hypothesis import strategies as st

import skshapes as sks


def test_combination_loss():
    """Test the linear combination of two losses."""

    l1 = sks.L2Loss()
    l2 = sks.LpLoss(p=3)

    circle_1 = sks.Circle(n_points=5)
    circle_2 = sks.Circle(n_points=5)
    circle_2.points += 0.1

    a = 0.5
    assert torch.allclose(
        l1(circle_1, circle_2) + a * l2(circle_1, circle_2),
        (l1 + a * l2)(circle_1, circle_2),
    )


@given(p=st.floats(min_value=0, max_value=10))
def test_lp_loss(p):
    loss = sks.LpLoss(p=p)

    X = torch.tensor(
        [[0, 0], [0, 0]],
        dtype=sks.float_dtype,
    )

    Y = torch.tensor(
        [[0, 1], [-1, 2]],
        dtype=sks.float_dtype,
    )

    # Square distances : [1, 5]
    # Lp norm must be 1^(p/2) + 5^(p/2) = 1 + 5^(p/2)

    shape_X = sks.PolyData(points=X)
    shape_Y = sks.PolyData(points=Y)

    assert torch.isclose(
        loss(shape_X, shape_Y), torch.tensor(1 + 5 ** (p / 2))
    )
