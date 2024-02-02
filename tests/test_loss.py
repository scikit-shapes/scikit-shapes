"""Tests for the loss functions."""
import torch

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
