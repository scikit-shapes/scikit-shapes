# ruff: noqa: ARG001 (unused arguments)

import pytest
import torch

from skshapes.morphing.intrinsic_deformation import metric_validation


def test_metric_validation():
    """Test the metric validation function.

    A metric must have the following signature:
    `def metric(points_sequence, velocities_sequence, edges, triangles)`,
    additional optional arguments are allowed. What is more, the output
    must be a scalar tensor, differentiable wrt the velocities.

    This test function assert that the validation function raises the correct
    errors when the metric does not satisfy the requirements on toy examples
    and check the validity of the metrics implemented in skshapes.
    """

    def valid_metric(
        points_sequence,
        velocities_sequence,
        edges,
        triangles,
    ):
        """This metric is valid."""

        return velocities_sequence.mean()

    def invalid_metric_numpy_output(
        points_sequence,
        velocities_sequence,
        edges,
        triangles,
    ):
        """This metric returns a numpy array."""
        return velocities_sequence.mean().detach().numpy()

    def invalid_metric_not_scalar(
        points_sequence,
        velocities_sequence,
        edges,
        triangles,
    ):
        """This metric returns a tensor with more than one element."""
        return velocities_sequence

    def invalid_metric_diff(
        points_sequence,
        velocities_sequence,
        edges,
        triangles,
    ):
        """This metric violates the differentiability requirement."""
        velocities_sequence = velocities_sequence.detach().clone()
        return torch.mean(velocities_sequence) + 1

    def invalid_metric_args(
        points_sequence,
        velocities_sequence,
        triangles,
    ):
        """This metric has no edges argument."""
        return torch.mean(velocities_sequence)

    with pytest.raises(ValueError, match="The argument edges is missing"):
        metric_validation(invalid_metric_args)

    with pytest.raises(
        ValueError,
        match="The metric must be differentiable wrt the velocities.",
    ):
        metric_validation(invalid_metric_diff)

    with pytest.raises(ValueError, match="The metric must return a tensor."):
        metric_validation(invalid_metric_numpy_output)

    with pytest.raises(ValueError, match="The metric must return a scalar."):
        metric_validation(invalid_metric_not_scalar)

    metric_validation(valid_metric)

    from skshapes.morphing.intrinsic_deformation import (
        as_isometric_as_possible,
        shell_energy_metric,
    )

    metric_validation(as_isometric_as_possible)
    metric_validation(shell_energy_metric)


test_metric_validation()
