"""Test the shell energy implementation."""

import pytest
import torch

import skshapes as sks

try:
    import pytorch_shell

    pytorch_shell_available = True
except ImportError:
    pytorch_shell_available = False


# Test that is skipped if pytorch_shell is not installed
@pytest.mark.skipif(
    not pytorch_shell_available, reason="pytorch_shell is not installed"
)
def test_shell_energy():
    """Test that the shell energy is the same as pytorch_shell (reference)."""
    source = sks.Sphere()
    target = sks.Sphere()
    target.points += torch.randn_like(target.points) * 0.05

    points_undef = source.points
    points_def = target.points
    triangles = source.triangles

    sks_bending = sks.bending_energy(
        points_undef=points_undef,
        points_def=points_def,
        triangles=triangles,
    )

    sks_membrane = sks.membrane_energy(
        points_undef=points_undef,
        points_def=points_def,
        triangles=triangles,
    )

    sks_shell = sks.shell_energy(
        points_undef=points_undef,
        points_def=points_def,
        triangles=triangles,
    )

    pytorch_shell_bending = pytorch_shell.bending_energy(
        points_undef=points_undef,
        points_def=points_def,
        triangles=triangles,
    )

    pytorch_shell_membrane = pytorch_shell.membrane_energy(
        points_undef=points_undef,
        points_def=points_def,
        triangles=triangles,
    )

    pytorch_shell_shell = pytorch_shell.shell_energy(
        points_undef=points_undef,
        points_def=points_def,
        triangles=triangles,
    )

    assert torch.allclose(sks_bending, pytorch_shell_bending)
    assert torch.allclose(sks_membrane, pytorch_shell_membrane)
    assert torch.allclose(sks_shell, pytorch_shell_shell)


def test_register_with_shellenergy():
    """Test that the registration works with shell energy."""
    source = sks.Sphere()
    target = sks.Sphere()
    loss = sks.L2Loss()
    model = sks.IntrinsicDeformation(
        n_steps=3,
        metric="shell_energy",
    )

    r = sks.Registration(loss=loss, model=model)

    r.fit(
        source=source,
        target=target,
    )
