import torch
import skshapes as sks
import pytest

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
    import skshapes as sks
    import pytorch_shell
    import torch

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
