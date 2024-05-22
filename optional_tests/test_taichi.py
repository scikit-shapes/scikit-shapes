"""Dummy test for testing the importation of taichi."""

import pytest

import skshapes as sks


@pytest.mark.skipif(not sks.taichi_installed, reason="Taichi is not installed")
def test_taichi():
    """This test imports taichi.

    If taichi is not installed, the test is skipped.
    """
    import taichi as ti

    ti.init(arch=ti.cpu)
    n = 320
    pixels = ti.field(dtype=float, shape=(n, n))
    pixels.fill(1.0)
    pixels_np = pixels.to_numpy()
    assert pixels_np.shape == (n, n)
    assert pixels_np.sum() == n**2
