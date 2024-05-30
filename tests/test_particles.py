"""Tests for the ParticleSystem class."""

import pytest
import taichi as ti
import torch
from hypothesis import given
from hypothesis import strategies as st

import skshapes as sks


@given(
    n_particles=st.integers(min_value=1, max_value=50),
    integral_dimension=st.integers(min_value=0, max_value=1),
    n_x=st.integers(min_value=1, max_value=10),
    n_y=st.integers(min_value=1, max_value=10),
    n_z=st.integers(min_value=1, max_value=10),
    barrier=st.none() | st.floats(min_value=0, max_value=10),
)
def test_dual_loss_grad_hessian(
    *,
    n_particles: int,
    integral_dimension: int,
    n_x: int,
    n_y: int,
    n_z: int,
    barrier: float | None,
):
    ti.init(arch=ti.cpu)

    X, Y = torch.meshgrid(
        torch.linspace(0, 1, n_x),
        torch.linspace(0, 1, n_y),
        indexing="ij",
    )
    domain = ((X - 0.5) ** 2 + (Y - 0.5) ** 2).sqrt() < 0.5
    domain = domain | (Y > 0.5) | (X > 0.8)
    domain = X > -1
    domain_volume = domain.int().sum().item()

    assert n_z > 0

    particles = sks.ParticleSystem(
        domain=domain,
        n_particles=n_particles,
        particle_type=sks.PowerCell2D,
        integral_dimension=integral_dimension,
    )
    particles.position = torch.rand((n_particles, 2)) * torch.tensor(
        [n_x, n_y]
    )
    particles.power = torch.rand((n_particles,)) * 4
    particles.volume = torch.rand((n_particles,)) * domain_volume / n_particles

    potentials = 10 * torch.rand((n_particles,))
    particles.offset = -potentials

    particles.compute_dual_loss(barrier=barrier)
    loss = particles.dual_loss
    grad = particles.dual_grad
    hess = particles.dual_hessian
    dt = 1e-5
    for _ in range(3):
        direction = torch.randn((n_particles,))
        particles.offset = -(potentials + dt * direction)
        particles.compute_dual_loss(barrier=barrier)

        loss_p = particles.dual_loss
        grad_p = particles.dual_grad
        hess_p = particles.dual_hessian

        assert (loss_p - loss) / dt == pytest.approx(
            grad @ direction, abs=1e-2
        )
        assert (grad_p - grad) / dt == pytest.approx(
            hess @ direction, abs=1e-2
        )
        assert (hess_p - hess) / dt == pytest.approx(0, abs=1e-2)
