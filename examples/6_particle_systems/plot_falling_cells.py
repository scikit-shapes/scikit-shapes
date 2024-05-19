"""
Creating a system of incompressible cells
==============================================

This notebooks shows how to simulate a simple population of cells in 2D and 3D.

"""

import taichi as ti

ti.init(arch=ti.cpu)

import time

import numpy as np
import torch
from matplotlib import animation
from matplotlib import pyplot as plt

import skshapes as sks

Nx, Ny, n_particles = 300, 200, 50
X, Y = torch.meshgrid(
    torch.linspace(0, 1, Nx),
    torch.linspace(0, 1, Ny),
    indexing="ij",
)
domain = ((X - 0.5) ** 2 + (Y - 0.5) ** 2).sqrt() < 0.5
domain = domain | (Y > 0.5) | (X > 0.8)
domain_volume = domain.int().sum().item()

plt.figure()
plt.imshow(domain.cpu().numpy().transpose(1, 0), origin="lower")
print(f"Total volume of the domain: {domain_volume:,} pixels")

###############################################################################
# Create a population of cells
#
particles = sks.ParticleSystem(
    domain=domain,
    n_particles=n_particles,
    particle_type=sks.AnisotropicPowerCell2D,
    integral_dimension=1,
)

# Define two populations of particles:
labels = (torch.arange(n_particles) > (n_particles // 2)).int()

# Define the properties of both types of cells:
powers = torch.tensor([2.0, 2.0])

# By default, the volume of each pixel is 1:
volumes = torch.tensor([0.3, 0.5]) * domain_volume / n_particles
precisions = torch.stack(
    [
        torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
    ]
)


# Load the particles with our desired attributes:
particles.position = torch.rand((n_particles, 2)) * torch.tensor(
    [Nx, Ny / 2]
) + torch.tensor([0, Ny / 2])
particles.precision_matrix = precisions[labels]
particles.power = powers[labels]
particles.volume = volumes[labels]

for _ in range(5):
    particles.volume_fit(rtol=0.05)
    particles.position = particles.barycenter

###############################################################################
# Simulate the cells falling under gravity
#

start = time.time()
cell_volumes = []

t = 0
dt = 0.1
gravity_force = torch.tensor([0, -10])
velocities = 5 * torch.randn(n_particles, 2)

fig, ax = plt.subplots(figsize=(12, 6))
frames = []


for it in range(11):
    t += dt
    recall = particles.barycenter - particles.position
    velocities += (recall + gravity_force) * dt
    particles.position = particles.position + velocities * dt

    particles.volume_fit(
        stopping_criterion="max error", rtol=0.05, verbose=True
    )
    cell_volumes.append(particles.cell_volume.cpu().numpy())

    frames.append(
        particles.display(
            ax=ax,
            particle_colors=1000
            * particles.relative_volume_error.cpu().numpy(),
            title=f"t = {it * dt:.2f}, CPU time {time.time() - start:.2f}s",
            line_width=1,
        )
    )
    frames.append(
        particles.display(
            ax=ax,
            particle_colors=-2 * particles.descent_direction.cpu().numpy(),
            title=f"t = {it * dt:.2f}, CPU time {time.time() - start:.2f}s",
            line_width=1,
        )
    )

fig.colorbar(particles._scalarmappable, ax=ax)
ani = animation.ArtistAnimation(fig, frames, interval=50)


###############################################################################
# Convergence study:
#

cell_volumes = np.array(cell_volumes)[:, 1:]

plt.figure()
plt.plot(cell_volumes)
plt.ylim(volumes.min() * 0.7, volumes.max() * 1.3)
