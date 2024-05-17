"""
Creating a system of incompressible cells
==============================================

This notebooks shows how to simulate a simple population of cells in 2D and 3D.

"""

import taichi as ti

ti.init(arch=ti.cpu)

import time
import torch
from matplotlib import pyplot as plt

import skshapes as sks

Nx, Ny, n_particles = 256, 128, 50

X, Y = torch.meshgrid(
    torch.linspace(0, 1, Nx),
    torch.linspace(1, 0, Ny),
    indexing="xy",
)
domain = ((X - 0.5) ** 2 + (Y - 0.5) ** 2).sqrt() < 0.5
domain = domain | (Y > 0.5)

particles = sks.ParticleSystem(
    domain=domain,
    n_particles=n_particles,
    particle_type=sks.AnisotropicPowerCell2D,
)

# Define two populations of particles:
labels = torch.randint(0, 2, (n_particles,))

# Define the properties of both types of cells:
powers = torch.tensor([2.0, 2.0])

# By convention, the total volume of the domain is equal to 1:
volumes = torch.tensor([0.2, 0.5]) / n_particles
precisions = torch.stack(
    [
        torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
    ]
)

# Load the particles with our desired attributes:
particles.position = Ny * (0.2 + 0.5 * torch.rand((n_particles, 2)))
particles.precision_matrix = precisions[labels]
particles.power = powers[labels]
particles.volume = volumes[labels]


def display(title=""):
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(
        particles.pixel_colors(
            particle_colors=labels.cpu().numpy(),
            line_width=1,
        ),
        extent=(0, Nx, Ny, 0),
        origin="upper",
    )
    if hasattr(particles, "cell_centers"):
        plt.scatter(
            particles.barycenter[:, 1],
            particles.barycenter[:, 0],
            c="g",
            s=9,
        )
    plt.axis([0, Nx, 0, Ny])
    plt.xticks([])
    plt.yticks([])


particles.volume_fit()
display()


###############################################################################
# Simulate the cells falling under gravity
#

start = time.time()

for it in range(5):
    pos = particles.barycenter
    # pos[:, 0] -= 2
    particles.position = pos.clamp(0, Ny - 1)
    converged = particles.volume_fit(verbose=True)

    end = time.time()
    print("Time:", end - start)

    if it % 1 == 0:
        display(title=f"Iteration {it}")

plt.show()
