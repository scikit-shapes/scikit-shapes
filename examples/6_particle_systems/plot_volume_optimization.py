"""
Finding cells with correct volumes
==============================================

This notebooks discusses the optimal transport solver which is behind the
particle system class.

"""

##################################################################
# Create a toy 1D problem with a single particle:

import collections

import taichi as ti

ti.init(arch=ti.cpu)

import torch
from matplotlib import pyplot as plt

import skshapes as sks

##################################################################
# First of all, define a utility function to display the dual cost, gradient
# and hessian of the dual optimal transport problem as a function of
# one of the dual potentials:
#


def display_optimization_graphs(
        particles: sks.ParticleSystem,
        particle_index: int = 0,
    ):

    good_potential = particles.volume[particle_index]
    potential_values = torch.linspace(-0.1 * good_potential, 2 * good_potential, 10001)

    record = collections.defaultdict(
        lambda: collections.defaultdict(lambda: torch.zeros(len(potential_values)))
    )

    for integral_dimension in [0, 1]:

        # Set the smoothness of the dual cost:
        particles.integral_dimension = integral_dimension

        # Reset the dual potentials:
        particles.init_dual_potentials()

        for i, potential in enumerate(potential_values):
            particles.seed_potential[particle_index] = potential
            particles.compute_dual_loss()
            record["loss"][integral_dimension][i] = particles.dual_loss
            record["grad"][integral_dimension][i] = particles.dual_grad[
                particle_index
            ]
            record["hessian"][integral_dimension][i] = particles.dual_hessian[
                particle_index, particle_index
            ]
            record["volume"][integral_dimension][i] = particles.cell_volume[particle_index]

        particles.init_dual_potentials()
        particles.fit_cells(
            method="L-BFGS-B" if integral_dimension == 0 else "Newton",
            verbose=True,
            max_iter=10,
        )
        print(particles.seed_potential[particle_index], particles.cell_volume)

    # Plot the results on 4 different figures, one for each key in the record
    for k in record:
        plt.figure()
        for integral_dimension in record[k]:
            plt.plot(
                potential_values,
                record[k][integral_dimension],
                label=f"Integral dimension {integral_dimension}",
            )

        plt.xlabel(f"Potential[{particle_index}]")
        plt.ylabel(k)
        if k == "volume":
            plt.axhline(y=particles.volume[0], color="g", linestyle="-")

        plt.title(f"{k} vs potential for the particle {particle_index}")
        plt.legend()
        plt.grid(True)

##################################################################
# A simple 1D example:
#

Nx, n_particles = 100, 9
domain = torch.ones(Nx)

positions = Nx * torch.linspace(0.103, 0.9, n_particles).view(-1, 1)
target_volumes = 0.6 * torch.ones(n_particles) * Nx / n_particles


particles = sks.ParticleSystem(
    domain=domain,
    particles=[
        sks.PowerCell(
            position=positions[k],
            volume=target_volumes[k],
            power=2,
        )
        for k in range(n_particles)
    ]
)

print(f"Positions: {positions}")
print(f"Target volumes: {target_volumes}")

display_optimization_graphs(particles)


##################################################################
# Now in 2D:
#

Nx, Ny, n_particles = 10, 10, 9
domain = torch.ones(Nx, Ny)

t = torch.linspace(0, 2 * 3.14, n_particles + 1)[:-1]
positions = torch.stack([
    Nx * .3 * (1 + torch.cos(t)),
    Ny * .3 * (1 + torch.sin(t)),
    ], dim=1)
target_volumes = 0.6 * torch.ones(n_particles) * Nx * Ny / n_particles


particles = sks.ParticleSystem(
    domain=domain,
    particles=[
        sks.PowerCell(
            position=positions[k],
            volume=target_volumes[k],
            power=2,
        )
        for k in range(n_particles)
    ]
)

print(f"Positions: {positions}")
print(f"Target volumes: {target_volumes}")

display_optimization_graphs(particles)
