import collections

import matplotlib as mpl
import numpy as np
import taichi as ti
import taichi.math as tm
import torch

from ..input_validation import convert_inputs, typecheck
from ..types import (
    Callable,
    FloatTensor,
    Int32Tensor,
    IntTensor,
    Literal,
    Particle,
)

ti_float = ti.f32
AMBIENT_SPACE_LABEL = -1
MASK_LABEL = -2


@ti.data_oriented
class ParticleSystem:
    """A collection of particles that may interact with each other.

    Parameters
    ----------
    particles : list[Particle]
        A list of particles that will be used to define the cells.
    domain : torch tensor
        A binary mask that defines the domain of the particles.
    block_size : int, optional
        The size of the blocks in the pixel grids.
    integral_dimension : int, optional
        Dimension of the volume elements that we use to compute cell volumes.
        A value of 0 corresponds to a simple summation over a discrete pixel grid.
        A value of 1 corresponds to a piecewise linear model along a cartesian wireframe.


    Examples
    --------

    See the corresponding [examples](../../../generated/gallery/#particle-systems)
    for a demonstration of the `ParticleSystem` class.
    """

    @convert_inputs
    @typecheck
    def __init__(
        self,
        *,
        particles: list[Particle],
        domain: IntTensor,
        block_size: int | None = None,
        integral_dimension: Literal[0, 1] = 1,
    ) -> None:

        self.__dict__["device"] = domain.device

        self._load_particles(particles)

        self.integral_dimension = integral_dimension

        self.domain = domain.type(torch.int32)
        # Check that the domain is binary
        if not ((self.domain == 0) | (self.domain == 1)).all():
            msg = "The domain must be a binary mask."
            raise ValueError(msg)

        # By default, pixels are unit cubes.
        self.pixel_volume = 1

        self.shape = self.domain.shape

        # Make sure that our code is compatible with 1D, 2D and 3D domains by
        # defining the appropriate index types and segment connectivities.
        if len(self.shape) == 1:
            self.dim = 1
            ind = ti.i
            self.i32vector = ti.types.vector(1, ti.i32)
            self.segment_connectivity = [(1,)]

        elif len(self.shape) == 2:
            self.dim = 2
            ind = ti.ij
            self.i32vector = ti.types.vector(2, ti.i32)
            self.segment_connectivity = [(1, 0), (0, 1)]

        elif len(self.shape) == 3:
            self.dim = 3
            ind = ti.ijk
            self.i32vector = ti.types.vector(3, ti.i32)
            self.segment_connectivity = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        else:
            msg = "Only 2D and 3D domains are supported."
            raise ValueError(msg)

        # We use a periodic domain in the Jump Flooding Algorithm
        # (but by default, we do not use periodic costs).
        # This means that we need to compute the modulo of the indices
        # with respect to the domain shape, which can be expensive
        # for non-power-of-two shapes.
        self.power_of_two = all(
            (n & (n - 1) == 0) and n != 0 for n in self.shape
        )
        if self.power_of_two:
            # Use the fact that e.g. n % 8 = n & 00000111 = n & 7 in binary
            self.bitwise_modulo = self.i32vector(self.shape) - 1
        else:
            self.bitwise_modulo = None

        # Create the grid of pixels, and the intermediate buffer
        # that will be used in the Jump Flooding Algorithm.
        # We use these grids to store the index of the cell that is associated
        # to each pixel.
        self.pixels = ti.field(ti.i32)
        self.pixels_buffer = ti.field(ti.i32)

        if block_size is None:
            # By default, the pixel grids are just vanilla C-contiguous arrays.
            ti.root.dense(ind, self.shape).place(self.pixels)
            ti.root.dense(ind, self.shape).place(self.pixels_buffer)
        else:
            # But Taichi also lets us encode them as collections of square patches.
            # This can be useful for performance, especially on GPUs.
            coarse = tuple(s // block_size for s in self.shape)
            block = tuple(block_size for _ in self.shape)
            # The lines below mean that the pixel grids are now stored as
            # a collection of square patches of size (block_size, block_size).
            ti.root.dense(ind, coarse).dense(ind, block).place(self.pixels)
            ti.root.dense(ind, coarse).dense(ind, block).place(
                self.pixels_buffer
            )

    def _load_particles(self, particles: list[Particle]):
        """Loads the particles from a list of dictionaries."""

        if len(particles) == 0:
            msg = "The list of particles must not be empty."
            raise ValueError(msg)

        n_particles = len(particles)

        particle_type = particles[0]._taichi_class
        # We let users directly edit cell attributes by overloading
        # self.__getattr__, self.__setattr__, and checking if the attributes
        # correspond to a property of self.particle_type.
        # As a consequence, and before doing anything, we must initialize
        # self.particle_type with the low-level "self.__dict__" API.
        self.__dict__["particle_type"] = particle_type

        if any(p._taichi_class != self.particle_type for p in particles):
            msg = "Currently, all particles must have the same type."
            raise NotImplementedError(msg)

        # Create the collection of cells, one for each particle:
        self.cells = self.particle_type.field(shape=(n_particles,))

        # Turn the list of dictionaries into a dictionary of lists
        # (one list per attribute of the particle type)
        attributes = collections.defaultdict(list)
        # {k: [] for k in self.particle_type.__dict__["members"]}
        for p in particles:
            for k, v in p.__dict__.items():
                if k != "_taichi_class":
                    attributes[k].append(v)

        # Load the attributes into the Taichi field
        for k, v in attributes.items():
            if type(v[0]) in (int, float):
                v_tensor = torch.tensor(v, device=self.device)
            else:
                v_tensor = torch.stack(v).to(device=self.device)

            try:
                getattr(self.cells, k).from_torch(v_tensor)
            except AttributeError:
                # k should be an attribute of self instead of self.cells
                self.__dict__[k] = v_tensor

        # By default, all particles have zero dual potential:
        self.taichi_seed_potential = ti.field(ti_float, shape=(n_particles,))
        # self.cell_potential.from_torch(torch.zeros(1 + n_particles, device=self.device))

        if False:
            self.seed_potential = torch.zeros(n_particles, device=self.device)
        else:
            self.seed_potential = self.volume.clone()
            assert self.seed_potential.shape == (n_particles,)
            assert self.seed_potential.device == self.device
            assert self.seed_potential.dtype == torch.float32

        # Init the relevant attributes for the taichi loops.
        # Our convention is that:
        # - the mask has index -2,
        # - the ambient space has index -1,
        # - the seeds have indices 0, 1, 2, ..., n_particles-1

        self.cell_indices = ti.ndrange(n_particles)

        return particle_type

    @property
    @typecheck
    def n_particles(self) -> int:
        """Returns the number of seeds in the system, excluding the ambient space.

        Returns
        -------
        int
            The number of particles in the system.
        """
        return self.cells.shape[0]

    @property
    @typecheck
    def domain_volume(self) -> float:
        """Returns the total available volume in the domain.

        Returns
        -------
        float
            The total volume of the domain.
        """
        return float(self.domain.sum().item() * self.pixel_volume)

    def __getattr__(self, attr):
        """Give direct access to the attributes of self.cells."""
        if attr in self.__dict__["particle_type"].__dict__["members"]:
            # Case 1: the attribute is a property of the particle type
            out = getattr(self.cells, attr)  # out is a Taichi field
            # This is the Taichi way to convert to a torch tensor
            return out.to_torch()
        else:
            # Case 2: the attribute belongs to the ParticleSystem itself
            try:
                return self.__dict__[attr]
            except KeyError as e:
                raise AttributeError from e

    def __setattr__(self, attr, value):
        """Give direct access to the attributes of self.cells."""
        if attr in self.particle_type.__dict__["members"]:
            getattr(self.cells, attr).from_torch(value)
        else:
            self.__dict__[attr] = value

    @ti.func
    def wrap_indices(self, pos):
        """Ensures periodic boundary conditions for the Jump Flooding Algorithm.

        Note that this function is much faster when the domain shape is a power of two.

        Parameters
        ----------
        pos : ti int32 vector
            The position to wrap.

        Returns
        -------
        ti int32 vector
            The wrapped position, with values in [0, shape[0]), [0, shape[1]), etc.
        """
        # Modulo the indices to have periodic boundary conditions
        if ti.static(self.power_of_two):
            # Use the fact that e.g. n % 8 = n & 00000111 = n & 7 in binary
            return pos & self.bitwise_modulo
        else:
            return ti.cast(tm.mod(pos, self.shape), ti.i32)

    @ti.func
    def _cost_raw(self, i, pixel):
        """Computes the cost of cell i at pixel.

        Parameters
        ----------
        i : int
            The index of the cell.
        pixel : ti int64 vector
            The pixel coordinates.

        Returns
        -------
        float
            The cost of the cell at the pixel.
        """
        # By convention, cell i=0 is associated to a constant, zero cost
        # We want to return 0, but do not want to use if-else statements
        # as they would slow down GPU code. Instead, for cell i=0, we compute
        # another cost function and "select away" its value.
        index = ti.select(i == AMBIENT_SPACE_LABEL, 0, i)
        # N.B.: the cell cost function takes as input a float32 vector
        #       with pixel coordinates
        c_i = self.cells[index].cost(ti.cast(pixel, dtype=ti_float))
        return ti.select(i == AMBIENT_SPACE_LABEL, 0.0, c_i)

    @ti.func
    def _cost_offset(self, i, pixel):
        """Computes the cost of cell i at pixel, offset by the dual potential.

        Parameters
        ----------
        i : int
            The index of the cell.
        pixel : ti int64 vector
            The pixel coordinates.

        Returns
        -------
        float
            The cost of the cell at the pixel, offset by the dual potential.
        """
        # Offset the cost by the dual potential of the cell
        index = ti.select(i == AMBIENT_SPACE_LABEL, 0, i)
        offset = ti.select(
            i == AMBIENT_SPACE_LABEL, 0.0, self.taichi_seed_potential[index]
        )
        return self._cost_raw(i, pixel) - offset

    @ti.func
    def _cell_init_position(self, i):
        """Returns the initial position of cell i."""
        return self.cells[i].position

    @ti.kernel
    def _compute_labels_bruteforce(self):
        """For each pixel, computes the index of the cell with the lowest cost.
        This is a brute-force method that is suitable for a small number of cells.
        For very large systems, the Jump Flooding Algorithm is more efficient.
        """

        for X in ti.grouped(self.pixels):
            best_cost = ti_float(0.0)
            best_index = AMBIENT_SPACE_LABEL

            for i in ti.ndrange(self.n_particles):
                cost = self._cost_offset(i, X)

                if cost < best_cost:
                    best_cost = cost
                    best_index = i

            self.pixels[X] = best_index

    @ti.kernel
    def _init_sites(self):
        """Puts one pixel per seed in the pixel grid to init the Jump Flood Algorithm."""
        # Clear the pixels grid:
        for ind in ti.grouped(self.pixels):
            self.pixels[ind] = AMBIENT_SPACE_LABEL  # Ambient space by default

        # Place one pixel per seed.
        # Note that we must be careful here to have at least one pixel per cell:
        # otherwise, the JFA algorithm will not work as expected.
        # For this reason, we should not run the seeding in parallel:
        # ti.loop_config(serialize=True)
        for i in ti.ndrange(self.n_particles):
            if i != AMBIENT_SPACE_LABEL:  # Skip the ambient space
                # Cast the position to integers, using floor instead of round
                pos = tm.floor(self._cell_init_position(i), dtype=ti.i32)
                # Wrap the indices to have periodic boundary conditions
                # This is also useful if the user provided cells that lay outside
                # the domain
                pos = self.wrap_indices(pos)

                if self.pixels[pos] == AMBIENT_SPACE_LABEL:
                    # If the pixel is not already taken
                    self.pixels[pos] = i  # Assign the seed index to the pixel
                else:  # Collision!
                    # As a quick fix, we look for the nearest free pixel in
                    # a neighborhood of increasing size.
                    pos2 = pos
                    for radius in range(1, 10):
                        for dpos in ti.grouped(
                            ti.ndrange(*(((-radius, radius + 1),) * self.dim))
                        ):
                            pos2 = pos + ti.cast(dpos, dtype=ti.i32)
                            pos2 = self.wrap_indices(pos2)
                            if self.pixels[pos2] == AMBIENT_SPACE_LABEL:
                                self.pixels[pos2] = i
                                break
                        if self.pixels[pos2] == i:
                            break

                    assert (
                        self.pixels[pos2] == i
                    ), f"Could not find a free pixel for seed {i}."

    @ti.func
    def _clear_buffer(self):
        """Sets all pixels in the buffer to the ambient space."""
        for ind in ti.grouped(self.pixels_buffer):
            self.pixels_buffer[ind] = AMBIENT_SPACE_LABEL

    @ti.func
    def _copy_buffer(self):
        """Copies the pixel buffer to the main pixel grid."""
        for ind in ti.grouped(self.pixels):
            self.pixels[ind] = self.pixels_buffer[ind]

    @ti.kernel
    def _jfa_step(self, step: ti.i32):
        """Implements one step of the Jump Flooding Algorithm.

        Parameters
        ----------
        step : int
            The step size of the Jump Flooding Algorithm.
            Typically, the JFA will use steps of size 1, N/2, N/4, N/8, etc.
        """
        self._clear_buffer()

        for X in ti.grouped(self.pixels):
            # Loop over 9 neighbors at distance step
            # and find the closest seed to the current pixel [i,j]
            best_cost = ti_float(0.0)
            best_index = AMBIENT_SPACE_LABEL  # Ambient space by default

            ti.loop_config(serialize=True)  # Unroll the loop of size 3^dim

            # DX is a 1D, 2D or 3D vector with values in [-1, 0, 1]
            for DX in ti.grouped(ti.ndrange(*(((-1, 2),) * self.dim))):
                # Describe a 3^dim neighborhood around the current pixel,
                # with periodic boundary conditions
                X2 = X + step * ti.cast(DX, ti.i32)
                X2 = self.wrap_indices(X2)  # X2 = X2 modulo shape

                # Check if the cost function that is associated to X2 is a better
                # fit for the current pixel than the previous best candidate:
                i2 = self.pixels[X2]
                cost = self._cost_offset(i2, X)

                if cost < best_cost:
                    best_cost = cost
                    best_index = i2

            # We must use an intermediate buffer to avoid overwriting the pixels
            # while the main loop is still running
            self.pixels_buffer[X] = best_index

        self._copy_buffer()

    @ti.kernel
    def _cell_centers_volumes_kernel(
        self,
        centers: ti.types.ndarray(dtype=ti.int64, element_dim=0),
        volumes: ti.types.ndarray(dtype=ti.int64, element_dim=0),
    ):
        """Computes in place the non-normalized barycenter and volume of each cell.
        We use int64 instead of float32 to avoid overflows: when calling this function,
        do not forget to convert to float32 and normalize the barycenters afterwards.

        Parameters
        ----------
        centers : (n_particles, dim) ti ndarray
            The sum of pixel coordinates in each cell.
        volumes : (n_particles,) ti ndarray
            The volumes of the cells.
        """
        for X in ti.grouped(self.pixels):
            i = self.pixels[X]
            # Ignore the mask and the ambient space, just keep the "seed" indices
            if i != MASK_LABEL and i != AMBIENT_SPACE_LABEL:
                for d in ti.static(range(self.dim)):
                    centers[i, d] += X[d]
                volumes[i] += 1

    def _cell_centers_volumes(self):
        """Computes the barycenter and volume of each cell."""
        # We use int64 instead of float32 to avoid overflows during the summation.
        # Otherwise, since self._cell_centers_volumes just adds "+1" for every pixel,
        # it would run into problems when cells have more than 16,777,217 pixels.
        # (this is the first integer that cannot be represented exactly as a float32)
        centers = torch.zeros(
            (self.n_particles, self.dim), device=self.device, dtype=torch.int64
        )
        volumes = torch.zeros(
            self.n_particles, device=self.device, dtype=torch.int64
        )
        self._cell_centers_volumes_kernel(centers, volumes)
        centers = centers.to(torch.float32)
        volumes = volumes.to(torch.float32)

        centers = centers / volumes[:, None]
        centers[volumes == 0.0, :] = 0.0  # Avoid division by zero

        self.cell_center = centers
        self.cell_volume = volumes

    @ti.kernel
    def _corrective_terms_1D(
        self,
        volume_corrections: ti.types.ndarray(dtype=ti_float, element_dim=0),
        ot_hessian: ti.types.ndarray(dtype=ti_float, element_dim=0),
    ) -> ti_float:
        """Computes corrective terms to go from a 0D integral to a 1D integral.

        Parameters
        ----------
        volume_corrections : (n_particles,) ti ndarray
            The volume corrections for the cells.
        ot_hessian : (n_particles, n_particles) ti ndarray
            The Hessian of the dual semi-discrete optimal transport problem.
            Since the dual problem is concave, the Hessian is negative.

        Returns
        -------
        float
            The corrective term for the integral of the min-cost function.
        """
        out = ti_float(0.0)
        for X in ti.grouped(self.pixels):
            for dpos in ti.static(self.segment_connectivity):
                Y = X + ti.cast(dpos, dtype=ti.i32)
                Y = self.wrap_indices(Y)

                i = self.pixels[X]  # Index of the best cell at pixel X
                j = self.pixels[Y]  # Index of the best cell at pixel Y

                # Are we on the boundary between two cells?
                # Exclude the mask, and disable a ruff tip that doesn't fit with taichi
                if i != j and j != MASK_LABEL and i != MASK_LABEL:
                    # Compute the costs of the two cells at the two pixels
                    # By definition of i, Mx >= mx
                    Mx = self._cost_offset(j, X)
                    mx = self._cost_offset(i, X)
                    hx = Mx - mx  # >= 0

                    # By definition of j, My >= my
                    My = self._cost_offset(i, Y)
                    my = self._cost_offset(j, Y)
                    hy = My - my  # >= 0

                    if hx > 0 or hy > 0:  # avoid divisions by 0
                        H = 1 / (hx + hy)
                        out += 0.5 * H * hx * hy

                        if i != AMBIENT_SPACE_LABEL:
                            volume_corrections[i] += 0.5 * H * (hx - hy)
                        if j != AMBIENT_SPACE_LABEL:
                            volume_corrections[j] += 0.5 * H * (hy - hx)

                        # TODO: use a sparse Hessian matrix instead of a dense one
                        if i != AMBIENT_SPACE_LABEL:
                            ot_hessian[i, i] -= H
                        if j != AMBIENT_SPACE_LABEL:
                            ot_hessian[j, j] -= H
                        if (
                            i != AMBIENT_SPACE_LABEL
                            and j != AMBIENT_SPACE_LABEL
                        ):
                            ot_hessian[i, j] += H
                            ot_hessian[j, i] += H

        return out

    @ti.kernel
    def _apply_mask(
        self,
        mask: ti.types.ndarray(dtype=ti.i32, element_dim=0),
    ):
        """Sets to MASK_LABEL the pixel labels where the mask is zero."""
        for X in ti.grouped(self.pixels):
            self.pixels[X] = ti.select(mask[X], self.pixels[X], MASK_LABEL)

    def compute_labels(
        self,
        init_step=None,
        n_steps=None,
        method: Literal["bruteforce", "JFA"] = "bruteforce",
    ):
        """Computes the Laguerre cells associated to the particles.

        This function updates the following attributes:

        - the pixel grid with the index of the best cell at each pixel,
        - the cell centers and volumes,
        - the integral of the min-cost function over the domain,
        - the Hessian of the semi-discrete optimal transport problem.

        Parameters
        ----------
        init_step : int, optional
            The initial step size for the Jump Flooding Algorithm.
            By default, the step size is set to the maximum of the domain shape.
            Users may indicate a smaller step size to speed up the algorithm:
            to ensure correctness, and assuming that seeds stay inside the cells,
            init_step should be larger than the expected diameter of the largest cell.
        n_steps : int, optional
            The number of steps to perform in the Jump Flooding Algorithm.
            By default, the number of steps is set to log2(init_step).
        method : str, optional
            The method to use to compute the cells. Possible values are "bruteforce"
            and "JFA" (Jump Flooding Algorithm).
        """
        print("X", end="")

        # Load the cell potentials from the torch tensor to the Taichi field
        self.taichi_seed_potential.from_torch(self.seed_potential)

        # We expect that the exact bruteforce method will be faster
        # when the number of cells is small (< 1000 ?)
        if method == "bruteforce":
            self._compute_labels_bruteforce()

        # The Jump Flooding Algorithm is more efficient for a large number of cells,
        # especially if we use a power-of-two domain and a small-ish init_step like
        # 16 or 32.
        elif method == "JFA":
            # Create a list of steps that looks like
            # [1, N/2, N/4, N/8, ... 1]
            steps = [1]  # Start with a step of 1, aka. "1+JFA"
            if init_step is None:
                init_step = int(np.ceil(max(self.shape) // 2))

            n = init_step
            while n > 0:
                steps.append(n)
                n = int(np.ceil(n // 2))

            # Limit the number of steps if required, typically for debugging
            if n_steps is not None:
                steps = steps[:n_steps]

            # Initialize the pixel grid with the seed positions
            self._init_sites()

            # Run the Jump Flooding Algorithm
            for step in steps:
                self._jfa_step(step)
        else:
            msg = f"Unknown method {method}."
            raise ValueError(msg)

        # Set pixel labels to -1 wherever domain == 0
        self._apply_mask(self.domain)

        # Recompute self.cell_center and self.cell_volume
        self._cell_centers_volumes()

        # Estimation of the integral of the best cost function over the domain
        # and volumes if the uniform measure on the domain is approximated
        # by a sum of Dirac masses at the pixel centers,
        # i.e. elements of dimension 0.
        self.integral_cost_raw = (self.pixel_cost_raw * self.domain).sum()

        # TODO: use a sparse Hessian matrix instead of a dense ones
        ot_hessian = torch.zeros(
            (self.n_particles, self.n_particles),
            device=self.device,
        )

        if self.integral_dimension == 0:
            self.ot_hessian = ot_hessian

        elif self.integral_dimension == 1:
            # If dimension == 1, we use 1D elements that correspond to the segments
            # between adjacent pixels, along the axes of the 1D, 2D or 3D domain.
            # Assuming a piecewise linear model for the cost function,
            # direct computations show that we must add corrective terms to the
            # cost integral along the boundaries of the cell,
            # and correct the cell volumes as well.
            volume_corrections = torch.zeros_like(self.cell_volume)

            # Compute the extra terms for the cost integral and the cell volumes
            # that correspond to a switch from a piecewise constant model on
            # pixel grids to a piecewise linear model on the segments between pixels.
            cost_correction = self._corrective_terms_1D(
                volume_corrections, ot_hessian
            )

            assert volume_corrections.shape == (self.n_particles,)
            assert self.seed_potential.shape == (self.n_particles,)
            cost_correction += (volume_corrections * self.seed_potential).sum()

            # N.B.: self._corrective_terms_1D makes self.dim passes over the pixels,
            # so don't forget to divide by self.dim to get the correct formulas.
            self.cell_volume += volume_corrections / self.dim
            self.integral_cost_raw += cost_correction / self.dim
            # The potential for the ambient space is fixed to zero,
            # so we remove the first row and column of the Hessian of the dual OT cost.
            self.ot_hessian = ot_hessian / self.dim

        # The code above assumed that pixels had a volume of 1:
        # don't forget to multiply by the pixel volume to get the correct formulas.
        self.integral_cost_raw *= self.pixel_volume
        self.cell_volume *= self.pixel_volume
        self.ot_hessian *= self.pixel_volume

    @property
    def volume_error(self):
        """Returns the errors between the cell volumes and the target volumes."""
        return self.cell_volume - self.volume

    @property
    def relative_volume_error(self):
        """Returns the relative errors between the cell volumes and the target volumes."""
        return self.volume_error / self.volume

    @typecheck
    def compute_dual_loss(self, barrier=None):
        """Updates the dual loss, gradient and Hessian of the semi-discrete OT problem.

        This function computes the concave dual objective of the semi-discrete optimal
        transport problem. It populates the following attributes:

        - dual_loss: the value of the dual objective
        - dual_grad: the gradient of the dual objective
        - dual_hessian: the Hessian of the dual objective

        Parameters
        ----------
        barrier : float, optional
            Optional barrier parameter that promotes positivity for the dual potentials.
        """

        assert self.seed_potential.shape == (self.n_particles,)

        # Compute the Laguerre cells, i.e. assign a label i(x) to each pixel x
        # N.B.: this also updates self.taichi_seed_potential
        self.compute_labels()

        # The gradient of the loss function is the difference between the current
        # cell volumes and the target volumes.
        concave_grad = self.volume - self.cell_volume
        assert concave_grad.shape == (self.n_particles,)

        # Compute the concave, dual objective of the semi-discrete optimal transport
        # problem (which is actually discrete-discrete here, since we use pixels
        # instead of proper Voronoi-Laguerre cells)
        if False:
            # First term: \sum_i v_i * f_i
            # (without the volume of the ambient space, that is associated to
            # a potential of zero)
            concave_loss = (self.volume * particle_potentials).sum()
            # Second term: integral of the min-cost function over the domain.
            # Recall that domain is a binary mask, and
            # self.pixel_costs[x] = min_i ( c_i(x) - seed_potentials[i] )
            concave_loss = concave_loss + self.cost_integral
        else:
            # For the sake of numerical stability, we use the following equivalent
            # formula that decomposes the cost function into two terms:
            # First, sum_i f_i * (v_i - # of pixels in cell i)
            concave_loss = (concave_grad * self.seed_potential).sum()
            # Second, sum_x c_{i(x)}(x)
            concave_loss = concave_loss + self.integral_cost_raw

        concave_hessian = self.ot_hessian
        assert concave_hessian.shape == (self.n_particles, self.n_particles)

        # By convention, we divide the dual cost by the volume of the domain
        # to get a meaningful value that is independent of the domain shape.
        concave_loss = concave_loss / self.domain_volume
        concave_grad = concave_grad / self.domain_volume
        concave_hessian = concave_hessian / self.domain_volume

        if barrier is not None:
            # Add a barrier to prevent the potentials from becoming negative,
            # which would correspond to a negative volume.
            # This prevents L-BFGS from going too far in the wrong direction.
            # We use a simple quadratic barrier, but other choices are possible.
            # The barrier is only active when the potentials are negative.
            # We use a small value to avoid numerical issues.
            barrier_loss = 0.5 * (self.seed_potential.clamp(max=0) ** 2).sum()
            concave_loss -= barrier * barrier_loss

            barrier_gradient = self.seed_potential.clamp(max=0)
            concave_grad -= barrier * barrier_gradient

            # TODO: use a sparse Hessian matrix instead of a dense one
            concave_hessian -= torch.diag(
                barrier * (self.seed_potential < 0).float()
            )

        self.dual_loss = concave_loss
        self.dual_grad = concave_grad
        self.dual_hessian = concave_hessian

    @typecheck
    def fit_cells(
        self,
        max_iter=10,
        barrier=None,
        warm_start=True,
        stopping_criterion: Literal[
            "average error", "max error"
        ] = "max error",
        rtol=1e-2,
        atol=0,
        method: Literal["L-BFGS-B", "Newton"] = "Newton",
        verbose=False,
    ):
        """Solves the semi-discrete optimal transport problem with volume constraints.

        This method adjusts the potentials of the particles to match the target volumes.
        To this end, it maximizes the concave dual objective of a semi-discrete
        optimal transport problem. Unlike genuine semi-discrete OT solvers
        (as studied by Quentin Merigot, Bruno Levy, Fernando De Goes, etc.)
        that compute Voronoi-Laguerre cells as polygons, we discretize the domain with
        pixels. This allows us to handle arbitrary cost functions, and thus arbitrary
        cell shapes and deformabilities.

        For reference, see:

        - "Anisotropic power diagrams for polycrystal modelling: efficient generation of
           curved grains via optimal transport", Maciej Buze, Jean Feydy, Steven M. Roper,
           Karo Sedighiani, David P. Bourne (2024), https://arxiv.org/abs/2403.03571

        - "An optimal transport model for dynamical shapes, collective motion and cellular
           aggregates", Antoine Diez, Jean Feydy (2024), https://arxiv.org/abs/2402.17086

        Parameters
        ----------
        max_iter : int, optional
            The maximum number of iterations.
        barrier : float, optional
            Optional barrier parameter for the Newton method,
            which promotes positivity for the dual potentials.
        warm_start : bool, optional
            If True, uses the current potentials as a starting point.
            Otherwise, starts from zero potentials.
        stopping_criterion : str, optional
            The stopping criterion for the optimization. Possible values are
            "average error" and "max error".
            If "average error", the stopping criterion is the average relative error on
            the volumes.
            If "max error", the stopping criterion is the maximum relative error on the
            volumes.
        rtol : float, optional
            The relative tolerance for the stopping criterion.
        atol : float, optional
            The absolute tolerance for the stopping criterion.
        method : str, optional
            The optimization method to use. Possible values are "L-BFGS-B" and "Newton".
        verbose : bool, optional
            If True, we print information to monitor convergence.
        """

        # Initial the dual potentials for each seed, i.e. each cell except
        # cell 0 that corresponds to the ambient space.
        # These dual potentials (denoted by "f_i" in our papers)
        # act on the cost functions as negative offsets, i.e.
        # they turn c_i(x) into c_i(x) - f_i.
        if not warm_start:
            # Reset to start from zero potentials
            self.seed_potential = torch.zeros(
                self.n_particles, device=self.device
            )
            # The dual OT problem is concave, so the starting point
            # should not have a significant impact on the final result
            # (assuming that different seeds correspond to distinct cost functions).

        if self.volume is None:
            msg = "The '.volume' attribute must be set to specify target volumes."
            raise ValueError(msg)

        # Check that the problem is feasible
        sum_target_volume = self.volume.sum().item()
        if sum_target_volume > self.domain_volume:
            msg = (
                f"The total target volume of the cells ({sum_target_volume})"
                + f"exceeds the available volume ({self.domain_volume})."
            )
            raise ValueError(msg)

        # There is no need to require grads, since we will compute the gradient manually:
        # seed_potentials.requires_grad = True

        if method == "Newton":
            # Use Newton's method with an explicit Hessian as in papers by
            # Quentin Merigot, Bruno Levy, etc.
            # This only works when integral_dimension == 1.
            if self.integral_dimension != 1:
                msg = "Newton's method is only available when self.integral_dimension == 1."
                raise ValueError(msg)

            if False:
                optimizer = torch.optim.SGD([seed_potentials], lr=1)
            else:
                # Small hack: use the LBFGS optimizer with max_iter = 1
                # to get access to a good line search algorithm.
                optimizer = torch.optim.LBFGS(
                    [self.seed_potential],
                    lr=1,
                    line_search_fn="strong_wolfe",
                    tolerance_grad=(atol + rtol * self.volume.min())
                    / self.domain_volume,
                    tolerance_change=1e-20,
                    max_iter=1,
                )

            def step(closure, current_value):
                # grad of the convex minimization problem
                grad = -self.dual_grad
                hessian = -self.dual_hessian  # hessian is >= 0
                # Make it > 0 to avoid numerical issues
                hessian = hessian + 1e-6 * torch.eye(
                    self.n_particles, device=self.device
                )
                # We solve the linear system Hessian * delta = -grad
                # TODO: using a sparse Hessian matrix and an algebraic multigrid solver
                direction = torch.linalg.solve(hessian, -grad)

                """
                direction_error = hessian @ direction + grad
                print("")
                print("Volume error:", self.volume_error)
                print("-Gradient:", -grad)
                print("Direction:", direction)
                print("Newton error:", direction_error.abs().max())
                """
                self.descent_direction = direction

                if True:
                    # Implement line search by hand
                    t = 1
                    old_potential = self.seed_potential.clone()
                    current_error = self.volume_error.abs().max()
                    while True:
                        self.seed_potential = old_potential + t * direction
                        new_value = closure()

                        if False:
                            if new_value < current_value:
                                break
                        elif self.volume_error.abs().max() < current_error:
                            print("Breaking at t=", t)
                            break
                        t /= 2
                        if t < 1e-8:
                            break
                else:
                    self.seed_potential.grad = -direction
                    optimizer.step(closure=closure)
                    new_value = closure()

                if verbose:
                    print(":", end="", flush=True)
                return new_value

            n_outer_iterations = max_iter

        elif method == "L-BFGS-B":
            # We use LBFGS here as it is both simple and versatile.
            # The tolerance on the gradient corresponds to a stopping
            # criterion for the inner loop of LBFGS.
            # This threshold is applied to the maximum of the absolute values of the gradient
            # coordinates: we choose a value that corresponds to a conservative relative error
            optimizer = torch.optim.LBFGS(
                [self.seed_potential],
                lr=1,
                line_search_fn="strong_wolfe",
                tolerance_grad=(atol + rtol * self.volume.min())
                / (10 * self.domain_volume),
                tolerance_change=1e-20,
                max_iter=max_iter,
            )

            def step(closure, _):
                optimizer.step(closure=closure)
                cl = closure()
                self.descent_direction = -self.seed_potential.grad
                return cl

            n_outer_iterations = 10

        def closure():
            # Zero the gradients on the dual seed_potentials
            optimizer.zero_grad()
            # If NaN, throw an error
            if torch.isnan(self.seed_potential).any():
                msg = "NaN detected in seed potentials."
                raise ValueError(msg)

            # Compute the dual OT loss, gradient and Hessian
            self.compute_dual_loss(barrier=barrier)
            # Since the dual problem is concave, we minimize -dual_loss
            convex_loss = -self.dual_loss
            self.seed_potential.grad = -self.dual_grad

            if verbose:
                print(".", end="", flush=True)

            return convex_loss

        current_value = closure()

        # Actual optimization loop
        if verbose:
            print(
                f"Dual cost: {-current_value.item():.15e} ", end="", flush=True
            )

        converged = False
        for outer_it in range(n_outer_iterations):
            current_value = step(closure, current_value)

            if verbose:
                lo = current_value.item()
                print(f"it{outer_it}={-lo:.15e}", end="", flush=True)

            error = self.volume_error.abs()
            threshold = atol + rtol * self.volume.abs()

            if stopping_criterion == "average error":
                stop = error.mean() < threshold.mean()
            elif stopping_criterion == "max error":
                stop = (error < threshold).all()

            rel_error = self.relative_volume_error.abs()
            if verbose:
                print(
                    f" -> {100 * rel_error.max():.2f}% max error, {100 * rel_error.mean():.2f}% mean error"
                )

            if stop:
                converged = True
                break

        if not stop:
            # Throw a warning if the optimization did not converge
            print("Warning: optimization did not converge.")

        # To be sure, we recompute the cells and loss with the final seed potentials
        self.compute_dual_loss(barrier=barrier)
        return converged

    @property
    @typecheck
    def pixel_labels(self) -> Int32Tensor:
        """Returns a canvas (same shape as domain) with the index of the cell at each pixel."""
        labels = self.pixels.to_torch()
        assert labels.shape == self.shape
        assert labels.dtype == torch.int32
        assert labels.device == self.device
        return labels

    @typecheck
    def pixel_colors(
        self,
        *,
        particle_colors: np.ndarray | None = None,
        color_map: str | Callable = "bwr",
        line_width: int = 0,
        line_color: np.ndarray | None = None,
        background_color: np.ndarray | None = None,
        mask_color: np.ndarray | None = None,
        blur_radius: int | float = 0,
        cmin=-100,
        cmax=100,
    ) -> np.ndarray:
        """Returns a canvas with the colors of the cells at each pixel.

        Parameters
        ----------
        TODO

        Returns
        -------
        torch tensor
            A canvas with the colors of the cells at each pixel.
        """

        if particle_colors is None:
            particle_colors = np.linspace(0, 1, self.n_particles)

        if len(np.array(particle_colors).shape) == 1:
            cmap = (
                color_map if callable(color_map) else mpl.colormaps[color_map]
            )
            self._scalarmappable = mpl.cm.ScalarMappable(
                cmap=cmap,
                norm=mpl.colors.Normalize(vmin=cmin, vmax=cmax),
            )
            # Colors are RGBA values in [0, 1]
            particle_colors = np.array(particle_colors, dtype=np.float64)
            # particle_colors = particle_colors - particle_colors.min()
            # particle_colors = particle_colors / max(
            #    1e-8, particle_colors.max()
            # )
            particle_colors = self._scalarmappable.to_rgba(particle_colors)

        particle_colors = mpl.colors.to_rgba_array(particle_colors)
        assert particle_colors.shape == (self.n_particles, 4)

        # By default, the background is white and the mask is black
        if background_color is None:
            if self.dim == 3:
                background_color = np.array([0.0, 0.0, 0.0, 0.0])
            else:
                background_color = np.array([1.0, 1.0, 1.0, 1.0])
        if mask_color is None:
            if self.dim == 3:
                mask_color = np.array([0.0, 0.0, 0.0, 0.0])
            else:
                mask_color = np.array([0.0, 0.0, 0.0, 1.0])

        # Our convention is that self.pixel_labels:
        # - is -2 for the mask
        # - is -1 for the ambient space
        # - is in [0, n_particles-1] for the particles
        cell_colors = np.concatenate(
            [
                particle_colors,
                mpl.colors.to_rgba_array(mask_color),
                mpl.colors.to_rgba_array(background_color),
            ],
            axis=0,
        )
        cell_colors = torch.from_numpy(cell_colors).to(self.device)

        canvas = cell_colors[self.pixel_labels]
        assert canvas.shape == (*self.shape, 4)

        if line_width > 0:
            if line_color is None:
                line_color = np.array([0.0, 0.0, 0.0, 1.0])
            line_color = mpl.colors.to_rgba_array(line_color)[0]
            assert line_color.shape == (4,)
            line_color = torch.from_numpy(line_color).to(self.device)
            canvas[self.contours(linewidth=line_width) == 1] = line_color

        canvas = canvas.cpu().numpy()

        if blur_radius > 0:
            import scipy

            # Gaussian smoothing of the colors
            canvas = scipy.ndimage.gaussian_filter(
                canvas,
                sigma=blur_radius,
                axes=tuple(range(self.dim)),
                mode="constant",
            )
            canvas[..., :3] /= 1e-8 + canvas[..., 3:]

        assert canvas.shape == (*self.shape, 4)
        return canvas

    @ti.kernel
    def _pixel_cost_raw(
        self, canvas: ti.types.ndarray(dtype=ti_float, element_dim=0)
    ):
        for ind in ti.grouped(canvas):
            label = self.pixels[ind]
            if label != MASK_LABEL:
                canvas[ind] = self._cost_raw(label, ind)
            else:
                canvas[ind] = 0.0

    @property
    @typecheck
    def pixel_cost_raw(self) -> FloatTensor:
        """Returns a canvas with the cost of the cells at each pixel.

        At each pixel x, we evaluate the cost function c_i(x) of the best cell i(x).

        Returns
        -------
        torch tensor
            A canvas with the cost of the cells at each pixel.
        """
        canvas = torch.zeros(self.shape, device=self.device)
        self._pixel_cost_raw(canvas)
        return canvas

    @ti.kernel
    def _contours(
        self,
        canvas: ti.types.ndarray(dtype=ti.i32, element_dim=0),
        linewidth: ti.i32,
    ):
        for X in ti.grouped(self.pixels):
            ref = self.pixels[X]
            canvas[X] = 0
            ti.loop_config(
                serialize=True
            )  # Unroll the loop over the neighborhood
            for DX in ti.grouped(
                ti.ndrange(*(((-linewidth, linewidth + 1),) * self.dim))
            ):
                X2 = X + ti.cast(DX, ti.i32)
                X2 = tm.clamp(X2, 0, self.shape - self.i32vector(1, 1))
                # The pixel that gets contoured is the one with the smallest index,
                # typically the ambient space (index -1) but we exclude the mask (index -2)
                if self.pixels[X2] > ref and ref != MASK_LABEL:
                    canvas[X] = 1

    @typecheck
    def contours(self, linewidth=1) -> Int32Tensor:
        """Returns a binary mask that highlights the contours of the cells.

        Parameters
        ----------
        linewidth : int, optional
            The width of the contours.

        Returns
        -------
        torch int32 tensor
            A canvas with the contours of the cells.
        """

        canvas = torch.zeros(self.shape, dtype=torch.int32)
        self._contours(canvas, linewidth=linewidth)
        return canvas

    def display(self, *, ax, title, **kwargs):
        artists = []
        artists.append(
            ax.text(0.5, 1.05, title, transform=ax.transAxes, ha="center")
        )
        artists.append(
            ax.imshow(
                self.pixel_colors(**kwargs).transpose(1, 0, 2),
                origin="lower",
                interpolation="spline36",
            )
        )

        if hasattr(self, "position"):
            artists.append(
                ax.scatter(
                    self.position[:, 0],
                    self.position[:, 1],
                    c="r",
                    s=9,
                )
            )

            if hasattr(self, "cell_center"):
                # Display segments between the positions and the cell centers
                # Create a list of segments to display
                segments = []
                for i in range(self.n_particles):
                    segments.append([self.position[i], self.cell_center[i]])
                # Create a line collection
                lc = mpl.collections.LineCollection(
                    segments, color="r", linewidth=1
                )
                artists.append(ax.add_collection(lc))

        return artists
