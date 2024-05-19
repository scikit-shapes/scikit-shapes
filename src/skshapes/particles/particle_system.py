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


@ti.data_oriented
class ParticleSystem:
    """A collection of particles that may interact with each other.

    Parameters
    ----------
    n_particles : int
        The number of particles in the system.
    particle_type : taichi type
        The type of the particles in the system.
    domain : torch tensor
        A binary mask that defines the domain of the particles.
    blocksize : int, optional
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
        n_particles: int,
        particle_type: Particle,
        domain: IntTensor,
        blocksize: int | None = None,
        integral_dimension: Literal[0, 1] = 0,
    ) -> None:

        # We let users directly edit cell attributes by overloading
        # self.__getattr__, self.__setattr__, and checking if the attributes
        # correspond to a property of self.particle_type.
        # As a consequence, and before doing anything, we must initialize
        # self.particle_type with the low-level "self.__dict__" API.
        self.__dict__["particle_type"] = particle_type

        self.integral_dimension = integral_dimension

        self.domain = domain.type(torch.int32)
        # Check that the domain is binary
        if not ((self.domain == 0) | (self.domain == 1)).all():
            msg = "The domain must be a binary mask."
            raise ValueError(msg)

        # By default, pixels are unit cubes.
        self.pixel_volume = 1

        self.shape = self.domain.shape
        self.device = self.domain.device

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

        # Create the collection of particles, with an additional particle
        # for the ambient space.
        self.cells = particle_type.field(shape=(1 + n_particles,))

        # By default, all particles have zero offset and unit scaling.
        # Most users won't touch the scaling, but the offset is used by the semi-discrete
        # optimal transport solver to enforce volume constraints on the Laguerre cells.
        # N.B.: The two lines below actually use the __setattr__ method,
        #       which target the attributes of self.cells[1:].
        self.offset = torch.zeros(n_particles, device=self.device)
        self.scaling = torch.ones(n_particles, device=self.device)

        # Create the grid of pixels, and the intermediate buffer
        # that will be used in the Jump Flooding Algorithm.
        # We use these grids to store the index of the cell that is associated
        # to each pixel.
        self.pixels = ti.field(ti.i32)
        self.pixels_buffer = ti.field(ti.i32)

        if blocksize is None:
            # By default, the pixel grids are just vanilla C-contiguous arrays.
            ti.root.dense(ind, self.shape).place(self.pixels)
            ti.root.dense(ind, self.shape).place(self.pixels_buffer)
        else:
            # But Taichi also lets us encode them as collections of square patches.
            # This can be useful for performance, especially on GPUs.
            coarse = tuple(s // blocksize for s in self.shape)
            block = tuple(blocksize for _ in self.shape)
            # The lines below mean that the pixel grids are now stored as
            # a collection of square patches of size (blocksize, blocksize).
            ti.root.dense(ind, coarse).dense(ind, block).place(self.pixels)
            ti.root.dense(ind, coarse).dense(ind, block).place(
                self.pixels_buffer
            )

    @property
    @typecheck
    def n_cells(self) -> int:
        """Returns the number of cells in the system, including the ambient space.

        Returns
        -------
        int
            The number of cells in the system.
        """
        return self.cells.shape[0]

    @property
    @typecheck
    def n_particles(self) -> int:
        """Returns the number of seeds in the system, excluding the ambient space.

        Returns
        -------
        int
            The number of particles in the system.
        """
        return self.n_cells - 1

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
        """Give direct access to the attributes of self.cells, minus the ambient space."""
        if attr in self.__dict__["particle_type"].__dict__["members"]:
            # Case 1: the attribute is a property of the particle type
            out = getattr(self.cells, attr)  # out is a Taichi field
            # This is the Taichi way to convert to a torch tensor
            out = out.to_torch()
            return out[1:]
        else:
            # Case 2: the attribute belongs to the ParticleSystem itself
            try:
                return self.__dict__[attr]
            except KeyError as e:
                raise AttributeError from e

    def __setattr__(self, attr, value):
        """Give direct access to the attributes of self.cells, minus the ambient space."""
        if attr in self.particle_type.__dict__["members"]:
            # N.B.: All the attributes of the ambient space are set to zero.
            val = torch.cat([torch.zeros_like(value[:1]), value])
            getattr(self.cells, attr).from_torch(val)
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

    @ti.kernel
    def compute_cells_bruteforce(self):
        """For each pixel, computes the index of the cell with the lowest cost.
        This is a brute-force method that is suitable for a small number of cells.
        For very large systems, the Jump Flooding Algorithm is more efficient.
        """

        for X in ti.grouped(self.pixels):
            best_cost = ti_float(1e10)
            best_index = 0  # Ambient space by default

            for i in ti.ndrange(self.cells.shape[0]):
                # N.B.: the cell cost function takes as input a float32 vector
                #       with pixel coordinates
                cost = self.cells[i].cost(ti.cast(X, dtype=ti_float))
                if cost < best_cost:
                    best_cost = cost
                    best_index = i

            self.pixels[X] = best_index

    @ti.kernel
    def init_sites(self):
        """Puts one pixel per seed in the pixel grid to init the Jump Flood Algorithm."""
        # Clear the pixels grid:
        for ind in ti.grouped(self.pixels):
            self.pixels[ind] = 0  # Ambient space by default

        # Place one pixel per seed.
        # Note that we must be careful here to have at least one pixel per cell:
        # otherwise, the JFA algorithm will not work as expected.
        # For this reason, we should not run the seeding in parallel:
        # ti.loop_config(serialize=True)
        for i in self.cells:
            if i > 0:  # Skip the ambient space
                # Cast the position to integers, using floor instead of round
                pos = tm.floor(self.cells[i].position, dtype=ti.i32)
                # Wrap the indices to have periodic boundary conditions
                # This is also useful if the user provided cells that lay outside
                # the domain
                pos = self.wrap_indices(pos)

                if self.pixels[pos] == 0:  # If the pixel is not already taken
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
                            if self.pixels[pos2] == 0:
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
            self.pixels_buffer[ind] = 0  # Set all pixels to ambient space

    @ti.func
    def _copy_buffer(self):
        """Copies the pixel buffer to the main pixel grid."""
        for ind in ti.grouped(self.pixels):
            self.pixels[ind] = self.pixels_buffer[ind]

    @ti.kernel
    def jfa_step(self, step: ti.i32):
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
            best_cost = ti_float(1e10)
            best_index = 0  # Ambient space by default

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
                # N.B.: the cell cost function takes as input a float32 vector
                #       with integer coordinates
                cost = self.cells[i2].cost(ti.cast(X, dtype=ti_float))
                if cost < best_cost:
                    best_cost = cost
                    best_index = i2

            # We must use an intermediate buffer to avoid overwriting the pixels
            # while the main loop is still running
            self.pixels_buffer[X] = best_index

        self._copy_buffer()

    @ti.kernel
    def _cell_centers_volumes(
        self,
        centers: ti.types.ndarray(dtype=ti_float, element_dim=0),
        volumes: ti.types.ndarray(dtype=ti_float, element_dim=0),
    ):
        """Computes in place the barycenter and volume of each cell.

        Parameters
        ----------
        centers : (n_cells, dim) ti ndarray
            The barycenters of the cells.
        volumes : (n_cells,) ti ndarray
            The volumes of the cells.
        """
        for X in ti.grouped(self.pixels):
            i = self.pixels[X]
            if i >= 0:  # Ignore the mask, i == -1
                for d in ti.static(range(self.dim)):
                    centers[i, d] += ti.cast(X[d], ti_float)
                volumes[i] += 1.0

        # Normalize the sum of the pixel coordinates to get the barycenter
        for i in ti.ndrange(self.n_cells):
            for d in ti.static(range(self.dim)):
                if volumes[i] > 0:
                    centers[i, d] /= volumes[i]
                else:
                    centers[i, d] = 0.0

    @ti.kernel
    def _corrective_terms_1D(
        self,
        volume_corrections: ti.types.ndarray(dtype=ti_float, element_dim=0),
        ot_hessian: ti.types.ndarray(dtype=ti_float, element_dim=0),
    ) -> ti_float:
        """Computes corrective terms to go from a 0D integral to a 1D integral.

        Parameters
        ----------
        volume_corrections : (n_cells,) ti ndarray
            The volume corrections for the cells.
        ot_hessian : (n_cells, n_cells) ti ndarray
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
                if i != j and (-1 not in {i, j}):  # Exclude the mask
                    # Compute the costs of the two cells at the two pixels
                    # By definition of i, Mx >= mx
                    Mx = self.cells[j].cost(ti.cast(X, dtype=ti_float))
                    mx = self.cells[i].cost(ti.cast(X, dtype=ti_float))
                    hx = Mx - mx  # >= 0

                    # By definition of j, My >= my
                    My = self.cells[i].cost(ti.cast(Y, dtype=ti_float))
                    my = self.cells[j].cost(ti.cast(Y, dtype=ti_float))
                    hy = My - my  # >= 0

                    if hx > 0 and hy > 0:
                        f = 1 / (hx + hy)
                        out += 0.5 * f * hx * hy
                        volume_corrections[i] += 0.5 * f * (hx - hy)
                        volume_corrections[j] += 0.5 * f * (hy - hx)

                        # TODO: use a sparse Hessian matrix instead of a dense one
                        ot_hessian[i, i] -= f
                        ot_hessian[j, j] -= f
                        ot_hessian[i, j] += f
                        ot_hessian[j, i] += f

        return out

    @ti.kernel
    def _apply_mask(
        self,
        mask: ti.types.ndarray(dtype=ti.i32, element_dim=0),
    ):
        """Sets to -1 the pixel labels where the mask is zero."""
        for X in ti.grouped(self.pixels):
            self.pixels[X] = ti.select(mask[X], self.pixels[X], -1)

    def compute_cells(
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
        # We expect that the exact bruteforce method will be faster
        # when the number of cells is small (< 1000 ?)
        if method == "bruteforce":
            self.compute_cells_bruteforce()

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
            self.init_sites()

            # Run the Jump Flooding Algorithm
            for step in steps:
                self.jfa_step(step)
        else:
            msg = f"Unknown method {method}."
            raise ValueError(msg)

        # Set pixel labels to -1 wherever domain == 0
        self._apply_mask(self.domain)

        # Compute basic information (centers and volumes) about the cells
        self.cell_centers = torch.zeros(
            (self.n_cells, self.dim), device=self.device
        )
        self.cell_volume = torch.zeros(self.n_cells, device=self.device)
        self._cell_centers_volumes(self.cell_centers, self.cell_volume)

        # Estimation of the integral of the min-cost function over the domain
        # and volumes if the uniform measure on the domain is approximated
        # by a sum of Dirac masses at the pixel centers,
        # i.e. elements of dimension 0.
        self.cost_integral = (self.pixel_costs * self.domain).sum()

        # TODO: use a sparse Hessian matrix instead of a dense ones
        ot_hessian = torch.zeros(
            (1 + self.n_particles, 1 + self.n_particles),
            device=self.device,
        )

        if self.integral_dimension == 0:
            self.ot_hessian = ot_hessian[1:, 1:]

        elif self.integral_dimension == 1:
            # If dimension == 1, we use 1D elements that correspond to the segments
            # between adjacent pixels, along the axes of the 1D, 2D or 3D domain.
            # Assuming a piecewise linear model for the cost function,
            # direct computations show that we must add corrective terms to the
            # cost_integral along the boundaries of the cell,
            # and correct the cell volumes as well.
            volume_corrections = torch.zeros_like(self.cell_volume)

            # Compute the extra terms for the cost integral and the cell volumes
            # that correspond to a switch from a piecewise constant model on
            # pixel grids to a piecewise linear model on the segments between pixels.
            cost_correction = self._corrective_terms_1D(
                volume_corrections, ot_hessian
            )

            # N.B.: self._corrective_terms_1D makes self.dim passes over the pixels,
            # so don't forget to divide by self.dim to get the correct formulas.
            self.cell_volume += volume_corrections / self.dim
            self.cost_integral += cost_correction / self.dim
            # The potential for the ambient space is fixed to zero,
            # so we remove the first row and column of the Hessian of the dual OT cost.
            self.ot_hessian = ot_hessian[1:, 1:] / self.dim

        # The code above assumed that pixels had a volume of 1:
        # don't forget to multiply by the pixel volume to get the correct formulas.
        self.cost_integral *= self.pixel_volume
        self.cell_volume *= self.pixel_volume
        self.ot_hessian *= self.pixel_volume

    @property
    def volume_error(self):
        """Returns the errors between the cell volumes and the target volumes."""
        return self.cell_volume[1:] - self.volume

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

        particle_potentials = -self.offset

        # Compute the Laguerre cells, i.e. assign a color to each pixel
        self.compute_cells()

        # Compute the concave, dual objective of the semi-discrete optimal transport
        # problem (which is actually discrete-discrete here, since we use pixels
        # instead of proper Voronoi-Laguerre cells)
        # First term: \sum_i v_i * f_i
        # (without the volume of the ambient space, that is associated to
        # a potential of zero)
        concave_loss = (self.volume * particle_potentials).sum()
        # Second term: integral of the min-cost function over the domain.
        # Recall that domain is a binary mask, and
        # self.pixel_costs[x] = min_i ( c_i(x) - seed_potentials[i] )
        concave_loss = concave_loss + self.cost_integral

        # By convention, we divide the dual cost by the volume of the domain
        # to get a meaningful value that is independent of the domain shape.
        concave_loss = concave_loss / self.domain_volume

        # The gradient of the loss function is the difference between the current
        # cell volumes and the target volumes. We normalize by the number of pixels
        # to get meaningful values and gradients in the optimization.
        # (Keeping things of order 1 is probably a good idea for numerical stability
        # and compatibility with default optimizer settings.)
        concave_grad = (
            self.volume - self.cell_volume[1:]
        ) / self.domain_volume

        concave_hessian = self.ot_hessian / self.domain_volume

        if barrier is not None:
            # Add a barrier to prevent the potentials from becoming negative,
            # which would correspond to a negative volume.
            # This prevents L-BFGS from going too far in the wrong direction.
            # We use a simple quadratic barrier, but other choices are possible.
            # The barrier is only active when the potentials are negative.
            # We use a small value to avoid numerical issues.
            barrier_loss = 0.5 * (particle_potentials.clamp(max=0) ** 2).sum()
            concave_loss -= barrier * barrier_loss

            barrier_gradient = particle_potentials.clamp(max=0)
            concave_grad -= barrier * barrier_gradient

            # TODO: use a sparse Hessian matrix instead of a dense one
            concave_hessian -= torch.diag(
                barrier * (particle_potentials < 0).float()
            )

        self.dual_loss = concave_loss
        self.dual_grad = concave_grad
        self.dual_hessian = concave_hessian

    @property
    @typecheck
    def barycenter(self) -> FloatTensor:
        """Returns the barycenters of the particles."""
        # Cell 0 is the ambient space, so we exclude it
        return self.cell_centers[1:]

    @typecheck
    def volume_fit(
        self,
        max_iter=10,
        barrier=None,
        warm_start=True,
        stopping_criterion: Literal[
            "average error", "max error"
        ] = "average error",
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
        if warm_start:
            # Reuse the current potentials (= -1 * offset_i) as a starting point
            seed_potentials = -self.offset
        else:
            # Just start from zero potentials
            seed_potentials = torch.zeros(self.n_particles, device=self.device)
            # The dual OT problem is concave, so the starting point
            # should not have a significant impact on the final result
            # (assuming that different seeds correspond to distinct cost functions).

        if self.volume is None:
            msg = "The volume of the cells must be set."
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
                    [seed_potentials],
                    lr=1,
                    line_search_fn="strong_wolfe",
                    tolerance_grad=(atol + rtol * self.volume.min())
                    / self.domain_volume,
                    tolerance_change=1e-20,
                    max_iter=1,
                )

            def step(closure, current_value):
                nonlocal seed_potentials

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
                self.descent_direction = direction

                if False:
                    # Implement line search by hand
                    t = 1
                    old_potentials = seed_potentials.clone()
                    while True:
                        seed_potentials = old_potentials + t * direction
                        new_value = closure()
                        if new_value < current_value:
                            break
                        t /= 2
                        if t < 1e-8:
                            break
                else:
                    seed_potentials.grad = -direction
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
                [seed_potentials],
                lr=1,
                line_search_fn="strong_wolfe",
                tolerance_grad=(atol + rtol * self.volume.min())
                / self.domain_volume,
                tolerance_change=1e-20,
                max_iter=max_iter,
            )

            def step(closure, _):
                optimizer.step(closure=closure)
                return closure()

            n_outer_iterations = 3

        def closure():
            # Zero the gradients on the dual seed_potentials
            optimizer.zero_grad()
            # If NaN, throw an error
            if torch.isnan(seed_potentials).any():
                msg = "NaN detected in seed potentials."
                raise ValueError(msg)

            # Replace the cost function c_i(x) (e.g. = |x-x_i|^2)
            # with a biased version c_i(x) - seed_potentials[i]
            self.offset = -seed_potentials
            # Compute the dual OT loss, gradient and Hessian
            self.compute_dual_loss(barrier=barrier)
            # Since the dual problem is concave, we minimize -dual_loss
            convex_loss = -self.dual_loss
            seed_potentials.grad = -self.dual_grad

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

        # To be sure, we recompute the cells with the final seed potentials
        self.offsets = -seed_potentials
        self.compute_cells()
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
                norm=mpl.colors.Normalize(vmin=-100, vmax=100),
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
        # - is -1 for the mask
        # - is 0 for the background
        # - is in [1, n_particles] for the particles
        cell_colors = np.concatenate(
            [
                mpl.colors.to_rgba_array(background_color),
                particle_colors,
                mpl.colors.to_rgba_array(mask_color),
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
    def _pixel_costs(
        self, canvas: ti.types.ndarray(dtype=ti_float, element_dim=0)
    ):
        for ind in ti.grouped(canvas):
            index = self.pixels[ind]
            if index >= 0:
                canvas[ind] = self.cells[index].cost(
                    ti.cast(I, dtype=ti_float)
                )
            else:
                canvas[ind] = 0.0

    @property
    @typecheck
    def pixel_costs(self) -> FloatTensor:
        canvas = torch.zeros(self.shape, device=self.device)
        self._pixel_costs(canvas)
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
                # typically the ambient space (index 0) but we exclude the mask (index -1)
                if self.pixels[X2] > ref and ref >= 0:
                    canvas[X] = 1

    def contours(self, linewidth=1):
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
        if hasattr(self, "barycenter"):
            artists.append(
                ax.scatter(
                    self.barycenter[:, 0],
                    self.barycenter[:, 1],
                    c="g",
                    s=9,
                )
            )
        return artists
