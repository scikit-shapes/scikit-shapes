import numpy as np
import taichi as ti

ti_float = ti.f32


@ti.dataclass
class PowerCell1D:
    """This class implements an isotropic power cell."""

    # Custom parameters for the cell
    position: ti.types.vector(1, ti_float)
    power: ti_float

    inv_scale: ti_float
    factor: ti_float

    @ti.func
    def cost(self, pixel):
        delta = (self.position.x - pixel.x) * self.inv_scale
        sq_dist = delta**2
        return (sq_dist ** (self.power / 2)) * self.factor


@ti.dataclass
class PowerCell2D:
    """This class implements an isotropic power cell."""

    # Custom parameters for the cell
    position: ti.types.vector(2, ti_float)
    power: ti_float

    inv_scale: ti_float
    factor: ti_float

    @ti.func
    def cost(self, pixel):
        delta = (self.position - pixel) * self.inv_scale
        sq_dist = delta.dot(delta)
        return (sq_dist ** (self.power / 2)) * self.factor


@ti.dataclass
class AnisotropicPowerCell2D:
    """This class implements an anisotropic power cell."""

    # Custom parameters for the cell
    position: ti.types.vector(2, ti_float)
    precision_matrix: ti.types.matrix(2, 2, ti_float)
    power: ti_float

    inv_scale: ti_float
    factor: ti_float

    @ti.func
    def cost(self, pixel):
        # dist = (self.position.x - pixel.x) ** 2 + (self.position.y - pixel.y) ** 2
        delta = (self.position - pixel) * self.inv_scale
        sq_dist = delta.dot(self.precision_matrix @ delta)
        return (sq_dist ** (self.power / 2)) * self.factor


class PowerCell:
    def __init__(
        self, *, position, volume, power=2, precision_matrix=None, scaling=None
    ):
        dim = len(position)
        if dim not in {1, 2, 3}:
            msg = "Only 1D, 2D and 3D cells are supported."
            raise ValueError(msg)

        self.position = position
        self.volume = volume
        self.power = power

        if scaling is None:
            # inv_scale is the inverse of r(volume), where r(volume) is the radius
            # of the Euclidean ball of volume equal to volume.
            if dim == 1:
                # r = volume / 2  (length of [-r, r] = 2 * r)
                self.inv_scale = 2 / volume
            elif dim == 2:
                # r = sqrt(volume / pi)  (area of the disk of radius r = pi * r^2)
                self.inv_scale = np.sqrt(np.pi / volume)
            elif dim == 3:
                # r = ((3 * volume) / (4 * pi))^(1/3)
                # (volume of the ball of radius r = 4/3 * pi * r^3)
                self.inv_scale = ((4 * np.pi) / (3 * volume)) ** (1 / 3)

            self.factor = volume
        else:
            self.inv_scale = 1.0
            self.factor = 1.0

        if precision_matrix is None:
            self._taichi_class = PowerCell1D if dim == 1 else PowerCell2D
        else:
            self.precision_matrix = precision_matrix
            self._taichi_class = AnisotropicPowerCell2D
