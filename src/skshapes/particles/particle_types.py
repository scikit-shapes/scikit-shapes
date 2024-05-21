import taichi as ti

ti_float = ti.f32


@ti.dataclass
class PowerCell1D:
    """This class implements an isotropic power cell."""

    # Custom parameters for the cell
    position: ti.types.vector(1, ti_float)
    power: ti_float

    @ti.func
    def cost(self, pixel):
        dist = (self.position.x - pixel.x) ** 2
        return dist ** (self.power / 2)


@ti.dataclass
class PowerCell2D:
    """This class implements an isotropic power cell."""

    # Custom parameters for the cell
    position: ti.types.vector(2, ti_float)
    power: ti_float

    @ti.func
    def cost(self, pixel):
        dist = (self.position.x - pixel.x) ** 2 + (
            self.position.y - pixel.y
        ) ** 2
        return dist ** (self.power / 2)


@ti.dataclass
class AnisotropicPowerCell2D:
    """This class implements an anisotropic power cell."""

    # Custom parameters for the cell
    position: ti.types.vector(2, ti_float)
    precision_matrix: ti.types.matrix(2, 2, ti_float)
    power: ti_float

    @ti.func
    def cost(self, pixel):
        # dist = (self.position.x - pixel.x) ** 2 + (self.position.y - pixel.y) ** 2
        dist = (self.position - pixel).dot(
            self.precision_matrix @ (self.position - pixel)
        )
        return dist ** (self.power / 2)
