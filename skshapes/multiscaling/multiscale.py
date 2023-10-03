"""This module contains the multiscale class."""
# TODO : how to check the type of shapes avoiding circular import ?
# TODO : add_points_data interface ? Deal with existing data ?
# TODO : landmark interface ?

from ..types import (
    typecheck,
    Optional,
    FloatSequence,
    IntSequence,
    int_dtype,
    float_dtype,
    Int1dTensor,
    NumericalTensor,
    Number,
    polydata_type,
)
from typing import List
from ..utils import scatter
import torch


class Multiscale:
    """A class to handle multiscale data."""

    @typecheck
    def __init__(
        self,
        shape: polydata_type,
        *,
        ratios: Optional[FloatSequence] = None,
        n_points: Optional[IntSequence] = None
    ) -> None:
        """Initialize a multiscale object from a reference shape.

        A dictionnary of shapes is created, with the reference shape at ratio 1.
        The other ratios are created by decimation.

        Args:
            shape (Shape): the shape to be multiscaled (will be reffered to as ratio 1)
            ratios (Sequence of floats): the ratios at which the shape is multiscaled
            n_points (Sequence of ints): the (approximative) number of points at each ratio
        """

        if ratios is None and n_points is None:
            raise ValueError("You must specify either ratios or n_points")

        if ratios is not None:
            assert n_points is None, "You cannot specify both ratios and n_points"
            assert 0 < max(ratios) <= 1, "ratios must be between 0 and 1"
            assert 0 < min(ratios) <= 1, "ratios must be between 0 and 1"

        if n_points is not None:
            assert (
                0 < max(n_points) <= shape.n_points
            ), "n_points must be between 0 and the number of points in the mesh"
            ratios = [1 - (npts / shape.n_points) for npts in n_points]

        self.shapes = dict()
        self.mappings_from_origin = dict()

        self.shapes[1] = shape

        from ..data import PolyData

        if type(shape) == PolyData and shape.is_triangle_mesh():
            from ..decimation import Decimation

            self.decimation_module = Decimation(target_reduction=float(1 - min(ratios)))
            self.decimation_module.fit(shape)

            for ratio in ratios:
                if ratio != 1:
                    self.add_ratio(float(ratio))

        else:
            raise NotImplementedError("Only triangle meshes are supported for now")

    @typecheck
    def add_ratio(self, ratio: float) -> None:
        """Add a ratio to the multiscale object.

        Args:
            ratio (float): ratio to be added
        """
        assert 0 < ratio < 1, "ratio must be between 0 and 1"
        if ratio in self.shapes.keys():
            # ratio already exists, do nothing
            pass
        else:
            self.shapes[ratio] = self.decimation_module.transform(
                self.shapes[1], target_reduction=1 - ratio
            )
            self.mappings_from_origin[ratio] = self.decimation_module._indice_mapping

    @typecheck
    def at(self, ratio: Number):
        """Return the shape at a given ratio.

        If the ratio does not exist, the closest ratio is returned.

        Args:
            ratio (Number): the ratio at which the shape is returned

        Returns:
            Shape: the shape at the given ratio
        """
        available_ratios = list(self.shapes.keys())
        # find closest ratio
        closest_ratio = min(available_ratios, key=lambda x: abs(x - ratio))
        return self.shapes[closest_ratio]

    @typecheck
    def indice_mapping(self, high_res: Number, low_res: Number) -> Int1dTensor:
        """Return the indice mapping from high to low resolution.

        The indice mapping is a 1d tensor of integers of length equal to the number of
        points at the high resolution ratio. Each element of the tensor is the index of
        the corresponding point at the low resolution ratio.

        Args:
            high_res (Number): the ratio of the high resolution shape
            low_res (Number): the ratio of the low resolution shape

        Returns:
            Int1dTensor: the indice mapping from high to low resolution
        """

        assert high_res >= low_res, "high_res must be greater than low_res"
        assert 0 < low_res <= 1, "low_res must be between 0 and 1"
        assert 0 < high_res <= 1, "high_res must be between 0 and 1"

        available_ratios = list(self.shapes.keys())
        high_res = min(available_ratios, key=lambda x: abs(x - high_res))
        low_res = min(available_ratios, key=lambda x: abs(x - low_res))

        if high_res == low_res:
            return torch.arange(self.shapes[high_res].n_points)

        elif high_res == 1:
            return self.mappings_from_origin[low_res]

        else:
            tmp = self.mappings_from_origin[high_res]
            tmp = scatter(src=torch.arange(len(tmp)), index=tmp, reduce="min")
            return self.mappings_from_origin[low_res][tmp]

    @typecheck
    def signal_from_high_to_low_res(
        self,
        signal: NumericalTensor,
        *,
        high_res: Number,
        low_res: Number,
        reduce="mean"
    ) -> NumericalTensor:
        """Propagate a signal from a resolution to a lower resolution.

        This operation is a scatter operation, with the indice mapping as index.
        A reduce operation is applied to the scatter. The available reduce operations
        are "sum", "min", "max", "mean".

        Args:
            signal (NumericalTensor): the signal to be propagated
            high_res (Number): the ratio of the high resolution shape
            low_res (Number): the ratio of the low resolution shape
            reduce (str, optional): the reduce option. Defaults to "sum".

        Returns:
            NumericalTensor: the signal at the low resolution ratio
        """
        assert (
            signal.shape[0] == self.at(high_res).n_points
        ), "signal must have the same number of points as the origin ratio"
        assert high_res >= low_res, "high_res must be greater than low_res"

        return scatter(
            src=signal,
            index=self.indice_mapping(high_res=high_res, low_res=low_res),
            reduce=reduce,
        )

    @typecheck
    def signal_from_low_to_high_res(
        self,
        signal: NumericalTensor,
        *,
        low_res: Number,
        high_res: Number,
        smoothing="constant"
    ) -> NumericalTensor:
        """Propagate a signal from a resolution to a higher resolution.

        This operation requires a smoothing operation. The available smoothing
        operations are :
        - "constant" : the signal is repeated at each point of the high resolution shape

        Args:
            signal (NumericalTensor): the signal to be propagated
            low_res (Number): the ratio of the low resolution shape
            high_res (Number): the ratio of the high resolution shape
            smoothing (str, optional): the smoothing option. Defaults to "constant".

        Returns:
            NumericalTensor: the signal at the high resolution ratio
        """
        assert (
            signal.shape[0] == self.at(low_res).n_points
        ), "signal must have the same number of points as the origin ratio"
        assert low_res <= high_res, "high_res must be smaller than low_res"

        if smoothing == "constant":
            return torch.index_select(
                signal,
                dim=0,
                index=self.indice_mapping(high_res=high_res, low_res=low_res),
            )

        else:
            raise NotImplementedError("Only constant smoothing is supported for now")

    @typecheck
    def signal_convolution(self, signal, signal_ratio, target_ratio, **kwargs):
        source = self.at(signal_ratio)
        target = self.at(target_ratio)

        assert (
            signal.shape[0] == source.n_points
        ), "signal must have the same number of points as the shape at signal ratio"

        # Store the kwargs to pass to the point convolution
        point_convolution_args = source._point_convolution.__annotations__.keys()
        convolutions_kwargs = dict()
        for arg in kwargs:
            if arg in point_convolution_args:
                convolutions_kwargs[arg] = kwargs[arg]

        C = source.point_convolution(target=target, **convolutions_kwargs)

        return C @ signal

    @property
    @typecheck
    def ratios(self) -> List[Number]:
        """Return the (sorted) available ratios."""
        tmp = list(self.shapes.keys())
        tmp.sort()
        return tmp

    @typecheck
    def add_point_data(
        self, signal, *, name, at=1, reduce="mean", smoothing="constant"
    ):
        pass


@typecheck
def edge_smoothing(
    signal: NumericalTensor,
    shape: polydata_type,
    weight_by_length: bool = False,
) -> NumericalTensor:
    assert signal.shape[0] == shape.n_points
    signal_device = signal.device
    signal = signal.to(shape.device)

    K = shape.mesh_convolution(weight_by_length=weight_by_length)

    return (K @ signal).to(signal_device)


@typecheck
def vector_heat_smooting(
    signal: NumericalTensor, shape: polydata_type
) -> NumericalTensor:
    try:
        import potpourri3d as pp3d
    except:
        raise ImportError("Please install potpourri3d to use vector heat smoothing")

    if shape.is_triangle_mesh():
        V, F = shape.points.cpu().numpy(), shape.triangles.cpu().numpy()
        if F.shape[0] == 3:
            F = F.T

        try:
            solver = pp3d.MeshVectorHeatSolver(V, F)
        except:
            solver = pp3d.PointCloudHeatSolver(V)

    else:
        solver = pp3d.PointCloudHeatSolver(shape.points.cpu().numpy())

    ext = solver.extend_vector(torch.arange(len(signal)).numpy(), signal.cpu().numpy())
    return torch.from_numpy(ext)
