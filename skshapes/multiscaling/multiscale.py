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
)
from ..utils import scatter
import torch


class Multiscale:
    """A class to handle multiscale data."""

    @typecheck
    def __init__(
        self,
        shape,
        *,
        scales: Optional[FloatSequence] = None,
        n_points: Optional[IntSequence] = None
    ) -> None:
        """Initialize a multiscale object from a reference shape.

        A dictionnary of shapes is created, with the reference shape at scale 1.
        The other scales are created by decimation.

        Args:
            shape (Shape): the shape to be multiscaled (will be reffered to as scale 1)
            scales (Sequence of floats): the scales at which the shape is multiscaled
            n_points (Sequence of ints): the (approximative) number of points at each scale
        """

        if scales is None and n_points is None:
            raise ValueError("You must specify either scales or n_points")

        if scales is not None:
            assert n_points is None, "You cannot specify both scales and n_points"
            assert 0 < max(scales) <= 1, "scales must be between 0 and 1"
            assert 0 < min(scales) <= 1, "scales must be between 0 and 1"

        if n_points is not None:
            assert (
                0 < max(n_points) <= shape.n_points
            ), "n_points must be between 0 and the number of points in the mesh"
            scales = [1 - (npts / shape.n_points) for npts in n_points]

        self.shapes = dict()
        self.mappings_from_origin = dict()

        self.shapes[1] = shape

        from ..data import PolyData

        if type(shape) == PolyData and shape.is_triangle_mesh():
            from ..decimation import Decimation

            self.decimation_module = Decimation(target_reduction=float(1 - min(scales)))
            self.decimation_module.fit(shape)

            for scale in scales:
                if scale != 1:
                    self.add_scale(float(scale))

        else:
            raise NotImplementedError("Only triangle meshes are supported for now")

    @typecheck
    def add_scale(self, scale: float) -> None:
        """Add a scale to the multiscale object.

        Args:
            scale (float): scale to be added
        """
        assert 0 < scale < 1, "scale must be between 0 and 1"
        if scale in self.shapes.keys():
            # scale already exists, do nothing
            pass
        else:
            self.shapes[scale] = self.decimation_module.transform(
                self.shapes[1], target_reduction=1 - scale
            )
            self.mappings_from_origin[scale] = self.decimation_module._indice_mapping

    @typecheck
    def at(self, scale: Number):
        """Return the shape at a given scale.

        If the scale does not exist, the closest scale is returned.

        Args:
            scale (Number): the scale at which the shape is returned

        Returns:
            Shape: the shape at the given scale
        """
        available_scales = list(self.shapes.keys())
        # find closest scale
        closest_scale = min(available_scales, key=lambda x: abs(x - scale))
        return self.shapes[closest_scale]

    @typecheck
    def indice_mapping(self, high_res: Number, low_res: Number) -> Int1dTensor:
        """Return the indice mapping from high to low resolution.

        The indice mapping is a 1d tensor of integers of length equal to the number of
        points at the high resolution scale. Each element of the tensor is the index of
        the corresponding point at the low resolution scale.

        Args:
            high_res (Number): the scale of the high resolution shape
            low_res (Number): the scale of the low resolution shape

        Returns:
            Int1dTensor: the indice mapping from high to low resolution
        """

        assert high_res >= low_res, "high_res must be greater than low_res"
        assert 0 < low_res <= 1, "low_res must be between 0 and 1"
        assert 0 < high_res <= 1, "high_res must be between 0 and 1"

        available_scales = list(self.shapes.keys())
        high_res = min(available_scales, key=lambda x: abs(x - high_res))
        low_res = min(available_scales, key=lambda x: abs(x - low_res))

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
            high_res (Number): the scale of the high resolution shape
            low_res (Number): the scale of the low resolution shape
            reduce (str, optional): the reduce option. Defaults to "sum".

        Returns:
            NumericalTensor: the signal at the low resolution scale
        """
        assert (
            signal.shape[0] == self.at(high_res).n_points
        ), "signal must have the same number of points as the origin scale"
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
            low_res (Number): the scale of the low resolution shape
            high_res (Number): the scale of the high resolution shape
            smoothing (str, optional): the smoothing option. Defaults to "constant".

        Returns:
            NumericalTensor: the signal at the high resolution scale
        """
        assert (
            signal.shape[0] == self.at(low_res).n_points
        ), "signal must have the same number of points as the origin scale"
        assert low_res <= high_res, "high_res must be smaller than low_res"

        if smoothing == "constant":
            return signal[self.indice_mapping(high_res=high_res, low_res=low_res)]
        else:
            raise NotImplementedError("Only constant smoothing is supported for now")

    # @typecheck
    # def signal_convolution(self, signal, signal_scale, target_scale, **kwargs):

    #     assert signal.shape[0] == self.at(signal_scale).n_points, "signal must have the same number of points as the shape at signal scale"
    #     from ..convolutions import point_convolution

    #     # Store the kwargs to pass to the point convolution
    #     point_convolution_args = point_convolution.__annotations__.keys()
    #     convolutions_kwargs = dict()
    #     for arg in kwargs:
    #         if arg in point_convolution_args:
    #             convolutions_kwargs[arg] = kwargs[arg]

    #     self.at(signal_scale).point_weights = torch.ones_like(signal)

    #     C = self.at(signal_scale).point_convolution(signal, **convolutions_kwargs)

    #     return C @ signal

    @typecheck
    def add_point_data(
        self, signal, *, name, at=1, reduce="mean", smoothing="constant"
    ):
        pass


def edge_smoothing(signal, shape, weight_by_length=False, gpu=True):
    assert signal.shape[0] == shape.n_points

    n_edges = shape.n_edges
    n_points = shape.n_points
    edges = shape.edges

    # Edge smoothing
    edges_revert = torch.zeros_like(edges)
    edges_revert[0], edges_revert[1] = edges[1], edges[0]

    indices = torch.cat((edges, edges_revert), dim=1)

    if not weight_by_length:
        values = torch.ones(2 * n_edges, dtype=torch.float32)
    else:
        values = shape.edge_lengths.repeat(2)

    if torch.cuda.is_available() and gpu:
        indices = indices.cuda()
        values = values.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    S = torch.sparse_coo_tensor(
        indices=indices, values=values, size=(n_points, n_points), device=device
    )

    degrees = S @ torch.ones(n_points, device=device)
    output = (S @ signal.to(device)) / degrees
    output = output.to(shape.device)

    return output
