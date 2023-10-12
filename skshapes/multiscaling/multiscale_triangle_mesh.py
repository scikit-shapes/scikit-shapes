from __future__ import annotations

from ..types import (
    typecheck,
    convert_inputs,
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
from typing import List, Literal
from ..utils import scatter
import torch


# class SingleScale(PolyData):
#     """SingleScale is a polydata wrapper with reference to a multiscale object."""

#     def __init__(self, polydata: polydata_type, multiscale: MultiscaleTriangleMesh, ratio=Number) -> None:
#         """Initialize a SingleScale object.

#         Args:
#             polydata (polydata_type): the polydata to be wrapped
#         """
#         points = polydata.points
#         edges = polydata.edges
#         triangles = polydata.triangles
#         device = polydata.device
#         landmarks = polydata.landmarks
#         point_data = polydata.point_data
#         try:
#             cache_size = polydata.cache_size
#         except:
#             cache_size = None
#         super().__init__(
#             points=points,
#             edges=edges,
#             triangles=triangles,
#             device=device,
#             landmarks=landmarks,
#             point_data=point_data,
#             cache_size=cache_size,
#         )

#         self.multiscale = multiscale
#         self.ratio = ratio


#     @property
#     def point_data(self) -> DataAttributes:
#         warn("To set the value of a point_data entry, use the __setitem__ method, and not the point_data property")
#         return self._point_data

#     @point_data.setter
#     @typecheck
#     def point_data(self, point_data_dict: dict) -> None:
#         if not isinstance(point_data_dict, DataAttributes):
#             # Convert the point_data to a DataAttributes object
#             # the from_dict method will check that the point_data are valid
#             point_data_dict = DataAttributes.from_dict(point_data_dict)

#         assert (
#             point_data_dict.n == self.n_points
#         ), "The number of points in the point_data entries should be the same as the number of points in the shape."
#         self._point_data = point_data_dict.to(self.device)

#     def __setitem__(self, key, value):
#         self._point_data[key] = value
#         # update the multiscale object

#     def __repr__(self):
#         return (
#             f"SingleScale object at ratio {self.ratio} of shape {self.n_points} points"
#         )


class MultiscaleTriangleMesh:
    """A class to handle multiscale data."""

    @typecheck
    def __init__(
        self,
        shape: polydata_type,
        *,
        ratios: Optional[FloatSequence] = None,
        n_points: Optional[IntSequence] = None,
        fine_to_coarse_policy: Optional[dict] = None,
        coarse_to_fine_policy: Optional[dict] = None,
    ) -> None:
        """Initialize a multiscale object from a reference shape. This class is particularly
        useful to handle multiscale signals.

        An instance of Multiscale contains a dict of shapes, with the ratios as keys.

        For signal propagation, policies can be specified for downscaling and upscaling.

        Policy for downscaling consists of a reduce operation and a pass_through_all_scales option:
        - reduce (str): the reduce operation. Available options are "sum", "min", "max", "mean"
        - pass_through_all_scales (bool): if True, the signal is propagated through all the ratios when
          dowscaling. If False, the signal is propagated directly from the origin ratio to the
            target ratio.

        Policy for upscaling consists of a smoothing operation and a pass_through_all_scales option:
        - smoothing (str): the smoothing operation. Available options are "constant", "mesh_convolution"
        - n_smoothing_steps (int): the number of smoothing operations to apply
        - pass_through_all_scales (bool): if True, the signal is propagated through all the ratios when

        Default policies are:
        - fine_to_coarse_policy = {"reduce": "mean", "pass_through_all_scales": False}
        - coarse_to_fine_policy = {"smoothing": "constant", "n_smoothing_steps": 1, "pass_through_all_scales": False}

        When adding a signal, the policies can be specified for the signal. If not specified, the default policies
        are used.

        To add a signal to the multiscale object, use the add_signal method. The signal can be specified as a tensor
        or as a key in the point_data of the shape at the origin ratio. If the signal is specified as a tensor, it
        is added to the point_data of the shape at the origin ratio. The signal is then propagated to the available
        ratios following the policies. Whan adding a new ratio after a signal has been added, the signal is updated.

        Be careful that modifying directly the point_data of a shape in the multiscale object will not update the
        signals at other scale. Please make sure that no modification is done directly on the point_data of a shape
        in the multiscale object after the call of the add_signal method.

        Args:
            shape (Shape): the shape to be multiscaled (will be reffered to as ratio 1)
            ratios (Sequence of floats): the ratios at which the shape is multiscaled
            n_points (Sequence of ints): the (approximative) number of points at each ratio
            fine_to_coarse_policy (dict, optional): the policy for downscaling. Defaults to None.
            coarse_to_fine_policy (dict, optional): the policy for upscaling. Defaults to None.
        """
        if not shape.is_triangle_mesh():
            raise ValueError("MultiscaleTriangleMesh only supports triangle meshes")

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
        self.signals = dict()

        self.coarse_to_fine_policy = (
            coarse_to_fine_policy
            if coarse_to_fine_policy is not None
            else {
                "smoothing": "constant",
                "n_smoothing_steps": 1,
                "pass_through_all_scales": False,
            }
        )
        self.fine_to_coarse_policy = (
            fine_to_coarse_policy
            if fine_to_coarse_policy is not None
            else {"reduce": "mean", "pass_through_all_scales": False}
        )

        self.shapes[1] = shape

        from ..decimation import Decimation

        self.decimation_module = Decimation(target_reduction=float(1 - min(ratios)))
        self.decimation_module.fit(shape)

        for ratio in ratios:
            if ratio != 1:
                self.add_ratio(float(ratio))

    @typecheck
    def add_ratio(self, ratio: float) -> None:
        """Add a ratio to the multiscale object and update the signals.

        Args:
            ratio (float): ratio to be added
        """
        assert 0 < ratio < 1, "ratio must be between 0 and 1"
        if ratio in self.shapes.keys():
            # ratio already exists, do nothing
            pass
        else:
            polydata = self.decimation_module.transform(
                self.shapes[1], target_reduction=1 - ratio
            )

            # self.shapes[ratio] = SingleScale(polydata, multiscale=self, ratio=ratio)
            self.shapes[ratio] = polydata
            self.mappings_from_origin[ratio] = self.decimation_module._indice_mapping
        self.update_signals()

    @typecheck
    def at(self, ratio: Number) -> polydata_type:
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
    def __getitem__(self, ratio: Number) -> polydata_type:
        """Return the shape at a given ratio.

        If the ratio does not exist, the closest ratio is returned.

        Args:
            ratio (Number): the ratio at which the shape is returned

        Returns:
            Shape: the shape at the given ratio
        """
        return self.at(ratio)

    @property
    def available_ratios(self):
        """Return the available ratios (decreasing order)."""
        available_ratios = list(self.shapes.keys())
        available_ratios.sort()
        return available_ratios[::-1]

    @convert_inputs
    @typecheck
    def add_signal(
        self,
        signal: Optional[NumericalTensor] = None,
        *,
        key: str,
        at: Number = 1,
        fine_to_coarse_policy: Optional[dict] = None,
        coarse_to_fine_policy: Optional[dict] = None,
    ):
        """Add a signal to the multiscale object.

        The signal can be specified as a couple key/tensor (can be multidimensional) or as a key, if the signal is already
        stored in the point_data of the shape at the desired ratio. Upscale and downscale policies can be specified for
        the signal. If not specified, the default policies of the multiscale instance are used.

        This function add (if needed) the signal to the point_data of the shape at ratio "at", and propagate the signal through
        the available ratios following the policies. After calling this function, the signal is available at all the available
        ratios.

        If a new ratio is added to the multiscale instance after the adding of the signal with add_ratio, all the signals are
        updated.

        Example :
        >>> M = sks.Multiscale(shape, ratios=[0.5, 0.1])
        >>> signal = torch.rand(M.at(0.5).n_points)
        >>> M.add_signal(key="signal", at=0.5)
        >>> M.at(0.1).point_data["signal"].shape[0] == M.at(0.1).n_points # True

        Args:
            key (str): _description_
            signal (Optional[NumericalTensor], optional): _description_. Defaults to None.
            at (Number, optional): _description_. Defaults to 1.
            fine_to_coarse_policy (Optional[dict], optional): _description_. Defaults to None.
            coarse_to_fine_policy (Optional[dict], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """

        if signal is None:
            try:
                signal = self.at(at).point_data[key]
            except:
                raise ValueError(
                    f"{key} not in the point_data of the shape at ratio {at}"
                )

        if fine_to_coarse_policy is None:
            fine_to_coarse_policy = self.fine_to_coarse_policy

        if coarse_to_fine_policy is None:
            coarse_to_fine_policy = self.coarse_to_fine_policy

        self.signals[key] = {
            "origin_ratio": at,
            "available_ratios": [at],
            "fine_to_coarse_policy": fine_to_coarse_policy,
            "coarse_to_fine_policy": coarse_to_fine_policy,
        }

        self.at(at)[key] = signal

        self.propagate_signal(
            signal_name=key,
            origin_ratio=at,
            fine_to_coarse_policy=fine_to_coarse_policy,
            coarse_to_fine_policy=coarse_to_fine_policy,
        )

    @typecheck
    def propagate_signal(
        self,
        signal_name: str,
        origin_ratio: Number,
        fine_to_coarse_policy: Optional[dict] = None,
        coarse_to_fine_policy: Optional[dict] = None,
    ) -> None:
        """Propagate a signal from the origin ratio to the available ratios following
        the propagation policies.
        """

        self.signals[signal_name]["available_ratios"] = []

        if fine_to_coarse_policy is None:
            fine_to_coarse_policy = self.fine_to_coarse_policy
        if coarse_to_fine_policy is None:
            coarse_to_fine_policy = self.coarse_to_fine_policy

        # Ratios that are greater than the origin ratio (ascending order)
        up_ratios = [r for r in self.available_ratios if r >= origin_ratio][::-1]

        # Ratios that are smaller than the origin ratio (descending order)
        down_ratios = [r for r in self.available_ratios if r <= origin_ratio]

        signal_origin = self.at(origin_ratio).point_data[signal_name]

        tmp = signal_origin
        coarse_ratio = origin_ratio

        for r in up_ratios[1:]:
            self.at(r).point_data[signal_name] = self.signal_from_coarse_to_fine(
                tmp,
                coarse_ratio=coarse_ratio,
                fine_ratio=r,
                **coarse_to_fine_policy,
            )
            self.signals[signal_name]["available_ratios"].append(r)
            if (
                "pass_through_all_scales" in coarse_to_fine_policy.keys()
                and coarse_to_fine_policy["pass_through_all_scales"]
            ):
                coarse_ratio = r
                tmp = self.at(r).point_data[signal_name]

        tmp = signal_origin
        fine_ratio = origin_ratio

        for r in down_ratios[1:]:
            self.at(r).point_data[signal_name] = self.signal_from_fine_to_coarse(
                tmp,
                fine_ratio=fine_ratio,
                coarse_ratio=r,
                **fine_to_coarse_policy,
            )
            self.signals[signal_name]["available_ratios"].append(r)
            if (
                "pass_through_all_scales" in fine_to_coarse_policy.keys()
                and fine_to_coarse_policy["pass_through_all_scales"]
            ):
                tmp = self.at(r).point_data[signal_name]
                fine_ratio = r

    @typecheck
    def update_signals(self) -> None:
        """Update the signals at the available ratios.

        It checks for new
        ratios for existing signals and redo the propagation of the concerned signals.
        """

        # This is not needed anymore, since we propagate the signals added with add_signal
        # and do not try to automatically detect the signals

        # check for new signals
        # for r in self.available_ratios:
        #     for signal in self.at(r, update_signals=False).point_data:
        #         # If the signal is not in the dict, add it and propagate it
        #         # accross the available ratios
        #         if signal not in self.signals.keys():
        #             self.signals[signal] = {
        #                 "origin_ratio": r,
        #                 "available_ratios": [r],
        #             }
        #             self.propagate_signal(
        #                 signal_name=signal,
        #                 origin_ratio=r,
        #             )

        # check for new ratios for existing signals
        for signal in self.signals.keys():
            for r in self.available_ratios:
                if r not in self.signals[signal]["available_ratios"]:
                    if "fine_to_coarse_policy" not in self.signals[signal].keys():
                        fine_to_coarse_policy = self.fine_to_coarse_policy
                    else:
                        fine_to_coarse_policy = self.signals[signal][
                            "fine_to_coarse_policy"
                        ]

                    if "coarse_to_fine_policy" not in self.signals[signal].keys():
                        coarse_to_fine_policy = self.coarse_to_fine_policy
                    else:
                        coarse_to_fine_policy = self.signals[signal][
                            "coarse_to_fine_policy"
                        ]

                    self.propagate_signal(
                        signal_name=signal,
                        origin_ratio=self.signals[signal]["origin_ratio"],
                        fine_to_coarse_policy=fine_to_coarse_policy,
                        coarse_to_fine_policy=coarse_to_fine_policy,
                    )

    @typecheck
    def indice_mapping(self, fine_ratio: Number, coarse_ratio: Number) -> Int1dTensor:
        """Return the indice mapping from high to low resolution.

        The indice mapping is a 1d tensor of integers of length equal to the number of
        points at the high resolution ratio. Each element of the tensor is the index of
        the corresponding point at the low resolution ratio.

        Args:
            fine_ratio (Number): the ratio of the high resolution shape
            coarse_ratio (Number): the ratio of the low resolution shape

        Returns:
            Int1dTensor: the indice mapping from high to low resolution
        """

        assert (
            fine_ratio >= coarse_ratio
        ), "fine_ratio must be greater than coarse_ratio"
        assert 0 < coarse_ratio <= 1, "coarse_ratio must be between 0 and 1"
        assert 0 < fine_ratio <= 1, "fine_ratio must be between 0 and 1"

        available_ratios = list(self.shapes.keys())
        fine_ratio = min(available_ratios, key=lambda x: abs(x - fine_ratio))
        coarse_ratio = min(available_ratios, key=lambda x: abs(x - coarse_ratio))

        if fine_ratio == coarse_ratio:
            return torch.arange(self.shapes[fine_ratio].n_points)

        elif fine_ratio == 1:
            return self.mappings_from_origin[coarse_ratio]

        else:
            tmp = self.mappings_from_origin[fine_ratio]
            tmp = scatter(src=torch.arange(len(tmp)), index=tmp, reduce="min")
            return self.mappings_from_origin[coarse_ratio][tmp]

    @convert_inputs
    @typecheck
    def signal_from_fine_to_coarse(
        self,
        signal: NumericalTensor,
        *,
        fine_ratio: Number,
        coarse_ratio: Number,
        reduce="mean",
        **kwargs,
    ) -> NumericalTensor:
        """Propagate a signal from a resolution to a lower resolution.

        This operation is a scatter operation, with the indice mapping as index.
        A reduce operation is applied to the scatter. The available reduce operations
        are "sum", "min", "max", "mean".

        Args:
            signal (NumericalTensor): the signal to be propagated
            fine_ratio (Number): the ratio of the high resolution shape
            coarse_ratio (Number): the ratio of the low resolution shape
            reduce (str, optional): the reduce option. Defaults to "sum".

        Returns:
            NumericalTensor: the signal at the low resolution ratio
        """
        assert (
            signal.shape[0] == self.at(fine_ratio).n_points
        ), "signal must have the same number of points as the origin ratio"
        assert (
            fine_ratio >= coarse_ratio
        ), "fine_ratio must be greater than coarse_ratio"

        return scatter(
            src=signal,
            index=self.indice_mapping(fine_ratio=fine_ratio, coarse_ratio=coarse_ratio),
            reduce=reduce,
        )

    @convert_inputs
    @typecheck
    def signal_from_coarse_to_fine(
        self,
        signal: NumericalTensor,
        *,
        coarse_ratio: Number,
        fine_ratio: Number,
        smoothing: Literal["constant", "mesh_convolution"] = "constant",
        n_smoothing_steps: int = 1,
        **kwargs,
    ) -> NumericalTensor:
        """Propagate a signal from a resolution to a higher resolution.

        This operation requires a smoothing operation. The available smoothing
        operations are :
        - "constant" : the signal is repeated at each point of the high resolution shape
        - "mesh_convolution" : the signal is smoothed using the mesh convolution operator

        Args:
            signal (NumericalTensor): the signal to be propagated
            coarse_ratio (Number): the ratio of the low resolution shape
            fine_ratio (Number): the ratio of the high resolution shape
            smoothing (str, optional): the smoothing option. Defaults to "constant".

        Returns:
            NumericalTensor: the signal at the high resolution ratio
        """
        assert (
            signal.shape[0] == self.at(coarse_ratio).n_points
        ), "signal must have the same number of points as the origin ratio"
        assert (
            coarse_ratio <= fine_ratio
        ), "fine_ratio must be smaller than coarse_ratio"

        fine_ratio_signal = torch.index_select(
            signal,
            dim=0,
            index=self.indice_mapping(fine_ratio=fine_ratio, coarse_ratio=coarse_ratio),
        )

        if smoothing == "constant":
            return fine_ratio_signal

        # Else, smooth the signal

        elif smoothing == "mesh_convolution":
            C = self.at(fine_ratio).mesh_convolution()

        # for _ in range(n_smoothing_steps):
        #     fine_ratio_signal = C @ fine_ratio_signal
        fine_ratio_signal = edge_smoothing(
            fine_ratio_signal,
            self.at(fine_ratio),
            n_smoothing_steps=n_smoothing_steps,
        )

        return fine_ratio_signal

    @convert_inputs
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


@convert_inputs
@typecheck
def edge_smoothing(
    signal: NumericalTensor,
    shape: polydata_type,
    weight_by_length: bool = False,
    n_smoothing_steps: int = 1,
) -> NumericalTensor:
    """Smooth a signal on a triangle mesh by edge smoothing.

    Args:
        signal (NumericalTensor): the signal to be smoothed
        shape (polydata_type): the triangle mesh on which the signal is defined
        weight_by_length (bool, optional): Defaults to False.
        n_smoothing_steps (int, optional): the number of smoothing operations to apply. Defaults to 1.

    Returns:
        NumericalTensor: the smoothed signal
    """
    assert signal.shape[0] == shape.n_points
    assert n_smoothing_steps >= 1, "n_smoothing_steps must be positive"
    signal_device = signal.device
    signal = signal.to(shape.device)

    K = shape.mesh_convolution(weight_by_length=weight_by_length)

    output = signal.clone()
    for _ in range(n_smoothing_steps):
        output = K @ output

    return output.to(signal_device)


# @convert_inputs
# @typecheck
# def vector_heat_smooting(
#     signal: NumericalTensor, shape: polydata_type
# ) -> NumericalTensor:
#     """Smooth a vector signal on a triangle mesh or a points cloud by vector heat smoothing using pp3d.

#     Args:
#         signal (NumericalTensor): the signal to be smoothed
#         shape (polydata_type): the triangle mesh or points cloud on which the signal is defined

#     Raises:
#         ImportError: potpourri3d must be installed to use vector heat smoothing

#     Returns:
#         NumericalTensor: the smoothed signal
#     """
#     try:
#         import potpourri3d as pp3d
#     except:
#         raise ImportError("Please install potpourri3d to use vector heat smoothing")

#     if shape.is_triangle_mesh():
#         V, F = shape.points.cpu().numpy(), shape.triangles.cpu().numpy()

#         try:
#             solver = pp3d.MeshVectorHeatSolver(V, F)
#         except:
#             solver = pp3d.PointCloudHeatSolver(V)

#     else:
#         solver = pp3d.PointCloudHeatSolver(shape.points.cpu().numpy())

#     ext = solver.extend_vector(torch.arange(len(signal)).numpy(), signal.cpu().numpy())
#     return torch.from_numpy(ext)
