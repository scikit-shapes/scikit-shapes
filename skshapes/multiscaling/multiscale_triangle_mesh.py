"""MultiscaleTriangleMesh class."""

from __future__ import annotations

from ..types import (
    typecheck,
    convert_inputs,
    FloatSequence,
    IntSequence,
    Int1dTensor,
    NumericalTensor,
    Number,
    polydata_type,
)
from typing import Literal, Union, Optional
from ..utils import scatter
import torch
from ..decimation import Decimation


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
        decimation_module: Optional[Decimation] = None,
    ) -> None:
        """Initialize a multiscale object.

        Parameters
        ----------
        shape
            The shape to be multiscaled (will be reffered to as ratio 1).
        ratios
            The ratios of points at which the shape is multiscaled.
        n_points
            The (approximative) number of points at each ratio.
        fine_to_coarse_policy
            The global policy for downscaling.
        coarse_to_fine_policy
            The policy for upscaling.
        decimation_module
            If not None, the decimation module to be used.

        Raises
        ------
        ValueError
            If the shape is not a triangle mesh.
        """
        if not shape.is_triangle_mesh():
            raise ValueError(
                "MultiscaleTriangleMesh only supports triangle meshes"
            )

        if ratios is None and n_points is None:
            raise ValueError("You must specify either ratios or n_points")

        if ratios is not None:
            assert (
                n_points is None
            ), "You cannot specify both ratios and n_points"
            assert 0 < max(ratios) <= 1, "ratios must be between 0 and 1"
            assert 0 < min(ratios) <= 1, "ratios must be between 0 and 1"

        if n_points is not None:
            assert (
                0 < max(n_points) <= shape.n_points
            ), "n_points must be between 0 and n_points"
            ratios = [1 - (npts / shape.n_points) for npts in n_points]

        self.shapes = {}
        self.mappings_from_origin = {}
        self.signals = {}

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

        if decimation_module is None:
            self.decimation_module = Decimation(
                target_reduction=float(1 - min(ratios))
            )
            self.decimation_module.fit(shape)

        else:
            if not hasattr(decimation_module, "actual_reduction_"):
                raise ValueError(
                    "The decimation module has not been fitted. Please call"
                    + " fit method of the decimation module before passing it"
                    + " to the Multiscale constructor."
                )
            self.decimation_module = decimation_module

        for ratio in ratios:
            if ratio != 1:
                self.add_ratio(float(ratio))

    @typecheck
    def add_ratio(self, ratio: float) -> None:
        """Add a ratio to the multiscale object and update the signals.

        Parameters
        ----------
        ratio
            The ratio to be added.
        """
        assert 0 < ratio < 1, "ratio must be between 0 and 1"
        if ratio in self.shapes.keys():
            # ratio already exists, do nothing
            pass
        else:
            polydata = self.decimation_module.transform(
                self.shapes[1], target_reduction=1 - ratio
            )

            self.shapes[ratio] = polydata
            self.mappings_from_origin[
                ratio
            ] = self.decimation_module.indice_mapping

    @typecheck
    def at(self, ratio: Number) -> polydata_type:
        """Return the shape at a given ratio.

        If the ratio does not exist, the closest ratio is returned.

        Parameters
        ----------
        ratio
            The ratio at which the shape is returned.

        Returns
        -------
        polydata_type
            The shape decimated at the given ratio.
        """
        available_ratios = self.ratios
        # find closest ratio
        closest_ratio = min(available_ratios, key=lambda x: abs(x - ratio))
        return self.shapes[closest_ratio]

    @typecheck
    def __getitem__(self, ratio: Number) -> polydata_type:
        """Return the shape at a given ratio."""
        return self.at(ratio)

    @convert_inputs
    @typecheck
    def propagate(
        self,
        signal: Optional[NumericalTensor] = None,
        *,
        key: str,
        from_ratio=None,
        to_ratio: Union[Number, FloatSequence, Literal["all"]] = "all",
        from_n_points: Optional[int] = None,
        to_n_points: Union[int, IntSequence, Literal["all"]] = "all",
        fine_to_coarse_policy: Optional[dict] = None,
        coarse_to_fine_policy: Optional[dict] = None,
    ):
        """Add a signal to the multiscale object.

        The signal can be specified as a couple key/tensor (can be
        multidimensional) or as a key, if the signal is already
        stored in the point_data of the shape at the desired ratio. Upscale and
        downscale policies can be specified for the signal. If not specified,
        the default policies of the multiscale instance are used.

        This function add (if needed) the signal to the point_data of the shape
        at ratio "at", and propagate the signal through the available ratios
        following the policies. After calling this function, the signal is
        available at all the available ratios.

        If a new ratio is added to the multiscale instance after the adding of
        the signal with add_ratio, all the signals are updated.


        Parameters
        ----------
        key
            If the signal is not specified, the key of the signal in the
            point_data of the shape at from_ratio.
        signal
            The signal to be added. Must be specified if key is not specified,
            and must have the same number of points as the shape at ratio
            `from_ratio`.
        from_ratio
            The ratio at which the signal is added.
        fine_to_coarse_policy
            The policy for downscaling the signal.
        coarse_to_fine_policy
            The policy for upscaling the signal.

        """
        if from_ratio is None and from_n_points is None:
            raise ValueError(
                "You must specify either from_ratio or from_n_points"
            )

        if from_ratio is not None:
            assert (
                from_n_points is None
            ), "You cannot specify both from_ratio and from_n_points"
            assert 0 < from_ratio <= 1, "from_ratio must be between 0 and 1"

        if from_n_points is not None:
            assert (
                0 < from_n_points <= self.at(1).n_points
            ), "from_n_points must be between 0 and n_points"
            from_ratio = 1 - (from_n_points / self.at(1).n_points)

        if signal is None:
            # If the signal is not specified, it must be in the point_data of
            # the shape at from_ratio
            if key not in self.at(from_ratio).point_data.keys():
                raise ValueError(
                    f"{key} not in the point_data of the shape at ratio"
                    + f" {from_ratio}"
                )
            signal = self.at(from_ratio).point_data[key]

        if fine_to_coarse_policy is None:
            fine_to_coarse_policy = self.fine_to_coarse_policy

        if coarse_to_fine_policy is None:
            coarse_to_fine_policy = self.coarse_to_fine_policy

        self.signals[key] = {
            "origin_ratio": from_ratio,
            "available_ratios": [from_ratio],
            "fine_to_coarse_policy": fine_to_coarse_policy,
            "coarse_to_fine_policy": coarse_to_fine_policy,
        }

        self.at(from_ratio)[key] = signal

        if from_n_points is not None:
            if to_n_points == "all":
                to_ratios = self.available_ratios
            elif isinstance(to_n_points, int):
                to_ratios = [1 - (to_n_points / self.at(1).n_points)]
            else:
                to_ratios = [
                    1 - (npts / self.at(1).n_points) for npts in to_n_points
                ]

        elif from_ratio is not None:
            if to_ratio == "all":
                to_ratios = self.available_ratios
            elif not hasattr(to_ratio, "__iter__"):
                to_ratios = [to_ratio]
            else:
                to_ratios = to_ratio

        self._propagate_signal(
            signal_name=key,
            origin_ratio=from_ratio,
            to_ratios=to_ratios,
            fine_to_coarse_policy=fine_to_coarse_policy,
            coarse_to_fine_policy=coarse_to_fine_policy,
        )

    @typecheck
    def _propagate_signal(
        self,
        signal_name: str,
        origin_ratio: Number,
        to_ratios: FloatSequence,
        fine_to_coarse_policy: Optional[dict] = None,
        coarse_to_fine_policy: Optional[dict] = None,
    ) -> None:
        """Propagate a signal.

        This function propagates a signal from the origin ratio to the
        available ratios following the propagation policy.

        It is not meant to be called directly, but through the propagate
        function.
        """
        self.signals[signal_name]["available_ratios"] = []

        if fine_to_coarse_policy is None:
            fine_to_coarse_policy = self.fine_to_coarse_policy
        if coarse_to_fine_policy is None:
            coarse_to_fine_policy = self.coarse_to_fine_policy

        # Ratios that are greater than the origin ratio (ascending order)
        up_ratios = [r for r in to_ratios if r >= origin_ratio][::-1]

        # Ratios that are smaller than the origin ratio (descending order)
        down_ratios = [r for r in to_ratios if r <= origin_ratio]

        signal_origin = self.at(origin_ratio).point_data[signal_name]

        tmp = signal_origin
        coarse_ratio = origin_ratio

        for r in up_ratios[1:]:
            self.at(r).point_data[
                signal_name
            ] = self.signal_from_coarse_to_fine(
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
            self.at(r).point_data[
                signal_name
            ] = self.signal_from_fine_to_coarse(
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
    def indice_mapping(
        self, fine_ratio: Number, coarse_ratio: Number
    ) -> Int1dTensor:
        """Return the indice mapping from high to low resolution.

        The indice mapping is a 1d tensor of integers of length equal to the
        number of points at the high resolution ratio. Each element of the
        tensor is the index of the corresponding point at the low resolution
        ratio.

        Parameters
        ----------
        fine_ratio
            The ratio of the high resolution shape.
        coarse_ratio
            The ratio of the low resolution shape.

        Returns
        -------
        Int1dTensor
            The indice mapping from high to low resolution.
        """
        assert (
            fine_ratio >= coarse_ratio
        ), "fine_ratio must be greater than coarse_ratio"
        assert 0 < coarse_ratio <= 1, "coarse_ratio must be between 0 and 1"
        assert 0 < fine_ratio <= 1, "fine_ratio must be between 0 and 1"

        available_ratios = list(self.shapes.keys())
        fine_ratio = min(available_ratios, key=lambda x: abs(x - fine_ratio))
        coarse_ratio = min(
            available_ratios, key=lambda x: abs(x - coarse_ratio)
        )

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

        This operation is a scatter operation, with the indice mapping as
        index.
        A reduce operation is applied to the scatter. The available reduce
        operations are "sum", "min", "max", "mean".

        Parameters
        ----------
        signal
            The signal to be propagated.
        fine_ratio
            The ratio of the high resolution shape.
        coarse_ratio
            The ratio of the low resolution shape.
        reduce
            The reduce option. Defaults to "sum".

        Returns
        -------
        NumericalTensor
            The signal at the low resolution ratio.
        """
        assert (
            signal.shape[0] == self.at(fine_ratio).n_points
        ), "signal must have the same number of points as the origin ratio"
        assert (
            fine_ratio >= coarse_ratio
        ), "fine_ratio must be greater than coarse_ratio"

        return scatter(
            src=signal,
            index=self.indice_mapping(
                fine_ratio=fine_ratio, coarse_ratio=coarse_ratio
            ),
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
        - "constant" : the signal is repeated at each point of the high
            resolution shape
        - "mesh_convolution" : the signal is smoothed using the mesh
            convolution operator

        Parameters
        ----------
        signal
            The signal to be propagated.
        coarse_ratio
            The ratio of the low resolution shape.
        fine_ratio
            The ratio of the high resolution shape.
        smoothing
            The smoothing option.

        Returns
        -------
        NumericalTensor
            The signal at the high resolution ratio.
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
            index=self.indice_mapping(
                fine_ratio=fine_ratio, coarse_ratio=coarse_ratio
            ),
        )

        if smoothing == "constant":
            return fine_ratio_signal

        # Else, smooth the signal

        elif smoothing == "mesh_convolution":
            fine_ratio_signal = edge_smoothing(
                fine_ratio_signal,
                self.at(fine_ratio),
                n_smoothing_steps=n_smoothing_steps,
            )

        return fine_ratio_signal

    @convert_inputs
    @typecheck
    def signal_convolution(self, signal, signal_ratio, target_ratio, **kwargs):
        """Convolve a signal between two ratios.

        Parameters
        ----------
        signal
            The signal to be convolved.
        signal_ratio
            The ratio of the signal.
        target_ratio
            The ratio of the target shape.
        kwargs
            The kwargs to be passed to the point convolution.

        Returns
        -------
        NumericalTensor
            The convolved signal.
        """
        source = self.at(signal_ratio)
        target = self.at(target_ratio)

        if signal.shape[0] != source.n_points:
            raise ValueError(
                "signal must have the same number of points as"
                + " the shape at signal ratio"
            )

        # Store the kwargs to pass to the point convolution
        point_convolution_args = (
            source._point_convolution.__annotations__.keys()
        )
        convolutions_kwargs = {}
        for arg in kwargs:
            if arg in point_convolution_args:
                convolutions_kwargs[arg] = kwargs[arg]

        C = source.point_convolution(target=target, **convolutions_kwargs)

        return C @ signal

    @property
    @typecheck
    def ratios(self) -> list[Number]:
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

    Parameters
    ----------
    signal
        The signal to be smoothed.
    shape
        The triangle mesh on which the signal is defined.
    weight_by_length
        Wetether to weight the smoothing by the length of the edges.
    n_smoothing_steps
        The number of repetition of smoothing operation.

    Returns
    -------
    NumericalTensor
        The smoothed signal.
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
