"""Generic Multiscale class."""

from __future__ import annotations


from typing import Optional
import torch
from ..types import (
    Number,
    shape_type,
    FloatTensor,
    Int1dTensor,
    IntSequence,
    NumberSequence,
    FineToCoarsePolicy,
    CoarseToFinePolicy,
)
from ..utils import scatter
from ..input_validation import typecheck, one_and_only_one, convert_inputs
from ..errors import NotFittedError
from ..decimation import Decimation


class Multiscale:
    """Multiscale representation of a shape.

    This class allows to represent a shape at different scales. The shape is
    represented at the origin scale and at different coarser scales. The
    coarser scales can be defined by three different parameters:

    - the ratio of the number of points between the origin scale and the
        coarser scale,
    - the number of points of the coarser scale,
    - the scale of the coarser scale.

    If landmarks are defined on the origin shape, they are propagated to the
    coarser scales.

    New scales can be added to the multiscale representation by calling the
    [`append`][skshapes.multiscaling.Multiscale.append] method. The new scale
    is defined by one of the three parameters described above.

    Existing scales can be retrieved by calling the
    [`at`][skshapes.multiscaling.Multiscale.at] # noqa E501 method. The scale
    is defined by one of the three parameters described above.

    Signals defined at any scale can be propagated to the other scales. The
    propagation is performed in both directions from the origin to the coarser
    scales and from the origin to the finer scales. The propagation is done by
    interpolation or smoothing depending on the policies defined by the
    `fine_to_coarse_policy` and `coarse_to_fine_policy` parameters of the
    [`propagate`][skshapes.multiscaling.Multiscale.propagate] method.

    Most of the methods of this class can be called with one of the `ratio`,
    `n_points` or `scale` parameters.

    Parameters
    ----------
    shape
        The shape at the origin scale.
    ratios
        The ratios of the coarser scales.
    n_points
        The number of points of the coarser scales.
    scales
        The scales of the coarser scales.
    decimation_module
        The decimation module to use to compute the coarser scales. If not
        provided, it is defined automatically.

    Raises
    ------
    NotImplementedError
        If the shape is not a triangle mesh.
    ValueError
        If none of the `ratios`, `n_points` or `scales` parameters are
        provided or if more than one of these parameters are provided.

    Examples
    --------
    ```python
    import skshapes as sks

    # load a shape
    shape = sks.Sphere()
    ratios = [0.5, 0.25, 0.125]
    multiscale = sks.Multiscale(shape=shape, ratios=ratios)
    ```

    """

    @one_and_only_one(parameters=["ratios", "n_points", "scales"])
    @typecheck
    def __init__(
        self,
        shape: shape_type,
        ratios: Optional[NumberSequence] = None,
        n_points: Optional[IntSequence] = None,
        scales: Optional[NumberSequence] = None,
        decimation_module=None,
    ) -> None:
        self.shape = shape

        if ratios is not None:
            pass
        elif n_points is not None:
            ratios = [n / shape.n_points for n in n_points]
        elif scales is not None:
            raise NotImplementedError("Scales are not implemented yet")

        if decimation_module is not None:
            if not hasattr(decimation_module, "ref_mesh_"):
                raise NotFittedError(
                    "The decimation module has not been fitted."
                )
            self._decimation_module = decimation_module

        if decimation_module is None:
            if shape.is_triangle_mesh():
                min_n_points = 1
                decimation_module = Decimation(n_points=min_n_points)

            else:
                raise NotImplementedError(
                    "Only triangle meshes are supported for now"
                )

            decimation_module.fit(shape)
            self._decimation_module = decimation_module

        self.shapes = {}
        self.mappings_from_origin = {}
        self.shapes[1] = shape

        for r in ratios:
            self.append(ratio=float(r))

    @one_and_only_one(parameters=["ratio", "n_points", "scale"])
    @typecheck
    def append(
        self,
        *,
        ratio: Optional[Number] = None,
        n_points: Optional[int] = None,
        scale: Optional[Number] = None,
    ) -> None:
        """Append a new shape.

        This function can be called with one of the `ratio`, `n_points` or
        `scale`.

        Parameters
        ----------
        ratio
            The target ratio from the initial number of points.
        n_points
            The target number of points.
        scale
            The target scale.

        """
        if n_points is not None:
            ratio = n_points / self.shape.n_points
        elif scale is not None:
            raise NotImplementedError("Scales are not implemented yet")

        if ratio in self.shapes.keys():
            # ratio already exists, do nothing
            pass
        else:
            new_shape, indice_mapping = self._decimation_module.transform(
                self.shape,
                ratio=ratio,
                return_indice_mapping=True,
            )

            self.shapes[ratio] = new_shape
            self.mappings_from_origin[ratio] = indice_mapping

    @one_and_only_one(parameters=["ratio", "n_points", "scale"])
    @typecheck
    def at(
        self,
        *,
        ratio: Optional[Number] = None,
        n_points: Optional[int] = None,
        scale: Optional[Number] = None,
    ) -> shape_type:
        """Get the shape at a given ratio, number of points or scale."""
        if n_points is not None:
            ratio = n_points / self.shape.n_points
        elif scale is not None:
            raise NotImplementedError("Scales are not implemented yet")

        # find clostest n_points
        available_ratios = self.shapes.keys()
        ratio = min(available_ratios, key=lambda x: abs(x - ratio))

        return self.shapes[ratio]

    @one_and_only_one(parameters=["from_ratio", "from_n_points", "from_scale"])
    @typecheck
    def propagate(
        self,
        signal_name: str,
        from_scale: Optional[Number] = None,
        from_ratio: Optional[Number] = None,
        from_n_points: Optional[int] = None,
        fine_to_coarse_policy: Optional[FineToCoarsePolicy] = None,
        coarse_to_fine_policy: Optional[CoarseToFinePolicy] = None,
    ) -> None:
        """Propagate a signal to the other scales.

        The signal must be defined at the origin scale. The propagation is
        performed in both directions from the origin to the coarser scales and
        from the origin to the finer scales.

        The propagation is parametrized by two policies: one for the fine to
        coarse propagation and one for the coarse to fine propagation. These
        policies are defined by the `fine_to_coarse_policy` and
        `coarse_to_fine_policy` parameters. If not provided, default policies
        are used. See the documentation of the
        [`FineToCoarsePolicy`][skshapes.types.FineToCoarsePolicy] and
        [`CoarseToFinePolicy`][skshapes.types.CoarseToFinePolicy] classes
        for more details.

        Parameters
        ----------
        signal_name
            The name of the signal to propagate.
        from_scale
            The scale of the origin shape.
        from_ratio
            The ratio of the origin shape.
        from_n_points
            The number of points of the origin shape.
        fine_to_coarse_policy
            The policy for the fine to coarse propagation.
        coarse_to_fine_policy
            The policy for the coarse to fine propagation.
        """
        if fine_to_coarse_policy is None:
            fine_to_coarse_policy = FineToCoarsePolicy()

        if coarse_to_fine_policy is None:
            coarse_to_fine_policy = CoarseToFinePolicy()

        available_ratios = self.shapes.keys()
        if from_n_points is not None:
            from_ratio = from_n_points / self.shape.n_points
        elif from_scale is not None:
            raise NotImplementedError("Scales are not implemented yet")

        from_ratio = min(available_ratios, key=lambda x: abs(x - from_ratio))

        if signal_name not in self.shapes[from_ratio].point_data.keys():
            raise ValueError(
                f"The signal {signal_name} is not available at the scale"
                + f" {from_ratio}"
            )

        # propagate the signal from the origin to the other scales
        ratio_lower = [r for r in available_ratios if r < from_ratio]
        ratio_higher = [r for r in available_ratios if r > from_ratio]

        ratio_lower = [from_ratio] + sorted(ratio_lower, reverse=True)
        ratio_higher = [from_ratio] + sorted(ratio_higher)

        for i in range(len(ratio_lower) - 1):
            source_ratio = ratio_lower[i]
            target_ratio = ratio_lower[i + 1]
            source_signal = self.at(ratio=source_ratio)[signal_name]
            target_signal = self._signal_from_one_scale_to_another(
                source_signal=source_signal,
                source_ratio=source_ratio,
                target_ratio=target_ratio,
                fine_to_coarse_policy=fine_to_coarse_policy,
                coarse_to_fine_policy=coarse_to_fine_policy,
            )
            self.at(ratio=target_ratio)[signal_name] = target_signal

        for i in range(len(ratio_higher) - 1):
            source_ratio = ratio_higher[i]
            target_ratio = ratio_higher[i + 1]
            source_signal = self.at(ratio=source_ratio)[signal_name]
            target_signal = self._signal_from_one_scale_to_another(
                source_signal=source_signal,
                source_ratio=source_ratio,
                target_ratio=target_ratio,
                fine_to_coarse_policy=fine_to_coarse_policy,
                coarse_to_fine_policy=coarse_to_fine_policy,
            )
            self.at(ratio=target_ratio)[signal_name] = target_signal

    @convert_inputs
    @typecheck
    def _signal_from_one_scale_to_another(
        self,
        source_signal: FloatTensor,
        *,
        source_ratio: Number,
        target_ratio: Number,
        fine_to_coarse_policy: FineToCoarsePolicy,
        coarse_to_fine_policy: CoarseToFinePolicy,
    ) -> FloatTensor:
        """Propagate a signal from one scale to another."""
        if source_ratio == target_ratio:
            return source_signal
        elif source_ratio > target_ratio:
            # propagate from fine to coarse
            reduce = fine_to_coarse_policy.reduce
            return scatter(
                src=source_signal,
                index=self.indice_mapping(
                    fine_ratio=source_ratio, coarse_ratio=target_ratio
                ),
                reduce=reduce,
            )

        else:
            # propagate from coarse to fine
            smoothing = coarse_to_fine_policy.smoothing
            n_smoothing_steps = coarse_to_fine_policy.n_smoothing_steps

            fine_ratio_signal = torch.index_select(
                source_signal,
                dim=0,
                index=self.indice_mapping(
                    fine_ratio=target_ratio, coarse_ratio=source_ratio
                ),
            )

            if smoothing == "constant":
                return fine_ratio_signal
            elif smoothing == "mesh_convolution":
                convolution = self.at(ratio=target_ratio).mesh_convolution()
                for _ in range(n_smoothing_steps):
                    fine_ratio_signal = convolution @ fine_ratio_signal
            else:
                raise NotImplementedError(
                    "This function is not implemented yet"
                )

            return fine_ratio_signal

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
