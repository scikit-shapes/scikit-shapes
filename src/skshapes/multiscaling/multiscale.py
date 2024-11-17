"""Generic Multiscale class."""

from __future__ import annotations

import torch

from ..decimation import Decimation
from ..errors import NotFittedError
from ..input_validation import convert_inputs, one_and_only_one, typecheck
from ..types import (
    CoarseToFinePolicy,
    FineToCoarsePolicy,
    FloatTensor,
    Int1dTensor,
    IntSequence,
    Number,
    NumberSequence,
    shape_type,
)
from ..utils import scatter


class Multiscale:
    """Multiscale representation of a shape.

    This class represents a shape using multiple levels of details,
    from coarse to fine scales.
    This is useful to speed-up computations such as multigrid simulations or level-of-detail rendering.
    This class takes as input a high-resolution shape and accepts three types of parameters to define the coarser representations in the hierarchy:

    - The number of points ``n_points`` of each coarser scale.
    - The typical ``scale`` or distance between two neighboring samples at each coarser scale.
    - The ``ratio`` of the number of points between the origin scale and each coarser scale. A ratio of 1.0 corresponds to the original high-resolution shape, whereas a ratio of 0.2 corresponds to a coarse representation that only uses 20% of the original point count.


    Most of the methods of this class can be called with one of the ``n_points``, ``scale`` or ``ratio`` parameters.

    We can retrieve existing coarse models at different scales using the
    :meth:`at<skshapes.multiscaling.multiscale.Multiscale.at>` method and add models at new sampling resolutions using the
    :meth:`append<skshapes.multiscaling.multiscale.Multiscale.append>` method.

    Signals (:attr:`~skshapes.data.polydata.PolyData.point_data`) defined at one scale can be propagated to the other scales using the :meth:`~skshapes.multiscaling.multiscale.Multiscale.propagate` method.
    This is illustrated in :ref:`this tutorial <multiscale_signal_propagation_example>`.

    Likewise, if landmarks are defined on the original shape, they are propagated to the coarser scales as illustrated in :ref:`this tutorial <multiscale_landmarks_example>`.


    Parameters
    ----------
    shape
        The shape at the original, finest scale.
    ratios
        The sampling ratios that describe the coarser scales.
    n_points
        The target numbers of points for each coarser scale.
    scales
        The sampling scales that describe the coarser scales.
    decimation_module
        The decimation module that should be used to compute the coarser representations of the shape. If not provided, it is defined automatically.

    Raises
    ------
    NotImplementedError
        If the shape is not a triangle mesh.
    ValueError
        If none of the ``ratios``, ``n_points`` or ``scales`` parameters are
        provided or if more than one of these parameters are provided.

    Examples
    --------

    .. testcode::

        import skshapes as sks

        shape = sks.Sphere()
        multiscale = sks.Multiscale(shape=shape, ratios=[0.5, 0.25])
        # Note that the original shape is always included in the multiscale representation
        # for a sampling ratio = 1.0
        for ratio in multiscale.ratios:
            print(f"at ratio={ratio}, {multiscale.at(ratio=ratio).n_points} points")

    .. testoutput::

        at ratio=1.0, 842 points
        at ratio=0.5, 421 points
        at ratio=0.25, 210 points

    See the :ref:`gallery <multiscale_examples>` for more examples.
    """

    @one_and_only_one(parameters=["ratios", "n_points", "scales"])
    @typecheck
    def __init__(
        self,
        shape: shape_type,
        ratios: NumberSequence | None = None,
        n_points: IntSequence | None = None,
        scales: NumberSequence | None = None,
        decimation_module=None,
    ) -> None:
        self.shape = shape

        if decimation_module is not None:
            if not hasattr(decimation_module, "ref_mesh_"):
                msg = "The decimation module has not been fitted."
                raise NotFittedError(msg)
            self._decimation_module = decimation_module

        if decimation_module is None:
            if shape.is_triangle_mesh():
                min_n_points = 1
                decimation_module = Decimation(n_points=min_n_points)

            else:
                msg = "Only triangle meshes are supported for now"
                raise NotImplementedError(msg)

            decimation_module.fit(shape)
            self._decimation_module = decimation_module

        self.shapes = {}
        self.mappings_from_origin = {}

        # Add the original shape at ratio = 1.0
        self.shapes[1.0] = shape

        if ratios is not None:
            for r in ratios:
                self.append(ratio=float(r))
        elif n_points is not None:
            for n in n_points:
                self.append(n_points=int(n))
        elif scales is not None:
            for s in scales:
                self.append(scale=float(s))

    @one_and_only_one(parameters=["ratio", "n_points", "scale"])
    @typecheck
    def append(
        self,
        *,
        ratio: Number | None = None,
        n_points: int | None = None,
        scale: Number | None = None,
    ) -> None:
        """Appends a new shape to the list of sub-sampled representations of the base shape.

        This function can be called with one of the `ratio`, `n_points` or
        `scale` parameters.

        Parameters
        ----------
        ratio
            The target ratio from the initial number of points.
        n_points
            The target number of points.
        scale
            The target scale.


        Examples
        --------

        .. testcode::

            import skshapes as sks

            shape = sks.Sphere()
            multiscale = sks.Multiscale(shape=shape, ratios=[0.5])
            multiscale.append(n_points=100)
            # N.B.: the original shape is always included in the multiscale representation
            #       for a sampling ratio = 1.0
            for ratio in multiscale.ratios:
                print(f"at ratio={ratio:.3f}, {multiscale.at(ratio=ratio).n_points} points")

        .. testoutput::

            at ratio=1.000, 842 points
            at ratio=0.500, 421 points
            at ratio=0.119, 100 points

        """
        if n_points is not None:
            ratio = n_points / self.shape.n_points
        elif scale is not None:
            msg = "Scales are not implemented yet"
            raise NotImplementedError(msg)

        if ratio in self.shapes:
            # ratio already exists, do nothing
            pass
        else:
            if n_points is not None:
                sampling = dict(n_points_strict=n_points)
            else:
                sampling = dict(ratio=ratio)
            new_shape, index_mapping = self._decimation_module.transform(
                self.shape,
                **sampling,
                return_index_mapping=True,
            )

            self.shapes[ratio] = new_shape
            self.mappings_from_origin[ratio] = index_mapping

    @property
    @typecheck
    def ratios(self) -> NumberSequence:
        """Return the available ratios of the shapes in the multiscale representation.

        The ratios are sorted in decreasing order.

        Examples
        --------

        .. testcode::

            import skshapes as sks

            shape = sks.Sphere()  # The sphere has 842 points
            multiscale = sks.Multiscale(shape=shape, n_points=[400, 100])
            # N.B.: the original shape is always included in the multiscale representation
            #       for a sampling ratio = 1.0
            print(multiscale.ratios)

        .. testoutput::

            [1.0, 0.4750593824228028, 0.1187648456057007]

        """
        return sorted(self.shapes.keys(), reverse=True)

    def best_key(self, *, ratio: Number) -> Number:
        """Return the best key in the shapes dictionary for a given, arbitrary ratio.

        This returns the smallest ratio (= most compact shape) which is still larger
        (= at least as precise) as the required sampling ratio.


        Examples
        --------

        .. testcode::

            import skshapes as sks

            shape = sks.Sphere()
            multiscale = sks.Multiscale(shape=shape, ratios=[0.5, 0.25])
            print(multiscale.best_key(ratio=0.6))
            print(multiscale.best_key(ratio=0.3))

        .. testoutput::

            1.0
            0.5

        """
        # Clamp the ratio to 1, i.e. n_points to shape.n_points
        # If the user asks for too many points, we simply return the raw shape.
        ratio = min(1, ratio)
        # Since ratio is <= 1 and 1 always belongs to self.ratios,
        # this is a non-empty minimization.
        return min(r for r in self.ratios if r >= ratio)

    @one_and_only_one(parameters=["ratio", "n_points", "scale"])
    @typecheck
    def at(
        self,
        *,
        ratio: Number | None = None,
        n_points: int | None = None,
        scale: Number | None = None,
    ) -> shape_type:
        """Returns the shape at a given ratio, number of points or scale.

        If the shape at the given ratio, number of points or scale does not exist,
        returns most compact (= coarsest) shape that is at least as precise.

        Examples
        --------

        .. testcode::

            import skshapes as sks

            shape = sks.Sphere()  # The sphere has 842 points
            multiscale = sks.Multiscale(shape=shape, n_points=[300, 100])
            print(multiscale.at(n_points=200).n_points)

        .. testoutput::

            300

        """
        if n_points is not None:
            ratio = n_points / self.shape.n_points
        elif scale is not None:
            msg = "Scales are not implemented yet"
            raise NotImplementedError(msg)

        return self.shapes[self.best_key(ratio=ratio)]

    @one_and_only_one(parameters=["from_ratio", "from_n_points", "from_scale"])
    @typecheck
    def propagate(
        self,
        signal_name: str,
        from_scale: Number | None = None,
        from_ratio: Number | None = None,
        from_n_points: int | None = None,
        fine_to_coarse_policy: FineToCoarsePolicy | None = None,
        coarse_to_fine_policy: CoarseToFinePolicy | None = None,
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

        Examples
        --------

        See the :ref:`multiscale_signal_propagation_example` tutorial for an example.
        """
        if fine_to_coarse_policy is None:
            fine_to_coarse_policy = FineToCoarsePolicy()

        if coarse_to_fine_policy is None:
            coarse_to_fine_policy = CoarseToFinePolicy()

        available_ratios = self.shapes.keys()
        if from_n_points is not None:
            from_ratio = from_n_points / self.shape.n_points
        elif from_scale is not None:
            msg = "Scales are not implemented yet"
            raise NotImplementedError(msg)

        from_ratio = min(available_ratios, key=lambda x: abs(x - from_ratio))

        if signal_name not in self.shapes[from_ratio].point_data:
            msg = f"{signal_name} not available at ratio {from_ratio}"
            raise KeyError(msg)

        # propagate the signal from the origin to the other scales
        ratio_lower = [r for r in available_ratios if r < from_ratio]
        ratio_higher = [r for r in available_ratios if r > from_ratio]

        ratio_lower = [from_ratio, *sorted(ratio_lower, reverse=True)]
        ratio_higher = [from_ratio, *sorted(ratio_higher)]

        for i in range(len(ratio_lower) - 1):
            source_ratio = ratio_lower[i]
            target_ratio = ratio_lower[i + 1]
            source_signal = self.at(ratio=source_ratio).point_data[signal_name]
            target_signal = self._signal_from_one_scale_to_another(
                source_signal=source_signal,
                source_ratio=source_ratio,
                target_ratio=target_ratio,
                fine_to_coarse_policy=fine_to_coarse_policy,
                coarse_to_fine_policy=coarse_to_fine_policy,
            )
            self.at(ratio=target_ratio).point_data[signal_name] = target_signal

        for i in range(len(ratio_higher) - 1):
            source_ratio = ratio_higher[i]
            target_ratio = ratio_higher[i + 1]
            source_signal = self.at(ratio=source_ratio).point_data[signal_name]
            target_signal = self._signal_from_one_scale_to_another(
                source_signal=source_signal,
                source_ratio=source_ratio,
                target_ratio=target_ratio,
                fine_to_coarse_policy=fine_to_coarse_policy,
                coarse_to_fine_policy=coarse_to_fine_policy,
            )
            self.at(ratio=target_ratio).point_data[signal_name] = target_signal

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

        if source_ratio > target_ratio:
            # propagate from fine to coarse
            reduce = fine_to_coarse_policy.reduce
            return scatter(
                src=source_signal,
                index=self.index_mapping(
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
                index=self.index_mapping(
                    fine_ratio=target_ratio, coarse_ratio=source_ratio
                ),
            )

            if smoothing == "constant":
                return fine_ratio_signal

            if smoothing == "mesh_convolution":
                convolution = self.at(ratio=target_ratio).mesh_convolution()
                for _ in range(n_smoothing_steps):
                    fine_ratio_signal = convolution @ fine_ratio_signal
            else:
                msg = "This function is not implemented yet"
                raise NotImplementedError(msg)

            return fine_ratio_signal

    @typecheck
    def index_mapping(
        self, fine_ratio: Number, coarse_ratio: Number
    ) -> Int1dTensor:
        """Returns the index mapping from a fine to a coarse resolution.

        The index mapping is a 1d tensor of integers whose length is equal to the
        number of points at the fine resolution ratio. Each element of the
        tensor is the index of the corresponding point at the coarse resolution
        ratio. This method is used internally to propagate signals from one
        scale to another.

        Parameters
        ----------
        fine_ratio
            The ratio of the high resolution shape.
        coarse_ratio
            The ratio of the low resolution shape.

        Returns
        -------
        Int1dTensor
            The index mapping from a fine to a coarse resolution.

        Examples
        --------

        .. testcode::

            import skshapes as sks

            shape = sks.Sphere()
            multiscale = sks.Multiscale(shape=shape, ratios=[0.5, 0.25])
            im = multiscale.index_mapping(fine_ratio=0.5, coarse_ratio=0.25)

            print(f"{multiscale.at(ratio=0.5).n_points} = {im.shape},", im.dtype)

        .. testoutput::

            421 = torch.Size([421]), torch.int64

        .. testcode::

            print(f"{multiscale.at(ratio=0.25).n_points} = {im.max()} + 1")

        .. testoutput::

            210 = 209 + 1

        .. testcode::

            print(im[:10])

        .. testoutput::

            tensor([53,  0,  0,  7,  1,  2,  2,  3,  4,  4])

        """
        assert (
            fine_ratio >= coarse_ratio
        ), "fine_ratio must be greater than coarse_ratio"
        assert 0 < coarse_ratio <= 1, "coarse_ratio must be between 0 and 1"
        assert 0 < fine_ratio <= 1, "fine_ratio must be between 0 and 1"

        fine_ratio = self.best_key(ratio=fine_ratio)
        coarse_ratio = self.best_key(ratio=coarse_ratio)

        if fine_ratio == coarse_ratio:
            return torch.arange(self.shapes[fine_ratio].n_points)

        if fine_ratio == 1:
            return self.mappings_from_origin[coarse_ratio]

        tmp = self.mappings_from_origin[fine_ratio]
        tmp = scatter(src=torch.arange(len(tmp)), index=tmp, reduce="min")
        return self.mappings_from_origin[coarse_ratio][tmp]

    @typecheck
    def __len__(self) -> int:
        """Returns the number of scales that are stored internally.

        Note that the original shape is always included in the multiscale representation
        for a sampling ratio = 1.0.

        Examples
        --------

        .. testcode::

            import skshapes as sks

            shape = sks.Sphere()
            multiscale = sks.Multiscale(shape=shape, ratios=[0.5, 0.25])
            print(multiscale.ratios, len(multiscale))

        .. testoutput::

            [1.0, 0.5, 0.25] 3

        """
        return len(self.shapes)
