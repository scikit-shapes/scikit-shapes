from ..input_validation import typecheck
from ..types import (
    PointMasses,
    PointVectorSignals,
)
from .neighborhoods import Neighborhoods


class TrivialNeighborhoods(Neighborhoods):
    def __init__(self, masses: PointMasses):
        super().__init__(
            masses=masses,
            scale=None,
            n_normalization_iterations=1,
            smoothing_method="auto",
            laplacian_method="auto",
        )
        self._compute_scaling()

    @typecheck
    def _smooth_without_scaling(
        self, signal: PointVectorSignals
    ) -> PointVectorSignals:
        assert signal.shape[0] == self.n_points
        return signal
