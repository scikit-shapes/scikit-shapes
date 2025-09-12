from __future__ import annotations

from dataclasses import dataclass

from ..input_validation import typecheck
from ..linear_operators import LinearOperator
from ..types import (
    Eigenvalues,
    Literal,
    PointEigenvectors,
)


@typecheck
@dataclass
class Spectrum:
    eigenvectors: PointEigenvectors
    laplacian_eigenvalues: Eigenvalues
    smoothing_eigenvalues: Eigenvalues
    diffusion_eigenvalues: Eigenvalues

    @staticmethod
    @typecheck
    def from_metric(
        *,
        n_modes: int,
        mass: LinearOperator,
        metric: LinearOperator,
        diffusion_method: Literal["exponential", "implicit euler"],
    ) -> Spectrum:
        eigendecomposition = metric.eigendecomposition(
            mass=mass, mode="smallest magnitude", n_modes=n_modes
        )
        eigenvectors = eigendecomposition.eigenvectors
        assert eigenvectors.shape == (
            n_modes,
            metric.n_points,
            metric.n_features,
        )

        # Our convention is that the Laplacian is positive semi-definite.
        # This is common in geometry processing, but opposite to the convention
        # in physics.
        laplacian_eigenvalues = eigendecomposition.eigenvalues
        assert (laplacian_eigenvalues >= 0).all()
        assert laplacian_eigenvalues.shape == (n_modes,)

        # The smoothing operator is the inverse of the Laplacian.
        smoothing_eigenvalues = 1 / laplacian_eigenvalues
        assert smoothing_eigenvalues.shape == (n_modes,)

        # We support different implementations of the diffusion operator.
        if diffusion_method == "exponential":
            diffusion_eigenvalues = (-laplacian_eigenvalues).exp()
        elif diffusion_method == "implicit euler":
            diffusion_eigenvalues = 1 / (1 + laplacian_eigenvalues)
        assert diffusion_eigenvalues.shape == (n_modes,)

        return Spectrum(
            eigenvectors=eigenvectors,
            laplacian_eigenvalues=laplacian_eigenvalues,
            smoothing_eigenvalues=smoothing_eigenvalues,
            diffusion_eigenvalues=diffusion_eigenvalues,
        )

    @staticmethod
    @typecheck
    def from_cometric(
        *,
        n_modes: int,
        mass: LinearOperator,
        cometric: LinearOperator,
        diffusion_method: Literal["exponential", "implicit euler"],
    ) -> Spectrum:
        eigendecomposition = cometric.eigendecomposition(
            mass=mass, mode="largest magnitude", n_modes=n_modes
        )
        eigenvectors = eigendecomposition.eigenvectors
        assert eigenvectors.shape == (
            n_modes,
            cometric.n_points,
            cometric.n_features,
        )

        smoothing_eigenvalues = eigendecomposition.eigenvalues
        assert (smoothing_eigenvalues >= 0).all()
        assert smoothing_eigenvalues.shape == (n_modes,)

        # The smoothing operator is the inverse of the Laplacian.
        # Our convention is that the Laplacian is positive semi-definite.
        # This is common in geometry processing, but opposite to the convention
        # in physics.
        laplacian_eigenvalues = 1 / smoothing_eigenvalues
        assert laplacian_eigenvalues.shape == (n_modes,)

        # We support different implementations of the diffusion operator.
        if diffusion_method == "exponential":
            diffusion_eigenvalues = (-laplacian_eigenvalues).exp()
        elif diffusion_method == "implicit euler":
            diffusion_eigenvalues = smoothing_eigenvalues / (
                1 + smoothing_eigenvalues
            )
        assert diffusion_eigenvalues.shape == (n_modes,)

        return Spectrum(
            eigenvectors=eigenvectors,
            laplacian_eigenvalues=laplacian_eigenvalues,
            smoothing_eigenvalues=smoothing_eigenvalues,
            diffusion_eigenvalues=diffusion_eigenvalues,
        )
