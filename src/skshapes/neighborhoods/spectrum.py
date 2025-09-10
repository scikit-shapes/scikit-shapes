from dataclasses import dataclass

from ..input_validation import typecheck
from ..linear_operators import LinearOperator
from ..types import (
    Eigenvalues,
    Function,
    Literal,
    Measure,
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
        n_components: int,
        mass: LinearOperator[Function, Measure],
        metric: LinearOperator[Function, Measure],
        diffusion_method: Literal[
            "exponential", "implicit euler", "truncated"
        ],
    ) -> "Spectrum":
        eigendecomposition = metric.eigendecomposition(
            mass=mass, mode="smallest magnitude", n_components=n_components
        )
        eigenvectors = eigendecomposition.eigenvectors
        assert eigenvectors.shape == (
            n_components,
            metric.n_points,
            metric.n_features,
        )

        # Our convention is that the Laplacian is positive semi-definite.
        # This is common in geometry processing, but opposite to the convention
        # in physics.
        laplacian_eigenvalues = eigendecomposition.eigenvalues
        assert (laplacian_eigenvalues >= 0).all()
        assert laplacian_eigenvalues.shape == (n_components,)

        # The smoothing operator is the inverse of the Laplacian.
        smoothing_eigenvalues = 1 / laplacian_eigenvalues
        assert smoothing_eigenvalues.shape == (n_components,)

        # We support different implementations of the diffusion operator.
        if diffusion_method == "exponential":
            diffusion_eigenvalues = (-laplacian_eigenvalues).exp()
        elif diffusion_method == "implicit euler":
            diffusion_eigenvalues = 1 / (1 + laplacian_eigenvalues)
        else:
            msg = f"Diffusion method '{diffusion_method}' is not implemented."
            raise NotImplementedError(msg)
        assert diffusion_eigenvalues.shape == (n_components,)

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
        n_components: int,
        mass: LinearOperator[Function, Measure],
        cometric: LinearOperator[Measure, Function],
        diffusion_method: Literal[
            "exponential", "implicit euler", "truncated"
        ],
    ) -> "Spectrum":
        eigendecomposition = cometric.eigendecomposition(
            mass=mass, mode="largest magnitude", n_components=n_components
        )
        eigenvectors = eigendecomposition.eigenvectors
        assert eigenvectors.shape == (
            n_components,
            cometric.n_points,
            cometric.n_features,
        )

        smoothing_eigenvalues = eigendecomposition.eigenvalues
        assert (smoothing_eigenvalues >= 0).all()
        assert smoothing_eigenvalues.shape == (n_components,)

        # The smoothing operator is the inverse of the Laplacian.
        # Our convention is that the Laplacian is positive semi-definite.
        # This is common in geometry processing, but opposite to the convention
        # in physics.
        laplacian_eigenvalues = 1 / smoothing_eigenvalues
        assert laplacian_eigenvalues.shape == (n_components,)

        # We support different implementations of the diffusion operator.
        if diffusion_method == "exponential":
            diffusion_eigenvalues = (-laplacian_eigenvalues).exp()
        elif diffusion_method == "implicit euler":
            diffusion_eigenvalues = smoothing_eigenvalues / (
                1 + smoothing_eigenvalues
            )
        else:
            msg = f"Diffusion method '{diffusion_method}' is not implemented."
            raise NotImplementedError(msg)
        assert diffusion_eigenvalues.shape == (n_components,)

        return Spectrum(
            eigenvectors=eigenvectors,
            laplacian_eigenvalues=laplacian_eigenvalues,
            smoothing_eigenvalues=smoothing_eigenvalues,
            diffusion_eigenvalues=diffusion_eigenvalues,
        )
