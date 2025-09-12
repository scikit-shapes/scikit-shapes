from __future__ import annotations

import numpy as np
import scipy
import torch

from ..input_validation import typecheck
from ..types import Callable, Signal


class LinearOperator:
    """General square matrix.

    This default implementation supports arbitrary linear operators and relies extensively
    on SciPy. Subclasses provide better performance for specific types of operators,
    such as sparse matrices or convolutional operators.
    """

    @typecheck
    def __init__(
        self,
        *,
        n_points: int,
        n_features: int,
        device: torch.device,
        matvec: Callable[[Signal], Signal] | None = None,
        rmatvec: Callable[[Signal], Signal] | None = None,
        scipy_solver: Callable[
            [scipy.sparse.linalg.LinearOperator, np.ndarray],
            tuple[np.ndarray, int],
        ] = None,
        symmetric: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:

        if symmetric:
            if rmatvec is not None:
                msg = "A symmetric operator cannot have a different rmatvec."
                raise ValueError(msg)
            rmatvec = matvec

        # User-provided matvec and rmatvec.
        # They are used by the default matvec implementation,
        # but may not be used by subclasses.
        self._matvec = matvec
        self._rmatvec = rmatvec

        if scipy_solver is None:
            scipy_solver = (
                scipy.sparse.linalg.cg
                if symmetric
                else scipy.sparse.linalg.bicg
            )

        self.scipy_solver = scipy_solver
        self.n_points = n_points
        self.n_features = n_features
        self.device = device
        self.dtype = dtype
        self.is_symmetric = symmetric

    @typecheck
    def __matmul__(self, x: Signal) -> Signal:
        """Matrix-vector product ``A @ x``.

        This method handles the device and dtype conversions for the underlying
        numerical routine implemented in ``self._operator``.
        """
        assert x.shape == (self.n_points, self.n_features)
        out = self.matvec(x.to(dtype=self.dtype, device=self.device))
        assert out.shape == (self.n_points, self.n_features)
        # Preserve the dtype and device of the input
        return out.to(dtype=x.dtype, device=x.device)

    @typecheck
    def matvec(self, x: Signal) -> Signal:
        """Matrix-vector product ``A @ x``."""
        return self._matvec(x)

    @typecheck
    def t(self) -> LinearOperator:
        """Transpose of the operator."""
        return LinearOperator(
            matvec=self._rmatvec,
            rmatvec=self._matvec,
            n_points=self.n_points,
            n_features=self.n_features,
            device=self.device,
            dtype=self.dtype,
            symmetric=self.is_symmetric,
        )

    @typecheck
    def _scipy_matvec(self, v: np.ndarray) -> np.ndarray:
        """Wraps the underlying operator for scipy compatibility.

        :meth:`self._matvec` works on ``(N, F)`` torch tensors,
        while scipy works on flattened ``(N * F,)`` numpy arrays.
        """
        N = self.n_points
        F = self.n_features
        v = torch.from_numpy(v).reshape((N, F))
        result = self @ v
        return result.reshape((N * F,)).numpy()

    @typecheck
    def to_scipy(self) -> scipy.sparse.linalg.LinearOperator:
        """Convert to a scipy sparse linear operator."""
        N = self.n_points
        F = self.n_features

        return scipy.sparse.linalg.LinearOperator(
            shape=(N * F, N * F),
            matvec=self._scipy_matvec,
            rmatvec=self.t()._scipy_matvec,
            dtype=self.dtype,
        )

    @staticmethod
    def from_scipy(
        *,
        operator: scipy.sparse.linalg.LinearOperator,
        n_points: int,
        n_features: int,
        symmetric: bool,
    ) -> LinearOperator:
        """Create a :class:`LinearOperator` from a scipy linear operator."""
        assert operator.shape == (n_points * n_features, n_points * n_features)

        def matvec(x: torch.Tensor) -> torch.Tensor:
            assert x.shape == (n_points, n_features)
            x_flat = x.reshape((n_points * n_features,)).numpy()
            y_flat = operator @ x_flat
            y = torch.from_numpy(y_flat).to(dtype=x.dtype, device=x.device)
            return y.reshape((n_points, n_features))

        def rmatvec(y: torch.Tensor) -> torch.Tensor:
            assert y.shape == (n_points, n_features)
            y_flat = y.reshape((n_points * n_features,)).numpy()
            x_flat = operator.rmatvec(y_flat)
            x = torch.from_numpy(x_flat).to(dtype=y.dtype, device=y.device)
            return x.reshape((n_points, n_features))

        return LinearOperator(
            matvec=matvec,
            rmatvec=None if symmetric else rmatvec,
            n_points=n_points,
            n_features=n_features,
            device=torch.device("cpu"),
            dtype=operator.dtype,
            symmetric=symmetric,
        )

    @typecheck
    def inverse(self) -> LinearOperator:
        N = self.n_points
        F = self.n_features
        scipy_operator = self.to_scipy()

        scipy_inv = scipy.sparse.linalg.LinearOperator(
            shape=(N * F, N * F),
            matvec=lambda b: self.scipy_solver(scipy_operator, b)[0],
            rmatvec=lambda b: self.scipy_solver(scipy_operator.T, b)[0],
            dtype=self.dtype,
        )

        return LinearOperator.from_scipy(
            operator=scipy_inv,
            n_points=self.n_points,
            n_features=self.n_features,
            symmetric=self.is_symmetric,
        )
