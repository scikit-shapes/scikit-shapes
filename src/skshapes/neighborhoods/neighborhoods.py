from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Literal

import numpy as np
import scipy
import scipy.sparse
import torch

from ..input_validation import convert_inputs, typecheck
from ..types import (
    Eigenvalues,
    Number,
    PointAnySignals,
    PointEigenvectors,
    PointMasses,
    PointSymmetricTensors,
    PointVectorSignals,
    neighborhoods_type,
    torch_to_np_dtypes,
)


def symmetric_function_to_linear_operator(
    *, function, n_points, dtype, device
):
    def wrapped_f(x):
        x_torch = torch.tensor(x, dtype=dtype, device=device)
        return function(x_torch).detach().cpu().numpy()

    return scipy.sparse.linalg.LinearOperator(
        shape=(n_points, n_points),
        matvec=wrapped_f,
        matmat=wrapped_f,
        rmatvec=wrapped_f,
        rmatmat=wrapped_f,
        dtype=torch_to_np_dtypes[dtype],
    )


@typecheck
@dataclass
class Spectrum:
    eigenvectors: PointEigenvectors
    laplacian_eigenvalues: Eigenvalues
    smoothing_eigenvalues: Eigenvalues


class Neighborhoods(neighborhoods_type):
    @typecheck
    def __init__(
        self,
        masses: PointMasses,
        scale: Number | None = None,
        n_normalization_iterations: int | None = None,
        smoothing_method: Literal[
            "auto", "exact", "exp(x)=1/(1-x)", "nystroem"
        ] = "auto",
        laplacian_method: Literal["auto", "exact", "log(x)=x-1"] = "auto",
    ):
        self.masses = masses
        self.n_points = self.masses.shape[0]
        self.device = self.masses.device
        self.dtype = self.masses.dtype
        self.scale = scale
        self.n_normalization_iterations = n_normalization_iterations
        self.smoothing_method = smoothing_method
        self.laplacian_method = laplacian_method

    def _compute_scaling(self):
        if self.n_normalization_iterations is None:
            if hasattr(self, "_laplacian"):
                self.n_normalization_iterations = 0
            else:
                self.n_normalization_iterations = 10

        assert self.masses.shape == (self.n_points,)
        self.scaling = torch.ones_like(self.masses).view(-1, 1)

        # _smooth_without_scaling expects and returns signals with shape (n_points, 1)
        for _ in range(self.n_normalization_iterations):
            denom = self._smooth_without_scaling(
                self.scaling * self.masses.view(-1, 1)
            )
            self.scaling = (self.scaling / denom).sqrt()

        self.scaling = self.scaling.view(-1)
        assert self.scaling.shape == (self.n_points,)

        # self.smooth_1 should be close to a vector of ones
        self.smooth_1 = self.smooth(
            torch.ones_like(self.masses),
            input_type="function",
            output_type="function",
        )
        assert self.smooth_1.shape == (self.n_points,)

    @typecheck
    def spectrum(
        self, n_modes: int, check_tolerance: Number = 1e-4
    ) -> Spectrum:
        # Wrap self.smooth as a SciPy LinearOperator
        def smooth_function_to_measure(f):
            return self.smooth(f, input_type="function", output_type="measure")

        smooth_operator = symmetric_function_to_linear_operator(
            function=smooth_function_to_measure,
            n_points=self.n_points,
            dtype=self.dtype,
            device=self.device,
        )
        # Compute the largest n_modes eigenmodes with mass matrix diag(masses):
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            smooth_operator,
            k=n_modes,
            which="LM",
            M=scipy.sparse.diags(self.masses.detach().cpu().numpy()),
            Minv=scipy.sparse.diags(1 / self.masses.detach().cpu().numpy()),
        )
        # Sort the eigenvalues and eigenvectors in ascending order
        eigenvalues = np.ascontiguousarray(eigenvalues[::-1])
        eigenvectors = np.ascontiguousarray(eigenvectors[:, ::-1])

        # Wrap eigenvalues and coordinates are torch tensors
        eigenvalues = torch.tensor(
            eigenvalues, dtype=self.dtype, device=self.device
        )
        eigenvectors = torch.tensor(
            eigenvectors, dtype=self.dtype, device=self.device
        )

        assert eigenvalues.shape == (n_modes,)
        assert eigenvectors.shape == (self.n_points, n_modes)

        # Make sure that the eigenvalues of smooth are all in [0, 1]
        assert (eigenvalues >= -check_tolerance).all(), eigenvalues
        assert (eigenvalues <= 1 + check_tolerance).all(), eigenvalues

        eigenvalues = eigenvalues.clamp(min=0, max=1)

        # Compute the Laplacian eigenvalues.
        # Note that we follow "mathematical" conventions, where the Laplacian
        # eigenvalues are negative as in "-omega^2".
        if self.laplacian_method in ["auto", "log(x)=x-1"]:
            laplacian_eigenvalues = eigenvalues - 1
        elif self.laplacian_method == "exact":
            laplacian_eigenvalues = eigenvalues.log()
        else:
            msg = f"Unsupported laplacian method: {self.laplacian_method}"
            raise NotImplementedError(msg)

        assert laplacian_eigenvalues.shape == (n_modes,)
        assert (
            laplacian_eigenvalues <= check_tolerance
        ).all(), laplacian_eigenvalues

        return Spectrum(
            eigenvectors=eigenvectors,
            laplacian_eigenvalues=laplacian_eigenvalues,
            smoothing_eigenvalues=eigenvalues,
        )

    @convert_inputs
    @typecheck
    def smooth(
        self,
        signal: PointAnySignals,
        input_type: Literal["function", "measure"] = "function",
        output_type: Literal["function", "measure"] = "function",
    ) -> PointAnySignals:
        if signal.shape[0] != self.n_points:
            msg = f"Expected signal to have shape ({self.n_points}, *), but got {signal.shape}."
            raise ValueError(msg)

        signal_shape = signal.shape
        signal = signal.to(self.device).view(self.n_points, -1)

        smooth_signal = self._smooth(
            signal, input_type=input_type, output_type=output_type
        )

        smooth_signal = smooth_signal.view(signal_shape)
        assert smooth_signal.shape == signal_shape
        assert smooth_signal.device == signal.device
        assert smooth_signal.dtype == signal.dtype

        return smooth_signal

    @convert_inputs
    @typecheck
    def smooth_symmetric(
        self,
        signal: PointSymmetricTensors,
        input_type: Literal["function", "measure"] = "function",
        output_type: Literal["function", "measure"] = "function",
        skew_symmetric_channels: list[bool] | None = None,
    ) -> PointSymmetricTensors:
        """



        Examples
        --------

        .. testcode::

            import skshapes as sks

            # Create three points in the plane R^2
            shape = sks.PolyData([[0, 0], [1, 0], [2, 0]])
            # Create a (n_points, 2, 2) collection of symmetric (2, 2) matrices
            s = [
                [[0, 2], [2, 2]],
                [[3, 0], [0, 1]],
                [[0, 0], [0, 5]],
            ]
            # Default smoothing of the 4 channels per point:
            smooth_ref = shape.point_neighborhoods(scale=0.5).smooth(s)
            # Faster smoothing that relies on the symmetry of the matrices:
            smooth_sym = shape.point_neighborhoods(scale=0.5).smooth_symmetric(s)

            print(smooth_sym.shape, smooth_sym.dtype)
            print((smooth_ref == smooth_sym).all())

        .. testoutput::

            torch.Size([3, 2, 2]) torch.float32
            tensor(True)

        .. testcode::

            import torch

            # Generate a simple 3D shape
            shape = sks.PolyData(torch.randn(10, 3))

            # smooth_symmetric also works with trailing "channel" dimensions
            s = torch.randn(10 * 3 * 3 * 5).view(10, 3, 3, 5)
            # We just make sure that s is symmetric with respect to dims 1 and 2
            s = 0.5 * (s + s.permute(0, 2, 1, 3))

            # Default smoothing of the 3*3*5 channels per point:
            smooth_ref = shape.point_neighborhoods(scale=0.2).smooth(s)
            # Faster smoothing that relies on the symmetry of the matrices:
            smooth_sym = shape.point_neighborhoods(scale=0.2).smooth_symmetric(s)

            print(smooth_sym.shape, smooth_sym.dtype)
            print((smooth_ref == smooth_sym).all())

        .. testoutput::

            torch.Size([10, 3, 3, 5]) torch.float32
            tensor(True)

        """
        N = self.n_points
        D = signal.shape[1]
        T = (D * (D + 1)) // 2
        trailing_dims = signal.shape[3:]
        # The number of channels is the product of the trailing dimensions
        C = reduce(mul, trailing_dims, 1)  # = 1 if trailing_dims is empty

        if signal.shape[0] != self.n_points:
            msg = f"Expected signal to have shape ({self.n_points}, ...), but got {signal.shape}."
            raise ValueError(msg)

        # @typecheck should handle ValueErrors for the assertion below
        assert signal.shape == (N, D, D, *trailing_dims)

        # For the sake of simplicity, reshape signal as a (N, D, D, C) tensor
        input_shape = signal.shape  # Save this for the final assertion
        signal = signal.view(N, D, D, C)

        if skew_symmetric_channels is None:
            skew_symmetric_channels = [False] * C

        symmetric_channels = [not skew for skew in skew_symmetric_channels]

        transpose_dims = [0, 2, 1, 3]
        sym_signal = signal.permute(transpose_dims)
        if not (
            sym_signal[..., symmetric_channels]
            == signal[..., symmetric_channels]
        ).all():
            msg = 'The input channels are not symmetric. Please consider using the "smooth()" method instead of "smooth_symmetric()".'
            raise ValueError(msg)

        if not (
            sym_signal[..., skew_symmetric_channels]
            == -signal[..., skew_symmetric_channels]
        ).all():
            msg = "The input channels are not skew-symmetric."
            raise ValueError(msg)

        if D == 1:
            tri_upper_mask = torch.tensor([[True]])
            sym_from_tri = torch.tensor([0])
        elif D == 2:
            tri_upper_mask = torch.tensor([[True, True], [False, True]])
            # [0, 1]
            # [1, 2]
            sym_from_tri = torch.tensor([0, 1, 1, 2])
        elif D == 3:
            tri_upper_mask = torch.tensor(
                [[True, True, True], [False, True, True], [False, False, True]]
            )
            # [0, 1, 2]
            # [1, 3, 4]
            # [2, 4, 5]
            sym_from_tri = torch.tensor([0, 1, 2, 1, 3, 4, 2, 4, 5])
        else:
            msg = "Symmetric smoothing is not implemented beyond shape (3, 3)"
            raise NotImplementedError(msg)

        tri_upper_mask = tri_upper_mask.to(self.device)
        sym_from_tri = sym_from_tri.to(self.device)
        tri_lower_mask = ~tri_upper_mask

        signal = signal.to(self.device)
        signal = signal[:, tri_upper_mask, :]
        assert signal.shape == (N, T, C)

        smooth_signal = self.smooth(
            signal, input_type=input_type, output_type=output_type
        )
        assert smooth_signal.shape == (N, T, C)

        smooth_signal = smooth_signal[:, sym_from_tri].view(N, D, D, C)

        # Apply a -1 factor to the skew-symmetric channels in the lower triangle.
        # NB: *= -1 is a in-place, non-differentiable operation so we use "-1 *" instead.
        if any(skew_symmetric_channels):
            smooth_signal[..., tri_lower_mask, skew_symmetric_channels] = (
                -smooth_signal[..., tri_lower_mask, skew_symmetric_channels]
            )

        # Check the (skew-)symmetry of the output
        sym_signal = smooth_signal.permute(transpose_dims)
        assert (
            sym_signal[..., symmetric_channels]
            == smooth_signal[..., symmetric_channels]
        ).all()
        assert (
            sym_signal[..., skew_symmetric_channels]
            == -smooth_signal[..., skew_symmetric_channels]
        ).all()

        smooth_signal = smooth_signal.view(N, D, D, *trailing_dims)
        assert smooth_signal.device == signal.device
        assert smooth_signal.dtype == signal.dtype
        assert smooth_signal.shape == input_shape

        return smooth_signal

    def _smooth(
        self,
        signal: PointVectorSignals,
        input_type: Literal["function", "measure"],
        output_type: Literal["function", "measure"],
    ) -> PointVectorSignals:
        assert signal.shape[0] == self.n_points
        assert signal.ndim == 2
        assert self.scaling.shape == (self.n_points,)
        assert self.masses.shape == (self.n_points,)

        if input_type == "measure":
            scaled_signal = self.scaling.view(-1, 1) * signal
        elif input_type == "function":
            scaled_signal = (self.scaling * self.masses).view(-1, 1) * signal

        smooth_signal = self._smooth_without_scaling(scaled_signal)

        if output_type == "measure":
            smooth_signal = (self.scaling * self.masses).view(
                -1, 1
            ) * smooth_signal
        elif output_type == "function":
            smooth_signal = (self.scaling).view(-1, 1) * smooth_signal

        assert smooth_signal.shape == signal.shape
        assert smooth_signal.device == signal.device
        assert smooth_signal.dtype == signal.dtype

        return smooth_signal

    def _smooth_without_scaling(
        self, signal: PointVectorSignals
    ) -> PointVectorSignals:
        if self.smoothing_method in ["exp(x)=1/(1-x)"]:
            # Implements a step of implicit Euler integration for the heat equation.
            # This corresponds to applying a linear operator with the same
            # eigenvectors as the Laplacian, but with eigenvalues lambda_i <= 0
            # replaced by 1/(1 - lambda_i) (~= exp(lambda_i) for negative lambda_i).
            msg = f"{signal.shape}"
            raise NotImplementedError(msg)
            # TODO: something like this, but with fast factorization for sparse
            # laplacians
            # smooth_signal = self._identity_minus_smooth_operator.solve(signal)

        msg = f"Unsupported smoothing method: {self.smoothing_method}."
        raise NotImplementedError(msg)

    @convert_inputs
    @typecheck
    def laplacian(
        self,
        signal: PointAnySignals,
        input_type: Literal["function", "measure"] = "function",
        output_type: Literal["function", "measure"] = "function",
    ) -> PointAnySignals:
        if signal.shape[0] != self.n_points:
            msg = f"Expected signal to have shape ({self.n_points}, *), but got {signal.shape}."
            raise ValueError(msg)

        signal_shape = signal.shape
        signal = signal.to(self.device).view(self.n_points, -1)

        if hasattr(self, "_laplacian"):
            laplacian_signal = self._laplacian(signal)
        elif self.laplacian_method in ["auto", "log(x)=x-1"]:
            smooth_signal = self.smooth(
                signal,
                input_type=input_type,
                output_type=output_type,
            )
            if input_type == output_type:
                identity_signal = signal
            elif input_type == "function" and output_type == "measure":
                identity_signal = signal * self.masses.view(-1, 1)
            elif input_type == "measure" and output_type == "function":
                identity_signal = signal / self.masses.view(-1, 1)
            else:
                msg = f"Unsupported input -> output types: {input_type} -> {output_type}"
                raise NotImplementedError(msg)
            laplacian_signal = smooth_signal - identity_signal
        else:
            msg = f"Unsupported laplacian method: {self.laplacian_method}"
            raise NotImplementedError(msg)

        laplacian_signal = laplacian_signal.view(signal_shape)
        assert laplacian_signal.shape == signal.shape
        assert laplacian_signal.device == signal.device
        assert laplacian_signal.dtype == signal.dtype
        return laplacian_signal
