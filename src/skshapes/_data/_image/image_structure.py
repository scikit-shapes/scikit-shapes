from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import pyvista
import torch

from ...input_validation import convert_inputs, one_and_only_one, typecheck
from ...types import IntTensor, Number, NumericalTensor
from ._indices_management import (
    apply_at_indices,
    get_at_indices,
    indices_difference,
    set_at_indices,
)
from .grid_structure import GridStructure

if TYPE_CHECKING:
    from . import Mask


class UniqueOutput(NamedTuple):
    output: torch.Tensor
    counts: torch.Tensor | None = None


@typecheck
def _apply_pointwise_unary_operation(
    operation: callable, *, image: ImageStructure, **kwargs
) -> ImageStructure | Mask:
    from . import Image, Mask, SparseImage

    if isinstance(image, Image):
        new_values = operation(image.values, **kwargs)

        if new_values.dtype == torch.bool:
            return Mask(indices=torch.argwhere(new_values), shape=image.shape)
        else:
            return Image(values=new_values)

    elif isinstance(image, SparseImage):
        new_values = operation(image._flat_values, **kwargs)
        new_constant = operation(image._constant[None, ...], **kwargs).squeeze(
            0
        )

        if new_values.dtype == torch.bool:
            if new_constant:
                flat_indices = torch.arange(
                    image.numel, dtype=torch.int64, device=image.device
                )
                flat_indices = indices_difference(
                    flat_indices, image._flat_indices[~new_values]
                )
            else:
                flat_indices = image._flat_indices[new_values]

            return Mask(flat_indices=flat_indices, shape=image.shape)
        else:
            return SparseImage(
                flat_indices=image._flat_indices,
                values=new_values,
                constant=new_constant,
                shape=image.shape,
            )
    else:
        msg = "Wrong image type for _apply_pointwise_unary_operation."
        raise TypeError(msg)


@typecheck
def _apply_pointwise_binary_operation(
    operation: callable,
    *,
    image1: ImageStructure,
    image2: ImageStructure | Number | torch.Tensor,
    **kwargs,
) -> ImageStructure | Mask:
    """
    (Image, Image) -> Image
    (Image, SparseImage) -> Image
    (SparseImage, Image) -> Image
    (SparseImage, SparseImage) -> SparseImage

    Parameters
    ----------
    operation
    image1
    image2
    kwargs

    Returns
    -------

    """
    from . import Image, Mask, SparseImage

    if isinstance(image2, Number | torch.Tensor):
        return _apply_pointwise_unary_operation(
            lambda u: operation(u, image2, **kwargs), image=image1
        )

    elif isinstance(image1, Image) and isinstance(image2, Image):
        new_values = operation(image1.values, image2.values, **kwargs)
        if new_values.dtype == torch.bool:
            return Mask(indices=torch.argwhere(new_values), shape=image1.shape)
        else:
            return Image(values=new_values)

    elif isinstance(image1, Image) and isinstance(image2, SparseImage):
        # TODO vérifier
        flat_indices2, flat_values2 = image2._flat_indices, image2._flat_values
        flat_values1 = get_at_indices(
            grid=image1.values, flat_indices=flat_indices2
        )

        flat_values_new = operation(flat_values1, flat_values2, **kwargs)
        new_values = operation(
            image1.values, image2._constant[None, ...], **kwargs
        )
        new_values = set_at_indices(
            grid=new_values,
            flat_indices=flat_indices2,
            new_values=flat_values_new,
        )

        if new_values.dtype == torch.bool:
            return Mask(indices=torch.argwhere(new_values), shape=image1.shape)
        else:
            return Image(values=new_values)

    elif isinstance(image1, Image) and isinstance(image2, SparseImage):
        return _apply_pointwise_binary_operation(
            lambda u, v: operation(v, u, **kwargs),
            image1=image2,
            image2=image1,
        )

    elif isinstance(image1, SparseImage) and isinstance(image2, SparseImage):
        union_indices, union_values, new_constant = apply_at_indices(
            operation=operation,
            flat_indices1=image1._flat_indices,
            flat_indices2=image2._flat_indices,
            flat_values1=image1._flat_values,
            flat_values2=image2._flat_values,
            constant1=image1._constant,
            constant2=image2._constant,
            **kwargs,
        )

        if union_values.dtype == torch.bool:
            if new_constant:
                flat_indices = torch.arange(
                    image1.numel, dtype=torch.int64, device=image1.device
                )
                flat_indices = indices_difference(
                    flat_indices, union_indices[~union_values]
                )
            else:
                flat_indices = union_indices[union_values]
            return Mask(flat_indices=flat_indices, shape=image1.shape)
        else:
            return SparseImage(
                flat_indices=union_indices,
                values=union_values,
                constant=new_constant,
                shape=image1.shape,
            )
    else:
        msg = "Wrong image types for _apply_pointwise_binary_operation."
        raise TypeError(msg)


class ImageStructure(GridStructure):
    @convert_inputs
    @typecheck
    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        values_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(shape=shape, device=device)

        self._values_shape = values_shape

        self._values_dim = len(values_shape)
        self._full_shape = self._shape + self._values_shape
        self._dtype = dtype

    @typecheck
    def __repr__(self) -> str:
        """String representation of the image.

        Returns
        -------
        str
            A succinct description of the image data.
        """

        def suffix(dtype):
            return str(dtype).split(".")[-1].split("'")[0]

        repr_str = f"skshapes.{self.__class__.__name__} (0x{id(self):x} on {self.device}, {suffix(self.dtype)}), a {self.dim}D image."
        repr_str += (
            f" Image shape: {self.shape}, Values shape: {self.values_shape}"
        )

        return repr_str

    ##########################
    #### Image properties ####
    ##########################

    @property
    @typecheck
    def values_shape(self) -> tuple[int, ...]:
        """For tensor-valued images, the shape of the value tensor."""
        return self._values_shape

    @property
    @typecheck
    def values_dim(self) -> int:
        """The dimension of the value tensors"""
        return len(self.full_shape) - self.dim

    @property
    @typecheck
    def full_shape(self) -> tuple[int, ...]:
        """The full shape of the image. Equivalent to ``shape + values_shape``."""
        return self._full_shape

    @property
    @typecheck
    def dtype(self) -> torch.dtype:
        """Dtype getter."""
        return self._dtype

    @property
    @typecheck
    def values(self) -> NumericalTensor:
        """Abstract function that provides the tensor of shape (shape_0,...shape_{dim-1}) containing the values of the
        image."""
        return None

    @one_and_only_one(["query_indices", "flat_query_indices"])
    @convert_inputs
    @typecheck
    def values_at(
        self,
        *,
        query_indices: IntTensor | None,
        flat_query_indices: IntTensor | None = None,
    ) -> NumericalTensor:
        """Get the values of the image at the indices specified in input."""

        return get_at_indices(
            grid=self.values,
            indices=query_indices,
            flat_indices=flat_query_indices,
            shape=self.shape,
        )

    ##############################
    #### Pointwise operations ####
    ##############################

    @typecheck
    def apply_pointwise(
        self,
        operation: callable,
        other: ImageStructure | None = None,
        **kwargs,
    ) -> ImageStructure | Mask:
        """
        Apply a function at each point of the image.

        The argument ``operation`` is the function that is applied to the image. It can map:
                1. A tensor shape ``(N,V_1,...,V_K)`` to a tensor of shape ``(N,V'_1,...,V'_K')``.
                2. Two tensors of shape ``(N,V_1,...,V_K)`` to a tensor of shape ``(N,V'_1,...,V'_K')``. Then, ``other``
                is used as the second argument of ``operation``.

        It computes the image whose values are ``operation(self.values, **kwargs)``, or ``operation(self.values, other, **kwargs)``
        if ``other`` is not None.

        The type of the image (Image or SparseImage) is chosen depending on the type of the inputs.
        If ``other`` is None, the type of the output will be the same as the ``self`` type. Otherwise, the output type
        is determined as follows:
        ``self`` of type:    ``other`` of type:          -> output of type:
            ``Image``            ``Image``                    ``Image``
            ``Image``            ``SparseImage``              ``Image``
            ``SparseImage``      ``Image``                    ``Image``
            ``SparseImage``      ``SparseImage``              ``SparseImage``


        Parameters
        ----------
        operation
            The function that is applied to the image.

        other
            If not None, it will be used as the second argument of the function ``operation``.

        **kwargs
            Additional keyword arguments passed to ``operation``.

        Returns
        -------

        """
        if other is None:
            return _apply_pointwise_unary_operation(
                operation=operation, image=self, **kwargs
            )
        else:
            return _apply_pointwise_binary_operation(
                operation=operation, image1=self, image2=other, **kwargs
            )

    def __add__(self, other) -> ImageStructure:
        def operation(u, v):
            return u.__add__(v)

        return self.apply_pointwise(operation=operation, other=other)

    def __eq__(self, other) -> Mask:
        def operation(u, v):
            return u.__eq__(v)

        return self.apply_pointwise(operation=operation, other=other)

    def __gt__(self, other) -> Mask:
        def operation(u, v):
            return u.__gt__(v)

        return self.apply_pointwise(operation=operation, other=other)

    def __lt__(self, other) -> Mask:
        def operation(u, v):
            return u.__lt__(v)

        return self.apply_pointwise(operation=operation, other=other)

    def __mul__(self, other) -> ImageStructure:
        def operation(u, v):
            return u.__mul__(v)

        return self.apply_pointwise(operation=operation, other=other)

    def __neg__(self) -> ImageStructure:
        def operation(u):
            return u.__neg__()

        return self.apply_pointwise(operation=operation)

    def __ne__(self, other) -> Mask:
        def operation(u, v):
            return u.__ne__(v)

        return self.apply_pointwise(operation=operation, other=other)

    def __sub__(self, other) -> ImageStructure:
        def operation(u, v):
            return u.__sub__(v)

        return self.apply_pointwise(operation=operation, other=other)

    def isin(self, test_elements: torch.Tensor) -> Mask:
        def operation(u):
            return torch.isin(u, test_elements)

        return self.apply_pointwise(operation=operation)

    ##############################
    #### Reduction operations ####
    ##############################

    @typecheck
    def unique(
        self, *, sorted: bool = True, return_counts: bool = False
    ) -> UniqueOutput:
        dim = None if len(self.values_shape) == 0 else 0
        return UniqueOutput(
            *self.values.unique(
                dim=dim, sorted=sorted, return_counts=return_counts
            )
        )

    def apply_reduction(
        self,
        operation: callable,  # TODO problème beartype
        *,
        requires_count: bool = True,
        **kwargs,
    ):
        if requires_count:
            uniques, counts = self.unique(return_counts=True)
            return operation(uniques, counts, **kwargs)
        else:
            return operation(self.unique(return_counts=False), **kwargs)

    def histogram(
        self, bins: int | torch.Tensor
    ) -> torch.return_types.histogram:
        """!-> cpu"""

        def operation(u, c):
            return torch.histogram(
                input=u, bins=bins, weight=c.to(dtype=u.dtype)
            )

        return self.apply_reduction(operation, requires_count=True)

    def max(self) -> torch.return_types.max:
        def operation(u):
            return u.max()

        return self.apply_reduction(operation, requires_count=False)

    def min(self) -> torch.return_types.min:
        def operation(u):
            return u.min()

        return self.apply_reduction(operation, requires_count=False)

    ###########################
    #### Plotting utilities ###
    ###########################

    @typecheck
    def plot(
        self,
        backend: Literal["pyvista", "vedo"] = "pyvista",
        **kwargs,
    ) -> None:

        if backend == "pyvista":
            pl = pyvista.Plotter(**kwargs)

            pl.add_volume(self.values.cpu().numpy())
            pl.show()
