"""Utils to handle indices when working with images and masks. #TODO changer

The functions of this module take two types of tensors inputs:
    - ``flat_indices``: int64 tensors of shape ``(N,)`` (or ``(N,C_1,...,C_L)`` when channeled indices are allowed).
    - ``flat_values``: tensors of shape ``(N,)`` or ``(N,V_1,...,V_K)``.

These tensors are used to describe D-dimensional structures in a flatten way. More precisely, given a D-dimensional
values tensor of shape ``(X_1,...,X_D)``, operations of the form:
    ``output[indices[...,0], indices[...,1],...,indices[...,D]] =  f(values[indices[...,0], indices[...,1],...,indices[...,D]])``
are performed using the following scheme:
    ``flat_output == f(flat_values[flat_indices])``.

"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch

from skshapes.errors import ShapeError
from skshapes.input_validation import one_and_only_one, typecheck

from ...types import Number

ImageShape = tuple[int, ...] | torch.Tensor


#############################
#### Indices conversions ####
#############################


@typecheck
def indices_periods(
    shape: ImageShape, *, device: str | torch.device | None = None
) -> torch.Tensor:
    """
    Computes the values to add to the indices of a flatten D-dimensional image to shift it by 1 in each axis.

    More precisely, for a 3-dimensional image ``A``, the output ``periods`` is a flat tensor with 3 values such that:
    ``A[i,j,k] = A.flatten()[i * periods[0] + j * periods[1] + k * periods[2]]``

    Parameters
    ----------
    shape
        The shape of the D-dimensional image the indices belong to.

    device
        The device on which to send the output.

    Returns
    -------
    torch.Tensor
        A tensor of ints of shape ``(D,)``.
    """
    if not isinstance(shape, torch.Tensor):
        shape = torch.tensor(shape, dtype=torch.int64, device=device)

    if device is None:
        device = shape.device
    else:
        shape = shape.to(device)

    return torch.cat(
        [
            shape[1:].flip(dims=[0]).cumprod(dim=0).flip(dims=[0]),
            torch.tensor([1], device=device),
        ]
    )


@typecheck
def flatten_indices(
    indices: torch.Tensor,
    *,
    shape: ImageShape,
) -> torch.Tensor:
    """Convert indices of shape ``(N,C_1,...,C_L,D)`` to flattened indices of shape ``(N,C_1,...,C_L)``.
    ``N`` corresponds to the number of points, and ``C_1,...,C_L`` correspond to the (optional) channel dimensions.
    If no batch is used, the output will be of shape ``(N,)``.

    The output flat_indices satisfies:
    ``A[indices[...,0], indices[...,1],...,indices[...,D-1]] == A.flatten()[flat_indices]``

    Parameters
    ----------
    indices
        A tensor of voxel indices of shape ``(N,D)`` or ``(N,C_1,...,C_L,D)``.

    shape
        The shape of the D-dimensional image the indices belong to.

    Returns
    -------
    torch.Tensor
        The flattened indices, in a tensor of shape ``(N,)`` or ``(N,C_1,...,C_L)``.

    """

    if indices.shape[-1] == len(shape):
        # flat_indices = indices[...,0] * periods[0] + indices[...,1] * periods[1] + ... + indices[...,D-1] * periods[D-1]
        flat_indices = (
            indices * indices_periods(shape, device=indices.device)
        ).sum(dim=-1)
    else:
        msg = f"Invalid tensor shape for indices ({indices.shape})."
        raise ShapeError(msg)

    return flat_indices


@typecheck
def expand_indices(
    flat_indices: torch.Tensor, *, shape: ImageShape
) -> torch.Tensor:
    """
    Convert flattened indices of shape ``(N,C_1,...,C_L)``  to expanded indices of shape ``(N,C_1,...,C_L,D)``.

    This is the inverse operation of ``flatten_indices``.

    Parameters
    ----------
    flat_indices
        A tensor of flattened indices of shape ``(N,)`` or ``(N,C_1,...,C_L)``

    shape
        The shape of the D-dimensional image the indices belong to.

    Returns
    -------
        The unflattened indices, in a tensor of shape ``(N,D)`` or ``(N,C_1,...,C_L,D)``.

    """
    if not isinstance(shape, torch.Tensor):
        shape = torch.tensor(
            shape, dtype=torch.int64, device=flat_indices.device
        )

    # indices[..., d] = (flat_indices // periods[d]) % shape[d]
    return (
        flat_indices[..., None]
        // indices_periods(shape, device=flat_indices.device)[None, :]
    ) % shape[None, :]


#########################
#### Grid operations ####
#########################


@typecheck
def flatten_grid(grid: torch.Tensor, *, shape: ImageShape) -> torch.Tensor:
    """
    Flatten the grid tensor along the D first dimensions, where D = len(shape).

    Parameters
    ----------
    grid
        A tensor of shape ``(X_1,...,X_D)`` or ``(X_1,...,X_D,V_1,...,V_K)``.

    shape
        The shape of the original unflattened input.

    Returns
    -------
    torch.Tensor
        Flattened tensor of shape ``(Q,)`` or ``(Q,V_1,...,V_K)``, with ``Q = X_1*...*X_D``.

    """
    return grid.reshape((-1,) + grid.shape[len(shape) :])


@typecheck
def expand_grid(flat_grid: torch.Tensor, *, shape: ImageShape) -> torch.Tensor:
    """
    Expand the input tensor.

    This is the inverse operation of ``flatten_grid``.

    Parameters
    ----------
    flat_grid
        A tensor of shape ``(Q,)`` or ``(Q,V_1,...,V_K)``.

    shape
        The shape of the original unflattened input.

    Returns
    -------
    torch.Tensor
        Expanded tensor of shape ``(X_1,...,X_D)`` or ``(X_1,...,X_D,V_1,...,V_K)``.

    """
    return flat_grid.reshape(shape + flat_grid.shape[1:])


@typecheck
@one_and_only_one(["grid", "flat_grid"])
@one_and_only_one(["indices", "flat_indices"])
def get_at_indices(
    *,
    grid: torch.Tensor | None = None,
    flat_grid: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
    flat_indices: torch.Tensor | None = None,
    shape: ImageShape | None = None,
) -> torch.Tensor:
    """
    Get the values of ``grid`` at ``indices``:
    ``output = grid[indices[...,0], indices[...,1],...,indices[...,D]]``

    The inputs can either be provided in their flatten form or in their expanded form.
    If channels are specified for the indices, the output shape will be of shape ``(N,C_1,...,C_L,V_1,...,V_K)``

    Parameters
    ----------
    grid
        A tensor of shape ``(X_1,...,X_D,V_1,...,V_K)``.

    flat_grid
        A tensor of shape ``(Q,V_1,...,V_K)``.

    indices
        A tensor of shape ``(N,D)`` or ``(N,C_1,...,C_L,D)``.

    flat_indices
        A tensor of flattened indices of shape ``(N,)`` or ``(N,C_1,...,C_L)``

    shape
        The shape of the original image (only used if ``values`` or ``indices`` is specified).

    Returns
    -------
    torch.Tensor
        A tensor of shape (N,C_1,...,C_L,V_1,...,V_K) with the values of ``grid`` or ``flat_grid`` at the positions
        given by ``indices`` or ``flat_indices``.

    """
    if flat_indices is None:
        flat_indices = flatten_indices(indices=indices, shape=shape)
    if flat_grid is None:
        flat_grid = flatten_grid(grid, shape=shape)

    return flat_grid[flat_indices]


@typecheck
@one_and_only_one(["grid", "flat_grid"])
@one_and_only_one(["indices", "flat_indices"])
def set_at_indices(
    *,
    grid: torch.Tensor | None = None,
    flat_grid: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
    flat_indices: torch.Tensor | None = None,
    shape: tuple[int, ...] | None = None,
    new_values: Number | bool | torch.Tensor,
) -> torch.Tensor:
    """
    Set the values of ``grid`` to ``new_values`` at ``indices``.

    Parameters
    ----------
    grid
        A tensor of shape ``(X_1,...,X_D,V_1,...,V_K)``.

    flat_grid
        A tensor of shape ``(Q,V_1,...,V_K)``.

    indices
        A tensor of shape ``(N,D)``.

    flat_indices
        A tensor of flattened indices of shape ``(N,)``

    new_values
        A tensor of shape ``(N,V_1,...,V_K)``

    shape
        The shape of the original image (only used if ``grid`` or ``indices`` is specified).

    Returns
    -------
    torch.Tensor
        The original tensor ``grid`` or ``flat_grid``, with values updated to ``new_values`` at the  positions given by
        ``indices`` or ``flat_indices``.

    """
    if flat_indices is None:
        flat_indices = flatten_indices(indices=indices, shape=shape)

    if flat_grid is None:
        flat_grid = flatten_grid(grid=grid, shape=shape)
        flat_grid[flat_indices] = new_values
        return expand_grid(flat_grid=flat_grid, shape=shape)
    else:
        flat_grid[flat_indices] = new_values
        return flat_grid


# TODO
@typecheck
@one_and_only_one(["query_indices", "flat_query_indices"])
def get_at_sparse_indices(
    *,
    flat_values: torch.Tensor | None = None,
    flat_indices: torch.Tensor | None = None,
    constant: torch.Tensor | None = None,
    query_indices: torch.Tensor | None = None,
    flat_query_indices: torch.Tensor | None = None,
    shape: ImageShape | None = None,
) -> torch.Tensor:
    """
    Get the values of ``grid`` at ``indices``:
    ``output = grid[indices[...,0], indices[...,1],...,indices[...,D]]``

    The inputs can either be provided in their flatten form or in their expanded form.
    If channels are specified for the indices, the output shape will be of shape ``(N,C_1,...,C_L,V_1,...,V_K)``

    Parameters
    ----------
    grid
        A tensor of shape ``(X_1,...,X_D,V_1,...,V_K)``.

    flat_grid
        A tensor of shape ``(Q,V_1,...,V_K)``.

    indices
        A tensor of shape ``(N,D)`` or ``(N,C_1,...,C_L,D)``.

    flat_indices
        A tensor of flattened indices of shape ``(N,)`` or ``(N,C_1,...,C_L)``

    shape
        The shape of the original image (only used if ``values`` or ``indices`` is specified).

    Returns
    -------
    torch.Tensor
        A tensor of shape (N,C_1,...,C_L,V_1,...,V_K) with the values of ``grid`` or ``flat_grid`` at the positions
        given by ``indices`` or ``flat_indices``.

    """
    if flat_query_indices is None:
        flat_query_indices = flatten_indices(
            indices=query_indices, shape=shape
        )

    reverse_index, valid = search_in(
        flat_origin=flat_query_indices, flat_destination=flat_indices
    )
    reverse_index = reverse_index.flatten()

    # Initialize the neighborhood values with the input constant (or with zeros if the constant is None)
    values_shape = flat_query_indices.shape + flat_values.shape[1:]
    if constant is None:
        values = torch.zeros(
            reverse_index.shape + flat_values.shape[1:],
            dtype=flat_values.dtype,
            device=flat_values.device,
        )
    else:
        values = (
            constant.reshape((1, *constant.shape))
            .expand(reverse_index.shape + constant.shape)
            .clone()
        )

    # If a shift_indices is in flat_indices, the corresponding value in the neighborhood is replaced by the corresponding
    # value of the input flat_values
    values[valid] = flat_values[reverse_index]
    return values.reshape(values_shape)


###########################
#### Sorting utilities ####
###########################


@typecheck
def check_sorted(flat_tensor: torch.Tensor) -> bool:
    """
    Check whether ``flat_tensor`` is sorted in increasing order.

    Parameters
    ----------
    flat_tensor
        A tensor of shape ``(N,)``.

    """
    return bool((flat_tensor[1:] >= flat_tensor[:-1]).all())


class SortByIndicesOutput(NamedTuple):
    output: torch.tensor
    flat_indices: torch.Tensor | None = None


@typecheck
def sort_by_indices(
    *,
    flat_values: torch.Tensor,
    flat_indices: torch.Tensor,
    return_indices: bool = False,
) -> SortByIndicesOutput:
    """
    Sort ``flat_values`` using the permutation that sorts ``flat_indices`` increasingly.

    If ``return_indices`` is ``True`` then ``flat_indices`` is also sorted in increasing order.

    Parameters
    ----------
    flat_values
        A tensor of shape ``(N,V_1,...,V_K)`` that will be sorted.

    flat_indices
        A tensor of shape ``(N,)`` that will determine the order in which ``flat_input`` will be sorted.

    return_indices
        If True, both ``flat_input`` and ``flat_indices`` will be sorted.

    Returns
    -------
    SortByIndicesOutput
        A named tuple ``(output, flat_indices)``, int64 tensors of shape ``(N,)`` containing the values of ``flat_values``
        and ``flat_indices`` after sorted.

    """

    if check_sorted(flat_indices):
        flat_input_out = flat_values
        flat_indices_out = flat_indices if return_indices else None
    else:
        sort_idx = torch.argsort(flat_indices)
        flat_input_out = flat_values[sort_idx]
        flat_indices_out = flat_indices[sort_idx] if return_indices else None

    return SortByIndicesOutput(
        output=flat_input_out, flat_indices=flat_indices_out
    )


######################################
#### Operations on sorted indices ####
######################################


@typecheck
def isin_sorted(
    elements: torch.Tensor,
    test_elements: torch.Tensor,
    invert: bool = False,
) -> torch.Tensor:
    """
    Implements the function ``torch.isin`` when the tensor ``test_elements`` is sorted.

    This implementation uses the function ``torch.searchsorted`` to check the inclusion of ``elements`` in
    ``test_elements`` in a memory and time efficient way.

    Parameters
    ----------
    elements
        Input elements

    test_elements
        Values against which to test for each input element

    invert
        If True, inverts the boolean return tensor, resulting in True values for elements not in ``test_elements``.
        Default: False

    Returns
    -------
        A boolean tensor of the same shape as ``elements`` that is True for elements in ``test_elements`` and False
        otherwise.

    """
    search_idx = torch.searchsorted(
        sorted_sequence=test_elements, input=elements
    )

    # Append a dummy value at the end of test_elements to avoid indexations errors when values of elements are larger
    # than every value of test_elements
    extended_test_elements = torch.cat(
        [test_elements, torch.tensor([-1], device=test_elements.device)]
    )

    if invert:
        return extended_test_elements[search_idx] != elements
    else:
        return extended_test_elements[search_idx] == elements


@typecheck
def search_in(
    *,
    flat_origin: torch.Tensor,
    flat_destination: torch.Tensor,
) -> (torch.Tensor, torch.Tensor):
    """
    Returns a tensor of bools ``isin`` indicating which values of ``flat_origin`` are in ``flat_destination``, and a
    tensor ``reverse_index`` such that ``flat_destination[reverse_index] = flat_origin[isin]``.

    The tensor ``flat_destination`` is assumed to be sorted in increasing order.

    Parameters
    ----------
    flat_origin

    flat_destination

    Returns
    -------

    """

    full_reverse_index = torch.searchsorted(flat_destination, flat_origin)
    flat_indices_extended = torch.cat(
        [flat_destination, torch.tensor([-1], device=flat_destination.device)]
    )
    isin = flat_indices_extended[full_reverse_index] == flat_origin

    reverse_index = full_reverse_index[isin]
    return reverse_index, isin


########################
#### Set Operations ####
########################


@typecheck
def indices_intersection(
    flat_indices1: torch.Tensor,
    flat_indices2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the set intersection between two set of flat sorted indices.

    Parameters
    ----------
    flat_indices1
        A tensor of flattened indices of shape ``(N,)``

    flat_indices2
        A tensor of flattened indices of shape ``(M,)``

    Returns
    -------
    torch.Tensor
        A tensor of indices of shape ``(P,)`` containing the indices that are present in both ``flat_indices1`` and
        ``flat_indices2``.

    """
    return flat_indices1[isin_sorted(flat_indices1, flat_indices2)]


@typecheck
def indices_difference(
    flat_indices1: torch.Tensor,
    flat_indices2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the set difference between two set of flat sorted indices.

    Parameters
    ----------
    flat_indices1
        A tensor of flattened indices of shape ``(N,)``

    flat_indices2
        A tensor of flattened indices of shape ``(M,)``

    Returns
    -------
    torch.Tensor
        A tensor of indices of shape ``(P,)`` containing the indices that are present in ``flat_indices1`` but not in
        ``flat_indices2``.

    """

    return flat_indices1[
        isin_sorted(flat_indices1, flat_indices2, invert=True)
    ]


@typecheck
def indices_symmetric_difference(
    flat_indices1: torch.Tensor,
    flat_indices2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the symmetric difference ("XOR") between two set of flat sorted indices.


    Parameters
    ----------
    flat_indices1
        A tensor of flattened indices of shape ``(N,)``

    flat_indices2
        A tensor of flattened indices of shape ``(M,)``

    Returns
    -------
    torch.Tensor
        A tensor of indices of shape ``(P,)`` containing the indices that are present either in ``indices1`` or
        ``indices2``, but not in both.

    """
    return torch.cat(
        [
            flat_indices1[
                isin_sorted(flat_indices1, flat_indices2, invert=True)
            ],
            flat_indices2[
                isin_sorted(flat_indices1, flat_indices2, invert=True)
            ],
        ]
    ).sort()[0]


@typecheck
def indices_union(
    flat_indices1: torch.Tensor,
    flat_indices2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the union of two set of indices.


    Parameters
    ----------
    flat_indices1
        A tensor of flattened indices of shape ``(N,)``

    flat_indices2
        A tensor of flattened indices of shape ``(M,)``

    Returns
    -------
    torch.Tensor
        A tensor of indices of shape ``(P,)`` containing the indices that are present either in ``flat_indices1`` or
        ``flat_indices2``.

    """
    return torch.cat([flat_indices1, flat_indices2]).unique(sorted=True)


####################################
#### Pointwise sparse operation ####
####################################


# @typecheck
def apply_at_indices(
    *,
    operation: callable,
    flat_indices1: torch.Tensor,
    flat_indices2: torch.Tensor,
    flat_values1: torch.Tensor,
    flat_values2: torch.Tensor,
    constant1: torch.Tensor,
    constant2: torch.Tensor,
    **kwargs,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Computes the operation ``operation(tensor1, tensor2, **kwargs)``, where ``tensori`` is the tensor with values
    ``flat_valuesi`` at the indices ``flat_indicesi`` and ``constanti`` elsewhere.

    The result is given as a triple ``(final_indices, final_values, final_constant)`` following the same conventions as
    the input tensors.

    The indices ``flat_indices1`` and ``flat_indices2`` are supposed to be sorted in increasing order.

    Parameters
    ----------
    operation
        Function that takes two tensors of shape ``(Q,V_1,...,V_K)`` and ``(Q,V_1,...,V_K)`` as input, and return
        a tensor of shape ``(Q,V'_1,...,V'_K')`` as output. It should allow broadcasting, ie if one of the input tensors
        have a first axis size of 1, the output should still be of shape ``(Q,V'_1,...,V'_K')``.

    flat_indices1
        A tensor of flattened indices of shape ``(N,)``

    flat_indices2
        A tensor of flattened indices of shape ``(M,)``

    flat_values1
        A tensor of values of shape ``(N,V_1,...,V_K)``

    flat_values2
        A tensor of values of shape ``(M,V_1,...,V_K)``

    constant1
        A tensor of values of shape ``(V_1,...,V_K)``

    constant2
        A tensor of values of shape ``(V_1,...,V_K)``

    **kwargs
        Keyword arguments passed to ``operation``.

    Returns
    -------

    """
    final_constant = operation(
        constant1[None, ...], constant2[None, ...], **kwargs
    ).squeeze(0)

    # Compute the indices that are either in flat_indices1 or flat_indices2
    final_indices, counts = torch.cat([flat_indices1, flat_indices2]).unique(
        sorted=True, return_counts=True
    )
    inverse_indices1 = torch.searchsorted(final_indices, flat_indices1)
    inverse_indices2 = torch.searchsorted(final_indices, flat_indices2)

    intersection = counts > 1

    # Boolean tensors that indicates the indices of flat_indices1, flat_indices2 that overlap
    isinter1, isinter2 = (
        intersection[inverse_indices1],
        intersection[inverse_indices2],
    )  # (N,), (M,)

    # Values of the SparseImage result after the operation (to be filled).
    final_values = torch.zeros(
        final_indices.shape + final_constant.shape,
        dtype=final_constant.dtype,
        device=final_constant.device,
    )  # (P,V)

    # Put the values of the images that doesn't intersect with the other one at their right positions:
    if ~isinter1.all():
        final_values[inverse_indices1[~isinter1]] = operation(
            flat_values1[~isinter1], constant2[None, ...], **kwargs
        )
    if ~isinter2.all():
        final_values[inverse_indices2[~isinter2]] = operation(
            constant1[None, ...], flat_values2[~isinter2], **kwargs
        )

    # Compute the values of the images at the intersections between the two indices sets:
    final_values[intersection] = operation(
        flat_values1[isinter1], flat_values2[isinter2], **kwargs
    )

    return final_indices, final_values, final_constant


#######################################
#### Neighborhood-based operations ####
#######################################


@typecheck
def shift_indices(
    flat_indices: torch.Tensor,
    *,
    offsets: torch.Tensor,
    shape: ImageShape,
    clip: bool = True,
) -> torch.Tensor:
    """
    Computes the indices of ``flat_indices`` shifted by ``offsets``.

    The parameter ``clip`` determines how the borders of the grid should be handled.

    Parameters
    ----------
    flat_indices
        A tensor of flattened indices of shape ``(N,)``

    offsets
        A tensor of indices of shape ``(C_1,...,C_L,D)``

    shape
        The shape of the original unflattened grid.

    clip
        If True, indices will be clipped at the borders of the images. Otherwise, they will loop periodically (as if the
        grid was toric).

    Returns
    -------
        A tensor of indices of shape ``(N, C_1,...,C_L)``
    """
    device = flat_indices.device

    if not isinstance(shape, torch.Tensor):
        shape = torch.tensor(shape, dtype=torch.int64, device=device)

    if clip:
        dim = len(shape)

        periods = indices_periods(shape)
        flat_indices = flat_indices.view(
            (flat_indices.shape[0],) + (1,) * (len(offsets.shape) - 1)
        )

        # (N,C_1,...,C_L)
        new_indices = torch.zeros(
            size=(flat_indices.shape[0],) + offsets.shape[:-1],
            dtype=torch.int64,
            device=device,
        )

        for axis in range(dim):
            new_axis_coords = (
                (flat_indices // periods[axis]) % shape[axis]
            ).repeat((1,) + offsets.shape[:-1])
            new_axis_coords.add_(offsets[None, ..., axis])
            new_axis_coords.clamp_(
                torch.tensor(0, device=device), shape[axis] - 1
            )
            new_axis_coords.mul_(periods[axis])
            new_indices += new_axis_coords
            new_axis_coords = None

        return new_indices

    else:
        flat_offsets = flatten_indices(indices=offsets, shape=shape)
        viewshape = (flat_indices.shape[0],) + (1,) * (len(flat_offsets.shape))

        return (
            flat_indices.squeeze(-1).view(viewshape) + flat_offsets[None, ...]
        ) % torch.prod(shape)


@typecheck
def neighborhood_adjacency(
    flat_indices: torch.Tensor,
    *,
    offsets: torch.Tensor,
    shape: ImageShape,
    clip: bool = True,
) -> (torch.Tensor, torch.Tensor):
    """
    Compute the adjacency matrix of the mask defined by ``flat_indices``, for the adjacency relation defined by ``offsets``.

    The output is a couple of tensors ``row`` and ``col`` that provides the adjacency matrix as a set of edges (following
    the ``scipy.sparse.csr_matrix`` data structure). Each pair ``(row[i], col[i])`` is such that ``indices[col[i]] - indices[row[i]]``
    is in ``offsets``.

    Parameters
    ----------
    flat_indices
        A tensor of flattened indices of shape ``(N,)``

    offsets
        A tensor of indices of shape ``(Q ,D)``

    shape
        The shape of the original unflattened grid.

    clip
        If True, indices will be clipped at the borders of the images. Otherwise, they will loop periodically (as if the
        grid was toric).

    Returns
    -------
        Two tensor of indices of shape ``(P,)``

    """
    shifted_indices = shift_indices(
        flat_indices=flat_indices, offsets=offsets, shape=shape, clip=clip
    )
    reverse_index, isin = search_in(
        flat_origin=shifted_indices, flat_destination=flat_indices
    )

    row = torch.argwhere(isin)[:, 0]
    col = reverse_index

    return row, col


@typecheck
def neighborhood_values(
    flat_indices: torch.Tensor,
    *,
    flat_mask_indices: torch.Tensor | None = None,
    flat_values: torch.Tensor,
    constant: torch.Tensor | None = None,
    offsets: torch.Tensor,
    shape: ImageShape,
    clip: bool = True,
) -> torch.Tensor:
    """
    Return the values of tensor ``tensor`` at the indices ``flat_mask_indices`` shifted by ``offsets``, where ``tensor``
    is the tensor with values ``flat_values`` at the indices ``flat_indices`` and ``constant`` elsewhere.

    ``output[i,q,c1,...,cl,v1,...,vk] = tensor[mask[i] + offsets[q,c1,...,cl]][v1,...,vk]``

    Parameters
    ----------
    flat_indices
        A tensor of flattened indices of shape ``(N,)``

    flat_mask_indices
        A tensor of indices of shape ``(M,)``. If None, ``flat_indices`` is used instead.

    flat_values
        A tensor of values of shape ``(N,V_1,...,V_K)``

    constant
        A tensor of shape ``(V_1,...,V_K)``

    offsets
        A tensor of indices of shape ``(C_1,...,C_L,D)``

    shape
        The shape of the original unflattened grid.

    clip
        If True, indices will be clipped at the borders of the images. Otherwise, they will loop periodically (as if the
        grid was toric).

    Returns
    -------
    torch.Tensor
        A tensor of indices of shape ``(N, C_1,...,C_L, V_1,...,V_K)`` containing the values of the input tensor at the
        neighborhoods of the positions given by ``flat_mask_indices`` shifted by ``offsets``.

    """
    if flat_mask_indices is None:
        flat_mask_indices = flat_indices

    # Compute the indices of the local neighborhoods
    shifted_indices = shift_indices(
        flat_indices=flat_mask_indices, offsets=offsets, shape=shape, clip=clip
    ).flatten()
    shifted_shape = shifted_indices.shape

    # Search whether shifted_indices are in the input flat_indices, and their position in the latter
    reverse_index, valid = search_in(
        flat_origin=shifted_indices, flat_destination=flat_indices
    )
    shifted_indices = None

    # Initialize the neighborhood values with the input constant (or with zeros if the constant is None)
    values_shape = (
        (flat_mask_indices.shape[0],)
        + offsets.shape[:-1]
        + flat_values.shape[1:]
    )
    if constant is None:
        values = torch.zeros(
            shifted_shape + flat_values.shape[1:],
            dtype=flat_values.dtype,
            device=flat_values.device,
        )
    else:
        values = (
            constant.reshape((1, *constant.shape))
            .expand(shifted_shape + constant.shape)
            .clone()
        )

    # If a shift_indices is in flat_indices, the corresponding value in the neighborhood is replaced by the corresponding
    # value of the input flat_values
    values[valid] = flat_values[reverse_index]

    return values.reshape(values_shape)


@typecheck
def scatter_convolution(
    flat_indices: torch.Tensor,
    *,
    flat_values: torch.Tensor | None = None,
    constant: torch.Tensor | None = None,
    offsets: torch.Tensor,
    shape: ImageShape,
    clip: bool = True,
    kernel: Literal["sum"] = "sum",
    weights: torch.Tensor | None = None,
) -> (torch.Tensor, torch.Tensor):
    """
    Computes the convolution of the tensor with values ``flat_values`` at the indices ``flat_indices`` and ``constant``
    elsewhere.

    Parameters
    ----------
    flat_indices
        A tensor of flattened indices of shape ``(N,)``

    flat_values
        A tensor of values of shape ``(N,)``

    constant
        A tensor of values of shape ``(,)``

    offsets
        A tensor of indices of shape ``(Q,D)``

    shape
        The shape of the original unflattened grid.

    clip
        If True, indices will be clipped at the borders of the images. Otherwise, they will loop periodically (as if the
        grid was toric).

    kernel
        The convolution kernel operation (used in the scatter_reduction).

    weights
        If None, the weights to apply to the local neighborhoods before convolution as a tensor of shape ``(Q,)``.

    Returns
    -------
        A couple of tensors of shape (P,) containing the indices and values of the resulting image.

    """
    if constant is not None and constant.neq(
        0
    ):  # TODO gérer constante + gérer images multivaluées
        msg = "Scatter convolution not implemented for constant != 0"
        raise ValueError(msg)

    shifted_indices = shift_indices(
        flat_indices=flat_indices, offsets=-offsets, shape=shape, clip=clip
    )  # (N, Q,)
    neigh_shape = shifted_indices.shape

    # Compute the unique indices of shifted_indices, and the positions of shifted_indices in new_indices
    # This is equivalent to ``new_indices, rev_index = shifted_indices.unique(reverse_indices=True)`` but here we compute
    # rev_index in a second time to save memory
    new_indices = shifted_indices.unique(sorted=True)
    rev_index = torch.searchsorted(new_indices, shifted_indices).flatten()
    shifted_indices = None

    # Compute the values that will be scattered to the final image
    if weights is None:
        neigh_values = flat_values.reshape(-1, 1).expand(neigh_shape).flatten()
    else:
        neigh_values = (flat_values[:, None] * weights[None, :]).flatten()

    # Compute the values that will be scattered to the final image
    new_values = torch.zeros(
        new_indices.shape, dtype=flat_values.dtype, device=flat_values.device
    )
    new_values = new_values.scatter_reduce_(
        dim=0,
        index=rev_index,
        src=neigh_values,
        reduce=kernel,
        include_self=False,
    )

    return new_indices, new_values


@typecheck
def scatter_boolean_convolution(
    flat_indices: torch.Tensor,
    *,
    offsets: torch.Tensor | None = None,
    shape: ImageShape,
    clip: bool = True,
    kernel: Literal["any", "all"] = "any",
) -> torch.Tensor:
    """
    Computes the convolution of the boolean tensor with values True at the indices ``flat_indices`` and False elsewhere.

    Parameters
    ----------
    flat_indices
        A tensor of flattened indices of shape ``(N,)``

    offsets
        A tensor of indices of shape ``(Q,D)``

    shape
        The shape of the original unflattened grid.

    clip
        If True, indices will be clipped at the borders of the images. Otherwise, they will loop periodically (as if the
        grid was toric).

    kernel
        The convolution kernel operation. If 'any', the resulting image will be True at each position whose neighborhood
        contains at least one True. If 'all', the resulting image will be True at each position whose neighborhood
        contains only True.

    Returns
    -------
        A tensor of shape (P,) containing the indices where the resulting image is True.

    """
    shifted_indices = shift_indices(
        flat_indices=flat_indices, offsets=offsets, shape=shape, clip=clip
    )

    if kernel == "any":
        output = shifted_indices.unique(sorted=True)
    elif kernel == "all":
        new_indices, counts = shifted_indices.unique(
            sorted=True, return_counts=True
        )
        output = new_indices[counts == offsets[:-1].numel()]
    else:
        msg = (
            "Scatter convolution not implemented for kernel != 'any' or 'all')"
        )
        raise Exception(msg)

    return output


def masked_convolution(
    flat_indices: torch.Tensor,
    *,
    flat_mask_indices: torch.Tensor,
    flat_values: torch.Tensor,
    constant: torch.Tensor | None = None,
    offsets: torch.Tensor,
    shape: ImageShape,
    clip: bool = True,
    kernel: callable | Literal["sum", "prod", "min", "max"] = "sum",
    # TODO gérer beartype param kernel
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the convolution of the tensor with values ``flat_values`` at the indices ``flat_indices`` and ``constant``
    elsewhere, at the indices determined by ``flat_mask_indices``.

    Parameters
    ----------
    flat_indices
        A tensor of flattened indices of shape ``(N,)``

    flat_mask_indices
        A tensor of indices of shape ``(M,)``.

    flat_values
        A tensor of values of shape ``(N,V_1,...,V_K)``

    constant
        A tensor of shape ``(V_1,...,V_K)``

    offsets
        A tensor of indices of shape ``(Q,C_1,...,C_L,D)``

    shape
        The shape of the original unflattened grid.

    clip
        If True, indices will be clipped at the borders of the images. Otherwise, they will loop periodically (as if the
        grid was toric).

    kernel
        The convolution kernel operation. It must be a function that takes a tensor of shape ``(M,Q,C_1,...,C_L,V_1,...,V_K)``
         as input and return and ``(Q,V_1,...,V_K)`` as input, and return a tensor of shape ``(M,C_1,...,C_L,V'_1,...,V'_K')`` as output.
         If weight is not None, it should also take a second input of shape ``(1,Q,C_1,...,C_L)`` as input.

    weights
        If not None, a tensor of shape ``(Q,C_1,...,C_L)`` that will be provided as additional input to the kernel.

    Returns
    -------
        A tensor of shape ``(M,C_1,...,C_L,V'_1,...,V'_K')`` containing the values of the convolution at the mask indices.

    """
    neigh_values = neighborhood_values(
        flat_indices=flat_indices,
        flat_mask_indices=flat_mask_indices,
        flat_values=flat_values,
        constant=constant,
        offsets=offsets,
        shape=shape,
        clip=clip,
    )

    if weights is not None:
        if kernel == "sum":

            def kernel(u, w):
                return (u * w).sum(dim=1)

        if kernel == "prod":

            def kernel(u, w):
                return (u * w).prod(dim=1)

        return kernel(neigh_values, weights[None, ...])
    else:
        if kernel == "sum":

            def kernel(u):
                return u.sum(dim=1)

        if kernel == "prod":

            def kernel(u):
                return u.prod(dim=1)

        if kernel == "max":

            def kernel(u):
                return u.max(dim=1)[0]

        if kernel == "min":

            def kernel(u):
                return u.min(dim=1)[0]

        return kernel(neigh_values)
