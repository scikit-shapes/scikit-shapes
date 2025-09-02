"""Utils to handle indices when working with images and masks."""

# TODO vérifier quelle convention choisir
# TODO est-ce qu'on veut faire des opérations ensemblistes sur indices batchés?

from __future__ import annotations

import torch

from ..errors import ShapeError


def _flatten_indices(
    indices: torch.Tensor, shape: torch.Tensor | tuple | None = None
) -> torch.Tensor:
    """Convert indices of various shapes to flattened indices of shape ``(N,B_1,...,B_K)``.
    ``N`` corresponds to the number of points, and ``B_1,...,B_K`` correspond to the (optional) batch dimensions.
    If no batch is used, the output will be of shape ``(N,)``.

    If indices is of shape ``(N,...,D)``, the output flat_indices satisfies:
    ``A[indices[...,0], indices[...,1],...,indices[...,D]] == A.flatten()[flat_indices]``

    Parameters
    ----------
    indices
        A tensor of voxel indices.
        It can have the following shapes (depending whether the indices are batched):
            1. ``(N,)``  or ``(N,B_1,...,B_K)``     -> Already flattened indices.
            2. ``(N,1)`` or ``(N,B_1,...,B_K,1)``   -> Already flattened indices, with last dimension unsqueezed.
            3. ``(N,D)`` or ``(N,B_1,...,B_K,D)``   -> Unflattened indices, the last dimension giving the tuple of D-coordinates.

    shape
        The shape of the D-dimensional image the indices belong to. It is only used when indices belong to case 3.

    Returns
    -------
    torch.Tensor
        The flattened indices, in a tensor of shape ``(N,)`` or ``(N,B_1,...,B_K)``.

    """

    if len(indices.shape) == 1:  # indices of shape (N,...)
        flat_indices = indices
    elif indices.shape[-1] == 1:  # Indices of shape (N,...,1)
        flat_indices = indices.squeeze(-1)
    elif indices.shape[-1] == len(shape):  # Indices of shape (N,...,D)
        if not isinstance(shape, torch.Tensor):
            shape = torch.tensor(
                shape, dtype=torch.int64, device=indices.device
            )

        periods = torch.cat(
            [
                shape[1:].flip(dims=[0]).cumprod(dim=0).flip(dims=[0]),
                torch.tensor([1], device=indices.device),
            ]
        )

        flat_indices = (indices * periods).sum(dim=-1)
    else:
        msg = "Invalid tensor shape for indices."
        raise ShapeError(msg)

    return flat_indices  # Indices of shape (N,...)


def _expand_indices(
    flat_indices: torch.Tensor, shape: tuple | torch.Tensor
) -> torch.Tensor:
    """
    Convert flattened indices of shape ``(N,B_1,...,B_K)``  to expanded indices of shape ``(N,B_1,...,B_K,D)``.

    This is the inverse operation of ``_flatten_indices``.


    Parameters
    ----------
    flat_indices
        A tensor of flattened indices of shape ``(N,)`` or ``(N,B_1,...,B_K)``

    shape
        The shape of the D-dimensional image the indices belong to. It is only used when indices belong to case 3.

    Returns
    -------
        The unflattened indices, in a tensor of shape ``(N,D)`` or ``(N,B_1,...,B_K,D)``.
    """

    if not isinstance(shape, torch.Tensor):
        shape = torch.tensor(
            shape, dtype=torch.int64, device=flat_indices.device
        )

    periods = torch.cat(
        [
            shape[1:].flip(dims=[0]).cumprod(dim=0).flip(dims=[0]),
            torch.tensor([1], device=flat_indices.device),
        ]
    )
    indices = flat_indices.squeeze(-1)[..., None] // periods[None, :]
    return indices % shape[None, :]


########################
#### Set Operations ####
########################


def _difference_indices(
    indices1: torch.Tensor, indices2: torch.Tensor
) -> torch.Tensor:
    """
    Compute the difference of two set of indices.
    The input indices must be flattened (see ``_flatten_indices``).

    Parameters
    ----------
    indices1
        A tensor of flattened indices of shape ``(N,)``

    indices2
        A tensor of flattened indices of shape ``(M,)``

    Returns
    -------
    torch.Tensor
        A tensor of indices of shape ``(P,)`` or ``(P,B_1,...,B_K)`` containing the indices that are present in
        ``indices1`` but not in ``indices2``.

    """
    discarded = torch.isin(indices1, indices2)
    return indices1[~discarded]


def _intersection_indices(
    indices1: torch.Tensor, indices2: torch.Tensor
) -> torch.Tensor:
    """
    Compute the intersection of two set of indices.
    The input indices must be flattened (see ``_flatten_indices``).
    If indices are batched, i.e. ``indices1`` have shape ``(N,B_1,...,B_K)`` and ``indices2`` have shape
    ``(M,B_1,...,B_K,D)``, the intersection will be performed on the set of K-uples defined by the batch dimensions.

    Parameters
    ----------
    indices1
        A tensor of flattened indices of shape ``(N,)`` or ``(N,B_1,...,B_K)``
    indices2
        A tensor of flattened indices of shape ``(M,)`` or ``(M,B_1,...,B_K)``

    Returns
    -------
    torch.Tensor
        A tensor of indices of shape ``(P,)`` or ``(P,B_1,...,B_K)`` containing the indices that are present in both
        ``indices1`` and ``indices2``.

    """
    concat_indices, counts = torch.cat([indices1, indices2], dim=0).unique(
        dim=0, return_counts=True
    )
    return concat_indices[torch.where(counts > 1)]


def _symmetric_difference_indices(
    indices1: torch.Tensor, indices2: torch.Tensor
) -> torch.Tensor:
    """
    Compute the symmetric difference ("XOR") of two set of indices.
    The input indices must be flattened (see ``_flatten_indices``).
    If indices are batched, i.e. ``indices1`` have shape ``(N,B_1,...,B_K)`` and ``indices2`` have shape
    ``(M,B_1,...,B_K,D)``, the union will be performed on the set of K-uples defined by the batch dimensions.

    Parameters
    ----------
    indices1
        A tensor of flattened indices of shape ``(N,)`` or ``(N,B_1,...,B_K)``
    indices2
        A tensor of flattened indices of shape ``(M,)`` or ``(M,B_1,...,B_K)``

    Returns
    -------
    torch.Tensor
        A tensor of indices of shape ``(P,)`` or ``(P,B_1,...,B_K)`` containing the indices that are present either in
        ``indices1`` or ``indices2``, but not in both.

    """
    concat_indices, counts = torch.cat([indices1, indices2], dim=0).unique(
        dim=0, return_counts=True
    )
    return concat_indices[torch.where(counts == 1)]


def _union_indices(
    indices1: torch.Tensor, indices2: torch.Tensor
) -> torch.Tensor:
    """
    Compute the union of two set of indices.
    The input indices must be flattened (see ``_flatten_indices``).
    If indices are batched, i.e. ``indices1`` have shape ``(N,B_1,...,B_K)`` and ``indices2`` have shape
    ``(M,B_1,...,B_K,D)``, the union will be performed on the set of K-uples defined by the batch dimensions.

    Parameters
    ----------
    indices1
        A tensor of flattened indices of shape ``(N,)`` or ``(N,B_1,...,B_K)``
    indices2
        A tensor of flattened indices of shape ``(M,)`` or ``(M,B_1,...,B_K)``

    Returns
    -------
    torch.Tensor
        A tensor of indices of shape ``(P,)`` or ``(P,B_1,...,B_K)`` containing the indices that are present either in
        ``indices1`` or ``indices2``.

    """
    return torch.cat([indices1, indices2], dim=0).unique(dim=0)
