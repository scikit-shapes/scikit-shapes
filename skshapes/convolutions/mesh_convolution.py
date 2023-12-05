"""Mesh convolution kernel."""

from ..types import float_dtype
from ..input_validation import typecheck

import torch
from .linear_operator import LinearOperator


@typecheck
def _mesh_convolution(
    self,
    weight_by_length: bool = False,
) -> LinearOperator:
    """Convolution kernel on a triangle mesh or a wireframe PolyData.

    Parameters
    ----------
    weight_by_length
        If True, the convolution kernel is weighted by the length of the edges.

    Raises
    ------
    ValueError
        If the mesh is not a triangle mesh or a wireframe PolyData.

    Returns
    -------
    LinearOperator
        A (N, N) convolution kernel.
    """
    if self.n_edges == 0:
        raise ValueError(
            "Mesh convolution is only defined on triangle meshes or "
            "wireframe PolyData."
        )
    n_edges = self.n_edges
    n_points = self.n_points
    edges = self.edges

    # Edge smoothing
    edges_revert = torch.zeros_like(edges, dtype=float_dtype)
    edges_revert[:, 0], edges_revert[:, 1] = edges[:, 1], edges[:, 0]

    indices = torch.cat((edges.T, edges_revert.T), dim=1)

    if not weight_by_length:
        values = torch.ones(2 * n_edges, dtype=float_dtype)
    else:
        values = self.edge_lengths.repeat(2).to(dtype=float_dtype)

    S = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(n_points, n_points),
        device=self.device,
    )

    degrees = S @ torch.ones(n_points, device=self.device, dtype=float_dtype)

    return LinearOperator(S, output_scaling=1 / degrees)
