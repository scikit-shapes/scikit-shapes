from ..types import (
    typecheck,
    NumericalTensor,
    polydata_type,
)

import torch
from .linear_operator import LinearOperator


@typecheck
def _mesh_convolution(
    self,
    weight_by_length: bool = False,
) -> LinearOperator:
    """Creates a convolution kernel on a triangle mesh or a wireframe polydata as a (N, N) operator.

    Args:
        weight_by_length (bool, optional): If True, the convolution kernel is
            weighted by the length of the edges. Defaults to False.

    Returns:
        LinearOperator: A (N, N) convolution kernel.
    """

    n_edges = self.n_edges
    n_points = self.n_points
    edges = self.edges

    # Edge smoothing
    edges_revert = torch.zeros_like(edges)
    edges_revert[:, 0], edges_revert[:, 1] = edges[:, 1], edges[:, 0]

    indices = torch.cat((edges.T, edges_revert.T), dim=1)

    if not weight_by_length:
        values = torch.ones(2 * n_edges, dtype=torch.float32)
    else:
        values = self.edge_lengths.repeat(2)

    S = torch.sparse_coo_tensor(
        indices=indices, values=values, size=(n_points, n_points), device=self.device
    )

    degrees = S @ torch.ones(n_points, device=self.device)

    return LinearOperator(S, output_scaling=1 / degrees)
