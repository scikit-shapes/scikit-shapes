"""Specific functions for triangle meshes."""

import numpy as np
import torch
from fast_edges_extraction import extract_edges

from ..types import Triangles, int_dtype


class EdgeTopology:
    """Topology of the edges of a triangle mesh.

    This class is initialized from the triangles connectivity of a triangle
    mesh. It computes :
    - the edges of the mesh
    - the degrees of the edges (number of adjacent triangles)

    For manifold edges (degree 2), it also computes :
    - the adjacent triangles
    - the adjacent points

    For boundary edges (degree 1), it also computes :
    - the adjacent triangle
    - the adjacent point

    Manifold edges and boundary edges can be accessed separately using the
    attributes `manifold_edges` and `boundary_edges`.

    The indices of manifold edges and boundary edges can be accessed using the
    attributes `is_manifold` and `is_boundary`.

    Parameters
    ----------
    triangles
        Triangles of the mesh
    """

    def __init__(
        self,
        triangles: Triangles,
    ) -> None:
        device = triangles.device

        triangles_numpy = triangles.detach().cpu().numpy().astype(np.int64)

        # make them contiguous
        triangles_numpy = np.ascontiguousarray(triangles_numpy)

        (
            edges,
            edge_degrees,
            adjacent_triangles,
            adjacent_points,
        ) = extract_edges(triangles_numpy, return_adjacency=True)

        edges = torch.from_numpy(edges).to(int_dtype).to(device)
        adjacent_triangles = (
            torch.from_numpy(adjacent_triangles).to(int_dtype).to(device)
        )
        adjacent_points = (
            torch.from_numpy(adjacent_points).to(int_dtype).to(device)
        )
        edge_degrees = torch.from_numpy(edge_degrees).to(int_dtype).to(device)

        self.edges = edges
        self.degrees = edge_degrees

        self.is_manifold = self.degrees == 2
        self.manifold_adjacent_triangles = adjacent_triangles[self.is_manifold]
        self.manifold_adjacent_points = adjacent_points[self.is_manifold]

        self.is_boundary = self.degrees == 1
        self.boundary_adjacent_triangles = adjacent_triangles[
            self.is_boundary, 0
        ]
        self.boundary_adjacent_points = adjacent_points[self.is_boundary, 0]

    @property
    def manifold_edges(self):
        """Manifold edges of the mesh."""
        return self.edges[self.is_manifold]

    @property
    def boundary_edges(self):
        """Boundary edges of the mesh."""
        return self.edges[self.is_boundary]
