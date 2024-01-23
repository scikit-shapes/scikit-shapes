"""Energy functions for triangle meshes."""
from typing import Optional, Union

import torch

from ..input_validation import typecheck
from ..types import (
    Float1dTensor,
    FloatScalar,
    Number,
    Points,
    PointsSequence,
    Triangles,
)
from .edge_topology import EdgeTopology
from .geometry import (
    dihedral_angles,
    edge_lengths,
    triangle_areas,
)


@typecheck
def bending_energy(
    *,
    points_undef: Union[Points, PointsSequence],
    points_def: Union[Points, PointsSequence],
    triangles: Triangles,
    edge_topology: Optional[EdgeTopology] = None,
) -> Union[FloatScalar, Float1dTensor]:
    """Compute the bending energy of the mesh.

    The mathematical formulation is given in page 4 of:
    https://ddg.math.uni-goettingen.de/pub/HeRuSc14.pdf

    The implementation provided here is a pytorch version of the cpp
    implementation available at:
    https://gitlab.com/numod/shell-energy/-/blob/main/src/bending_energy.cpp

    Parameters
    ----------
    points_undef : torch.Tensor
        The undeformed points of the mesh. Shape: (n_points, dim) for a single
        mesh and (n_points, n_poses, dim) for a sequence of poses of the same
        mesh.

    points_def : torch.Tensor
        The deformed points of the mesh. Shape: (n_points, dim) for a single
        mesh and (n_points, n_poses, dim) for a sequence of poses of the same
        mesh.

    triangles : torch.Tensor
        The triangles of the mesh(es). Shape: (n_triangles, 3).

    weight : float, optional
        The weight of the bending energy. Default: 0.001.

    """
    if edge_topology is None:
        edge_topology = EdgeTopology(triangles)

    # Compute difference of dihedral angles
    dihedral_angles_def = dihedral_angles(
        points=points_def,
        triangles=triangles,
        edge_topology=edge_topology,
    )

    dihedral_angles_undef = dihedral_angles(
        points=points_undef,
        triangles=triangles,
        edge_topology=edge_topology,
    )

    del_theta = dihedral_angles_def - dihedral_angles_undef

    # Compute areas per edge (sum of adjacent triangles areas)
    t_a = triangle_areas(points=points_undef, triangles=triangles)
    areas = t_a[edge_topology.manifold_adjacent_triangles].sum(dim=1)

    # Compute lengths per edge
    lengths_squared = (
        edge_lengths(points=points_undef, triangles=triangles) ** 2
    )
    lengths_squared = lengths_squared[edge_topology.is_manifold]

    return (3 * (del_theta**2) * (lengths_squared / areas)).sum(dim=0)


@typecheck
def membrane_energy(
    *,
    points_undef: Union[Points, PointsSequence],
    points_def: Union[Points, PointsSequence],
    triangles: Triangles,
    edge_topology: Optional[EdgeTopology] = None,
) -> Union[FloatScalar, Float1dTensor]:
    """Compute the membrane energy of the mesh.

    The mathematical formulation is given by equation (8) of:
    https://ddg.math.uni-goettingen.de/pub/HeRuWaWi12_final.pdf

    The implementation provided here is a pytorch version of the cpp
    implementation available at:
    https://gitlab.com/numod/shell-energy/-/blob/main/src/membrane_energy.cpp

    Parameters
    ----------
    points_undef : torch.Tensor
        The undeformed points of the mesh. Shape: (n_points, dim) for a single
        mesh and (n_points, n_poses, dim) for a sequence of poses of the same
        mesh.

    points_def : torch.Tensor
        The deformed points of the mesh. Shape: (n_points, dim) for a single
        mesh and (n_points, n_poses, dim) for a sequence of poses of the same
        mesh.

    triangles : torch.Tensor
        The triangles of the mesh(es). Shape: (n_triangles, 3).

    """
    if edge_topology is None:
        edge_topology = EdgeTopology(triangles)

    a = points_def[triangles[:, 0]]  # i
    b = points_def[triangles[:, 1]]  # j
    c = points_def[triangles[:, 2]]  # k

    ei = c - b
    ej = a - c
    ek = a - b

    ei_norm = (ei**2).sum(dim=-1)
    ej_norm = (ej**2).sum(dim=-1)
    ek_norm = (ek**2).sum(dim=-1)

    a = points_undef[triangles[:, 0], :]
    b = points_undef[triangles[:, 1], :]
    c = points_undef[triangles[:, 2], :]

    ei = c - b
    ej = a - c
    ek = a - b

    trace = (
        ei_norm * (ej * ek).sum(dim=-1)
        + ej_norm * (ek * ei).sum(dim=-1)
        - ek_norm * (ei * ej).sum(dim=-1)
    )

    areas_def = triangle_areas(
        points=points_def,
        triangles=triangles,
        edge_topology=edge_topology,
    )
    areas_undef = triangle_areas(
        points=points_undef,
        triangles=triangles,
        edge_topology=edge_topology,
    )

    mu = 1.0
    lambd = 1.0

    return torch.sum(
        (mu * trace / 8 + lambd * (areas_def**2) / 4) / areas_undef
        - areas_undef
        * (
            (mu / 2 + lambd / 4) * (2 * torch.log(areas_def / areas_undef))
            + mu
            + lambd / 4
        ),
        dim=0,
    )


@typecheck
def shell_energy(
    *,
    points_undef: Union[Points, PointsSequence],
    points_def: Union[Points, PointsSequence],
    triangles: Triangles,
    edge_topology: Optional[EdgeTopology] = None,
    weight: Number = 0.001,
) -> Union[FloatScalar, Float1dTensor]:
    """Compute the shell energy.

    The shell energy is defined as the sum of the membrane and weight * bending
    energies.

    Parameters
    ----------
    points_undef : torch.Tensor
        The undeformed points of the mesh. Shape: (n_points, dim) for a single
        mesh and (n_points, n_poses, dim) for a sequence of poses of the same
        mesh.

    points_def : torch.Tensor
        The deformed points of the mesh. Shape: (n_points, dim) for a single
        mesh and (n_points, n_poses, dim) for a sequence of poses of the same
        mesh.

    triangles : torch.Tensor
        The triangles of the mesh(es). Shape: (n_triangles, 3).

    weight : float, optional
        The weight of the bending energy. Default: 0.001.

    """
    if edge_topology is None:
        edge_topology = EdgeTopology(triangles)

    m_energy = membrane_energy(
        points_undef=points_undef,
        points_def=points_def,
        triangles=triangles,
        edge_topology=edge_topology,
    )
    b_energy = bending_energy(
        points_undef=points_undef,
        points_def=points_def,
        triangles=triangles,
        edge_topology=edge_topology,
    )
    return m_energy + weight * b_energy
