"""H2 energy from Elastic Shape Analysis of Surfaces with Second-Order Sobolev Metrics: A Comprehensive Numerical Framework

Code: https://github.com/emmanuel-hartman/H2_SurfaceMatch/tree/main
"""

import torch

from ..input_validation import typecheck
from ..types import (
    Float3dTensor,
    FloatScalar,
    Triangles,
    float_dtype,
    int_dtype,
)

##############################################################################################################################
# Discrete Differential Geometry Helper Functions from DDG.py
##############################################################################################################################


def batchDot(dv1, dv2):
    """Parallel computation of batches of dot products.

    Input:
        - dv1 [Vxd torch tensor]
        - dv2 [Vxd torch tensor]

    Output:
        - tensor of dot products between corresponding rows of dv1 and dv2 [Vx1 torch tensor]
    """

    return torch.einsum("bi,bi->b", dv1, dv2)


def getSurfMetric(V, F):
    """Computation of the Riemannian metric evaluated at the faces of a triangulated surface.

    Input:
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        - F: faces of the triangulated surface [nFx3 torch tensor]

    Output:
        - g: Riemannian metric evaluated at each face of the triangulated surface [nFx2x2 torch tensor]
    """

    # Number of faces
    nF = F.shape[0]

    # Preallocate tensor for Riemannian metric
    alpha = torch.zeros((nF, 3, 2), dtype=F.dtype, device=V.device)

    # Compute Riemannian metric at each face
    V0, V1, V2 = (
        V.index_select(0, F[:, 0]),
        V.index_select(0, F[:, 1]),
        V.index_select(0, F[:, 2]),
    )

    alpha[:, :, 0] = V1 - V0
    alpha[:, :, 1] = V2 - V0

    alpha_f32 = alpha.to(dtype=torch.float32)

    return torch.matmul(alpha_f32.transpose(1, 2), alpha_f32).to(int_dtype)


def getMeshOneForms(V, F):
    """Computation of the Riemannian metric evaluated at the faces of a triangulated surface.

    Input:
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        - F: faces of the triangulated surface [nFx3 torch tensor]

    Output:
        - alpha: One form evaluated at each face of the triangulated surface [nFx3x2 torch tensor]
    """

    # Number of faces
    nF = F.shape[0]

    # Preallocate tensor for Riemannian metric
    alpha = torch.zeros((nF, 3, 2)).to(dtype=float_dtype, device=V.device)

    # Compute Riemannian metric at each face
    V0, V1, V2 = (
        V.index_select(0, F[:, 0]),
        V.index_select(0, F[:, 1]),
        V.index_select(0, F[:, 2]),
    )

    alpha[:, :, 0] = V1 - V0
    alpha[:, :, 1] = V2 - V0

    return alpha


def getLaplacian(V, F):
    """Computation of the mesh Laplacian operator of a triangulated surface evaluated at one of its tangent vectors h.

    Input:
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        - F: faces of the triangulated surface [nFx3 torch tensor]

    Output:
        - L: function that will evaluate the mesh Laplacian operator at a tangent vector to the surface [function]
    """

    # Number of vertices and faces
    nV, nF = V.shape[0], F.shape[0]

    # Get x,y,z coordinates of each face
    face_coordinates = V[F]
    v0, v1, v2 = (
        face_coordinates[:, 0],
        face_coordinates[:, 1],
        face_coordinates[:, 2],
    )

    # Compute the area of each face using Heron's formula
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)  # lengths of each side of the faces
    C = (v0 - v1).norm(dim=1)
    s = 0.5 * (A + B + C)  # semi-perimeter
    area = (
        (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-6).sqrt()
    )  # Apply Heron's formula and clamp areas of small faces for numerical stability

    # Compute cotangent expressions for the mesh Laplacian operator
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 2.0

    # Find indices of adjacent vertices in the triangulated surface (i.e., edge list between vertices)
    ii = F[:, [1, 2, 0]]
    jj = F[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, nF * 3)

    # Define function that evaluates the mesh Laplacian operator at one of the surface's tangent vectors
    def L(h):
        """Function that evaluates the mesh Laplacian operator at a tangent vector to the surface.

        Input:
            - h: tangent vector to the triangulated surface [nVx3 torch tensor]

        Output:
            - Lh: mesh Laplacian operator of the triangulated surface applied to one its tangent vectors h [nVx3 torch tensor]
        """

        # Compute difference between tangent vectors at adjacent vertices of the surface
        hdiff = h[idx[0]] - h[idx[1]]

        # Evaluate mesh Laplacian operator by multiplying cotangent expressions of the mesh Laplacian with hdiff
        values = torch.stack([cot.view(-1)] * 3, dim=1) * hdiff

        # Sum expression over adjacent vertices for each coordinate
        Lh = torch.zeros((nV, 3)).to(dtype=float_dtype, device=V.device)
        Lh[:, 0] = Lh[:, 0].scatter_add(0, idx[1, :], values[:, 0])
        Lh[:, 1] = Lh[:, 1].scatter_add(0, idx[1, :], values[:, 1])
        Lh[:, 2] = Lh[:, 2].scatter_add(0, idx[1, :], values[:, 2])

        return Lh

    return L


def getVertAreas(V, F):
    """Computation of vertex areas for a triangulated surface.

    Input:
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        - F: faces of the triangulated surface [nFx3 torch tensor]

    Output:
        - VertAreas: vertex areas [nVx1 torch tensor]
    """

    # Number of vertices
    nV = V.shape[0]

    # Get x,y,z coordinates of each face
    face_coordinates = V[F]
    v0, v1, v2 = (
        face_coordinates[:, 0],
        face_coordinates[:, 1],
        face_coordinates[:, 2],
    )

    # Compute the area of each face using Heron's formula
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)  # lengths of each side of the faces
    C = (v0 - v1).norm(dim=1)
    s = 0.5 * (A + B + C)  # semi-perimeter
    area = (
        (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-6).sqrt()
    )  # Apply Heron's formula and clamp areas of small faces for numerical stability

    # Compute the area of each vertex by averaging over the number of faces that it is incident to
    idx = F.view(-1)
    incident_areas = torch.zeros(nV, dtype=float_dtype, device=V.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    incident_areas.scatter_add_(0, idx, val)

    return 2 * incident_areas / 3.0 + 1e-24


def getNormal(F, V):
    """Computation of normals at each face of a triangulated surface.

    Input:
        - F: faces of the triangulated surface [nFx3 torch tensor]
        - V: vertices of the triangulated surface [nVx3 torch tensor]

    Output:
        - N: vertex areas [nFx1 torch tensor]
    """

    # Compute normals at each face by taking the cross product between edges of each face that are incident to its x-coordinate
    V0, V1, V2 = (
        V.index_select(0, F[:, 0]),
        V.index_select(0, F[:, 1]),
        V.index_select(0, F[:, 2]),
    )

    return 0.5 * torch.cross(V1 - V0, V2 - V0)


def computeBoundary(F):
    """Determining if a vertex is at the boundary of the mesh of a triagulated surface.

    Input:
        - F: faces of the triangulated surface [nFx3 ndarray]

    Output:
        - BoundaryIndicatorOfVertex: boolean vector indicating which vertices are at the boundary of the mesh [nVx1 boolean ndarray]

    Note: This is a CPU computation
    """

    import numpy as np
    import scipy

    # Get number of vertices and faces
    nF = F.shape[0]
    nV = F.max() + 1

    # Find whether vertex is at the boundary of the mesh
    Fnp = F  # F.detach().cpu().numpy()
    rows = Fnp[:, [0, 1, 2]].reshape(3 * nF)
    cols = Fnp[:, [1, 2, 0]].reshape(3 * nF)
    vals = np.ones(3 * nF, dtype=int)
    E = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(nV, nV))
    E -= E.transpose()
    i, j = E.nonzero()
    BoundaryIndicatorOfVertex = np.zeros(nV, dtype=bool)
    BoundaryIndicatorOfVertex[i] = True
    BoundaryIndicatorOfVertex[j] = True

    return BoundaryIndicatorOfVertex


def getGabMetric(alpha, xi1, xi2, g, dg1, dg2, dn1, dn2, a, b, c, d):
    n = g.shape[0]  # noqa: F841
    areas = torch.sqrt(torch.det(g)).to(dtype=float_dtype)
    ginv = torch.inverse(g)
    ginvdg1 = torch.matmul(ginv, dg1)
    ginvdg2 = torch.matmul(ginv, dg2)
    A = 0
    B = 0
    C = 0
    D = 0
    if a > 0:
        afunc = torch.einsum("bii->b", torch.matmul(ginvdg1, ginvdg2))
        A = a * torch.sum(afunc * areas)
    if b > 0:
        bfunc1 = torch.einsum("bii->b", ginvdg1)
        bfunc2 = torch.einsum("bii->b", ginvdg2)
        bfunc = bfunc1 * bfunc2
        B = b * torch.sum(bfunc * areas)
    if c > 0:
        cfunc = torch.einsum("bi,bi->b", dn1, dn2)
        C = c * torch.sum(cfunc * areas)
    if d > 0:
        xi1_0 = torch.matmul(
            torch.matmul(alpha, ginv),
            torch.matmul(xi1.transpose(1, 2), alpha)
            - torch.matmul(alpha.transpose(1, 2), xi1),
        )
        xi2_0 = torch.matmul(
            torch.matmul(alpha, ginv),
            torch.matmul(xi2.transpose(1, 2), alpha)
            - torch.matmul(alpha.transpose(1, 2), xi2),
        )
        dfunc = torch.einsum(
            "bii->b",
            torch.matmul(xi1_0, torch.matmul(ginv, xi2_0.transpose(1, 2))),
        )
        D = d * torch.sum(dfunc * areas)
    return A + B + C + D


def getGabNorm(alpha, xi, g, dg, dn, a, b, c, d):

    return getGabMetric(
        alpha=alpha,
        xi1=xi,
        xi2=xi,
        g=g,
        dg1=dg,
        dg2=dg,
        dn1=dn,
        dn2=dn,
        a=a,
        b=b,
        c=c,
        d=d,
    )


def getH2Metric(M, dv1, dv2, a0, a1, b1, c1, d1, a2, F_sol):
    """
    Parameters
    ----------
    M
        Vertices of the mesh

    dv1
        Deformation vector 1

    dv2
        Deformation vector 2

    F_sol
        Faces of the mesh

    Returns
    -------
    torch.tensor
        The H2 metric <dv1, dv2> at M
    """
    b1 = (b1 - a1) / 8
    M1 = M + dv1
    M2 = M + dv2
    enr = 0
    if a2 > 0 or a0 > 0:
        A = getVertAreas(M, F_sol)
    if a2 > 0:
        L = getLaplacian(M, F_sol)
        NL = batchDot(L(dv1), L(dv2)) / A
        enr += a2 * torch.sum(NL)
    if a1 > 0 or b1 > 0 or c1 > 0:

        alpha0 = getMeshOneForms(M, F_sol)
        g0 = getSurfMetric(M, F_sol)
        n0 = getNormal(F_sol, M)

        alpha1 = getMeshOneForms(M1, F_sol)
        g1 = getSurfMetric(M1, F_sol)
        n1 = getNormal(F_sol, M1)
        dg1 = g1 - g0
        dn1 = n1 - n0
        xi1 = alpha1 - alpha0

        alpha2 = getMeshOneForms(M2, F_sol)
        g2 = getSurfMetric(M2, F_sol)
        n2 = getNormal(F_sol, M2)
        dg2 = g2 - g0
        dn2 = n2 - n0
        xi2 = alpha2 - alpha0

        enr += getGabMetric(
            alpha0, xi1, xi2, g0, dg1, dg2, dn1, dn2, a1, b1, c1, d1
        )
    if a0 > 0:
        Ndv = A * batchDot(dv1, dv2)
        enr += a0 * torch.sum(Ndv)
    return enr


def GetH2Norm(M, dv, a0, a1, b1, c1, d1, a2, F_sol):
    """
    Parameters
    ----------
    M
        Vertices of the mesh

    dv
        Deformation vector

    F_sol
        Faces of the mesh

    Returns
    -------
    torch.tensor
        The H2 norm ||dv|| at M
    """
    return getH2Metric(
        M=M,
        dv1=dv,
        dv2=dv,
        a0=a0,
        a1=a1,
        b1=b1,
        c1=c1,
        d1=d1,
        a2=a2,
        F_sol=F_sol,
    )


def getPathEnergyH2(geod, a0, a1, b1, c1, d1, a2, F_sol, stepwise=False):
    """Get the full path energy of the H2 metric

    Parameters
    ----------
    geod
        The geodesic path (successive vertices of the path)
    a0
        a0 weight
    a1
        a1 weight
    b1
        b1 weight
    c1
        c1 weight
    d1
        d1 weight
    a2
        a2 weight
    F_sol
        Faces of the mesh
    stepwise, optional
        _description_, by default False

    Returns
    -------
        _description_
    """
    N = geod.shape[0]
    diff = geod[1:, :, :] - geod[:-1, :, :]
    midpoints = geod[0 : N - 1, :, :] + diff / 2
    # diff=diff*N
    enr = 0

    b1 = (b1 - a1) / 8

    step_enr = torch.zeros((N - 1, 1), dtype=float_dtype)
    alpha0 = getMeshOneForms(geod[0], F_sol)
    g0 = getSurfMetric(geod[0], F_sol)
    n0 = getNormal(F_sol, geod[0])
    for i in range(N - 1):
        dv = diff[i]
        if a2 > 0 or a0 > 0:
            M = getVertAreas(geod[i], F_sol)
        if a2 > 0:
            L = getLaplacian(midpoints[i], F_sol)
            L = L(dv)
            NL = batchDot(L, L) / M
            enr += a2 * torch.sum(NL) * N
        if a1 > 0 or b1 > 0 or c1 > 0 or d1 > 0:
            alpha1 = getMeshOneForms(geod[i + 1], F_sol)
            g1 = getSurfMetric(geod[i + 1], F_sol)
            n1 = getNormal(F_sol, geod[i + 1])
            xi = alpha1 - alpha0
            dg = g1 - g0
            dn = n1 - n0
            enr += (
                getGabNorm(
                    getMeshOneForms(midpoints[i], F_sol),
                    xi,
                    getSurfMetric(midpoints[i], F_sol),
                    dg,
                    dn,
                    a1,
                    b1,
                    c1,
                    d1,
                )
                * N
            )
            g0 = g1
            n0 = n1
            alpha0 = alpha1
        if a0 > 0:
            Ndv = M * batchDot(dv, dv)
            enr += a0 * torch.sum(Ndv) * N
        if stepwise:
            if i == 0:
                step_enr[0] = enr
            else:
                step_enr[i] = enr - torch.sum(step_enr[0:i])
    if stepwise:
        return enr, step_enr
    return enr


# Scikit-shapes energy function


@typecheck
def H2_energy(
    points_sequence: Float3dTensor,
    triangles: Triangles | None = None,
) -> FloatScalar:
    """H2 energy"""
    if triangles is None:
        msg = "This metric requires triangles to be defined"
        raise AttributeError(msg)

    geod = points_sequence
    F_sol = triangles

    a0 = 1.0
    a1 = 1.0
    b1 = 1.0
    c1 = 1.0
    d1 = 1.0
    a2 = 1.0

    stepwise = False  # noqa: F841

    return getPathEnergyH2(geod, a0, a1, b1, c1, d1, a2, F_sol, stepwise=False)
