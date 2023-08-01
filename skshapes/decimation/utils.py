import numpy as np
import numba as nb


from time import time
from ..types import typecheck, FloatArray, IntArray
from typing import Tuple


def compute_edges(triangles, repeated=False):
    repeated_edges = np.concatenate(
        [
            triangles[[0, 1], :],
            triangles[[1, 2], :],
            triangles[[0, 2], :],
        ],
        axis=1,
    )

    repeated_edges.sort(axis=0)

    repeated_edges
    # Remove the duplicates and return
    if not repeated:
        return np.unique(repeated_edges, axis=1)

    else:
        ordering = np.lexsort(repeated_edges)
        return repeated_edges[:, ordering]


@nb.jit(nopython=True, fastmath=True, cache=True)
def initialize_quadrics(points, triangles):
    quadrics = np.zeros((points.shape[0], 11))

    for i in range(triangles.shape[1]):
        p0, p1, p2 = points[triangles[:, i], :]
        n = np.cross(p1 - p0, p2 - p0)
        area2 = np.sqrt((np.sum(n * n))) / 2
        n /= 2 * area2

        d = -np.sum(n * p0)

        # Compute the quadric for this triangle
        tmp = np.zeros(4)
        tmp[0:3] = n
        tmp[3] = d

        Q = np.zeros(11 + 4 * 3)
        Q[0] = n[0] * n[0]
        Q[1] = n[0] * n[1]
        Q[2] = n[0] * n[2]
        Q[3] = n[0] * d
        Q[4] = n[1] * n[1]
        Q[5] = n[1] * n[2]
        Q[6] = n[1] * d
        Q[7] = n[2] * n[2]
        Q[8] = n[2] * d
        Q[9] = d * d
        Q[10] = 1
        Q = Q * area2

        for j in triangles[:, i]:
            quadrics[j, 0:11] += Q

    # Normalize the quadrics
    return quadrics


@nb.jit(nopython=True, fastmath=True, cache=True)
def compute_boundary_quadrics(points, repeated_edges, triangles):
    boundary_quadrics = np.zeros((points.shape[0], 11))

    n_edges = repeated_edges.shape[1]
    n_boundary_edges = 0
    # Identify boundary edges
    boundary = True
    e0, e1 = repeated_edges[:, 0]
    for i in range(1, n_edges):
        if repeated_edges[0, i] == e0 and repeated_edges[1, i] == e1:
            boundary = False

        else:
            if boundary:
                n_boundary_edges += 1

                for j in range(triangles.shape[1]):
                    t = triangles[:, j]
                    if (
                        (t[0] == e0 and t[1] == e1)
                        or (t[1] == e0 and t[2] == e1)
                        or (t[0] == e0 and t[2] == e1)
                        or (t[0] == e1 and t[1] == e0)
                        or (t[1] == e1 and t[2] == e0)
                        or (t[0] == e1 and t[2] == e0)
                    ):
                        for k in t:
                            if k != e0 and k != e1:
                                t0 = k
                        t1 = points[e0]
                        t2 = points[e1]

                        u = t2 - t1
                        v = t1 - t0
                        n = (
                            v - (np.sum(u * v) / np.sum(u * u)) * u
                        )  # n is orthogonal to the boundary edge [e0, e1]
                        n = n / np.sqrt(np.sum(n * n))  # normalize n
                        w = np.sum(
                            u * u
                        )  # the weight corresponds to the square length of the boundary edge

                        d = -np.sum(n * t1)
                        Q = np.zeros(11 + 4 * 3)
                        Q[0] = n[0] * n[0]
                        Q[1] = n[0] * n[1]
                        Q[2] = n[0] * n[2]
                        Q[3] = n[0] * d
                        Q[4] = n[1] * n[1]
                        Q[5] = n[1] * n[2]
                        Q[6] = n[1] * d
                        Q[7] = n[2] * n[2]
                        Q[8] = n[2] * d
                        Q[9] = d * d
                        Q[10] = 1
                        Q = Q * w

                        for indice in range(11):
                            boundary_quadrics[e0][indice] += Q[indice]
                            boundary_quadrics[e1][indice] += Q[indice]

            e0, e1 = repeated_edges[:, i]
            boundary = True

    return boundary_quadrics


@nb.jit(nopython=True, fastmath=True, cache=True)
def compute_cost(edge, quadrics, points):
    pt0, pt1 = edge
    tmpQuad = quadrics[pt0] + quadrics[pt1]
    A = np.zeros((3, 3))
    b = np.zeros(3)

    A[0][0] = tmpQuad[0]
    A[0][1] = tmpQuad[1]
    A[1][0] = tmpQuad[1]
    A[0][2] = tmpQuad[2]
    A[2][0] = tmpQuad[2]
    A[1][1] = tmpQuad[4]
    A[1][2] = tmpQuad[5]
    A[2][1] = tmpQuad[5]
    A[2][2] = tmpQuad[7]

    b[0] = -tmpQuad[3]
    b[1] = -tmpQuad[6]
    b[2] = -tmpQuad[8]

    error = 1e-10

    norm = np.max(np.sqrt(np.sum(A * A, axis=1)))
    if np.linalg.det(A) / (norm**3) > error:
        x = np.linalg.solve(A, b)

    else:
        pt0 = points[pt0]
        pt1 = points[pt1]

        v = pt1 - pt0

        tmp2 = np.zeros(3)
        tmp2[0] = A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2]
        tmp2[1] = A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2]
        tmp2[2] = A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2]

        if np.sum(tmp2 * tmp2) > error:
            tmp = np.zeros(3)
            tmp[0] = A[0][0] * pt0[0] + A[0][1] * pt0[1] + A[0][2] * pt0[2]
            tmp[1] = A[1][0] * pt0[0] + A[1][1] * pt0[1] + A[1][2] * pt0[2]
            tmp[2] = A[2][0] * pt0[0] + A[2][1] * pt0[1] + A[2][2] * pt0[2]
            tmp = b - tmp
            c = np.sum(tmp * tmp2) / np.sum(tmp2 * tmp2)

            x = pt0 + c * v

        else:
            x = 0.5 * (pt0 + pt1)

    cost = 0.0
    newpoint = np.concatenate((x, np.array([1.0])))
    counter = 0
    for i in range(4):
        cost += newpoint[i] * newpoint[i] * tmpQuad[counter]
        counter += 1
        for j in range(i + 1, 4):
            cost += 2 * newpoint[i] * newpoint[j] * tmpQuad[counter]
            counter += 1

    return cost, x


@nb.jit(nopython=True, fastmath=True, cache=True)
def intialize_costs(edges, quadrics, points):
    n_edges = edges.shape[1]
    costs = np.zeros(n_edges)
    newpoints = np.zeros((n_edges, 3))

    for i in range(n_edges):
        costs[i], newpoints[i] = compute_cost(edges[:, i], quadrics, points)

    return costs, newpoints


@nb.jit(nopython=True, fastmath=True, cache=True)
def collapse(
    edges,
    costs,
    newpoints,
    quadrics,
    points,
    n_points_to_remove=5000,
):
    indices_toremove = np.zeros(n_points_to_remove, dtype=np.int64)
    collapses = np.zeros((n_points_to_remove, 2), dtype=np.int64)
    newpoints_history = np.zeros((n_points_to_remove, 3), dtype=np.float32)
    n_points_removed = 0

    # the edges with infinite cost will be at the end of the array
    noninf_limit = len(edges)
    while n_points_removed < n_points_to_remove:
        ### FAST #####
        indice = np.argmin(costs[0:noninf_limit])
        ##############
        e0, e1 = edges[indice]
        ### FAST #####
        quadrics[e0] += quadrics[e1]
        newpoint = newpoints[indice]
        points[e0] = newpoint
        collapses[n_points_removed][0] = e0
        collapses[n_points_removed][1] = e1
        newpoints_history[n_points_removed] = newpoint
        indices_toremove[n_points_removed] = e1
        ##############

        n_points_removed += 1

        # Update the edges
        i = 0
        # the edges with indices > noninf_limit are the ones with infinite cost
        # they are not considered
        while i < noninf_limit:
            # Update the connectivity e0 <- e1
            if edges[i][0] == e1:
                edges[i][0] = e0
            if edges[i][1] == e1:
                edges[i][1] = e0

            # Update the cost of the impacted edges (they have e0 as vertex)
            if (edges[i][0] == e0 or edges[i][1] == e0) and edges[i][0] != edges[i][1]:
                costs[i], newpoints[i] = compute_cost(edges[i], quadrics, points)

            # If the edge is degenerated, remove it
            if edges[i][0] == edges[i][1]:
                noninf_limit -= 1
                ### FAST #####
                costs[i] = costs[noninf_limit]
                edges[i] = edges[noninf_limit]
                newpoints[i] = newpoints[noninf_limit]
                ##############
                i -= 1

            i += 1

    new_vertices = np.zeros((points.shape[0] - n_points_to_remove, 3), dtype=np.float32)
    ### FAST #####
    indices_to_remove = np.sort(indices_toremove)
    j = 0
    counter = 0
    for i in range(points.shape[0]):
        if i == indices_to_remove[j]:
            j += 1
        else:
            new_vertices[counter] = points[i]
            counter += 1
    ##############

    return new_vertices, collapses, newpoints_history


@nb.jit(nopython=True, fastmath=True, cache=True)
def _replay_loop(points, quadrics, collapses_history):
    for i in range(len(collapses_history)):
        e0, e1 = collapses_history[i]
        quadrics[e0] += quadrics[e1]
        newpoint = compute_cost([e0, e1], quadrics, points)[1]
        points[e0] = newpoint

    return points


@typecheck
def replay_decimation(
    points,
    triangles,
    collapses_history,
):
    quadrics = initialize_quadrics(points, triangles)
    repeated_edges = compute_edges(triangles, repeated=True)
    boundary_quadrics = compute_boundary_quadrics(points, repeated_edges, triangles)
    quadrics += boundary_quadrics

    newpoints = _replay_loop(points, quadrics, collapses_history)
    keep = np.setdiff1d(
        np.arange(len(points)), collapses_history[:, 1]
    )  # Indices of the points that must be kept after decimation

    return newpoints[keep]


@typecheck
def decimate(
    points,
    triangles,
    target_reduction: float = 0.5,
    running_time: bool = False,
):
    """Apply the quadric decimation algorithm to a mesh.

    Args:
        points (_type_): _description_
        triangles (_type_): _description_
        target_reduction (float, optional): _description_. Defaults to 0.5.
        print_compute_time (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    assert target_reduction > 0.0 and target_reduction < 1.0

    if running_time:
        times = dict()

    start = time()
    quadrics = initialize_quadrics(points, triangles)
    if running_time:
        times["initialize_quadrics"] = time() - start

    # Are there boundary edges?
    start = time()
    repeated_edges = compute_edges(triangles, repeated=True)
    boundary_quadrics = compute_boundary_quadrics(points, repeated_edges, triangles)
    if running_time:
        times["check_boundary_constraints"] = time() - start

    quadrics += boundary_quadrics
    # Compute the cost for each edge
    start = time()
    edges = compute_edges(triangles)
    costs, target_points = intialize_costs(edges, quadrics, points)

    n_points_to_remove = int(target_reduction * points.shape[0])
    if running_time:
        times["initialize_costs"] = time() - start

    start = time()
    output_points, collapses, newpoints = collapse(
        edges=edges.T,
        costs=costs,
        newpoints=target_points,
        quadrics=quadrics,
        points=points,
        n_points_to_remove=n_points_to_remove,
    )

    if running_time:
        times["collapse"] = time() - start
        times["total"] = sum(times.values())

    if not running_time:
        return output_points, collapses, newpoints
    else:
        return output_points, collapses, newpoints, times


# Old functions
if False:

    @nb.jit(nopython=True, fastmath=True, cache=True)
    def det3x3(mat):
        det = (
            mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2])
            - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
            + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0])
        )
        return det

    # This function finds the solution of system of
    # linear equations using cramer's rule
    @nb.jit(nopython=True, fastmath=True, cache=True)
    def solve3x3(A, b):
        # Matrix d using coeff as given in
        # cramer's rule
        D = det3x3(A)

        D1 = det3x3(
            [
                [b[0], A[0][1], A[0][2]],
                [b[1], A[1][1], A[1][2]],
                [b[2], A[2][1], A[2][2]],
            ]
        )

        D2 = det3x3(
            [
                [A[0][0], b[0], A[0][2]],
                [A[1][0], b[1], A[1][2]],
                [A[2][0], b[2], A[2][2]],
            ]
        )

        D3 = det3x3(
            [
                [A[0][0], A[0][1], b[0]],
                [A[1][0], A[1][1], b[1]],
                [A[2][0], A[2][1], b[2]],
            ]
        )

        x = D1 / D
        y = D2 / D
        z = D3 / D

        return np.array([x, y, z])

    @nb.jit(nopython=True, fastmath=True, cache=True)
    def solve2(tmpQuad):
        D = det3x3(
            [
                [tmpQuad[0], tmpQuad[1], tmpQuad[2]],
                [tmpQuad[1], tmpQuad[4], tmpQuad[5]],
                [tmpQuad[2], tmpQuad[5], tmpQuad[7]],
            ]
        )

        D1 = det3x3(
            [
                [-tmpQuad[3], tmpQuad[1], tmpQuad[2]],
                [-tmpQuad[6], tmpQuad[4], tmpQuad[5]],
                [-tmpQuad[8], tmpQuad[5], tmpQuad[7]],
            ]
        )

        D2 = det3x3(
            [
                [tmpQuad[0], -tmpQuad[3], tmpQuad[2]],
                [tmpQuad[1], -tmpQuad[6], tmpQuad[5]],
                [tmpQuad[2], -tmpQuad[8], tmpQuad[7]],
            ]
        )

        D3 = det3x3(
            [
                [tmpQuad[0], tmpQuad[1], -tmpQuad[3]],
                [tmpQuad[1], tmpQuad[4], -tmpQuad[6]],
                [tmpQuad[2], tmpQuad[5], -tmpQuad[8]],
            ]
        )

        x = D1 / D
        y = D2 / D
        z = D3 / D

        return np.array([x, y, z])

    @nb.jit(nopython=True, fastmath=True, cache=True)
    def _decimate(points, alphas, collapses_history):
        """
        This function applies the decimation to a mesh that is in correspondence with the reference mesh given the information about successive collapses.

        Args:
            points (np.ndarray): the points of the mesh to decimate.
            alphas (np.ndarray): the list of alpha coefficients such that when (e0, e1) collapses : e0 <- alpha * e0 + (1-alpha) * e1
            collapses_history (np.ndarray): the history of collapses, a list of edges (e0, e1) that have been collapsed. The convention is that e0 is the point that remains and e1 is the point that is removed.

        Returns:
            points (np.ndarray): the decimated points.

        """

        for i in range(len(collapses_history)):
            e0, e1 = collapses_history[i]
            points[e0] = alphas[i] * points[e0] + (1 - alphas[i]) * points[e1]

        return points

    @nb.jit(nopython=True, fastmath=True, cache=True)
    def _compute_alphas(points, collapses_history, newpoints_history):
        """
        This function computes the alpha coefficients such that when (e0, e1) collapses to the point x, the projection of x on the line (e0, e1) is alpha * e0 + (1-alpha) * e1

        Args:
            points (np.ndarray): the points of the mesh to decimate.
            collapses_history (np.ndarray): the history of collapses, a list of edges (e0, e1) that have been collapsed. The convention is that e0 is the point that remains and e1 is the point that is removed.
            newpoints_history (np.ndarray): the list of new points that have been computed during the collapses.
        """

        alphas = np.zeros(len(collapses_history))

        for i in range(len(collapses_history)):
            e0, e1 = collapses_history[i]

            alpha = np.linalg.norm(newpoints_history[i] - points[e1]) / np.linalg.norm(
                points[e0] - points[e1]
            )
            points[e0] = alpha * points[e0] + (1 - alpha) * points[e1]
            alphas[i] = alpha

        return alphas
