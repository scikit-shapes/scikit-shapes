# distutils: language = c++

cimport cython

from libc.math cimport sqrt
cimport numpy as cnp
cnp.import_array()
import numpy as np
from libcpp.vector cimport vector

INT_DTYPE = np.int64
FLOAT_DTYPE = np.double
ctypedef cnp.int64_t INT_DTYPE_t
ctypedef cnp.double_t FLOAT_DTYPE_t


def extract_edges(FLOAT_DTYPE_t [:, :] points, INT_DTYPE_t [:, :] triangles):

    # assert points.shape[1] == 3
    assert triangles.shape[0] == 3, "Triangles must be a 3xn_triangles array"

    # if needed convert to contiguous arrays
    # this is important for reading the data in C++
    # points = np.ascontiguousarray(points, dtype=FLOAT_DTYPE)
    triangles = np.ascontiguousarray(triangles, dtype=INT_DTYPE)

    cdef int n_points = points.shape[0]
    cdef vector[vector[int]] neighbors = vector[vector[int]](n_points)

    #Compute neighbors
    cdef int i, j, k, l
    cdef int n_triangles = triangles.shape[1]

    cdef INT_DTYPE_t [:, :] edge_faces = -1 * np.ones((2, 3 * n_triangles), dtype=INT_DTYPE)
    cdef INT_DTYPE_t [:, :] edge_indices = -1 * np.ones((2, 3 * n_triangles), dtype=INT_DTYPE)
    cdef INT_DTYPE_t [:] n_adjacent_triangles = np.zeros(3 * n_triangles, dtype=INT_DTYPE)

    # edges store the (repeated edges)
    cdef INT_DTYPE_t [:, :] edges = np.zeros((2, 3 * n_triangles), dtype=INT_DTYPE)

    for i in range(n_triangles):

        edges[0, 3 * i] = triangles[0, i]
        edges[1, 3 * i] = triangles[1, i]

        edges[0, 3 * i + 1] = triangles[0, i]
        edges[1, 3 * i + 1] = triangles[2, i]

        edges[0, 3 * i + 2] = triangles[1, i]
        edges[1, 3 * i + 2] = triangles[2, i]

    # Sort edges by lexicographic sort
    edges = np.sort(np.asarray(edges), axis=0)
    cdef INT_DTYPE_t[:] order = np.lexsort((np.asarray(edges[1, :]), np.asarray(edges[0, :])))

    # Remove duplicates
    cdef INT_DTYPE_t[:, :] tight_edges = np.zeros((2, 3 * n_triangles), dtype=INT_DTYPE)
    cdef INT_DTYPE_t n_keep = 1

    cdef INT_DTYPE_t[:] triangles_ref = np.repeat(np.arange(n_triangles), repeats=3)
    cdef INT_DTYPE_t[:] indice_ref = np.tile(np.array([2, 1, 0], dtype=INT_DTYPE), reps=n_triangles)

    tight_edges[:, 0] = edges[:, order[0]]
    edge_faces[0, 0] = triangles_ref[order[0]]
    edge_indices[0, 0] = indice_ref[order[0]]
    n_adjacent_triangles[0] = 1

    for i in range(1, 3 * n_triangles):
        if (edges[0, order[i]] != edges[0, order[i - 1]]) or (edges[1, order[i]] != edges[1, order[i - 1]]):
            tight_edges[:, n_keep] = edges[:, order[i]]
            edge_faces[0, n_keep] = triangles_ref[order[i]]
            edge_indices[0, n_keep] = indice_ref[order[i]]
            n_adjacent_triangles[n_keep] = 1
            n_keep += 1
        else:
            edge_faces[1, n_keep - 1] = triangles_ref[order[i]]
            edge_indices[1, n_keep - 1] = indice_ref[order[i]]
            n_adjacent_triangles[n_keep - 1] += 1

    return np.asarray(tight_edges)[:, :n_keep]
