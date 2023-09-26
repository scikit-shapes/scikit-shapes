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

    cdef int n_points = points.shape[0]
    cdef vector[vector[int]] neighbors = vector[vector[int]](n_points)

    #Compute neighbors
    cdef int i, j, k, l
    cdef int n_triangles = triangles.shape[1]

    cdef INT_DTYPE_t [:, :] edges = np.zeros((2, 3 * n_triangles), dtype=INT_DTYPE)

    for i in range(n_triangles):

        edges[0, 3 * i] = triangles[0, i]
        edges[1, 3 * i] = triangles[1, i]
        edges[0, 3 * i + 1] = triangles[0, i]
        edges[1, 3 * i + 1] = triangles[2, i]
        edges[0, 3 * i + 2] = triangles[1, i]
        edges[1, 3 * i + 2] = triangles[2, i]

    # Lexicographic sort
    cdef INT_DTYPE_t[:] order = np.lexsort((np.asarray(edges[1, :]), np.asarray(edges[0, :])))
    
    # Remove duplicates
    cdef INT_DTYPE_t[:, :] tight_edges = np.zeros((2, 3 * n_triangles), dtype=INT_DTYPE)
    cdef INT_DTYPE_t n_keep = 1

    tight_edges[:, 0] = edges[:, order[0]]
    for i in range(1, 3 * n_triangles):
        if (edges[0, order[i]] != edges[0, order[i - 1]]) or (edges[1, order[i]] != edges[1, order[i - 1]]):
            tight_edges[:, n_keep] = edges[:, order[i]]
            n_keep += 1

    return np.asarray(tight_edges)[:, :n_keep]