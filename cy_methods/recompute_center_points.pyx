#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.int_t DTYPE_t

def recompute_center_points(np.ndarray data, np.ndarray labels, int k):
    cdef int N = data.shape[0]
    cdef int V = data.shape[1]
    cpdef np.ndarray centers = np.ndarray(shape=(k, V), dtype=DTYPE)
    cpdef np.ndarray dist = np.ndarray(shape=(k, N), dtype=DTYPE)

    for c in range(k):
        c_point = data[labels == c].mean(0)
        dist[c, :] = ((data-c_point)**2).sum(1)
        centers[c, :] = c_point

    return dist.argmin(0), dist, centers

