import numpy as np
cimport numpy as np

from random import sample

class KmeansVariationErr(Exception): pass

def kMeans(np.ndarray data, int k, int max_iters = 1000000):
    """An implementation of the kMeans algorithmn using numpy

    Arguments:
        data {np.ndarray(shape=(n,r))} -- a two-dimensional numpy array where axis=0 represents each observation
        k {int} -- the number of clusters
        max_iters -- control the maximum number of times that the algorithmn can run

    Returns:
        centers {np.ndarray(shape=(k,r))} -- a list of centeroid points
        labels {np.ndarray(shape=(n))} -- the labels for each observation in the data
    """

    cdef int N_ROWS = data.shape[0]
    cdef int N_COLS = data.shape[1]
    cpdef np.ndarray centers_hist = np.ndarray(shape=(k, N_COLS, 2), dtype=np.float)
    cdef np.ndarray dist_hist = np.ndarray(shape=(k, N_ROWS, 2), dtype=np.float)
    cpdef np.ndarray label_hist = np.ndarray(shape=(N_ROWS, 3), dtype=np.int)
    cdef np.ndarray c_point

    # Checking uniqueness for initial start
    cdef np.ndarray unique_vals = np.unique(data, axis=0)
    if unique_vals.shape[0] < k:
        raise KmeansVariationErr("Not enough variation found in the provided data")

    # initial values
    centers_hist[:, :, 0] = unique_vals[
        sample( range(unique_vals.shape[0]), k )
    ]
    dist_hist[:, :, 0] = np.array([ ((data-c)**2).sum(1) for c in centers_hist[:, :, 0] ])
    label_hist[:,1] = dist_hist[:, :, 0].argmin(0)

    # compute first iteration manually
    for c in range(k):
        c_point = data[label_hist[:,1] == c].mean(0)
        dist_hist[c, :, 1] = ((data-c_point)**2).sum(1)
        centers_hist[c, :, 1] = c_point
    label_hist[:, 2] = dist_hist[:, :, 1].argmin(0)

    # Simply return if no movement occured
    if (label_hist[:, 1] == label_hist[:,2]).all():
        return centers_hist[:,:,-1], label_hist[:,-1]

    # a while-loop limited in scope
    for _ in range(int(max_iters)):
        # recompute
        for c in range(k):
            c_point = data[label_hist[:,-1] == c].mean(0)
            dist_hist[c, :, 1] = ((data-c_point)**2).sum(1)
            centers_hist[c, :, 1] = c_point
        label_hist[:, 2] = dist_hist[:, :, 1].argmin(0)

        # check if same as last
        if (label_hist[:,-1] == label_hist[:, 1]).all():
            return centers_hist[:,:,-1], label_hist[:,-1]

        # check if same as the one from two steps ago (oscillations)
        if (label_hist[:,-1] == label_hist[:, 0]).all():
            return centers_hist[:,:,-1], label_hist[:,-1]

        # update storage
        centers_hist[:,:,0] = centers_hist[:,:,1]
        dist_hist[:,:,0] = dist_hist[:,:,1]

        # the label history should be kept at size=2
        # otherwise we risk a memory leak (which could only happen if the algorithmn runs for too long)
        label_hist[:, 0] = label_hist[:, 1]
        label_hist[:, 1] = label_hist[:, 2]

    print("WARN: Reached maximum iters, convergence not guaranteed")
    return centers_hist[:,:,-1], label_hist[:,-1]





