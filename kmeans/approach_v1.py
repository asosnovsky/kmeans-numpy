from random import sample
from typing import Tuple
import numpy as np

class KmeansVariationErr(Exception): pass


def kMeans(data: np.ndarray, k: int, max_iters=1E10) -> Tuple[np.ndarray, np.ndarray]:
    """An implementation of the kMeans algorithmn using numpy

    Arguments:
        data {np.ndarray(shape=(n,r))} -- a two-dimensional numpy array where axis=0 represents each observation
        k {int} -- the number of clusters
        max_iters -- control the maximum number of times that the algorithmn can run

    Returns:
        centers {np.ndarray(shape=(k,r))} -- a list of centeroid points
        labels {np.ndarray(shape=(n))} -- the labels for each observation in the data
    """

    # Checking uniqueness for initial start
    unique_vals = np.unique(data, axis=0)
    if unique_vals.shape[0] < k:
        raise KmeansVariationErr(
            "Not enough variation found in the provided code")

    # initial values
    centers = np.array(sample(list(unique_vals), k))
    dist = np.array([ ((data-c)**2).sum(1) for c in centers ])

    # keep a record of labels (this will always be of size 2)
    label_hist = [dist.argmin(0)]

    # compute first iteration manually
    new_centers = np.array([ data[label_hist[0]==c].mean(0) for c in range(k) ])
    new_dist = np.array([ ((data-c)**2).sum(1) for c in new_centers ])
    labels = new_dist.argmin(0)

    # Simply return if no movement occured
    if (labels == label_hist[0]).sum() == data.shape[0]:
        return centers, labels

    # Update storage
    centers = new_centers
    dist = new_dist
    label_hist.append(labels)

    # a while-loop limited in scope
    for _ in range(int(max_iters)):
        # recompute
        new_centers = np.array([ data[label_hist[1]==c].mean(0) for c in range(k) ])
        new_dist = np.array([ ((data-c)**2).sum(1) for c in new_centers ])
        labels = new_dist.argmin(0)

        # check if same as last
        if (labels == labels[1]).sum() == data.shape[0]:
            return centers, labels

        # check if same as the one from two steps ago (oscillations)
        if (labels == label_hist[0]).sum() == data.shape[0]:
            return centers, labels

        # update storage
        centers = new_centers
        dist = new_dist
        # the label history should be kept at size=2
        # otherwise we risk a memory leak (which could only happen if the algorithmn runs for too long)
        label_hist[0] = label_hist[1]
        label_hist[1] = labels

    print("WARN: Reached maximum iters, convergence not guaranteed")
    return centers, labels





