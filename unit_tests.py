import kmeans.cy.kmeans as cy_kmeans
import kmeans.approach_v1 as v1_kmeans
import kmeans.approach_v2 as v2_kmeans
from random import randint, seed
from typing import Callable
import numpy as np
import unittest

modules = [
    v1_kmeans,
    v2_kmeans,
    cy_kmeans
]

# ============================ #
#   Util Methods
# ============================ #
def relevel_cls_labs(labs: np.ndarray) -> np.ndarray:
    """re-level the labels
    
    Arguments:
        labs {np.ndarray(shape=(N,1))}
    
    Returns:
        np.ndarray(shape=(N,1))
    """
    nlabs = np.ndarray(shape=labs.shape, dtype=int)
    cls, first = np.unique(labs, return_index=True)
    for nidx, (_, c) in enumerate(sorted(zip(first.tolist(), cls.tolist()))):
        nlabs[labs == c] = nidx
    return nlabs

# ============================ #
#   All Unit Tests
# ============================ #
class TestModule(unittest.TestCase):
    def test_mod3(self):
        """This test generate excatly 3 distinct classes on a 1-d plane
        """
        data = np.array([
            [(i % 3)]
            for i in range(1000)
        ])
        for mod in modules:
            out = mod.kMeans(data, 3)
            self.assertTrue(
                (relevel_cls_labs(out[1]) == data.flatten()).all()
            )

    def test_mod7(self):
        """This test generate excatly 7 distinct classes on a 1-d plane
        """
        data = np.array([
            [(i % 7)]
            for i in range(1000)
        ])
        for mod in modules:
            out = mod.kMeans(data, 7)
            self.assertTrue(
                (relevel_cls_labs(out[1]) == data.flatten()).all()
            )


    def test_mod3_mod2(self):
        """This test generate excatly 6 distinct classes on a 2-d plane
            Note, that the end result is validated against the expected sum of the differences for each class
                the releveled classes will be 0,1,2,3,4,5 , so their differences would be -1,-1,-1,-1,-1,5,
                if predicted correctly, the classes will return ordered, so the process should become determenistic
        """
        data = np.array([
            [i % 3, i % 2]
            for i in range(1000)
        ])
        for mod in modules:
            out = mod.kMeans(data, 6)
            self.assertTrue(
                np.diff(relevel_cls_labs(out[1])).sum(),
                (999-166)-166*5
            )

    def test_variation_err(self):
        """This tests ensures that the algo raises an exception during impossible cases
        """
        data = np.array([
            [0, 0]
            for i in range(1000)
        ])
        for mod in modules:
            with self.assertRaises(mod.KmeansVariationErr):
                mod.kMeans(data, 7)

# ============================ #
#   Run Tests
# ============================ #
unittest.main()
