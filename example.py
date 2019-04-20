import numpy as np
from kmeans.approach_v1 import kMeans, KmeansVariationErr
from random import randint

# Example 1
# I generate some data where the x-coords is the modulus of the index by 3, 
# this ensures that will 3 unique clusters
data = np.array([
    [ ( i % 3 ) , randint(0, 50) ]
    for i in range(1000)
])
kMeans(data, 3)

# Example 2
# this test that an error is thrown when expected
try:
    data = np.array([
        [0, 0]
        for i in range(1000)
    ])
    kMeans(data, 3)
except KmeansVariationErr as kerr:
    print("succesfully failed")
