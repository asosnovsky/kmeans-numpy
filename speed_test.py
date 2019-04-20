from timeit import timeit

REPS = 10000
DATA_STEP = """
from random import randint, seed
import numpy as np
seed(0)
data = np.array([
    [ ( i % 3 ) , randint(0, 50) ]
    for i in range(1000)
])
num_of_clusters = 3
"""

print("   Comparing the execution of various implementation")
print("\n".join([
    "way #{} = {:.4f}ms".format(i, 1000*t/REPS)
    for i, t in 
    enumerate([
        timeit(
            number=REPS,
            stmt="kMeans(data, num_of_clusters)",
            setup=f"from kmeans.approach_v1 import kMeans\n{DATA_STEP} "
        ),
        timeit(
            number=REPS,
            stmt="kMeans(data, num_of_clusters)",
            setup=f"from kmeans.approach_v2 import kMeans\n{DATA_STEP} "
        ),
        timeit(
            number=REPS,
            stmt="kMeans(data, num_of_clusters)",
            setup=f"from kmeans.approach_v3 import kMeans\n{DATA_STEP} "
        ),
        timeit(
            number=REPS,
            stmt="kMeans(data, num_of_clusters)",
            setup=f"from kmeans.approach_v4 import kMeans\n{DATA_STEP} "
        ),
        timeit(
            number=REPS,
            stmt="cy_kmeans(data, num_of_clusters)",
            setup=f"from kmeans import cy_kmeans\n{DATA_STEP} "
        )
    ])
]))
print("sklearn = {:.4f}ms".format(
    1000*timeit(
        number=REPS,
        stmt="KMeans(n_clusters=num_of_clusters).fit(data)",
        setup=f"from sklearn.cluster import KMeans\n{DATA_STEP}"
    )/REPS
))
