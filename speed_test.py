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

exec_types = [
    f"from kmeans.approach_v1 import kMeans\n{DATA_STEP}",
    f"from kmeans.approach_v2 import kMeans\n{DATA_STEP}",
    f"from kmeans import cy_kmeans as kMeans\n{DATA_STEP}"
]

print("Comparing the execution of various implementation")
for i, exc_type in enumerate(exec_types, 1):
    t = timeit(number=REPS, stmt="kMeans(data, num_of_clusters)", setup=exc_type)
    print(
        "  > v{} = {:.2f}ms (total = {:.2f}s)".format(i, 1000*t/REPS, t)
    )

sk_t = timeit(
    number=REPS,
    stmt="KMeans(n_clusters=num_of_clusters).fit(data)",
    setup=f"from sklearn.cluster import KMeans\n{DATA_STEP}"
)
print(
    "  > v{} = {:.2f}ms (total = {:.2f}s)".format(i, 1000*sk_t/REPS, sk_t)
)
print(f"each test ran {REPS} times")

#   Comparing the execution of various implementation
# way  v1 = 2.8790ms
# way  v2 = 2.9453ms
# way  v3 = 2.9453ms
# sklearn = 32.9870ms
