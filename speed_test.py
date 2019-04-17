from timeit import timeit

my_way = timeit(
    number=100,
    stmt="""kMeans(data, 3)""",
    setup="""
from kmeans import kMeans
from random import randint
import numpy as np

# I generate some data where the x-coords is the modulus of the index by 3, 
# this ensures that will 3 unique clusters
data = np.array([
    [ ( i % 3 ) , randint(0, 50) ]
    for i in range(1000)
])
"""
)

sklearn_way = timeit(
    number=100,
    stmt="KMeans(n_clusters=3).fit(data)",
    setup="""
import numpy as np
from sklearn.cluster import KMeans
from random import randint

data = np.array([
    [ ( i % 3 ) , randint(0, 50) ]
    for i in range(1000)
])
"""
)

print("""
    Comparing the execution of this code with sklearn

        - This Code = %2.2fs
        - Sklearn = %2.2fs
        - Times Faster = %2.0fx
""" % (my_way, sklearn_way, sklearn_way/my_way) )
