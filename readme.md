# kMeans Implementation Tests

This repo contains 5 different implementation of the kmeans algorithmn.

This implementation utilizes numpy to write a very simple implementation of kMeans. 

While simple at its core, I also demonstrate that this is an efficient implementation of the algorithmn as it exceeds a common standard implementation found in the sklearn library. (see [speed_text.py](./speed_test.py))

While simple at its core this implementation will assume convergance if one of the following conditions is met:

1. No movement is detected in the labels since the iteration
2. Oscillations is detected between the latest and the earlier labels

Additionally, this implementation provide us with the following guards:

1. An exception is raised if there is not enough variation in the dataset
2. A warning is read if no convergence is detected

# File Descriptions

1. [kmeans.py](./kmeans.py): the location of the implementation
2. [example.py](./example.py): the example of how the algo runs
3. [speed_test.py](./speed_test.py): a speed test compared against sklearn