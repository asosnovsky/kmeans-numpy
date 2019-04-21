# kMeans Implementation Tests

This repo contains 3 different implementation of the kmeans algorithmn.

- v1: heavily reliant numpy version
- v2: same as v1, but certain operations have been combined
- v3: a cython version of v2

Additionally, I compare this implementation to the standard implementation found in the sklearn library. (see [speed_text.py](./speed_test.py))

While simple at its core this implementation will assume convergance if one of the following conditions is met:

1. No movement is detected in the labels since the iteration
2. Oscillations is detected between the latest and the earlier labels

Additionally, this implementation provide us with the following guards:

1. An exception is raised if there is not enough variation in the dataset
2. A warning is read if no convergence is detected

# File Descriptions

1. [kmeans](./kmeans): the location of the various implementations
2. [example.py](./example.py): the example of how the algo runs
3. [unit_tests.py](./unit_tests.py): some unit tests that ensure some basic errors are caught and that all algorithmns provide similar outputs
4. [speed_test.py](./speed_test.py): a speed test compared against sklearn

# Speed Test Results


| Version | Time per round | Total Time | Number of rounds |
|---------|----------------|------------|------------------|
| v1      | 2.88ms         | 28.82s     | 10,000 |
| v2 | 2.40ms | 24.02s | 10,000 |
| v3 (v2-cy) | 2.43ms | 24.29s | 10,000 |
| benchmark (sklearn) | 23.44ms | 234.39s | 10,000 | 

