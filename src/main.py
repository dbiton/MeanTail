import numpy as np
from scipy.special import zeta
from collections import Counter

from dist_counters import DistCounters


def main():
    zipf_param = 2
    stream_size = 100000
    np.random.seed(0)
    stream = np.random.zipf(zipf_param, stream_size)
    harmonic_approx = zeta(zipf_param)
    dist = lambda index: (1 / harmonic_approx) * (index + 1) ** (-zipf_param)
    estimator = DistCounters(1000, dist)
    actual_counts = Counter(stream)
    for v in stream:
        estimator.update(v, 1)
    estimate_counts = [estimator.query(k) for k in actual_counts.keys()]
    total_error = sum([abs(e0 - e1) for e0, e1 in zip(actual_counts, estimate_counts)])
    print(total_error)


if __name__ == "__main__":
    counters_count = 10
    keys_count = 100
    distribution = lambda index : 1 / keys_count
    dist_counters = DistCounters(counters_count, distribution)
    for i in range(keys_count):
        print(i)
        dist_counters.update(i, 10)
    for i in dist_counters.keys:
        dist_counters.query(i), 10, "Counter value should be updated to 10"
