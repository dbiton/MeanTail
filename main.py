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
    dist = lambda index : (1 / harmonic_approx) * (index+1) ** (-zipf_param)
    estimator = DistCounters(1000, dist)
    actual_counts = Counter(stream)
    for v in stream:
        estimator.update(v, 1)
    estimate_counts = [estimator.query(k) for k in actual_counts.keys()]
    total_error = sum([abs(e0-e1) for e0, e1 in zip(actual_counts, estimate_counts)])
    print(total_error)
      

if __name__ == "__main__":
    main()
