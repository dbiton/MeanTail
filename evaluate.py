from space_saving import SpaceSaving
from dist_counters import DistCounters
import distribution as dist

from collections import Counter
from random import randint
from time import time


def evaluate(estimator, stream):
    for e in stream:
        estimator.update(e, 1)
    actual_counts = Counter(stream)
    differences = [abs(estimator.query(k) - v) for k, v in actual_counts.items()]
    return sum(differences)


if __name__ == "__main__":
    stream_size = 100000
    key_count = 100
    dists = [
        dist.BinomialDistribution(key_count),
        dist.ExponentialDistribution(key_count),
        dist.NormalDistribution(key_count),
        dist.PoissonDistribution(key_count),
        dist.UniformDistribution(key_count),
    ]
    for d in dists:
        probability_function = d.probability
        ss = SpaceSaving(10)
        dc = DistCounters(20, probability_function)
        stream = d.generate(stream_size)
        t0 = time()
        ss_error = evaluate(ss, stream)
        t1 = time()
        dc_error = evaluate(dc, stream)
        t2 = time()
        print(d.__class__.__name__)
        print("space saving:", ss_error, t1 - t0)
        print("dist counters:", dc_error, t2 - t1)
