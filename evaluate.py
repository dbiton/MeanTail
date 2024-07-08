from space_saving import SpaceSaving
from dist_counters import DistCounters
import distribution as dist
from logger import logger

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
    stream_size = 1000
    key_count = 10
    dists = [
        dist.BinomialDistribution(key_count),
        dist.ExponentialDistribution(key_count),
        dist.NormalDistribution(key_count),
        dist.PoissonDistribution(key_count),
        dist.UniformDistribution(key_count),
    ]
    for d in dists:
        probability_function = d.probability
        ss = SpaceSaving(3)
        dc = DistCounters(6, probability_function)
        stream = d.generate(stream_size)
        t0 = time()
        ss_error = evaluate(ss, stream)
        t1 = time()
        dc_error = evaluate(dc, stream)
        t2 = time()
        logger.info(d.__class__.__name__)
        logger.info(f"[space saving] error: {round(ss_error,2)} time: {round(t1 - t0,2)}")
        logger.info(f"[dist counters] error: {round(dc_error,2)} time: {round(t2 - t1,2)}")
