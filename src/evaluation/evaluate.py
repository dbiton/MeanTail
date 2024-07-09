import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from estimators.space_saving import SpaceSaving
from estimators.dist_counters import DistCounters
from estimators.count_min import CountMin
import misc.distribution as dist
from misc.logger import logger

from collections import Counter
from random import randint
from time import time
import numpy as np


def evaluate(estimator, stream):
    for e in stream:
        estimator.update(e, 1)
    actual_counts = Counter(stream)
    differences = [abs(estimator.query(k) - v) for k, v in actual_counts.items()]
    return sum(differences)


if __name__ == "__main__":
    np.random.seed(42069)
    stream_size = 500
    key_count = 50
    estimator_size = 16
    cm_depth = 4
    dists = [
        dist.BinomialDistribution(key_count),
        dist.ExponentialDistribution(key_count),
        dist.NormalDistribution(key_count),
        dist.PoissonDistribution(key_count),
        dist.UniformDistribution(key_count),
    ]
    for d in dists:
        logger.info(d.__class__.__name__)
        probability_function = d.probability
        ss = SpaceSaving(estimator_size / 2)
        dc = DistCounters(estimator_size, probability_function)
        cm = CountMin(estimator_size // cm_depth, cm_depth)
        stream = d.generate(stream_size)
        for estimator in [ss, dc, cm]:
            t0 = time()
            ss_error = evaluate(estimator, stream)
            t1 = time()
            estimator_name = estimator.__class__.__name__
            logger.info(f"[{estimator_name}] error: {round(ss_error,2)} time: {round(t1 - t0,2)}")