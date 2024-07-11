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

import matplotlib.pyplot as plt

def evaluate(estimator, stream):
    for e in stream:
        estimator.update(e, 1)
    actual_counts = Counter(stream)
    estimate_counts = {k: estimator.query(k) for k in actual_counts}
    diff_counts = {k: abs(estimate_counts[k]-a) for k, a in actual_counts.items()}
    diff_counts_values_sorted = sorted(diff_counts.values(), reverse=True)    
    return sum(diff_counts.values()) / len(actual_counts), diff_counts_values_sorted



if __name__ == "__main__":
    np.random.seed(42069)
    stream_size = 1000000
    key_count = 100000
    estimator_size = 10000
    cm_depth = 5
    dists = [
        dist.ExponentialDistribution(key_count),
        dist.NormalDistribution(key_count),
        dist.UniformDistribution(key_count),
    ]
    
    data = {}
    
    for d in dists:
        probability_name = d.__class__.__name__
        logger.info(probability_name)
        probability_function = d.probability
        ss = SpaceSaving(estimator_size // 1.5)
        dc = DistCounters(estimator_size, probability_function)
        cm = CountMin(estimator_size // cm_depth, cm_depth)
        stream = d.generate(stream_size)

        data[probability_name] = {}
        
        for estimator in [ss, dc, cm]:
            t0 = time()
            ss_error, deltas = evaluate(estimator, stream)
            t1 = time()
            estimator_name = estimator.__class__.__name__
            logger.info(f"[{estimator_name}] aae: {round(ss_error,2)} time: {round(t1 - t0,2)}")
            data[probability_name][estimator_name] = deltas
    
    fig, axs = plt.subplots(nrows=len(data),ncols=len(data), figsize=(10, 10))
    for i_prob, probability_name in enumerate(data):
        for i_est, estimator_name in enumerate(data[probability_name]):
            ax = axs[i_prob][i_est]
            diff_counts_values_sorted = data[probability_name][estimator_name]
            ax.plot(np.arange(len(diff_counts_values_sorted)), diff_counts_values_sorted)
            ax.set_xlabel('rank')
            ax.set_ylabel('absolute error')
            ax.set_title(f'{probability_name} {estimator_name}')
    plt.tight_layout()
    plt.show()
