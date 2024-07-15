import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from estimators.space_saving import SpaceSaving
from estimators.dist_counters import DistCounters
from estimators.rap import RandomAdmissionPolicy
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
    errors = {k: abs(estimate_counts[k]-a) for k, a in actual_counts.items()}
    diff_counts_values_sorted = sorted(errors.values(), reverse=True)    
    aae = sum(errors.values()) / len(actual_counts)
    are = sum([errors[k]/v for k, v in actual_counts.items()]) / len(actual_counts)
    return aae, are, diff_counts_values_sorted

def narrow():
    stream_size = 10000
    key_count = 1000
    estimator_size = 32
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
        ss = SpaceSaving(estimator_size)
        dc = DistCounters(estimator_size * 1.5, probability_function)
        cm = RandomAdmissionPolicy(estimator_size)
        stream = d.generate(stream_size)

        data[probability_name] = {}
        
        for estimator in [ss, dc, cm]:
            t0 = time()
            aae, are, deltas = evaluate(estimator, stream)
            t1 = time()
            estimator_name = estimator.__class__.__name__
            logger.info(f"[{estimator_name}] aae: {round(aae,2)} are: {round(are,2)} time: {round(t1 - t0,2)}")
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

def broad():
    min_stream_size = 1000
    max_stream_size = 100000
    key_count = 1000
    estimator_size = 256
    data_points = 16
    dists = [
        dist.ExponentialDistribution(key_count),
        dist.NormalDistribution(key_count)
    ]
    
    data = {d.__class__.__name__ : {} for d in dists}
    
    for d in dists:
        probability_name = d.__class__.__name__
        logger.info(probability_name)
        probability_function = d.probability
        ss = SpaceSaving(estimator_size)
        dc = DistCounters(estimator_size * 1.5, probability_function)
        cm = RandomAdmissionPolicy(estimator_size)
        estimators = [ss, dc, cm]
        data[probability_name] = {e.__class__.__name__ : [] for e in estimators}

        for estimator in estimators:
            for stream_size in np.linspace(min_stream_size, max_stream_size, data_points):
                stream = d.generate(int(stream_size))
                t0 = time()
                aae, are, deltas = evaluate(estimator, stream)
                t1 = time()
                estimator_name = estimator.__class__.__name__
                logger.info(f"[{estimator_name}] stream_size: {stream_size} aae: {round(aae,2)} are: {round(are,2)} time: {round(t1 - t0,2)}")
                data[probability_name][estimator_name].append((stream_size, are))
    
    fig, axs = plt.subplots(ncols=len(dists), figsize=(10, 10))
    for i_prob, probability_name in enumerate(data):
        axs[i_prob].set_xlabel('stream size')
        axs[i_prob].set_ylabel('mean average error')
        axs[i_prob].set_title(f'{probability_name}')
        for i_est, estimator_name in enumerate(data[probability_name]):
            streamsizes = [p[0] for p in data[probability_name][estimator_name]]
            ares = [p[1] for p in data[probability_name][estimator_name]]
            axs[i_prob].plot(streamsizes, ares, label=estimator_name)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.seterr(all='raise')
    np.random.seed(42069)
    broad()
    
