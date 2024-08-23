import math
from random import random
import numpy as np
import scipy.stats as stats

NONE_INT = np.iinfo(np.uint64).max


class AutoDistCounters:
    def __init__(self, size: int):
        self.mean = 0
        self.variance = 1
        self.total_counter = 0
        self.size = 0
        self.keys = np.full(size + 1, NONE_INT, dtype=np.uint64)
        self.keys_indice = dict()

    def update_distribution(self, index):
        log_index = math.log(index + 1)
        self.total_counter += 1
        delta = log_index - self.mean
        self.mean += delta / self.total_counter
        delta2 = log_index - self.mean
        self.variance += delta * delta2
    
    def normal_cdf(self, x):
        std_dev = math.sqrt(self.variance)
        return stats.norm.cdf(x, loc=self.mean, scale=std_dev)

    def estimate_counter(self, index):
        log_index_min = math.log(index + 0.5)
        log_index_max = math.log(index + 1.5)
        pmf_value = self.normal_cdf(log_index_max) - self.normal_cdf(log_index_min)
        return pmf_value * self.total_counter

    def rebalance_estimate(self, index, value):
        curr_index = index
        tresh = self.swap_probability(curr_index, value)
        while random() <= tresh and curr_index > 0:
            self.swap(curr_index)
            curr_index -= 1
            tresh = self.swap_probability(curr_index, value)

    def swap_probability(self, index, value):
        if index == 0:
            return 0
        value_small = self.estimate_counter(index + 1)
        value_large = self.estimate_counter(index)
        difference = value_large - value_small
        if difference == 0:
            return 0
        # consider swapping more than a single step up
        probability = min(value / difference, 1)
        return probability

    def swap(self, index):
        self.keys[index], self.keys[index - 1] = self.keys[index - 1], self.keys[index]
        self.keys_indice[self.keys[index]], self.keys_indice[self.keys[index-1]] = index, index - 1

    def update(self, key, value):
        index = self.find_index_of_key(key)
        if index != NONE_INT:
            self.update_distribution(index)
            if index > 0:
                self.rebalance_estimate(index, 1)
        else:
            self.keys[self.size] = key
            self.keys_indice[key] = self.size
            self.rebalance_estimate(self.size, 1)
            if self.keys[-1] != NONE_INT:
                del self.keys_indice[self.keys[-1]]
                self.keys[-1] = NONE_INT
            self.size = min(len(self.keys) - 1, self.size + 1)


    def find_index_of_key(self, key):
        if key not in self.keys_indice:
            return NONE_INT
        else:
            return self.keys_indice[key]       

    def query(self, key):
        index = self.find_index_of_key(key)
        if index == NONE_INT:
            return 0
        estimate = self.estimate_counter(index + 1)
        return max(1, round(estimate))
