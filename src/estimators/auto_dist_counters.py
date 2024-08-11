import math
from random import random
from scipy.stats import lognorm
from scipy.integrate import quad
import numpy as np

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
        self.total_counter += 1
        log_index = math.log(index + 1)
        old_mean = self.mean
        self.mean += (log_index - self.mean) / self.total_counter
        self.variance += (log_index - old_mean) * (log_index - self.mean)

    @staticmethod
    def lognormal_pdf(x, stddev, scale):
        mu = np.log(scale)  # Convert scale to mean of the underlying normal distribution
        sigma = stddev  # Standard deviation of the underlying normal distribution
        return (1 / (x * sigma * np.sqrt(2 * np.pi))) * \
            np.exp(-0.5 * ((np.log(x) - mu) ** 2) / (sigma ** 2))

    def estimate_counter(self, index):
        stddev = np.sqrt(self.variance)
        scale = np.exp(self.mean)
        pmf_value = self.lognormal_pdf(index, stddev, scale) - self.lognormal_pdf(index + 1, stddev, scale)
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
