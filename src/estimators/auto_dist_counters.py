import math
from random import random
from scipy.stats import lognorm
from scipy.integrate import quad
import numpy as np


class AutoDistCounters:
    def __init__(self, size):
        self.mean = 0
        self.variance = 1
        self.total_counter = 0
        self.size = size
        self.keys = []

    def update_distribution(self, index):
        self.total_counter += 1
        log_index = math.log(index+1)
        old_mean = self.mean
        self.mean += (log_index - self.mean) / self.total_counter
        self.variance += (log_index - old_mean) * (log_index - self.mean)

    def estimate_counter(self, index):
        if index <= 0:
            raise Exception('illegal index')
        stddev = np.sqrt(self.variance)
        scale = np.exp(self.mean)
        pdf = lognorm(s=stddev, scale=scale).pdf
        pmf_value, _ = quad(pdf, index, index + 1)
        return pmf_value * self.total_counter

    def rebalance_estimate(self, index, value):
        curr_index = index
        tresh = self.swap_probability(curr_index, value)
        while tresh >= 1:
            self.swap(curr_index)
            curr_index -= 1
            tresh = self.swap_probability(curr_index, value)
        if random() < tresh and curr_index > 0:
            self.swap(curr_index)

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
        tmp = self.keys[index]
        self.keys[index] = self.keys[index - 1]
        self.keys[index - 1] = tmp

    def update(self, key, value):
        if key in self.keys:
            index = self.keys.index(key)
            for _ in range(value):
                self.update_distribution(index)
            if index > 0:
                self.rebalance_estimate(index, value)
        else:
            self.keys.append(key)
            self.rebalance_estimate(len(self.keys)-1, value)
            while len(self.keys) >= self.size:
                self.keys.pop()

    def query(self, key):
        if key in self.keys:
            index = self.keys.index(key)
            estimate = self.estimate_counter(index + 1)
            return max(1, round(estimate))
        else:
            return 0
