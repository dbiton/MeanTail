import math
from random import random
import numpy as np
import scipy.stats as stats

NONE_INT = np.iinfo(np.uint64).max


class AutoDistCounters:
    def __init__(self, size: int):
        self.mean = 0
        self.M2 = 0
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
        self.M2 += delta * delta2
    
    def get_variance(self):
        if self.total_counter <= 2:
            return 1
        return self.M2 / self.total_counter
    
    @staticmethod
    def distribution_pdf(x, mean, variance):
        normalizing_constant = 1 / (math.sqrt(2 * math.pi * variance))
        exponent = math.exp(-((x - mean) ** 2) / (2 * variance))
        pdf_value = normalizing_constant * exponent
        return pdf_value
    
    def estimate_counter(self, index):
        variance = self.get_variance()
        if variance == 0:
            return 1
        log_index = math.log(index+1)
        cdf = self.distribution_pdf(log_index, self.mean, variance)
        return self.total_counter * cdf

    def rebalance_estimate(self, index, value):
        curr_index = index
        tresh = self.swap_probability(curr_index, value)
        while random() <= tresh and curr_index > 0:
            self.swap(curr_index)
            curr_index -= 1
            tresh = self.swap_probability(curr_index, value)
        return curr_index

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

    def update_distribution_tail(self):
        distribution = stats.norm(loc=self.mean, scale=np.sqrt(self.get_variance()))
        # Find the CDF value at the threshold 
        cdf_value = distribution.cdf(math.log(self.size))
        # Generate a random value from the uniform distribution
        u = np.random.uniform(cdf_value, 1)
        # Use the inverse CDF (ppf) to get a value from the normal distribution
        log_index = distribution.ppf(u)
        self.total_counter += 1
        delta = log_index - self.mean
        self.mean += delta / self.total_counter
        delta2 = log_index - self.mean
        self.M2 += delta * delta2
    
    def update(self, key, value):
        index = self.find_index_of_key(key)
        if index != NONE_INT:
            if index > 0:
                rebalanced_index = self.rebalance_estimate(index, 1)
                self.update_distribution(rebalanced_index)
            else:
                self.update_distribution(0)
        else:
            self.keys[self.size] = key
            self.keys_indice[key] = self.size
            rebalanced_index = self.rebalance_estimate(self.size, 1)
            # key was not added to our keys - it remains in the tail
            if self.keys[-1] != NONE_INT:
                self.update_distribution_tail()
                del self.keys_indice[self.keys[-1]]
                self.keys[-1] = NONE_INT
            else:
                self.update_distribution(rebalanced_index)
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
