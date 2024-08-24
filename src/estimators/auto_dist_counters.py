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
        if x <= 0:
            return 0  # PDF is 0 for x <= 0 in log-normal distribution
        stddev = np.sqrt(variance)
        coefficient = 1 / (x * stddev * np.sqrt(2 * np.pi))
        exponent = -((np.log(x) - mean) ** 2) / (2 * variance)
        return coefficient * np.exp(exponent)
    
    @staticmethod
    def distribution_cdf(x, mean, variance):
        if x <= 0:
            return 0  # CDF is 0 for x <= 0 in log-normal distribution
        stddev = np.sqrt(variance)
        z = (np.log(x) - mean) / stddev
        return stats.norm.cdf(z)
    
    def estimate_counter(self, index):
        variance = self.get_variance()
        if variance == 0:
            return 1
        pmf = self.distribution_pdf(index + 0.5, self.mean, variance) - self.distribution_pdf(index + 1.5, self.mean, variance)
        return self.total_counter * pmf

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
        value_small = self.estimate_counter(index)
        value_large = self.estimate_counter(index - 1)
        difference = value_large - value_small
        if difference == 0:
            return 0
        # consider swapping more than a single step up
        probability = min(value / difference, 1)
        return probability

    def swap(self, index):
        self.keys[index], self.keys[index - 1] = self.keys[index - 1], self.keys[index]

    def update_distribution_tail(self):
        distribution = stats.norm(loc=self.mean, scale=np.sqrt(self.get_variance()))
        # Find the CDF value at the threshold 
        cdf_value = distribution.cdf(math.log(self.size))
        # Generate a random value from the uniform distribution
        u = np.random.uniform(cdf_value, 1)
        # Use the inverse CDF (ppf) to get a value from the normal distribution
        log_index = float(distribution.ppf(u))
        self.total_counter += 1
        delta = log_index - self.mean
        self.mean += delta / self.total_counter
        delta2 = log_index - self.mean
        self.M2 += delta * delta2
    
    def update(self, key, value):
        index = self.find_index_of_key(key)
        if index != NONE_INT:
            self.update_distribution(index)
            if index > 0:
                index = self.rebalance_estimate(index, 1)
        else:
            self.keys[self.size] = key
            index = self.rebalance_estimate(self.size, 1)
            if index < len(self.keys) - 1:
                self.update_distribution(index)
            else:
                self.total_counter += 1
            self.keys[-1] = NONE_INT
            self.size = min(len(self.keys) - 1, self.size + 1)


    def find_index_of_key(self, key):
        indice = np.where(self.keys == key)[0]
        if len(indice) == 0:
            return NONE_INT
        elif len(indice) == 1:
            index = int(indice[0])
            return index
        else:
            raise Exception('duplicate keys!')

    def query(self, key):
        index = self.find_index_of_key(key)
        if index == NONE_INT:
            return 0
        estimate = self.estimate_counter(index)
        return max(1, estimate)
