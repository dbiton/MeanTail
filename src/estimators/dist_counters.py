from random import random


class DistCounters:
    def __init__(self, size, distribution):
        self.distribution = distribution
        self.size = size
        self.keys = []
        self.total_counter = 0

    def estimate_counter(self, index):
        return self.distribution(index) * self.total_counter

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

    def update(self, key, value):
        self.total_counter += value
        if key in self.keys:
            index = self.keys.index(key)
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
            return self.estimate_counter(index)
        else:
            return 0
