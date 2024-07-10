import numpy as np
from scipy.stats import norm, uniform, binom, poisson, expon

class Distribution:
    def __init__(self, probabilities):
        self.sorted_probabilities = np.sort(probabilities)

    def generate(self, n):
        raise NotImplementedError("Generate method not implemented!")

    def probability(self, n):
        return self.sorted_probabilities[n]

class NormalDistribution(Distribution):
    def __init__(self, domain, probability_out_of_bound=0.001):
        self.domain = domain

        self.mean = domain / 2
        self.std = domain / 8

        x_values = np.arange(domain)
        probabilities = norm.pdf(x_values, self.mean, self.std)
        super().__init__(probabilities)

    def generate(self, n):
        return np.clip(np.round(np.random.normal(self.mean, self.std, n)), 0, self.domain)

class UniformDistribution(Distribution):
    def __init__(self, domain):
        self.low = 0
        self.high = domain
        probabilities = np.full(domain, 1/domain) 
        super().__init__(probabilities)

    def generate(self, n):
        return np.round(np.random.uniform(self.low, self.high, n))


class ExponentialDistribution(Distribution):
    def __init__(self, domain, probability_out_of_bound=0.001):
        self.scale = - domain / np.log(probability_out_of_bound)
        x_values = np.arange(domain)
        probabilities = expon.pdf(x_values)
        self.domain = domain
        super().__init__(probabilities)

    def generate(self, n):
        return np.clip(np.round(np.random.exponential(self.scale, n)), 0, self.domain)

