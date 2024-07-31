import numpy as np
import scipy
from scipy.stats import norm, uniform, binom, poisson, expon, zipf

class Distribution:
    def __init__(self, probabilities):
        self.sorted_probabilities = sorted(probabilities, reverse=True)

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
    def __init__(self, domain, probability_out_of_bounds=0.01):
        self.upper_bound = -np.log(probability_out_of_bounds)
        x_values = np.linspace(0, self.upper_bound, domain)
        probabilities = expon.pdf(x_values)
        probabilities *= 1 / sum(probabilities) # normalize
        self.domain = domain
        super().__init__(probabilities)

    def generate(self, n):
        return np.clip(np.round(self.domain / self.upper_bound * np.random.exponential(1, n)), 0, self.domain)

class ZipfianDistribution(Distribution):
    def __init__(self, domain, a=1.5):
        self.domain = domain
        self.a = a
        super().__init__([])
   
    def probability(self, n):
        H_N_a = scipy.special.zetac(self.a) + 1  # zetac(a) gives H(∞, a) - 1
        # Calculate the probability
        P_n = (1 / n**self.a) / H_N_a
        return P_n    
    
    def generate(self, n):
        return np.random.zipf(self.s, n) % self.domain