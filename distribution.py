import numpy as np
from scipy.stats import norm, uniform, binom, poisson, expon

class Distribution:
    def __init__(self, range):
        self.range = range

    def generate(self, n):
        raise NotImplementedError("Generate method not implemented!")

    def probability(self, item):
        raise NotImplementedError("Probability method not implemented!")

class NormalDistribution(Distribution):
    def __init__(self, range):
        super().__init__(range)
        self.mean = 0
        self.std = range / 6

    def generate(self, n):
        return np.random.normal(self.mean, self.std, n)

    def probability(self, item):
        return norm.pdf(item, self.mean, self.std)

class UniformDistribution(Distribution):
    def __init__(self, range):
        super().__init__(range)
        self.low = 0
        self.high = range

    def generate(self, n):
        return np.random.uniform(self.low, self.high, n)

    def probability(self, item):
        return uniform.pdf(item, loc=self.low, scale=self.high - self.low)

class BinomialDistribution(Distribution):
    def __init__(self, range):
        super().__init__(range)
        self.trials = range
        self.p = 0.5

    def generate(self, n):
        return np.random.binomial(self.trials, self.p, n)

    def probability(self, item):
        return binom.pmf(item, self.trials, self.p)

class PoissonDistribution(Distribution):
    def __init__(self, range):
        super().__init__(range)
        self.lam = range / 2

    def generate(self, n):
        return np.random.poisson(self.lam, n)

    def probability(self, item):
        return poisson.pmf(item, self.lam)

class ExponentialDistribution(Distribution):
    def __init__(self, range):
        super().__init__(range)
        self.scale = range / 2

    def generate(self, n):
        return np.random.exponential(self.scale, n)

    def probability(self, item):
        return expon.pdf(item, scale=self.scale)

# Example usage
if __name__ == "__main__":
    n = 10
    range = 10

    distributions = [
        NormalDistribution(range),
        UniformDistribution(range),
        BinomialDistribution(range),
        PoissonDistribution(range),
        ExponentialDistribution(range)
    ]

    for dist in distributions:
        samples = dist.generate(n)
        probs = [dist.probability(x) for x in samples]
        print(f"{dist.__class__.__name__} Samples:", samples)
        print(f"{dist.__class__.__name__} Probabilities:", probs)
