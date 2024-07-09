import numpy as np
from scipy.stats import norm, uniform, binom, poisson, expon

class Distribution:
    def __init__(self, range):
        self.range = range

    def generate(self, n):
        raise NotImplementedError("Generate method not implemented!")

    def probability(self, n):
        raise NotImplementedError("Probability method not implemented!")

class NormalDistribution(Distribution):
    def __init__(self, range):
        super().__init__(range)
        self.mean = 0
        self.std = range / 6

    def generate(self, n):
        return np.random.normal(self.mean, self.std, n)

    def probability(self, n):
        # Assuming n-th most probable integer is around the mean
        x_values = np.arange(self.mean - 3*self.std, self.mean + 3*self.std, 0.1)
        probabilities = norm.pdf(x_values, self.mean, self.std)
        sorted_indices = np.argsort(probabilities)[::-1]
        n_most_probable_index = sorted_indices[n - 1]
        return probabilities[n_most_probable_index]

class UniformDistribution(Distribution):
    def __init__(self, range):
        super().__init__(range)
        self.low = 0
        self.high = range

    def generate(self, n):
        return np.random.uniform(self.low, self.high, n)

    def probability(self, n):
        # All items have the same probability in uniform distribution
        return 1 / self.range

class BinomialDistribution(Distribution):
    def __init__(self, range):
        super().__init__(range)
        self.trials = range
        self.p = 0.5

    def generate(self, n):
        return np.random.binomial(self.trials, self.p, n)

    def probability(self, n):
        probabilities = [binom.pmf(k, self.trials, self.p) for k in range(self.trials + 1)]
        sorted_probabilities = sorted(probabilities, reverse=True)
        return sorted_probabilities[n - 1]

class PoissonDistribution(Distribution):
    def __init__(self, range):
        super().__init__(range)
        self.lam = range / 2

    def generate(self, n):
        return np.random.poisson(self.lam, n)

    def probability(self, n):
        # Since the Poisson distribution is discrete and potentially infinite, we approximate
        x_values = np.arange(0, 10*self.lam)
        probabilities = poisson.pmf(x_values, self.lam)
        sorted_indices = np.argsort(probabilities)[::-1]
        n_most_probable_index = sorted_indices[n - 1]
        return probabilities[n_most_probable_index]

class ExponentialDistribution(Distribution):
    def __init__(self, range):
        super().__init__(range)
        self.scale = range / 2

    def generate(self, n):
        return np.random.exponential(self.scale, n)

    def probability(self, n):
        # Assuming n-th most probable integer is around the mode (which is 0 for exponential)
        x_values = np.linspace(0, 10*self.scale, 1000)
        probabilities = expon.pdf(x_values, scale=self.scale)
        sorted_indices = np.argsort(probabilities)[::-1]
        n_most_probable_index = sorted_indices[n - 1]
        return probabilities[n_most_probable_index]

# Example usage
if __name__ == "__main__":
    range = 10
    n = 3

    distributions = [
        NormalDistribution(range),
        UniformDistribution(range),
        BinomialDistribution(range),
        PoissonDistribution(range),
        ExponentialDistribution(range)
    ]

    for dist in distributions:
        print(f"{dist.__class__.__name__} {n}-th most probable value probability: {dist.probability(n)}")
