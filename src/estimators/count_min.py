import numpy as np
import mmh3

class CountMin:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.seeds = np.random.randint(1, 100000, size=depth)

    def _hash(self, index, i):
        return mmh3.hash(index, self.seeds[i]) % self.width

    def update(self, index, value):
        for i in range(self.depth):
            hash_index = self._hash(index, i)
            self.table[i, hash_index] += value

    def query(self, index):
        min_estimate = float('inf')
        for i in range(self.depth):
            hash_index = self._hash(index, i)
            min_estimate = min(min_estimate, self.table[i, hash_index])
        return min_estimate
