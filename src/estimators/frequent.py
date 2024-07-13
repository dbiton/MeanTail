class Frequent:
    def __init__(self, size):
        self.decrements = 0
        self.size = size
        self.counters = {}

    def update(self, index, value):
        if index in self.counters:
            self.counters[index] += value
        else:
            if len(self.counters) >= self.size:
                self.decrements += 1
                keys_to_remove = [k for k, v in self.counters.items() if v - self.decrements <= 0]
                self.counters = {k: v for k, v in self.counters.items() if k not in keys_to_remove}
            if len(self.counters) < self.size:
                self.counters[index] = value

    def query(self, index):
        return self.counters.get(index, 0) - self.decrements