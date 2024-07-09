class SpaceSaving:
    def __init__(self, size):
        self.size = size
        self.counters = {}

    def update(self, index, value):
        if index in self.counters:
            self.counters[index] += value
        else:
            if len(self.counters) >= self.size:
                index_smallest = min(self.counters, key=self.counters.get)
                self.counters[index] = self.counters[index_smallest] + value
                del self.counters[index_smallest]
            else:
                self.counters[index] = value

    def query(self, index):
        return self.counters.get(index, 0)