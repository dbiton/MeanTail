class SpaceSaving:
    def __init__(self, size):
        self.size = size
        self.counters = {}

    def update(self, index, value):
        if index in self.counters:
            self.counters[index] += value
        else:
            if len(self.counters) >= self.size:
              index_smallest = min(self.counters, key=lambda k: self.counters[k])
              self.counters[index] = self.counters[index_smallest] + value
              del self.counters[index_smallest]
            else:
              self.counters[index] = value 

    def query(self, index):
        if index in self.counters:
            return self.counters[index]
        else:
            return 0