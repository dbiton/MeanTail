from random import random

class RandomAdmissionPolicy:
    def __init__(self, size):
        self.size = size
        self.counters = {}
  
    def memory_usage(self):
      return self.size * 2 * 4

    def update(self, index, value):
        if index in self.counters:
            self.counters[index] += value
        else:
          if len(self.counters) < self.size:
            self.counters[index] = value
          else:
            min_counter_index = min(self.counters, key=self.counters.get)
            min_counter = self.counters[min_counter_index]
            thresh = 1 / (min_counter + 1)
            if random() < thresh:
              del self.counters[min_counter_index]
              self.counters[index] = min_counter + value

    def query(self, index):
        return self.counters.get(index, 0)