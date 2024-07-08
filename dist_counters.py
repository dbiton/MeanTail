from random import randrange

class DistCounters:
  def __init__(self, size, distribution):
    self.distribution = distribution
    self.size = size
    self.keys = []
    self.total_counter = 0

  def estimate_counter(self, index):
    return self.distribution(index + 1) * self.total_counter 

  def swap_probability(self, index, value):
    value_small = self.estimate_counter(index)
    value_large = self.estimate_counter(index - 1)
    difference = value_large - value_small
    if difference < 0:
      x = 0
    if difference == 0:
      return 0
    # consider swapping more than a single step up
    probability = min(value / difference, 1)
    return probability

  def swap(self, index):
    tmp = self.keys[index]
    self.keys[index] = self.keys[index-1]
    self.keys[index-1] = tmp
  
  def update(self, key, value):
    self.total_counter += value
    if self.keys.count(key) > 0:
      index = self.keys.index(key)
      if index > 0:
        tresh = self.swap_probability(index, value)
        if randrange(0, 1) >= tresh:
          self.swap(index)
    else:
      if len(self.keys) < self.size:
        self.keys.append(key)
      else:
        tresh = self.swap_probability(self.size-1, value)
        if randrange(0, 1) >= tresh:
          self.keys[-1] = key

  def query(self, key):
    if self.keys.count(key) > 0:
      index = self.keys.index(key)
      return self.estimate_counter(index)
    else:
      return 0