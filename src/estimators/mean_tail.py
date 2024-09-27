from random import random, randrange
import numpy as np

class MeanTail:
    def __init__(self, size, mem_percentage_tail=0.1):
        self.counters_size = int(size * (1 - mem_percentage_tail))
        self.tail_size = int(size * mem_percentage_tail * 2)
        self.counters = {}
        self.tail = []
        self.tail_total = 0

    def tail_average(self):
        return self.tail_total / len(self.tail)

    def attempt_promote_to_counters(self, key, value):
        min_counter_key = min(self.counters, key=self.counters.get)
        min_counter = self.counters[min_counter_key]
        tail_average = self.tail_average()
        divisor = max(1, 1 + min_counter - tail_average)
        thresh = value / divisor
        # swap from tail to counters
        if random() < thresh:
            del self.counters[min_counter_key]
            self.counters[key] = tail_average + value
            self.tail.remove(key)
            self.tail.append(min_counter_key)
            self.tail_total += min_counter - tail_average
        else:
            self.tail_total += value

    def attempt_promote_to_tail(self, key, value):
        tail_average = self.tail_average()
        thresh = value / (tail_average + 1)
        if random() < thresh:
            self.tail_total += value
            self.tail[randrange(len(self.tail))] = key
    
    def update(self, key, value):
        if key in self.counters:
            self.counters[key] += value
        elif key in self.tail:
            self.attempt_promote_to_counters(key, value)
        elif len(self.counters) < self.counters_size:
            self.counters[key] = value
        elif len(self.tail) < self.tail_size:
            self.tail_total += value
            self.tail.append(key)
        else:
            self.attempt_promote_to_tail(key, value)

    def query(self, key):
        estimate = self.counters.get(key, 0)
        if estimate != 0:
            return estimate
        if key in self.tail:
            return max(1, round(self.tail_average()))
        return 0