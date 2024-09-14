from random import random, randrange

BYTES_PER_COUNTER = 4
BYTES_PER_KEY = 4


class RangeCounters:
    def __init__(self, size, mem_percentage_tail=0):
        self.size = size * (1 - mem_percentage_tail)
        self.counters = {}
        self.tail = []
        self.tail_total = 0
        coe_tail_mem = (BYTES_PER_COUNTER + BYTES_PER_KEY) / BYTES_PER_COUNTER
        self.tail_size = size * mem_percentage_tail * coe_tail_mem

    def tail_average(self):
        return self.tail_total / len(self.tail)

    def update_tail(self, key, value):
        min_counter_index = min(self.counters, key=self.counters.get)
        min_counter = self.counters[min_counter_index]
        tresh = 1 / (min_counter + 1)
        # assert min_counter >= self.tail_average()
        if random() < tresh:
            del self.counters[min_counter_index]
            self.counters[key] = self.tail_average() + value
            self.tail_total -= self.tail_average()
        else:
            self.tail_total += value

    def update(self, key, value):
        if key in self.counters:
            self.counters[key] += value
        elif key in self.tail:
            self.update_tail(key, value)
        else:
            if len(self.counters) < self.size:
                self.counters[key] = value
            elif len(self.tail) < self.tail_size:
                self.tail.append(key)
            else:
                tail_counter = self.tail_average()
                tresh = 1 / (tail_counter + 1)
                if random() < tresh:
                    self.tail_total += value
                    self.tail[randrange(len(self.tail))] = key

    def query(self, key):
        estimate = self.counters.get(key, 0)
        if estimate != 0:
            return estimate
        if key in self.tail:
            return self.tail_total / len(self.tail)
        return 0