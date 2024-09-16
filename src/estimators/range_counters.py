from random import random, randrange
import numpy as np


class RangeCounters:
    def __init__(self, size, mem_percentage_tail=0.1):
        self.counters_size = int(size * (1 - mem_percentage_tail))
        self.tail_size = int(size * mem_percentage_tail * 2)
        self.counters = {}
        self.tail = []
        self.tail_total = 0

    def tail_average(self):
        return round(max(1, self.tail_total / len(self.tail)))

    def rebalance(self):
        tail_average = self.tail_average()
        min_counter_key = min(self.counters, key=self.counters.get)
        min_counter = self.counters[min_counter_key]
        if tail_average > 1 > 0.75 * min_counter and len(self.counters) > 1:
            print("tail!", len(self.counters), len(self.tail))
            del self.counters[min_counter_key]
            self.counters_size -= 1
            self.tail_size += 2
            self.tail.append(min_counter_key)
            self.tail_total += min_counter
        elif 1 < tail_average < 0.25 * min_counter and len(self.tail) > 2:
            print("hh!", len(self.counters), len(self.tail))
            del self.counters[min_counter_key]
            self.tail.pop()
            self.counters[self.tail.pop()] = tail_average
            self.counters_size += 1
            self.tail_size -= 2
            self.tail_total -= tail_average * 2


    def attempt_promote_to_counters(self, key, value):
        min_counter_key = min(self.counters, key=self.counters.get)
        min_counter = self.counters[min_counter_key]
        tail_average = self.tail_average()
        tresh = (value + tail_average) / (1 + min_counter)
        # swap from tail to counters
        if random() < tresh:
            del self.counters[min_counter_key]
            self.counters[key] = tail_average + value
            self.tail.remove(key)
            self.tail.append(min_counter_key)
            self.tail_total -= tail_average + min_counter
        else:
            self.tail_total += value

    def attempt_promote_to_tail(self, key, value):
        tail_average = self.tail_average()
        tresh = value / (tail_average + 1)
        if random() < tresh:
            self.tail_total += value
            self.tail[randrange(len(self.tail))] = key

    def update(self, key, value):
        if key in self.counters:
            self.counters[key] += value
        elif key in self.tail:
            self.attempt_promote_to_counters(key, value)
        else:
            if len(self.counters) < self.counters_size:
                self.counters[key] = value
            elif len(self.tail) < self.tail_size:
                self.tail.append(key)
            else:
               self.attempt_promote_to_tail(key, value)
        
        if len(self.tail) > 0:
            min_counter_key = min(self.counters, key=self.counters.get)
            min_counter = self.counters[min_counter_key]
            tail_average = self.tail_average()
            # print("taillen", len(self.tail), "tailavg", tail_average, "counterlen", len(self.counters), "countermin", min_counter)
            self.rebalance()

    def query(self, key):
        estimate = self.counters.get(key, 0)
        if estimate != 0:
            return estimate
        if key in self.tail:
            return max(1, self.tail_total / len(self.tail))
        return 0