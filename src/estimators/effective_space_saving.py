from random import random

class EffectiveSpaceSaving:
    def __init__(self, size, mem_percentage_candidates):
        self.size_counters = int(size * (1-mem_percentage_candidates))
        self.size_candidates = int(size * mem_percentage_candidates)
        self.counters = {}
        self.candidates = {}
 
    def update(self, index, value):
        if index in self.counters:
            self.counters[index] += value
        elif len(self.counters) < self.size_counters:
            self.counters[index] = value
        else:
            if index in self.candidates:
                self.candidates[index] += value
            elif len(self.candidates) < self.size_candidates:
                self.candidates[index] = value
            else:
                s = sum(self.candidates.values())
                largest_window_flow = max(self.candidates, key=self.candidates.get)
                smallest_counters_flow = min(self.counters, key=self.counters.get)
                self.counters[largest_window_flow] = self.candidates[largest_window_flow] + self.counters[smallest_counters_flow]
                del self.counters[smallest_counters_flow]
                self.candidates = {k: v for k, v in self.candidates.items() if random() <= v/s}


    def query(self, index):
        return self.counters.get(index, 0)