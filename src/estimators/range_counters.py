from random import random, randint

class Range:
    def __init__(self, size: int) -> None:
        self.size = size
        self.keys = []
        self.average = 0
    
    def __contains__(self, key: int) -> bool:
        return key in self.keys
    
    def remove(self, key: int) -> None:
        self.keys.remove(key)
    
    def update(self, key: int) -> str:
        if key not in self.keys:            
            if len(self.keys) < self.size:
                self.keys.append(key)
                return "insert"
            else:
                tresh = 1 / (self.average + 1)
                if random() < tresh:
                    index = randint(0, self.size-1)
                    self.keys[index] = key
                    return "insert"
                else:
                    return "reject"
        else:
            return "present"
            
    
class RangeCounters:
    def __init__(self, size: int):
        self.size = size
        self.counters = {}
        self.ranges = []
    
    def add_range(self, size: int) -> None:
        assert len(self.ranges) == 0 or self.ranges[-1].size < size
        range = Range(size)
        self.ranges.append(range) 
    
    def promote(self, range_index: int, key: int):
        if range_index > 0:
            range_next = self.ranges[range_index-1]
            result = range_next.update(key)
            if result == "insert":
                range_curr = self.ranges[range_index]
                range_curr.remove(key)
        else:
            min_counter_key = min(self.counters, key=self.counters.get)
            min_counter = self.counters[min_counter_key]
            tresh = 1 / (min_counter + 1)
            if random() < tresh:
                del self.counters[min_counter_key]
                self.counters[key] = min_counter + 1
                range_curr = self.ranges[0]
                range_curr.remove(key)
    
    def update(self, key: int) -> None:
        # already tracked
        if key in self.counters:
            self.counters[key] += 1
        else:
            for i, range in enumerate(self.ranges):
                if key in range:
                    self.promote(i, key)
                    return
        
        # key is not tracked
        if len(self.counters) < self.size:
            self.counters[key] = 1
        else:
            range = self.ranges[-1]
            range.update(key)


    def query(self, key: int) -> int:
        estimate = self.counters.get(key, 0)
        if estimate == 0:
            for range in self.ranges:
                if key in range:
                    return range.average
        return 0