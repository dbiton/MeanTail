from random import random, randrange

class RangeCounters:
    def __init__(self, size, mem_percentage_tail=0.5):
        self.size = int(size * (1 - mem_percentage_tail))
        self.counters = {}
        self.tail = []
        self.tail_total = 0
        self.tail_size = int(size * mem_percentage_tail * 2)

    def tail_average(self):
        return self.tail_total / self.tail_size

    def rebalance(self):
        tail_average = self.tail_average()
        min_counter_key = min(self.counters, key=self.counters.get)
        min_counter = self.counters[min_counter_key]
        if tail_average > 2 * min_counter:
            print("tail!", len(self.counters), len(self.tail))
            del self.counters[min_counter_key]
            self.size -= 1
            self.tail_size += 2
            self.tail.append(min_counter_key)
            self.tail_total += min_counter
        if min_counter > 2 * tail_average and len(self.tail) >= 2:
            print("hh!", len(self.counters), len(self.tail))
            del self.counters[min_counter_key]
            self.tail.pop()
            self.counters[self.tail.pop()] = tail_average
            self.size += 1
            self.tail_size -= 2
            self.tail_total -= tail_average * 2


    def update_tail(self, key, value):
        min_counter_key = min(self.counters, key=self.counters.get)
        min_counter = self.counters[min_counter_key]
        tail_average = self.tail_average()
        tresh = tail_average / min_counter
        # swap from tail to counters
        if random() < tresh:
            del self.counters[min_counter_key]
            self.counters[key] = tail_average + value
            self.tail.remove(key)
            self.tail.append(min_counter_key)
            self.tail_total -= tail_average + min_counter
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
        if len(self.counters) + len(self.tail) >= self.size + self.tail_size:
            self.rebalance()

    def query(self, key):
        estimate = self.counters.get(key, 0)
        if estimate != 0:
            return estimate
        if key in self.tail:
            return max(1, self.tail_total / len(self.tail))
        return 0