from space_saving import SpaceSaving
from dist_counters import DistCounters

from collections import Counter
from random import randint
from time import time

def evaluate(estimator, stream):
  for e in stream:
    estimator.update(e, 1)
  actual_counts = Counter(stream)
  differences = [abs(estimator.query(k) - v) for k, v in actual_counts.items()]
  return sum(differences)

if __name__=="__main__":
  stream_size = 1000000
  key_count = 1000
  uniform_dist = lambda index: 1 / key_count
  ss = SpaceSaving(100)
  dc = DistCounters(200, uniform_dist)
  stream = [randint(0, key_count-1) for _ in range(stream_size)]
  t0 = time()
  ss_error = evaluate(ss, stream)
  t1 = time()
  dc_error = evaluate(dc, stream)
  t2 = time()
  print("space saving:", ss_error, t1-t0)
  print("dist counters:", dc_error, t2-t1)