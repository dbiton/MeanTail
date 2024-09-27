import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter
import matplotlib.pyplot as plt

from estimators.rap import RandomAdmissionPolicy
from estimators.mean_tail import MeanTail


def read_trace(file_path, n=None):
    with open(file_path, "r") as file:
        if n is not None:
            return [int(next(file).strip()) for _ in range(n)]
        else:
            return [int(line.strip()) for line in file]

trace_file = "src/traces/youtube.trace"
trace_len = 1000000
estimator_len = 2**10
print('read trace...')
trace = read_trace(trace_file, trace_len)
print('find actual counts...')
actual_counts = Counter(trace)

rap = RandomAdmissionPolicy(estimator_len)
mt = MeanTail(estimator_len, 0.125)

print('update:')
i = 0
for v in trace:
    if i % (len(trace) // 5) == 0:
        print('*', end="", flush=True)
    i += 1
    rap.update(v, 1)
for v in trace:
    if i % (len(trace) // 5) == 0:
        print('*', end="", flush=True)
    i += 1
    mt.update(v, 1)
print("")
print('query:')
i = 0
rap_estimates = {}
mt_estimates = {}
for k in actual_counts.keys():
    if i % (len(actual_counts) // 10) == 0:
        print('*', end="", flush=True)
    i += 1
    rap_estimates[k] = rap.query(k)
    mt_estimates[k] = mt.query(k)
print("")
vs = [(k, actual_counts[k], rap_estimates[k], mt_estimates[k]) for k in actual_counts.keys()]
vs = sorted(vs, key=lambda v: v[1])
xs = list(range(len(vs)))
ys_actual = [v[1] for v in vs]
ys_rap = [v[2] for v in vs]
ys_mt = [v[3] for v in vs]
es_rap = [(y_rap - y_actual)**2 for y_rap, y_actual in zip(ys_rap, ys_actual)]
es_mt = [(y_mt - y_actual)**2 for y_mt, y_actual in zip(ys_mt, ys_actual)]


print("RAP", np.average(es_rap), "MT", np.average(es_mt))

plt.figure()
plt.plot(xs, ys_actual, 'b:', label='Data')
plt.plot(xs, ys_rap, 'g-', label=f'RAP')
plt.plot(xs, ys_mt, 'r-', label=f'MT')
# plt.xscale("log")
# plt.yscale("log")
plt.title(len(trace))
plt.legend()
plt.show()