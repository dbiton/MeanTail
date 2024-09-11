import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter
import matplotlib.pyplot as plt

from estimators.rap import RandomAdmissionPolicy
from estimators.range_counters import RangeCounters


def read_trace(file_path, n=None):
    with open(file_path, "r") as file:
        if n is not None:
            return [int(next(file).strip()) for _ in range(n)]
        else:
            return [int(line.strip()) for line in file]

trace_file = "src/traces/trace.txt"
trace_len = 100000
estimator_len = 1500
trace = read_trace(trace_file, trace_len)
actual_counts = Counter(trace)
rap = RandomAdmissionPolicy(estimator_len)
rc = RangeCounters(estimator_len//2)
rc.add_range(estimator_len)
for v in trace:
    rap.update(v, 1)
    rc.update(v)
rap_estimates = {k: rap.query(k) for k in actual_counts.keys()}
rc_estimates = {k: rc.query(k) for k in actual_counts.keys()}
vs = [(k, actual_counts[k], rap_estimates[k], rc_estimates[k]) for k in actual_counts.keys()]
vs = sorted(vs, key=lambda v: v[1])
xs = list(range(len(vs)))
ys_actual = [v[1] for v in vs]
ys_rap = [v[2] for v in vs]
ys_rc = [v[3] for v in vs]
es_rap = [abs(y_rap - y_actual) for y_rap, y_actual in zip(ys_rap, ys_actual)]
es_rc = [abs(y_rc - y_actual) for y_rc, y_actual in zip(ys_rc, ys_actual)]


print("RAP", np.average(es_rap), "RC", np.average(es_rc))

plt.figure(figsize=(8, 8))
plt.plot(xs, ys_actual, 'b:', label='Data')
plt.plot(xs, ys_rap, 'g-', label=f'RAP')
plt.plot(xs, ys_rc, 'r-', label=f'RC')
# plt.xscale("log")
# plt.yscale("log")
plt.title(len(trace))
plt.legend()
plt.show()