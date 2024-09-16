import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from estimators.count_min import CountMin
from estimators.frequent import Frequent
from estimators.space_saving import SpaceSaving
from estimators.rap import RandomAdmissionPolicy
from estimators.range_counters import RangeCounters



def read_trace(file_path, n=None):
    with open(file_path, "r") as file:
        if n is not None:
            return [int(next(file).strip()) for _ in range(n)]
        else:
            return [int(line.strip()) for line in file]


def calculate_mse(estimator, actual_counts):
    squared_errors = np.zeros(len(actual_counts))
    i = 0
    for k, v in actual_counts.items():
        v_est = estimator.query(k)
        squared_error = (v_est - v) ** 2
        squared_errors[i] = squared_error
        i += 1
    return squared_errors.mean()


def main():
    trace_file = "src/traces/trace.txt"
    trace_len = 100000
    print("read trace...")
    trace = read_trace(trace_file, trace_len)
    print("find actual counts...")
    actual_counts = Counter(trace)

    estimator_lengths = []
    mse_values = {"FR": [], "SS": [], "RAP": [], "RC": []}

    for estimator_exp in np.linspace(4,12,32):
        print("*", end="", flush=True)
        estimator_length = 2 ** estimator_exp
        estimator_lengths.append(estimator_length)
        estimators = {
            "FR": Frequent(estimator_length),
            "SS": SpaceSaving(estimator_length),
            "RAP": RandomAdmissionPolicy(estimator_length),
            "RC": RangeCounters(estimator_length),
        }

        for estimator_name, estimator in estimators.items():
            for k in trace:
                estimator.update(k, 1)
            mse = calculate_mse(estimator, actual_counts)
            mse_values[estimator_name].append(mse)

    # Plotting MSE for different estimators
    plt.figure(figsize=(10, 6))

    for estimator_name, mse_list in mse_values.items():
        plt.plot(estimator_lengths, mse_list, label=estimator_name, marker='o')

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Estimator Length (log scale)")
    plt.ylabel("Mean Squared Error (log scale)")
    plt.title("MSE for Different Estimators")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
