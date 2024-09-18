import math
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
from multiprocessing import Pool
from datetime import datetime

# Function to read trace files
def read_trace(file_path, n=None):
    with open(file_path, "r") as file:
        if n is not None:
            return [int(next(file).strip()) for _ in range(n)]
        else:
            return [int(line.strip()) for line in file]

def calculate_mse(estimator, actual_counts):
    errors = np.zeros(len(actual_counts))
    i = 0
    estimates = [estimator.query(k) for k in actual_counts]
    errors = [(v_est - v)**2 for v_est, v in zip(estimates, actual_counts.values())]
    return np.mean(errors)

# Process each trace file
def process_trace(trace_file):
    linestyles = ['-', '--', ':', "-."] 
    markers = ['o', 's', 'D', '^']
    trace_len = 10000
    
    # Get the start time
    start_time = datetime.now()
    print(f"Start processing {trace_file} at {start_time}")
    
    trace = read_trace(trace_file, trace_len)
    actual_counts = Counter(trace)

    estimator_lengths = []
    mse_values = {"SS": [], "RAP": [], "RC": [], "FR": []}

    log2_count_keys = math.ceil(math.log2(len(actual_counts))) - 1

    # Run experiments with different estimators
    for estimator_exp in np.linspace(4,log2_count_keys,log2_count_keys-3):
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

    # Plotting the results
    plt.figure()
    i = 0
    for estimator_name, mse_list in mse_values.items():
        plt.plot(estimator_lengths, mse_list, label=estimator_name, marker=markers[i], linestyle=linestyles[i])
        i += 1

    plt.legend()
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Estimator Length")
    plt.ylabel("Mean Squared Error")
    plt.tight_layout()
    
    # Save the graph with the trace file name
    output_file = f"{os.path.splitext(os.path.basename(trace_file))[0]}.png"
    plt.savefig(output_file)

    # Get the end time and print it
    end_time = datetime.now()
    print(f"Finished processing {trace_file} at {end_time}, saved as {output_file}")
    print(f"Duration: {end_time - start_time}")

# Main function to execute in parallel
def main():
    trace_dir = "src/traces/"
    
    # Get all .trace files in the directory
    trace_files = [os.path.join(trace_dir, f) for f in os.listdir(trace_dir) if f.endswith(".trace")]

    # Run the trace processing in parallel
    with Pool() as pool:
        pool.map(process_trace, trace_files)

if __name__ == "__main__":
    main()
