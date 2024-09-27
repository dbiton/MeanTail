import math
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from estimators.count_min import CountMin
from estimators.frequent import Frequent
from estimators.effective_space_saving import EffectiveSpaceSaving
from estimators.space_saving import SpaceSaving
from estimators.rap import RandomAdmissionPolicy
from estimators.mean_tail import MeanTail
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
    return np.average(errors)

# Process each trace file
def process_trace(trace_file):
    linestyles = ['-', '--', ':', "-."] * 2 
    markers = ['o', 's', 'D', '^', '*', '>'] * 2
    trace_len = 100000
    
    # Get the start time
    start_time = datetime.now()
    print(f"Start processing {trace_file} at {start_time}")
    
    trace = read_trace(trace_file, trace_len)
    actual_counts = Counter(trace)

    estimator_lengths = []
    mse_values = {}

    log2_count_keys = math.ceil(math.log2(len(actual_counts)))

    # Run experiments with different estimators
    for estimator_exp in np.linspace(log2_count_keys-3,log2_count_keys,20):
        estimator_length = 2 ** estimator_exp
        estimator_lengths.append(estimator_length)
        estimators = {
            # "FR": Frequent(estimator_length),
            # "SS": SpaceSaving(estimator_length),
            # "ESS": EffectiveSpaceSaving(estimator_length, 0.125),
            "RAP": RandomAdmissionPolicy(estimator_length),
            "MT16": MeanTail(estimator_length, 0.0625),
            "MT8": MeanTail(estimator_length, 0.125),
            "MT4": MeanTail(estimator_length, 0.25),
        }
        for estimator_name, estimator in estimators.items():
            for k in trace:
                estimator.update(k, 1)
            mse = calculate_mse(estimator, actual_counts)
            if estimator_name not in mse_values:
                mse_values[estimator_name] = []
            mse_values[estimator_name].append(mse)

    # Plotting the results
    plt.figure()
    i = 0
    for estimator_name, mse_list in mse_values.items():
        plt.plot(estimator_lengths, mse_list, label=estimator_name, marker=markers[i], linestyle=linestyles[i])
        i += 1

    legend = plt.legend()
    plt.xscale("log", base=2)
    plt.yscale("log", base=10)
    plt.xlabel("Estimator Length")
    plt.ylabel("Mean Squared Error")
    plt.tight_layout()
    
    # Save the graph with the trace file name
    output_file = f"{os.path.splitext(os.path.basename(trace_file))[0]}.png"
    plt.savefig(output_file)

    # save legend as figure
    fig  = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('legend', dpi="figure", bbox_inches=bbox)
    
    # Get the end time and print it
    end_time = datetime.now()
    print(f"Finished processing {trace_file} at {end_time}, saved as {output_file}")
    print(f"Duration: {end_time - start_time}")

# Main function to execute in parallel
def main():
    trace_dir = "C:/Users/User/Desktop/Projects/DistCounters/src/traces"
    
    # Get all .trace files in the directory
    trace_files = [os.path.join(trace_dir, f) for f in os.listdir(trace_dir) if f.endswith(".trace")]

    # Run the trace processing in parallel
    with Pool() as pool:
        pool.map(process_trace, trace_files)

if __name__ == "__main__":
    main()
