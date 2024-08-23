import sys
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from misc.logger import logger
from estimators.dist_counters import DistCounters
from estimators.auto_dist_counters import AutoDistCounters
from estimators.rap import RandomAdmissionPolicy
from estimators.space_saving import SpaceSaving
import misc.distribution as dist
from evaluation.fit_zipfian import estimate_params

def lognormal_fit_mean_variance(x, mean, variance):
    stddev = np.sqrt(variance)
    scale = np.exp(mean)
    return stats.lognorm.pdf(x, stddev, scale=scale)

def read_trace(file_path, n=None):
    with open(file_path, "r") as file:
        if n is not None:
            return [int(next(file).strip()) for _ in range(n)]
        else:
            return [int(line.strip()) for line in file]

def evaluate(estimator, stream):
    for i, e in enumerate(stream):
        estimator.update(e, 1)
    actual_counts = Counter(stream)
    estimate_counts = {k: estimator.query(k) for k in actual_counts}
    errors = {k: abs(estimate_counts[k] - a) for k, a in actual_counts.items()}
    aae = sum(errors.values())
    are = sum([errors[k] / v for k, v in actual_counts.items()]) / len(actual_counts)
    return are, aae

def power_law_fit(x, a, b):
    return a * np.power(x, -b)

def estimate_params(packets: list, adc_length: int):
    counter = Counter(packets)
    packets_counts = np.array(sorted(list(counter.values()), key=lambda x: -x))
    frequency = packets_counts / len(packets)
    rank = np.arange(1, len(frequency) + 1)
    rank_hh = rank[:adc_length]
    frequency_hh = frequency[:adc_length]
    
    print("fit log-normal...")
    # Log-normal fit    
    lognormal_params, _ = curve_fit(lognormal_fit_mean_variance, rank, frequency, p0=[1.0, 1.0])
    mean_lognormal, variance_lognormal = lognormal_params
    lognormal_residuals = frequency - lognormal_fit_mean_variance(rank, *lognormal_params)
    lognormal_ssr = np.sum(lognormal_residuals**2)
    
    print("fit power law...")
    # Power-law fit
    power_law_params, _ = curve_fit(power_law_fit, rank, frequency, p0=[1.0, 1.1])
    a_power_law, b_power_law = power_law_params
    power_law_residuals = frequency - power_law_fit(rank, *power_law_params)
    power_law_ssr = np.sum(power_law_residuals**2)
    
    print("fit adc...")
    # ADC fit
    adc = AutoDistCounters(adc_length)
    for p in packets:
        adc.update(p, 1) 
    adc_params = adc.mean, adc.get_variance()
    adc_mean, adc_variance = adc_params
    
    # Plot the data and the fitted curves
    plt.figure(figsize=(10, 6))
    plt.plot(rank, frequency, 'b-', label='Data')
    plt.plot(rank, lognormal_fit_mean_variance(rank, *adc_params), 'g:', label=f'ADC Log-normal fit (mean={adc_mean:.2f}, variance={adc_variance:.2f})')
    plt.plot(rank_hh, lognormal_fit_mean_variance(rank_hh, *lognormal_params), 'y-', label=f'Log-normal fit (mean={mean_lognormal:.2f}, variance={variance_lognormal:.2f})')
    plt.plot(rank, power_law_fit(rank, *power_law_params), 'r-', label=f'Power law fit (a={a_power_law:.2f}, b={b_power_law:.2f})')    
    plt.title(len(packets))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def worker(trace_len):
    print(f"Processing trace_len = {trace_len}...")
    trace = read_trace("src/traces/trace.txt", trace_len)
    adc_len = int(trace_len * 0.01)
    estimate_params(trace, adc_len)
    return trace_len

def main():
    start_len = 10000
    end_len = 10000000
    num_jumps = 64
    
    step_size = (end_len - start_len) // num_jumps
    trace_len_values = range(start_len, end_len + step_size, step_size)

    # Create a pool of workers, one for each trace_len
    with mp.Pool() as pool:
        pool.map(worker, trace_len_values)

if __name__ == "__main__":
    main()
