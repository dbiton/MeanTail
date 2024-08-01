import sys
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from misc.logger import logger
from estimators.dist_counters import DistCounters
from estimators.rap import RandomAdmissionPolicy
from estimators.space_saving import SpaceSaving
import misc.distribution as dist
from evaluation.fit_zipfian import estimate_params

def read_trace(file_path, n=None):
    with open(file_path, "r") as file:
        if n is not None:
            return [int(next(file).strip()) for _ in range(n)]
        else:
            return [int(line.strip()) for line in file]

def evaluate(estimator, stream):
    for e in stream:
        estimator.update(e, 1)
    actual_counts = Counter(stream)
    estimate_counts = {k: estimator.query(k) for k in actual_counts}
    errors = {k: abs(estimate_counts[k] - a) for k, a in actual_counts.items()}
    are = sum([errors[k] / v for k, v in actual_counts.items()]) / len(actual_counts)
    return are

def main():
    trace_file = "C:\\Users\\Dvir\\Desktop\\Projects\\DistCounters\\traces\\nyc.txt"
    trace_lengths = range(1000, 10000, 1000)
    trace_ratio = 1 / 4

    results = {'Trace Length': [], 'DistCounters': [], 'RAP': [], 'SpaceSaving': []}

    with ThreadPoolExecutor() as executor:
        futures = []
        for trace_len in trace_lengths:
            futures.append(executor.submit(process_trace, trace_file, trace_len, trace_ratio))

        for future in as_completed(futures):
            trace_len, dc_result, rap_result, ss_result = future.result()
            results['Trace Length'].append(trace_len)
            results['DistCounters'].append(dc_result)
            results['RAP'].append(rap_result)
            results['SpaceSaving'].append(ss_result)
            print(f"Trace Length: {trace_len}, DC: {dc_result} RAP: {rap_result}, SS: {ss_result}")

    plot_results(results)

def dc_best_possible_are(stream, prob, estimator_length):
    actual_counts = Counter(stream)
    estimate_counts = {k: 0 for k in actual_counts}
    
    for n, (k, _) in enumerate(actual_counts.most_common()):
        if n < estimator_length:
            estimate_counts[k] = prob(n+1) * len(stream)
        else:
            break
    
    errors = {k: abs(estimate_counts[k] - a) for k, a in actual_counts.items()}
    are = sum([errors[k] / v for k, v in actual_counts.items()]) / len(actual_counts)
    return are

def process_trace(trace_file, trace_len, trace_ratio):
    logger.info("reading trace...")
    trace = read_trace(trace_file, trace_len)
    logger.info("finding zipf parameter...")
    zipf_param = estimate_params(trace)["Zipfian"]
    logger.info(f"zipf parameter is {zipf_param}")
    probability_function = dist.ZipfianDistribution(None, zipf_param).probability
    estimator_size = trace_ratio * trace_len
    dc = DistCounters(estimator_size * 2, probability_function)
    rap = RandomAdmissionPolicy(estimator_size)
    ss = SpaceSaving(estimator_size)
    dc_limit = dc_best_possible_are(trace, probability_function, estimator_size * 2)
    dc_result = evaluate(dc, trace)
    rap_result = evaluate(rap, trace)
    ss_result = evaluate(ss, trace)
    return trace_len, dc_result, dc_limit, rap_result, ss_result, 

def plot_results(results):
    plt.figure(figsize=(10, 5))
    plt.plot(results['Trace Length'], results['DistCounters'], label='DistCounters', marker='o')
    plt.plot(results['Trace Length'], results['RAP'], label='RAP', marker='x')
    plt.plot(results['Trace Length'], results['SpaceSaving'], label='SpaceSaving', marker='s')
    plt.xlabel('Trace Length')
    plt.ylabel('ARE (Average Relative Error)')
    plt.title('Estimator Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    trace_len = 10000
    trace_file = "C:\\Users\\Dvir\\Desktop\\Projects\\DistCounters\\traces\\nyc.txt"
    result = process_trace(trace_file, trace_len, 1/4)
    print(result)
