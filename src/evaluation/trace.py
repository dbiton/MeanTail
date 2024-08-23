import sys
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from misc.logger import logger
from estimators.dist_counters import DistCounters
from estimators.auto_dist_counters import AutoDistCounters
from estimators.rap import RandomAdmissionPolicy
from estimators.space_saving import SpaceSaving
import misc.distribution as dist
from evaluation.fit_zipfian import estimate_params, lognormal_fit

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

def sort_results_by_estimator_length(results):
    # Combine the lists into tuples and sort them by the first element (Estimator Length)
    combined = sorted(zip(results['Estimator Length'], results['DistCounters'], results['RAP'], results['SpaceSaving']))

    # Unzip the sorted tuples back into separate lists
    sorted_estimator_length, sorted_dist_counters, sorted_rap, sorted_space_saving = zip(*combined)

    # Update the results dictionary with sorted values
    results['Estimator Length'] = list(sorted_estimator_length)
    results['DistCounters'] = list(sorted_dist_counters)
    results['RAP'] = list(sorted_rap)
    results['SpaceSaving'] = list(sorted_space_saving)

    return results

def main():
    trace_file = "src/traces/trace.txt"
    trace_len = 100000
    max_ratio = 0.1
    step_size = max_ratio / 10
    trace_ratios = np.arange(step_size, max_ratio, step_size)

    results = {'Estimator Length': [], 'AutoDistCounters': [], 'DistCounters': [], 'RAP': [], 'SpaceSaving': [], 'DistCountersLimit': []}

    with ProcessPoolExecutor() as executor:
        futures = []
        for trace_ratio in trace_ratios:
            futures.append(executor.submit(process_trace, trace_file, trace_len, trace_ratio))

        for future in as_completed(futures):
            iter_result = future.result()
            results['Estimator Length'].append(iter_result['EST LEN'])
            results['DistCounters'].append(iter_result['DC ARE'])
            results['AutoDistCounters'].append(iter_result['ADC ARE'])
            results['DistCountersLimit'].append(iter_result['DC LIMIT LOGNORM'])
            results['RAP'].append(iter_result['RAP ARE'])
            results['SpaceSaving'].append(iter_result['SS ARE'])
            # print(f"Trace Length: {trace_len}, DC: {dc_result} RAP: {rap_result}, SS: {ss_result}")

    plot_results(sort_results_by_estimator_length(results))

def dc_best_possible_are(stream, prob, estimator_length):
    actual_counts = Counter(stream)
    estimate_counts = {k: 0 for k in actual_counts}
    logger.info(f"stream unique counts {len(actual_counts)}")
    for n, (k, _) in enumerate(actual_counts.most_common()):
        if n < estimator_length:
            estimate_counts[k] = prob(n) * len(stream)
        else:
            break
    errors = {k: abs(estimate_counts[k] - a) for k, a in actual_counts.items()}
    are = sum([errors[k] / v for k, v in actual_counts.items()]) / len(actual_counts)
    return are

def process_trace(trace_file, trace_len, trace_ratio):
    logger.info("reading trace...")
    trace = read_trace(trace_file, trace_len)
    logger.info("finding zipf parameter...")
    params = estimate_params(trace)
    zipf_param = params["Zipfian"]
    log_normal_param = params["Log-normal"]
    estimator_size = trace_ratio * trace_len
    rap = RandomAdmissionPolicy(estimator_size)
    ss = SpaceSaving(estimator_size)
    log_normal_function = lambda x: lognormal_fit(x, log_normal_param[0], log_normal_param[1])
    logger.info('checking lognormal errors...')
    dc_limit_log_normal = None # dc_best_possible_are(trace, log_normal_function, estimator_size * 2)
    adc = AutoDistCounters(int(estimator_size * 1))
    dc = DistCounters(int(estimator_size * 1), log_normal_function)
    logger.info('eval dc...')
    dc_result = 0,0 #evaluate(dc, trace)
    logger.info('eval adc...')
    adc_result = evaluate(adc, trace)
    logger.info('eval rap...')
    rap_result = evaluate(rap, trace)
    logger.info('eval ss...')
    ss_result = evaluate(ss, trace)
    result = {"TRACE LEN": trace_len, "DC ARE": dc_result, "DC LIMIT LOGNORM": dc_limit_log_normal, "RAP ARE": rap_result, "ADC ARE": adc_result, "SS ARE": ss_result, "EST LEN": estimator_size}
    logger.info(result)
    return result 

def plot_results(results):
    est_lengths = results['Estimator Length']
    dc_are = [result[0] for result in results['DistCounters']]
    adc_are = [result[0] for result in results['AutoDistCounters']]
    rap_are = [result[0] for result in results['RAP']]
    ss_are = [result[0] for result in results['SpaceSaving']]
    dc_limit_are = [result for result in results['DistCountersLimit']]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Subplot 1: ARE
    ax.plot(est_lengths, dc_are, label='DistCounters', marker='o')
    ax.plot(est_lengths, adc_are, label='AutoDistCounters', marker='s')
    ax.plot(est_lengths, rap_are, label='RAP', marker='x')
    ax.plot(est_lengths, ss_are, label='SpaceSaving', marker='s')
    # ax.plot(trace_lengths, dc_limit_are, label='DistCountersLimit', marker='D')
    ax.set_xlabel('Estimator Length')
    ax.set_ylabel('ARE (Average Relative Error)')
    ax.set_title('ARE (Average Relative Error) Comparison')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
