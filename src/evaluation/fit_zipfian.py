from collections import Counter
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import zetac

# Define the fitting function for Zipfian distribution
def zipfian_fit(x, a):
    return np.float_power(x, -a) / zetac(a)

def estimate_zipfian_param(packets: list):
    counter = Counter(packets)
    packets_counts = np.array(sorted(list(counter.values()), key=lambda x: -x))
    frequency = packets_counts / len(packets)
    rank = np.arange(1, len(frequency) + 1)
    params, _ = curve_fit(zipfian_fit, rank, frequency, p0=[1.1], bounds=(1, np.inf))
    # Extract the parameter a
    a_estimated = params[0]
    return a_estimated