from collections import Counter
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import zetac
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Define the fitting functions
def zipfian_fit(x, a):
    return np.float_power(x, -a) / zetac(a)

def power_law_fit(x, a, b):
    return a * np.power(x, -b)

def lognormal_fit(x, s, scale):
    return lognorm.pdf(x, s, scale=scale)

def estimate_params(packets: list):
    counter = Counter(packets)
    packets_counts = np.array(sorted(list(counter.values()), key=lambda x: -x))
    frequency = packets_counts / len(packets)
    rank = np.arange(1, len(frequency) + 1)
    
    # Zipfian fit
    zipf_params, _ = curve_fit(zipfian_fit, rank, frequency, p0=[1.1], bounds=(1, np.inf))
    a_zipf = zipf_params[0]
    zipfian_residuals = frequency - zipfian_fit(rank, *zipf_params)
    zipfian_ssr = np.sum(zipfian_residuals**2)
    
    # Power-law fit
    power_law_params, _ = curve_fit(power_law_fit, rank, frequency, p0=[1.0, 1.1])
    a_power_law, b_power_law = power_law_params
    power_law_residuals = frequency - power_law_fit(rank, *power_law_params)
    power_law_ssr = np.sum(power_law_residuals**2)
    
    # Log-normal fit
    lognormal_params, _ = curve_fit(lognormal_fit, rank, frequency, p0=[1.0, 1.0])
    s_lognormal, scale_lognormal = lognormal_params
    lognormal_residuals = frequency - lognormal_fit(rank, *lognormal_params)
    lognormal_ssr = np.sum(lognormal_residuals**2)
    
    # Plot the data and the fitted curves
    plt.figure(figsize=(10, 6))
    plt.plot(rank, frequency, 'b-', label='Data')
    plt.plot(rank, zipfian_fit(rank, *zipf_params), 'r--', label=f'Zipfian fit (a={a_zipf:.2f})')
    plt.plot(rank, power_law_fit(rank, *power_law_params), 'g:', label=f'Power law fit (a={a_power_law:.2f}, b={b_power_law:.2f})')
    plt.plot(rank, lognormal_fit(rank, *lognormal_params), 'y-', label=f'Log-normal fit (s={s_lognormal:.2f}, scale={scale_lognormal:.2f})')
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Various Distribution Fits')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"SSR for Zipfian fit: {zipfian_ssr:.4f}")
    print(f"SSR for Power law fit: {power_law_ssr:.4f}")
    print(f"SSR for Log-normal fit: {lognormal_ssr:.4f}")
    
    return {
        "Zipfian": a_zipf,
        "Power Law": (a_power_law, b_power_law),
        "Log-normal": (s_lognormal, scale_lognormal)
    }