from collections import Counter
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import zetac
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.integrate import quad

# Define the fitting functions
def zipfian_fit(x, a):
    return np.float_power(x, -a) / zetac(a)

def power_law_fit(x, a, b):
    return a * np.power(x, -b)

def lognormal_fit_mean_variance(x, mean, variance):
    # Convert variance to standard deviation (s)
    stddev = np.sqrt(variance)
    # Convert mean to scale parameter (exp(mean))
    scale = np.exp(mean)
    # Compute the PDF
    return lognorm.pdf(x, stddev, scale=scale)

def lognormal_fit(x, s, scale):
    return lognorm.pdf(x, s, scale=scale)

def lognormal_integer_fit(x, s, scale):
    # Define the PDF function
    def pdf(t):
        return lognorm.pdf(t, s, scale=scale)
    # Integrate the PDF over the interval [x, x+1]
    total_probability, _ = quad(pdf, x, x + 1)
    return total_probability

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
    
    '''
    # Log-normal fit mean variance
    len_hh = int(len(packets) * 0.00005)
    lognormal_mean_variance_params, _ = curve_fit(lognormal_fit_mean_variance, rank[:len_hh], frequency[:len_hh], p0=[1.0, 1.0])
    m_lognormal, v_lognormal = lognormal_mean_variance_params
    lognormal__residuals = frequency - lognormal_fit_mean_variance(rank, *lognormal_params)
    lognormal__ssr = np.sum(lognormal__residuals**2)
    
    # Plot the data and the fitted curves
    plt.figure(figsize=(10, 6))
    plt.plot(rank, frequency, 'b-', label='Data')
    plt.plot(rank, zipfian_fit(rank, *zipf_params), 'r--', label=f'Zipfian fit (a={a_zipf:.2f})')
    plt.plot(rank, power_law_fit(rank, *power_law_params), 'g:', label=f'Power law fit (a={a_power_law:.2f}, b={b_power_law:.2f})')
    plt.plot(rank, lognormal_fit(rank, *lognormal_params), 'y-', label=f'Log-normal fit (s={s_lognormal:.2f}, scale={scale_lognormal:.2f})')
    plt.plot(rank, lognormal_fit_mean_variance(rank, *lognormal_mean_variance_params), 'c-', label=f'Log-normal mean-variance fit (mean={m_lognormal:.2f}, variance={v_lognormal:.2f})')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Various Distribution Fits')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    
    print(f"SSR for Zipfian fit: {zipfian_ssr:.4f}")
    print(f"SSR for Power law fit: {power_law_ssr:.4f}")
    print(f"SSR for Log-normal fit: {lognormal_ssr:.4f}")
    
    return {
        "Zipfian": a_zipf,
        "Power Law": (a_power_law, b_power_law),
        "Log-normal": (s_lognormal, scale_lognormal)
    }