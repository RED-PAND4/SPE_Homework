import random
import numpy as np
import matplotlib.pyplot as plt

# Constants
P_tx = 0.1  
N = 1.6e-4  
num_samples = 1000 
num_sample_exp = 1000
num_trials = 1000  # number of trials
num_pairs = 100  # number of pairs per trial


# expoenntial variable ξtx
e_mean = 1# 1/lambda
def sample_exponential(mean):
    return -mean * np.log(1 - random.random())

exp_samples = [sample_exponential(e_mean) for n in range(num_sample_exp)]
e_tx_index = random.randint(0, num_sample_exp-1)
e_txI_index = random.randint(0, num_sample_exp-1)
while e_txI_index == e_tx_index:
            e_txI_index = random.randint(0, num_sample_exp-1)
e_tx = exp_samples[e_tx_index]
e_I = exp_samples[e_txI_index]




def compute_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def compute_snr_f(d_tx_rx):
    return (P_tx * e_tx/ (N * d_tx_rx**2))

def compute_sinr_f(d_tx_rx, d_txI_rx):
    return ((P_tx*e_tx*d_tx_rx**(-2)) / (N + P_tx*e_I*d_txI_rx**(-2)))


def simulate_trial_snr_f():
    success_count = 0
    for n in range(num_pairs):
        tx_index = random.randint(0, num_samples-1)
        rx_index = random.randint(0, num_samples-1)
        while rx_index == tx_index:
            rx_index = random.randint(0, num_samples-1)
        d_tx_rx = compute_distance(samples[tx_index], samples[rx_index])
        snr = compute_snr_f(d_tx_rx)
        if snr > 8:
            success_count += 1
    return success_count / num_pairs

def simulate_trial_sinr_f():
    success_count = 0
    for n in range(num_pairs):
        tx_index = random.randint(0, num_samples-1)
        rx_index = random.randint(0, num_samples-1)
        txI_index = random.randint(0, num_samples-1)
        while rx_index == tx_index or txI_index == tx_index or rx_index == txI_index:
            rx_index = random.randint(0, num_samples-1)
            txI_index = random.randint(0, num_samples-1)
        d_tx_rx = compute_distance(samples[tx_index], samples[rx_index])
        d_txI_rx = compute_distance(samples[tx_index], samples[txI_index])
        snr = compute_sinr_f(d_tx_rx, d_txI_rx)
        if snr > 8:
            success_count += 1
    return success_count / num_pairs

def sample_vector():
    while True:
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        u = random.uniform(0, 10)
        if u <= abs((-1)*x1 +10 - x2):
            if u <= abs(x1-x2):
                break
    return (x1, x2)


samples = [sample_vector() for n in range(num_samples)]


success_probabilities_snr = []
for n in range(num_trials):
    trial = simulate_trial_snr_f()
    success_probabilities_snr.append(trial)

success_probability_snr = np.mean(success_probabilities_snr)
confidence_interval_snr = np.percentile(success_probabilities_snr, [2.5, 97.5])
print(f"SNR:Success probability: {success_probability_snr:.4f}")
print(f"SNR:95% confidence interval: [{confidence_interval_snr[0]:.4f}, {confidence_interval_snr[1]:.4f}]")


success_probabilities_sinr = []
for n in range(num_trials):
    trial = simulate_trial_snr_f()
    success_probabilities_sinr.append(trial)

success_probability_sinr = np.mean(success_probabilities_sinr)
confidence_interval_sinr = np.percentile(success_probabilities_sinr, [2.5, 97.5])
print(f"SINR:Success probability: {success_probability_sinr:.4f}")
print(f"SINR:95% confidence interval: [{confidence_interval_sinr[0]:.4f}, {confidence_interval_sinr[1]:.4f}]")