import random
import matplotlib.pyplot as plt
import numpy as np

# Constants
P_tx = 0.1  
N = 1.6e-4  
num_samples = 1000 
num_trials = 1000  # number of trials
num_pairs = 100  # number of pairs per trial

def compute_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def compute_sinr(d_tx_rx, d_txI_rx):
    return ((P_tx*d_tx_rx**(-2)) / (N + P_tx*d_txI_rx**(-2)))

def simulate_trial():
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
        snr = compute_sinr(d_tx_rx, d_txI_rx)
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


success_probabilities = []
for n in range(num_trials):
    trial = simulate_trial()
    success_probabilities.append(trial)

success_probability = np.mean(success_probabilities)
confidence_interval = np.percentile(success_probabilities, [2.5, 97.5])
print(f"Success probability: {success_probability:.4f}")
print(f"95% confidence interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")