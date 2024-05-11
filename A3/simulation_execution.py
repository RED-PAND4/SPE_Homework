import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utility.compute_theoretical_statistic import *
from utility.batch_means import compute_batch_means_statistics
from utility.event import *
from simulation import simulation
from plotting import Plotting
from plotting import *


# setting simulation parameters
# mandatory l < mu
l = 1 # other 3 value l=mu, l<<<<mu, ot
mu = 2 # other 3 value
rho = l / mu
sim_time =100000
gen = np.random.default_rng(seed=41)

#theoretical values 
avg_packets_in_system_th = avg_packet_in_sys(rho)

packets, queue_occupation = simulation(sim_time, l, mu, gen)

total_width = np.sum(queue_occupation["width"].values)
avg_packets_in_system_sim = (
    np.sum(queue_occupation["packets_in_system"].values * queue_occupation["width"].values) / total_width
)

print(
    f"""Average number of packets in the queue (theory): {avg_packets_in_system_th} \t
        Average number of packets in the queue (simulation): {avg_packets_in_system_sim}"""
)

# Plotting
plots = Plotting(l, mu, sim_time, packets, queue_occupation)

# Plot distribution of arrival times and service times just to check they follow theoretical distributions
plots.plot_base_fun()

plots.plot_mean_in_time(queue_occupation)

plots.plot_system_occupation(avg_packets_in_system_sim)

# Plot autocorrelation to decide batch size for batch means
plots.plot_auto_correlation()


# batch variables
(
    grand_mean,
    ci_amplitude,
    batch_means,
    intervals,
) = compute_batch_means_statistics(Statistics.PACKET_IN_SYSTEM, queue_occupation, 10000, 4000, 0.95)
#---


plots.plot_batch_means(
    batch_means, intervals
)

print(
    f"""Average number of packets in the queue (theory): {avg_packets_in_system_th} \t
        Average number of packets in the queue with batch means(simulation): {grand_mean} +- {ci_amplitude}"""
) 


plt.show()

# print("Number of packets: ", number_of_packets)
# print("Packets: ", packets)
# print("Queue occupation: ", queue_occupation)


# Peak of waiting times should move to the right the closer rho is to 1
#plots.plot_waiting_times_distribution()

# avg_waiting_time_th = avg_waiting_time(rho, l, mu) 
# avg_response_time_th = avg_total_time(rho, l)
# avg_queue_length_th = avg_queue_length(rho)

# #Check if average time in the system is equal to the theoretical value using batch means
# print(
#     f"""Average response time (theory): {avg_response_time_th} \t
#       Average response time (simulation): {grand_mean} +- {ci_amplitude}"""
# )

# #Check if average time waiting is equal to the theoretical value using batch means
# print(
#     f"""Average waiting time (theory): {avg_waiting_time_th} \t
#         Average waiting time (simulation): {grand_mean_wt} +- {ci_amplitude_wt}"""
# )
