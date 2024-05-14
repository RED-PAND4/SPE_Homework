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

def simulation_plot(l, mu, sim_time, lag, warmup, batch_size, ser=None):
    rho = l / mu
    
    
    #theoretical values 
    avg_packets_in_system_th = avg_packet_in_sys(rho)
    
    queues = []
    packets, queue_occupation = simulation(sim_time, l, mu, ser)
    packets2, queue_occupation2 = simulation(sim_time, l, mu, ser)
    packets3, queue_occupation3 = simulation(sim_time, l, mu, ser)
    packets4, queue_occupation4 = simulation(sim_time, l, mu, ser)
    packets5, queue_occupation5 = simulation(sim_time, l, mu, ser)
    queues.append(queue_occupation)
    queues.append(queue_occupation2)
    queues.append(queue_occupation3)
    queues.append(queue_occupation4)
    queues.append(queue_occupation5)
    

    # Plotting
    plots = Plotting(l, mu, sim_time, packets, queue_occupation, ser)
    
    # Plot distribution of arrival times and service times just to check they follow theoretical distributions
    plots.plot_base_fun()
    
    plots.plot_mean_in_time(queues)
    
    
    # Plot autocorrelation to decide batch size for batch means
    plots.plot_auto_correlation(lag)
    
    
    # batch variables
    (
        grand_mean,
        ci_amplitude,
        batch_means,
        intervals,
    ) = compute_batch_means_statistics(Statistics.PACKET_IN_SYSTEM, queue_occupation, batch_size, warmup, 0.95)
    #---
    
    
    plots.plot_batch_means(batch_means, intervals )
    
    plots.plot_system_occupation(grand_mean, queues[2])

    print(
        f"""Average number of packets in the queue (theory - no pareto): {avg_packets_in_system_th} \t
            Average number of packets in the queue with batch means(simulation): {grand_mean} +- {ci_amplitude}"""
    ) 
    
    
    plt.show()


#l, mu, sim_time, lag, warm up, batch size, pareto
simulation_plot(0.1, 2,  1000000, 30000, 10000, 5000)
simulation_plot(1,   2,  100000,   400, 20000, 2000)
simulation_plot(1.8, 2,  100000,   1300, 40000, 2500)
simulation_plot(1.99, 2,  300000, 20000, 40000, 20000)



simulation_plot(0.1, 2,  1000000, 30000, 10000, 5000)
simulation_plot(0.1, 2, 1000000, 30000, 20000, 5000, "pareto")
simulation_plot(0.2, 2, 1000000, 30000, 20000, 5000, "pareto")
simulation_plot(0.2, 0.66,  1000000, 30000, 10000, 5000)
simulation_plot(0.2, 0.66, 1000000, 30000, 20000, 5000, "pareto")



