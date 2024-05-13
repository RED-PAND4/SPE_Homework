import matplotlib.pyplot as plt
import pandas.plotting as pdplt #type: ignore
import scipy.stats as stats #type: ignore
import pandas as pd
import numpy as np
from utility.compute_theoretical_statistic import *
from enum import Enum
from statsmodels.graphics.tsaplots import plot_acf

class Statistics(Enum):
    # WAITING_TIME = 0
    # RESPONSE_TIME = 1
    PACKET_IN_SYSTEM = 0


class Plotting:
    def __init__(self, l, mu, sim_time, packets, queue_occupation):
        self.l = l
        self.mu = mu
        self.sim_time = sim_time
        self.packets = packets
        self.queue_occupation = queue_occupation
        self.rho = l/mu 
        self.avg_packets_in_system_th = avg_packet_in_sys(self.rho)
        self.avg_waiting_time_th = avg_waiting_time(self.rho, l, mu) 
        self.avg_response_time_th = avg_total_time(self.rho, l)
        self.avg_queue_length_th = avg_queue_length(self.rho)

    def plot_base_fun(self):
        f, ax = plt.subplots(2, figsize=(5, 5))

        # Check if arrivals are uniformly distributed
        x = np.linspace(0, self.sim_time)
        ax[0].hist(
            self.packets["arrival_time"],
            bins="auto",
            density=True,
            label="Arrival times",
        )
        ax[0].plot(
            x,
            stats.uniform.pdf(x, 0, self.sim_time),
            label="Uniform distribution",
        )
        ax[0].set_title("Arrival distribution")
        ax[0].legend()

        # Check if service times are exponentially distributed both through a histogram and a Kolmogorov-Smirnoff test
        x = np.linspace(0, 3)
        exp = stats.expon(scale=1 / self.mu)
        ax[1].hist(
            self.packets['departure_time'] - self.packets["server_time"],
            bins="auto",
            density=True,
            label="Service times",
        )
        ax[1].plot(x, exp.pdf(x), label="Exponential distribution")
        ax[1].set_title("Service distribution")
        ax[1].legend() 

    def plot_system_occupation(self, sim_mean):
        f, ax = plt.subplots(1, figsize=(5, 5))
        ax.step(
            self.queue_occupation["width"].cumsum(),
            self.queue_occupation["packets_in_system"],
            label="System occupation",
        )
        ax.axhline(self.avg_packets_in_system_th, label="Theoretical mean", color="b")
        ax.axhline(sim_mean, label="Simulation mean", color="r")
        ax.legend()

        # ax[1].hist(self.queue_occupation["packets_in_system"], bins='auto', width=1)
        # #print(self.queue_occupation["packets_in_system"].mode())
        # ax[1].set_title("System occupation")
        # ax[1].legend()
        # f.suptitle("System occupation")

    def plot_auto_correlation(self):
        #f, ax = plt.subplots(1, figsize=(10, 5))
        # pdplt.autocorrelation_plot(self.queue_occupation["packets_in_system"])
        # ax.set_title("Autocorrelation of packets")

        data = self.queue_occupation
        data = data[['time', 'packets_in_system']].set_index(['time'])
        plot_acf(data, use_vlines=True, lags=75, marker=" ", auto_ylims=True)
        

    def plot_confidence_interval(
        self, x, mean, ci, color="green", horizontal_line_width=0.15
    ):
        left = x - horizontal_line_width / 2
        top = mean - ci
        right = x + horizontal_line_width / 2
        bottom = mean + ci
        plt.plot([x, x], [top, bottom], color=color)
        plt.plot([left, right], [top, top], color=color)
        plt.plot([left, right], [bottom, bottom], color=color)
        plt.plot(x, mean, "o", color="#f44336")

    def plot_batch_means(self, batch_means, intervals):
        f, ax = plt.subplots(1, figsize=(5, 5))
        for i in range(len(batch_means)):
            self.plot_confidence_interval(i + 1, batch_means[i], intervals[i])
            ax.axhline(self.avg_packets_in_system_th, color="r", label="Theoretical mean")
            ax.set_title("Batch means of packets")

        #ax.legend()  

    def plot_mean_in_time(self, queues):
        f, ax = plt.subplots(1, figsize=(5, 5))
        for queue in queues: 
            avg_history=[]
            tot_packet = 0
            for i in range(len(queue)):
                tot_packet += queue["packets_in_system"][i]* queue["width"][i]
                time = queue["time"][i] + queue["width"][i]
                avg_history.append(tot_packet/time)
            plt.plot(queue["time"], avg_history)
        ax.axhline(self.avg_packets_in_system_th, color="r", label="Theoretical mean")
        ax.set_title("Means in time")

    