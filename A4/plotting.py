import matplotlib.pyplot as plt
import pandas.plotting as pdplt #type: ignore
import scipy.stats as stats #type: ignore
import pandas as pd
import numpy as np
from enum import Enum
from statsmodels.graphics.tsaplots import plot_acf

class Statistics(Enum):
    NODES_SPEED = 0


class Plotting:
    def __init__(self, sim_time, speeds):
        self.sim_time = sim_time
        self.speeds = speeds
        

    def plot_auto_correlation(self, lag):
        f, ax = plt.subplots(1, figsize=(10, 5))
        # pdplt.autocorrelation_plot(self.speeds["packets_in_system"])
        # ax.set_title("Autocorrelation of packets")

        data = self.speeds
        data = data[['time', 'speed']].set_index(['time'])
        plot_acf(data, ax = ax, use_vlines=True, lags=lag, marker=" ")
        ax.set_ylabel("probability of data being correlated")
        ax.set_xlabel("number of lags")

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
        mean = self.speeds.mean()
        ax.axhline(mean["speed"], color="r", label="mean")
        ax.set_title("Batch means")
        ax.set_ylabel("Mean of speed")
        ax.set_xlabel("Batch number")
        ax.legend()  


    