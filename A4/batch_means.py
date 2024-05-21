import numpy as np
import math
import scipy.stats as stats #type: ignore
from plotting import Statistics
import pandas.plotting as pdplt #type: ignore
import matplotlib.pyplot as plt
#from statsmodels.graphics.tsaplots import plot_acf



# to remove the warmup period 

def compute_batch_means_statistics(type, speeds, batch_size, warmup_time, z):
    speeds = speeds.loc[speeds['time'] > warmup_time]

    if type == Statistics.PACKET_IN_SYSTEM:
        #batches = [speeds[i * batch_size : (i + 1) * batch_size] for i in range(int(len(speeds) / batch_size))]
        #print(speeds)
        num_batches = math.ceil(len(speeds) / batch_size)
        batches = np.array_split(speeds, num_batches)
    else:
        raise ValueError("Invalid type")
    #batches = batches[:-1] # remove last batch because is smaller than the others

    number_batches = len(batches)
    print(f"Number batches: {number_batches} of size {batch_size}", flush=True)

    
    #compute means
    batch_means = []
    for b in batches:  
        #print(b)  
        tot = b.mean()
        #print(tot["speed"])
        batch_means.append(tot["speed"])

    eta = stats.norm.ppf((1 + z) / 2) #enough batches to use CLT
    ci_s = []
    for b in batches:
        tot = b.mean()
        ci = eta * np.std(tot["speed"]) / np.sqrt(batch_size)
        #print(ci)
        ci_s.append(ci)

    # compute g_mean
    g_mean = np.mean(batch_means)

    # compute variance
    var = 1 / (number_batches - 1) * np.sum([(b - g_mean) ** 2 for b in batch_means])

    ci_amplitude = eta * np.sqrt(var / number_batches)

    return g_mean, ci_amplitude, batch_means, ci_s

