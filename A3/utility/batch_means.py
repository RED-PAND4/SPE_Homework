import numpy as np
import numbers
import scipy.stats as stats #type: ignore
from plotting import Statistics
import pandas.plotting as pdplt #type: ignore
import matplotlib.pyplot as plt



# to remove the warmup period 

def compute_batch_means_statistics(type, queue_occupation, batch_size, warmup_time, z):
    queue_occupation = queue_occupation.loc[queue_occupation['time'] > warmup_time]

    if type == Statistics.PACKET_IN_SYSTEM:
        batches = [queue_occupation[i * batch_size : (i + 1) * batch_size] for i in range(int(len(queue_occupation) / batch_size))]
    else:
        raise ValueError("Invalid type")
    batches = batches[:-1] # remove last batch because is smaller than the others

    number_batches = len(batches)
    print(f"Number batches: {number_batches} of size {batch_size}")


    #batch_means = [np.mean(batch) for batch in batches]

    t=0
    batch_means = []
    #compute means
    for b in batches:
        #print(b[0].shape)        
        total_width = np.sum(b["width"].values)
        tot = np.sum(b["packets_in_system"].values * b["width"].values) / total_width
        # t=1
        # tot = 0
        # for element in b:
        #     total_width = np.sum(element[2].values)
        #     tot = np.sum(element[1].values * element[2].values) / total_width
    
            # #print(element.shape)
            # print(element)
            # if not isinstance(element[0], numbers.Number):
            #     print("continue")
            #     continue
            # print("should be pakets:", element[0])
            # print("should be width:" , element[1])
            # t += element[1]
            # tot += element[1] * element[0]
        batch_means.append(tot)

    eta = stats.norm.ppf((1 + z) / 2) #enough batches to use CLT
    ci_s = [eta * np.std(b) / np.sqrt(batch_size) for b in batches]

    # correlation?
    f, ax = plt.subplots(1)
    pdplt.autocorrelation_plot(batch_means, ax=ax)
    ax.set_title("autocorrelation for batch means")

    # compute g_mean
    g_mean = np.mean(batch_means)

    # compute variance
    var = 1 / (number_batches - 1) * np.sum([(b - g_mean) ** 2 for b in batch_means])

    ci_amplitude = eta * np.sqrt(var / number_batches)

    return g_mean, ci_amplitude, batch_means, ci_s
