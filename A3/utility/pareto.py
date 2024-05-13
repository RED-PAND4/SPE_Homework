import numpy as np
import matplotlib.pyplot as plt



def pareto_dist(a, m):
    pareto_num = np.random.pareto(m= 0.5, a=1.5)
    return pareto_num