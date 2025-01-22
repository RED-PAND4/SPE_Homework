from graph import Bianconi_Barabasi_network
from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import alpha
from scipy.stats import arcsine
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


#creates a simulation of the model printing the final network.
# m: the number of connection a node will make
# n: the number of new nodes to be added
# distribution: the distribution used to generate the fitnesses
def run_printed_simulation(m, n, distribution):
    nw = Bianconi_Barabasi_network(m,distribution)
    for _ in range(0,n,1):
        nw.add_node() 
    # nw.print_fitnesses()
    # nw.plot()
    nw.print_top(30)
    nw.plot_all()
    nw.print_top(20)
    return nw.get_nodes()
    # nw.plot_npm_of_chosen_nodes()

#creates a simulation of the model without printing the final network.
# m: the number of connection a node will make
# n: the number of new nodes to be added
# distribution: the distribution used to generate the fitnesses
# top: the number of nodes with the highest fitness to be printed
def run_printless_simulation(m, n, distribution, top):
    nw = Bianconi_Barabasi_network(m,distribution)
    for _ in range(0,n,1):
        nw.add_node() 
    nw.print_top(top)
    return nw.get_nodes()



def degree_distribution_fit(links_number):
    counts, bin_edges, patches = plt.hist(links_number, bins=int(n/4))
    print(counts)
    print(bin_edges)
    def func(x, a, b, c, d):
        return a*np.exp(-c*(x-b))+d
    delta = (bin_edges[1] - bin_edges[0])/2
    xx = [x+delta for x in bin_edges[:-1]]
    print("delta:",delta)
    print(xx)
    popt, pcov = curve_fit(func, xx,counts, [50,200,0.001,0])
    print("popt:", popt)
    # plt.plot(xx,y)
    x=np.linspace(0,bin_edges[-1],n)
    plt.plot(xx,func(xx,*popt))
    plt.show()

m=7
n=2
distribution = expon
constant = rv_discrete(name='constant', values=([1], [1.0]))
nodes = run_printed_simulation(m,n, distribution)
# nodes = run_printless_simulation(m,n, distribution, 20)

#if distribution has finite domain, do exponential fitness distribution
if(distribution == uniform or distribution == arcsine):

    links_number = [n.links for n in nodes]
    links_number.sort(reverse=True)
    # print(links_number)
    degree_distribution_fit(links_number)
    # plt.hist(links_number,bins=int(n/4))
    # plt.show()