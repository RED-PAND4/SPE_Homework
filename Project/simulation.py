from graph import Bianconi_Barabasi_network
from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import alpha
from scipy.stats import arcsine
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import time
constant = rv_discrete(name='constant', values=([1], [1.0]))

nw = None

def run_simulation(m,n, distribution, plotted=True, seed=None):
    dist=None
    match distribution:
        case "uniform":
            dist = uniform(scale=5)
        case "constant":
            dist = constant
        case "expon":
            dist = expon
        case "arcsine":
            dist = arcsine
        case "alpha":
            dist = alpha(1.5)
        case _:
            dist = uniform(scale=5)

    global nw
    nw = Bianconi_Barabasi_network(m,dist, s=seed)

    for _ in range(0,n,1):
        nw.add_node() 
    nw.print_top(20)

    if(plotted):
        nw.plot_all()
    nodes = nw.get_nodes()
    coeff = nw.calculate_clustering_coefficient()
    print(coeff)
    if distribution in ["uniform", "constant", "arcsine"]:
        # print("HEREEEE")
        links_number = [n.links for n in nodes]
        links_number.sort(reverse=True)
        # print(links_number)
        degree_distribution_fit(links_number)
    else:
        # print("In here")
        sorted_nodes = sorted(nw.nodes, key=lambda node: node.links)
        node_id = sorted_nodes[-1].id
        print(sorted_nodes[-1].id)
        nw.plot_probability_in_time(sorted_nodes[-1].id)

        def func(x,a,b,c,d):
            return a-b/(np.power((x+c),d))
        
        node_prob = [prob["probability"][0] for prob in nw.probabilities_nodes if prob["node"]==node_id]
        x=np.linspace(0,len(node_prob),len(node_prob))
        xx = np.linspace(0, len(node_prob),10000)
        # popt, pcov = curve_fit(func, x, node_prob, [1,1,1,1])
        # popt, pcov = curve_fit(func2, x, node_prob, [0.5,0.5,0.5,0.5],bounds=([0,-np.inf, 0,0], [np.inf, np.inf, np.inf,1]))
        popt, _ = curve_fit(func, x, node_prob, [0.5,0.5,0.5,0.5])

        plt.plot(xx,func(xx,*popt))
        plt.show()

    return nw.get_nodes()

#fits the degree distribution of a run using a finite-domain distribution to an exponential
def degree_distribution_fit(links_number):
    counts, bin_edges, _ = plt.hist(links_number, bins=int(nw.next_id/6), align="left")
    
    def func(x, a, b, c):
        return a * np.power((x-b), -2.78) +c
    
    delta = (bin_edges[1] - bin_edges[0])/2
    xx = np.array([x+delta for x in bin_edges[:-1]])
    # print("delta:",delta)
    # print(xx)
    # popt, pcov = curve_fit(func, xx,counts, [50,200,0.001,0])
    # popt, pcov = curve_fit(func2, xx,counts, [1,1,1])
    popt, _ = curve_fit(func, xx,counts, [1,0.1,0.1], bounds=([0,-np.inf, 0], [np.inf, np.inf, np.inf]))


    print("popt:", popt)
    # plt.plot(xx,y)
    x=np.linspace(0,bin_edges[-1],20000)
    plt.plot(xx,func(xx,*popt), "r")
    plt.show()


m=3
n=2000
distribution = "alpha"
# distribution = alpha(1.5)
# distribution = uniform(scale=5)

# nodes = run_printed_simulation(m,n, distribution)
total=0

#if distribution has finite domain, do exponential fitness distribution
#Run the simulations
#m is the number of links each new node establishes. The network will be initialized with m interconnected node
#n in the number on nodes to add
#distribution is a string specifying the distribution used to draw fitnesses.
#can be uniform, alpha, arcsine, expon and constant
#plotted can be set to true or false, and determine wether the netwoe will be plotted
#If the distribution has a finite domain, link degree distribution fitting will be performed 
#If the distribution has a non finite domain, the probability in time of the top node will be fitted (if possible)
#To ensure that the fitting of a non-finite domain distribution is likely, I suggest setting m to either 1 or 2
nodes = run_simulation(m,n, distribution, plotted=False, seed=123)