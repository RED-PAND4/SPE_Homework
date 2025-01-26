from graph import Bianconi_Barabasi_network
from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import alpha
from scipy.stats import arcsine
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import re
import time
from statsmodels.nonparametric.smoothers_lowess import lowess

constant = rv_discrete(name='constant', values=([1], [1.0]))

nw = None

def run_simulation(m,n, distribution, plot_type="all", seed=None):
    dist=None
    uniform_pattern = r'^uniform(\d+(\.\d+)?)$'
    alpha_pattern = r'^alpha(\d+(\.\d+)?)$'
    match distribution:
        case "constant":
            dist = constant
        case "expon":
            dist = expon
        case "arcsine":
            dist = arcsine
        case _ if re.match(uniform_pattern, distribution):
            scale_str = re.match(uniform_pattern, distribution).group(1)
            scale = float(scale_str)
            dist = uniform(scale=scale)
        case _ if re.match(alpha_pattern, distribution):
            a_str = re.match(alpha_pattern, distribution).group(1)
            a = float(a_str)
            dist = alpha(a)
        case _:
            dist = uniform(scale=1)

    global nw
    interactive = plot_type == "all"
    nw = Bianconi_Barabasi_network(m,dist, s=seed, interactive=interactive)
    
    for _ in range(0,n,1):
        nw.add_node() 
    nw.print_top(20)

    match plot_type:
        case "all":
            nw.plot_network()
            nw.plot_graphs()
            
        case "graphs":
            nw.plot_graphs()




    # print(coeff)
    if distribution in ["uniform", "constant", "arcsine"]:
        # print("HEREEEE")
        links_number = [n.links for n in nodes]
        links_number.sort(reverse=True)
        # print(links_number)
        degree_distribution_fit(links_number)
    else:
        sorted_nodes = sorted(nw.nodes, key=lambda node: node.links)
        node_id = sorted_nodes[-1].id
        nw.plot_probability_in_time(sorted_nodes[-1].id)
        try:
            def func(x,a,b,c,d):
                return a-b/(np.power((x+c),d))
            
            node_prob = [prob["probability"][0] for prob in nw.probabilities_nodes if prob["node"]==node_id]
            x=np.linspace(0,len(node_prob),len(node_prob))
            xx = np.linspace(0, len(node_prob),10000)
            # popt, pcov = curve_fit(func, x, node_prob, [1,1,1,1])
            # popt, pcov = curve_fit(func2, x, node_prob, [0.5,0.5,0.5,0.5],bounds=([0,-np.inf, 0,0], [np.inf, np.inf, np.inf,1]))
            popt, _ = curve_fit(func, x, node_prob, [0.5,0.5,0.5,0.5])

            plt.plot(xx,func(xx,*popt))
        except:
            print("Fit was not possible for probability in time")

    plt.show()
    return nw.get_nodes()

#fits the degree distribution of a run using a finite-domain distribution to an exponential
def degree_distribution_fit(links_number):
    f, ax = plt.subplots(1, figsize=(5, 5))
    ax.set_title("degree distribution of finite-domain distribution")
    ax.set_xlabel("degree")
    ax.set_ylabel("number of nodes")
    counts, bin_edges, _ = plt.hist(links_number, bins=int(nw.next_id/6), align="left")
    
    def func(x, a, b, c):
        return a * np.power((x-b), -2.78) +c
    
    delta = (bin_edges[1] - bin_edges[0])/2
    xx = np.array([x+delta for x in bin_edges[:-1]])
    popt, _ = curve_fit(func, xx,counts, [1,0.1,0.1], bounds=([0,-np.inf, 0], [np.inf, np.inf, np.inf]))


    # plt.plot(xx,y)
    x=np.linspace(0,bin_edges[-1],20000)
    plt.plot(xx,func(xx,*popt), "r")
    plt.show()


m=2
n=1000
distribution = "uniform10"

# nodes = run_printed_simulation(m,n, distribution)
total=0


#Run the simulations
#if distribution has finite domain, do exponential fitness distribution

#"m" is the number of links each new node establishes. The network will be inidtialized with m interconnected node

#"n" in the number on nodes to add

#"distribution" is a string specifying the distribution used to draw fitnesses.
#can be "uniformX", "alphaX", "arcsine", "expon" and "constant"
#in uniformX and alphaX, X is either an integer or a float number (using .). example: uniform1, uniform1.5, alpha4.0
#they represent the scale (in case of uniform) and the a parameter (in case of alpha)

#plot_type determines what will be plotted
#"all" plots everything, including the network in interactive mode
#"graphs" plots only the graph and not the network
#"none" does not plot anything, and only shows the top 20 nodes

#If the distribution has a finite domain, link degree distribution fitting will be performed 
#If the distribution has a non finite domain, the probability in time of the top node will be fitted (if possible)
#To ensure that the fitting of a non-finite domain distribution is likely, I suggest setting m to either 1 or 2

nodes = run_simulation(m,n, distribution, plot_type="graphs", seed=None)


#TODO: SMOOTH OUT IL PLOT DEL CLUS. COEFF, O FAI IL FIT (SMOOTH CON LOWESS, CHIEDI A CHATGPT)
#TODO: IN PLOT CLUS. COEFF, PLOTTA IN COLORI DIVERSI I TOP X NODI