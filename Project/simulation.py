from graph import Bianconi_Barabasi_network
from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import alpha
from scipy.stats import arcsine
from scipy.stats import rv_discrete


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


constant = rv_discrete(name='constant', values=([1], [1.0]))
run_printed_simulation(7,10, arcsine())
# run_printless_simulation(7,1, uniform, 50)


