import numpy as np
from node import Node
from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import alpha
from scipy.stats import arcsine
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import matplotlib.colors as mcolors
import networkx as nx
import random

class Bianconi_Barabasi_network:
    def __init__(self,n,m, dist):
        # n = number of starting nodes in the nw
        # m = number of connections a node can make when joining (< n)
        # dist = distribution for the fitnesses
        self.nodes=[]
        self.edges=set()
        self.distribution=dist
        self.m=m
        self.next_id=n
        self.probabilities_nodes = []

        #Creation of the starting nodes
        for i in range(0,n,1):
            self.nodes.append(Node(i,self.distribution.rvs(size=1)))
            
        #Connecting the starting nodes in a circular layout
        for i in range(0,n,1):
            if( ((i+1)%n, i) not in self.edges):
                self.edges.add((i,(i+1)%n))
                self.nodes[i].add_link()
                self.nodes[(i+1)%n].add_link()
                
    #Print all nodes and connections
    def print_all(self):
        for n in self.nodes:
            n.print_node()
        print(self.edges)

    #Print fitnesses of all nodes
    def print_fitnesses(self):
        for n in self.nodes:
            print("Fitness node ",n.id,":",n.fitness)

    #Add a node to the network
    def add_node(self):
        node = Node(self.next_id,self.distribution.rvs(size=1))
        self.next_id+=1
        self.generate_links(node)
        self.nodes.append(node)

    #Generating links for a new node
    def generate_links(self, new_node):
        connected=set()
        for _ in range(0,self.m,1):
            total=0
            comulative_prob=0
            #Calculating the Sum(Ki*ni) (excluding nodes already connected to the new node)
            for n in self.nodes:
                if n.id in connected:
                    continue
                total+=n.fitness*n.links 
    
            #randomly selecting a new node to connect to 
            x=random.random()           
            for n in self.nodes:
                if n.id in connected:
                    continue
                comulative_prob+=(n.fitness*n.links)/total
                #print("cumulative prob:",comulative_prob)
                self.probabilities_nodes.append({'time_new_node':len(self.nodes), 'node': n.id, 'probability': (n.fitness*n.links)/total})
                if (x<=comulative_prob):
                    connected.add(n.id)
                    n.add_link()
                    new_node.add_link()
                    self.edges.add((new_node.id,n.id))
                    x=2
                    #break

    def plot(self):

        fig, ax = plt.subplots()


        G = nx.Graph()
        for n in self.nodes:
            G.add_node(n.id)
        G.add_edges_from(self.edges)
        # Calculate node sizes based on degree
        node_degrees = dict(G.degree())  # Get the degree of each node
        degrees = np.array(list(node_degrees.values()))
        min_degree = degrees.min()
        max_degree = degrees.max()
        norm_degrees = (degrees - min_degree) / (max_degree - min_degree)

        cmap = mcolors.LinearSegmentedColormap.from_list("my_cmap", ["red", "yellow", "green"])
        
        node_colors = [cmap(norm_degree) for norm_degree in norm_degrees]

        node_sizes = [(node_degrees[node]-self.m+3) * 15 for node in G.nodes()]  # Scale node sizes

        # Plot the graph
        # pos = nx.spring_layout(G, k=1)  # positions for all nodes
        # pos = nx.nx_pydot.graphviz_layout(G)
        # pos = nx.kamada_kawai_layout(G)  # positions for all node
        pos = nx.circular_layout(G)  # positions for all node
        # Draw the nodes with sizes based on degree
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)

        # Draw the edges
        nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.5, edge_color='gray')

        # Draw the labels
        # nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

        for node, (x, y) in pos.items():
           text(x, y, node, fontsize=np.log(node_sizes[node]*50), ha='center', va='center')
        # Show the plot
        ax.axis('off')
        fig.set_facecolor('white')

        plt.title("Graph Visualization with Node Sizes Based on Degree")
        plt.show()
        
    def plot_probability_in_time(self, number_node):
        f, ax = plt.subplots(1, figsize=(5, 5))
        print(self.probabilities_nodes)
        print("----")
        node_prob = [prob["probability"] for prob in self.probabilities_nodes if prob["node"]==number_node]
        print(node_prob)
        #for prob in self.probabilities_nodes: 
            # avg_history=[]
            # tot_packet = 0
            # for i in range(len(queue)):
            #     tot_packet += queue["packets_in_system"][i]* queue["width"][i]
            #     time = queue["time"][i] + queue["width"][i]
            #     avg_history.append(tot_packet/time)
            #if prob["node"]==number_node:
                
        plt.plot(node_prob)
        ax.set_title("Probability in time")
        ax.set_ylabel('Probability of Node')
        ax.set_xlabel("new nodes")



#Bianconi_Barabasi_network(n,m,dist)
# n = number of starting nodes
# m = number of connections every new node can make when joining the network. must be <=n
# dist = distribution of the fitnesses. Must be >0

# nw = Bianconi_Barabasi_network(15,7,uniform())
# nw = Bianconi_Barabasi_network(15,7,expon())
# nw = Bianconi_Barabasi_network(15,7,alpha(a=1))
nw = Bianconi_Barabasi_network(5,3,arcsine())
# nw.print_all()
for i in range(0,15,1):
    nw.add_node() 

# nw.print_all()
nw.print_fitnesses()
#for n in nw.nodes:
nw.plot_probability_in_time(6)
nw.plot()