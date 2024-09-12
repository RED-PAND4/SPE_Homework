import numpy as np
from node import Node
from scipy.stats import uniform
from scipy.stats import expon
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, text
import matplotlib.colors as mcolors
import networkx as nx
import random

class Bianconi_Barabasi_network:
    def __init__(self,n,m, distribution):
        self.nodes=[]
        self.edges=set()
        self.distribution=distribution
        self.m=m
        self.next_id=n

        for i in range(0,n,1):
            self.nodes.append(Node(i,self.distribution.rvs(size=1)))
        for i in range(0,n,1):
            if( ((i+1)%n, i) not in self.edges):
                self.edges.add((i,(i+1)%n))
                self.nodes[i].add_link()
                self.nodes[(i+1)%n].add_link()
                
    def print_all(self):
        for n in self.nodes:
            n.print_node()
        print(self.edges)

    def add_node(self):
        node = Node(self.next_id,self.distribution.rvs(size=1))
        self.next_id+=1
        self.generate_links(node)
        self.nodes.append(node)

    def generate_links(self, new_node):
        connected=set()
        # while len(connected)<self.m:
        #     total=0
        #     for n in self.nodes:
        #         if n.id not in connected:
        #             total+=n.fitness*n.links

        for _ in range(0,self.m,1):
            total=0
            comulative_prob=0
            
            x=random.random()
            # print("x:",x)
            # print("connected:",connected)
            for n in self.nodes:
                if n.id in connected:
                    continue
                total+=n.fitness*n.links
            # print("total:",total)

            for n in self.nodes:
                if n.id in connected:
                    continue
                comulative_prob+=(n.fitness*n.links)/total
                # print("cumulative prob:",comulative_prob)
                if (x<=comulative_prob):
                    # print("found:",n.id)
                    connected.add(n.id)
                    n.add_link()
                    new_node.add_link()
                    self.edges.add((new_node.id,n.id))

                    break

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

        node_sizes = [node_degrees[node] * 15 for node in G.nodes()]  # Scale node sizes

        # Plot the graph
        # pos = nx.spring_layout(G, k=1)  # positions for all nodes
        # pos = nx.nx_pydot.graphviz_layout(G)
        # pos = nx.kamada_kawai_layout(G)  # positions for all node
        pos = nx.circular_layout(G)  # positions for all node
        # Draw the nodes with sizes based on degree
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)

        # Draw the edges
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, edge_color='black')

        # Draw the labels
        # nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

        for node, (x, y) in pos.items():
           text(x, y, node, fontsize=np.log(node_sizes[node]*50), ha='center', va='center')
        # Show the plot
        ax.axis('off')
        fig.set_facecolor('white')

        plt.title("Graph Visualization with Node Sizes Based on Degree")
        plt.show()
        

    

nw = Bianconi_Barabasi_network(15,7,uniform())
nw.print_all()
for i in range(0,100,1):
    nw.add_node() 

# nw.print_all()
nw.plot()