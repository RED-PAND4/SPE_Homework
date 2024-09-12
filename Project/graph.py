from node import Node
from scipy.stats import uniform
import matplotlib.pyplot as plt
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
            print("x:",x)
            print("connected:",connected)
            for n in self.nodes:
                if n.id in connected:
                    continue
                total+=n.fitness*n.links
            # print("total:",total)

            for n in self.nodes:
                if n.id in connected:
                    continue
                comulative_prob+=(n.fitness*n.links)/total
                print("cumulative prob:",comulative_prob)
                if (x<=comulative_prob):
                    print("found:",n.id)
                    connected.add(n.id)
                    n.add_link()
                    new_node.add_link()
                    self.edges.add((new_node.id,n.id))

                    break

    def plot(self):
        G = nx.Graph()
        for n in self.nodes:
            G.add_node(n.id)
        G.add_edges_from(self.edges)
        # Calculate node sizes based on degree
        node_degrees = dict(G.degree())  # Get the degree of each node
        node_sizes = [node_degrees[node] * 15 for node in G.nodes()]  # Scale node sizes

        # Plot the graph
        pos = nx.spring_layout(G)  # positions for all nodes

        # Draw the nodes with sizes based on degree
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue')

        # Draw the edges
        nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.5, edge_color='gray')

        # Draw the labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        # Show the plot
        plt.title("Graph Visualization with Node Sizes Based on Degree")
        plt.show()
        

    

nw = Bianconi_Barabasi_network(15,10,uniform())
nw.print_all()
for i in range(0,50,1):
    nw.add_node() 

# nw.print_all()
nw.plot()