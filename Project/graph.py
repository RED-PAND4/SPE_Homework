import numpy as np
from node import Node
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import networkx as nx
import random

class Bianconi_Barabasi_network:
    def __init__(self,m, distribution):
        self.nodes=[]
        self.edges=set()
        self.distribution=distribution
        self.connections_number = m
        self.next_id=0
        
        for _ in range(0,m,1):
            self.add_node()
            # self.print_all()
        # for i in range(0,n,1):
        #     self.nodes.append(Node(i,self.distribution.rvs(size=1)))
        # for i in range(0,n,1):
        #     if( ((i+1)%n, i) not in self.edges):
        #         self.edges.add((i,(i+1)%n))
        #         self.nodes[i].add_link()
        #         self.nodes[(i+1)%n].add_link()
                
    #Print all nodes and connections
    def print_all(self):
        for n in self.nodes:
            n.print_node()
        # print(self.edges)

    #Print fitnesses of all nodes
    def print_fitnesses(self):
        for n in self.nodes:
            print("Fitness node ",n.id,":",n.fitness)

    def print_top(self,n):
        sorted_nodes = sorted(self.nodes, key=lambda x: x.links, reverse=True)
        for x in sorted_nodes[:n]:
            print("Node: ", x.id,", Links: ",x.links, ", Fitness: ",x.fitness)

    def add_node(self):
        node = Node(self.next_id,self.distribution.rvs(size=1))
        # print("Sono qui")
        self.next_id+=1
        self.generate_links(node)
        self.nodes.append(node)

    #Generating links for a new node
    def generate_links(self, new_node):
        # print("generando links")
        connected=set()
        for _ in range(0,self.connections_number,1):
            total=0
            comulative_prob=0
            
            x=random.random()
            # print("len connected:",len(connected), " len(self.nodes):",len(self.nodes))
            if(len(connected) == len(self.nodes)):
                break

            if(len(self.nodes)==1):
                self.nodes[0].add_link()
                new_node.add_link()
                self.edges.add((new_node.id,self.nodes[0].id))
            # if(len(self))
            # print("x:",x)
            # print("connected:",connected)
            # print("Generating anyway")
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
                    self.chosen_nodes.append((n.links*n.fitness)/total)
                    connected.add(n.id)
                    n.add_link()
                    new_node.add_link()
                    self.edges.add((new_node.id,n.id))
                    x=2
                    # break
        
    def get_probabilities_nodes(self):
        return pd.DataFrame(self.probabilities_nodes)
    
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

        node_sizes = [(node_degrees[node]-self.connections_number+3) * 15 for node in G.nodes()]  # Scale node sizes

        # Plot the graph
        # pos = nx.spring_layout(G, k=1)  # positions for all nodes
        # pos = nx.nx_pydot.graphviz_layout(G)
        pos = nx.kamada_kawai_layout(G)  # positions for all node
        # pos = nx.circular_layout(G)  # positions for all node
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
        
    def plot_probability_in_time(self, number_node):
        f, ax = plt.subplots(1, figsize=(5, 5))
        # print(self.probabilities_nodes)
        # print("----")
        # for x in self.probabilities_nodes:
        #     print(x)
        node_prob = [prob["probability"] for prob in self.probabilities_nodes if prob["node"]==number_node]
        # node_prob = [(prob["probability"], prob["time_new_node"]) for prob in self.probabilities_nodes if prob["node"]==number_node]
        # values = set(map(lambda x:x[1], node_prob))
        # print(values)
        # newlist = [[y[0] for y in node_prob if y[1]==x] for x in values]
        # print(newlist)
        # print(node_prob)

        # newlist = self.get_probabilities_nodes()
        # newlist = newlist[newlist['node']==number_node]
        # print("only the node -- ")
        # print(newlist)
        # newlist = newlist.groupby('time_new_node')['probability'].sum()
        # print("groupby -- ")
        # print(newlist)
        # ax.plot(newlist.index, newlist.values, marker='o')
        ax.set_title("Probability in time, node id:"+str(number_node))
        ax.set_ylabel('Probability of Node')
        ax.set_xlabel("new nodes")
         
        plt.plot(node_prob)
