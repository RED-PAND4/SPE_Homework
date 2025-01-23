import numpy as np
from node import Node
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import networkx as nx
import random
import pandas as pd
import threading
import time
import mplcursors


class Bianconi_Barabasi_network:
    def __init__(self,m, distribution):
        self.nodes=[]
        self.edges=list()
        self.distribution=distribution
        self.connections_number = m
        self.next_id=0
        self.probabilities_nodes=[]
        self.chosen_nodes=[]
        self.G = nx.Graph()
        self.fig, self.ax = plt.subplots()
        self.running = False
        self.annotation = None
        
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
        # plt.ion()        
    #Print all nodes and connections
    def print_all(self):
        for n in self.nodes:
            n.print_node()
        # print(self.edges)
    # plt.ion()

    def get_nodes(self):
        return self.nodes
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
        #no link generation if no other nodes are present
        connected=set()
        total = 0
        if (len(self.nodes)==0):
            return
        
        #only one otehr node is present with zero links: connect to it
        if(len(self.nodes)==1):
            self.nodes[0].add_link()
            new_node.add_link()
            self.edges.append((new_node.id, self.nodes[0].id))
            return
        
        for n in self.nodes:
            total+=n.fitness*n.links 
        for _ in range(0,self.connections_number,1):
            comulative_prob=0
            
            x=random.random()
            # print("len connected:",len(connected), " len(self.nodes):",len(self.nodes))
            if(len(connected) == len(self.nodes)):
                return

            if(len(self.nodes)==1):
                self.nodes[0].add_link()
                new_node.add_link()
                self.edges.append((new_node.id,self.nodes[0].id))
            # if(len(self))
            # print("x:",x)
            # print("connected:",connected)
            # print("Generating anyway")
            
    
            #randomly selecting a new node to connect to 
            x=random.random()           
            for n in self.nodes:
                if n.id in connected:
                    continue
                comulative_prob+=(n.fitness*n.links)/total
                #print("cumulative prob:",comulative_prob)
                if (x<=comulative_prob):
                    self.probabilities_nodes.append({'time_new_node':len(self.nodes), 'node': n.id, 'probability': (n.fitness*n.links)/total, 'chosen':True})
                    # self.probabilities_nodes[-1]
                    self.chosen_nodes.append((n.links*n.fitness)/total)
                    total -= (n.links*n.fitness)
                    connected.add(n.id)
                    n.add_link()
                    new_node.add_link()
                    self.edges.append((new_node.id,n.id))
                    x=2
                    # break
                else:
                    self.probabilities_nodes.append({'time_new_node':len(self.nodes), 'node': n.id, 'probability': (n.fitness*n.links)/total, 'chosen':False})
        
    def get_probabilities_nodes(self):
        return pd.DataFrame(self.probabilities_nodes)
    
    
    def loop_task(self):
        while self.running:
            self.update_graph_new_node(None)
            time.sleep(0.2)

    def start_loop(self, event):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.loop_task)
            self.thread.start()
            print("Started loop.")

    def stop_loop(self, event):
        """Stop the loop by setting the flag to False."""
        self.running = False
        print("Stopped loop.")

    def update_graph_new_node(self, event):
        adding = False
        
        if(self.G.number_of_nodes() == 0): #first time function is called, initialization
            for n in self.nodes:
                self.G.add_nodes_from([(n.id, {"info":"fitness:"+str(n.fitness)})])
            self.G.add_edges_from(self.edges)
        else: #not first time function is called, addign of a new node
            new_node_id = self.nodes[-1].id
            new_node_fitness = self.nodes[-1].fitness
            self.add_node()
            self.G.add_nodes_from([(new_node_id, {"info":"fitness:"+str(new_node_fitness)})])
            self.G.add_edges_from(self.edges[-self.connections_number:])
            adding = True

        # Recalculate node sizes and colors
        node_degrees = dict(self.G.degree())
        degrees = np.array(list(node_degrees.values()))
        min_degree = degrees.min()
        max_degree = degrees.max()
        if max_degree == min_degree:
            norm_degrees = np.zeros_like(degrees)
        else:
            norm_degrees = (degrees - min_degree) / (max_degree - min_degree)

        cmap = mcolors.LinearSegmentedColormap.from_list("my_cmap", ["red", "yellow", "green"])
        node_colors = [cmap(norm_degree) for norm_degree in norm_degrees]
        node_sizes = [(node_degrees[node]-self.connections_number+3) * 15 for node in self.G.nodes()]  # Scale node sizes

        # Clear and redraw the graph
        self.ax.clear()
        self.ax.set_title("Graph Visualization with Node Sizes Based on Degree")

        pos = nx.circular_layout(self.G)
        scatter = nx.draw_networkx_nodes(self.G, pos, node_size=node_sizes, node_color=node_colors, ax=self.ax)
        
        if(adding):
            nx.draw_networkx_edges(self.G, pos, edgelist = self.edges[:-self.connections_number], width=0.3, alpha=0.5, edge_color="gray", ax=self.ax)
            nx.draw_networkx_edges(self.G, pos, edgelist = self.edges[-self.connections_number:], width=0.5, alpha=0.5, edge_color="red", ax=self.ax)
            s = "Added node "+str(new_node_id)+" with fitness = "+str(new_node_fitness)
            self.ax.text(0.1, 0.0, s, fontsize=10, ha='left', va='bottom', transform=self.ax.transAxes)
            # Refresh the plot
            self.ax.axis("off")
        else:
            self.ax.axis('off')
            self.fig.set_facecolor('white')

            nx.draw_networkx_edges(self.G, pos, width=0.3, alpha=0.5, edge_color='gray')

        # Create buttons only once when the graph is initialized (not on every update)
        if not hasattr(self, 'buttons_created'):  # Check if buttons are already created
            self.buttons_created = True
            btn_ax = self.fig.add_axes([0.05, 0.08, 0.15, 0.08])
            start_btn_ax = self.fig.add_axes([0.85, 0.23, 0.10, 0.08])
            stop_btn_ax = self.fig.add_axes([0.85, 0.08, 0.10, 0.08])

            new_node_btn = Button(btn_ax, 'Add node')
            new_node_btn.on_clicked(self.update_graph_new_node)

            start_btn = Button(start_btn_ax, 'Start')
            start_btn.on_clicked(self.start_loop)

            stop_btn = Button(stop_btn_ax, 'Stop')
            stop_btn.on_clicked(self.stop_loop)

        for node, (x, y) in pos.items():
            self.ax.text(x, y, node, fontsize=np.log(node_sizes[node]*50), ha='center', va='center')
            
        cursor = mplcursors.cursor(scatter, hover=True)
        cursor.connect("add", self.custom_annotation)    
        self.fig.canvas.mpl_connect("button_press_event", self.remove_annotation)  # Clear annotation when clicking background

        if(not adding):
            plt.show()  
        else:
            self.fig.canvas.draw()


    def custom_annotation(self, sel):
        if (self.annotation):
            self.annotation.set_visible(False)
            self.fig.canvas.draw_idle()  

        sel.annotation.set_text("Fitness:\n"+str(self.nodes[sel.index].fitness))
        # sel.annotation.set_text(f"Node: {sel.index + 1}")
        self.annotation = sel.annotation  # Save a reference to the annotation

    def remove_annotation(self, event):
        if self.annotation:  # If there's an annotation, clear it
            self.annotation.set_visible(False)
            self.fig.canvas.draw_idle()  
    def plot_probability_top_links(self):
        sorted_nodes = sorted(self.nodes, key=lambda node: node.links)
        print(sorted_nodes[-1].id)
        self.plot_probability_in_time(sorted_nodes[-1].id)

    #Plot probability of being chosen of the top node
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

    #Shows the pprobability that each chosen node had when it was chosen
    def plot_probability_of_chosen_nodes(self):
        f, ax = plt.subplots(1, figsize=(5, 5))
        ax.set_title("Probability of nodes at the time of choice")
        ax.set_ylabel('Probability of Node')
        ax.set_xlabel("new nodes")
        plt.plot(self.chosen_nodes)

    def plot_all(self):
        self.update_graph_new_node(None)
        self.plot_probability_top_links()
        self.plot_probability_of_chosen_nodes()
        plt.show()