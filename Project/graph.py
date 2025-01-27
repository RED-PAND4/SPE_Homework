import numpy as np
from node import Node
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
import networkx as nx
import random
import pandas as pd
import threading
import time
import mplcursors
from scipy.optimize import curve_fit


#simulates a Bianconi-Barabasi network
class Bianconi_Barabasi_network:

    #m is the number of connection each new node will make. The graph will be initialized with m fully interconnected nodes
    #distribution is the distribution that is used to generate the fitnesses.
    def __init__(self,m, distribution, s=None, interactive = False):
        #s is a seed used, if provided, for random number generator and variate draw
        self.seed=s
        self.nodes=[] #stores the list of nodes
        self.edges=list() #stores the edges between nodes
        self.distribution=distribution #the distribution used for the fitnesses
        self.connections_number = m #the number of connections of each new node
        self.next_id=0 #the id of the next node to be added
        self.probabilities_nodes=[] #probability and other informations of chosen nodes
        self.chosen_nodes=[] #probabilities that each node had at the time of it's choosing
        self.G = nx.Graph() #The graph to plot the network
        
        self.fig, self.ax = (None, None)
        if(interactive):
            self.fig, self.ax = plt.subplots()
        self.running = False #Flag used to start/stop automatic simulation
        self.annotation = None #annotataion shown when hovering over a node
        self.rng=None
        #adding the starting m nodes
        for _ in range(0,m,1):
            self.add_node()

        if s != None:
            random.seed(s)
            self.rng = np.random.default_rng(seed=s)

    #Print all nodes and connections
    def print_all(self):
        for n in self.nodes:
            n.print_node()
        # print(self.edges)
    # plt.ion()

    #returns list of nodes
    def get_nodes(self):
        return self.nodes
    
    #Print fitnesses of all nodes
    def print_fitnesses(self):
        for n in self.nodes:
            print("Fitness node ",n.id,":",n.fitness)

    #print the n nodes with the highest number of links in descending order
    def print_top(self,n):
        sorted_nodes = sorted(self.nodes, key=lambda x: x.links, reverse=True)
        for x in sorted_nodes[:n]:
            print("Node: ", x.id,", Links: ",x.links, ", Fitness: ",x.fitness)
    
    def variate_draw(self):
        if self.seed != None:
            return self.distribution.rvs(size=1, random_state=self.rng)
        else:
            return self.distribution.rvs(size=1)

    #adds a node to the network
    def add_node(self):
        i = self.variate_draw()
        node = Node(self.next_id,i)
        self.next_id+=1
        self.generate_links(node)
        self.nodes.append(node)

    #Generating links for a new node
    def generate_links(self, new_node):

        connected=set()
        total = 0
        #if no other node exists: no node generation
        if (len(self.nodes)==0):
            return
        
        #if only one other node is present with zero links: connect to it
        if(len(self.nodes)==1):
            self.nodes[0].add_link()
            new_node.add_link()
            self.edges.append((new_node.id, self.nodes[0].id))
            return
        
        #calculate summation of fitness*links of all nodes
        for n in self.nodes:
            total+=n.fitness*n.links 
        
        #generates new links
        for _ in range(0,self.connections_number,1):
            comulative_prob=0
            
            x=random.random()
            # x = np.random.rand(1,1)[0]

            #is number of connected nodes is equal to number of all nodes: already connected to all of them, returns
            if(len(connected) == len(self.nodes)):
                return
    
            #randomly selecting a new node to connect to
            x=random.random()           
            for n in self.nodes:
                if n.id in connected:
                    continue
                comulative_prob+=(n.fitness*n.links)/total
                
                #node is identified
                if (x<=comulative_prob): 
                    self.probabilities_nodes.append({'time_new_node':len(self.nodes), 'node': n.id, 'probability': (n.fitness*n.links)/total, 'chosen':True})
                    self.chosen_nodes.append((n.links*n.fitness)/total)
                    total -= (n.links*n.fitness)
                    connected.add(n.id)
                    n.add_link()
                    new_node.add_link()
                    n.add_neighbour(new_node.id)
                    new_node.add_neighbour(n.id)
                    self.edges.append((new_node.id,n.id))
                    break
                # else:
                #     self.probabilities_nodes.append({'time_new_node':len(self.nodes), 'node': n.id, 'probability': (n.fitness*n.links)/total, 'chosen':False})
        
    def get_probabilities_nodes(self):
        return pd.DataFrame(self.probabilities_nodes)
    
    #task to be executed while looping
    def loop_task(self):
        while self.running:
            self.update_graph_new_node(None)
            time.sleep(0.2)

    #starts the loop in a new thread and sets the flag to True
    def start_loop(self, event):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.loop_task)
            self.thread.start()
            print("Started loop.")

    #stops the loop by setting the flag to False
    def stop_loop(self, event):
        self.running = False
        print("Stopped loop.")

    #Adds a new node to the network and updates the plotted graph
    def update_graph_new_node(self, event):
        #used to discern wether this function is called the first time to plot all the nodes
        #or a second moment to add a new node
        adding = False
        
        if(self.G.number_of_nodes() == 0): #first time function is called, initialization
            #adds all existing nodes and edges
            for n in self.nodes:
                self.G.add_nodes_from([(n.id, {"info":"fitness "+str(n.fitness)})])
            self.G.add_edges_from(self.edges)
        else: #not first time function is called, adding of a new node
            new_node_id = self.nodes[-1].id
            new_node_fitness = self.nodes[-1].fitness
            self.add_node()
            self.G.add_nodes_from([(new_node_id, {"info":"fitness "+str(new_node_fitness)})])
            self.G.add_edges_from(self.edges[-self.connections_number:])
            adding = True

        # Calculates node sizes and colors based on number of links
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

        #layout used to arrange the nodes in the plot
        #Cirular is suggested, as it makes it easier to understand, works best with the automatic simulation and it's the fastest
        #spring and kawai provide alternative views, but are slower. I suggest using them for networks with less than 300 nodes
        #graphviz is the most computationally intensive, I suggest to use it only with relatively small networks (<100 nodes)

        pos = nx.circular_layout(self.G)
        # pos = nx.spring_layout(self.G, k=2)  # positions for all nodes
        # pos = nx.drawing.nx_pydot.graphviz_layout(self.G, prog='dot')
        # pos = nx.kamada_kawai_layout(self.G)  # positions for all node

        scatter = nx.draw_networkx_nodes(self.G, pos, node_size=node_sizes, node_color=node_colors, ax=self.ax)
        
        if(adding):
            #drawn every existing edge in gray and the edges added by the new node in red
            nx.draw_networkx_edges(self.G, pos, edgelist = self.edges[:-self.connections_number], width=0.3, alpha=0.5, edge_color="gray", ax=self.ax)
            nx.draw_networkx_edges(self.G, pos, edgelist = self.edges[-self.connections_number:], width=1.0, alpha=0.5, edge_color="red", ax=self.ax)
            #displays text information about the added node
            s = "Added node "+str(new_node_id)+" with fitness = "+str(new_node_fitness)
            self.ax.text(0.1, 0.0, s, fontsize=10, ha='left', va='bottom', transform=self.ax.transAxes)
            # Refresh the plot
            self.ax.axis("off")
        else:
            #draws all edges in gray
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
        
        #add cursors and connects it to the annotation showing function
        cursor = mplcursors.cursor(scatter, hover=True)
        cursor.connect("add", self.custom_annotation)    
        self.fig.canvas.mpl_connect("button_press_event", self.remove_annotation)  # Clear annotation when clicking background


        if(not adding):
            plt.show()  
        else:
            self.fig.canvas.draw()

    #modifies the annotation when hovering on different nodes
    def custom_annotation(self, sel):
        #removes annotation if already existent
        if (self.annotation):
            self.annotation.set_visible(False)
            self.fig.canvas.draw_idle()  
        sel.annotation.set_text("Node: "+str(self.nodes[sel.index].id)+"\nFitness: "+str(round(self.nodes[sel.index].fitness[0],3)))
        self.annotation = sel.annotation  # Save a reference to the annotation

    #removes annotation when clicking on background
    def remove_annotation(self, event):
        if self.annotation:  # If there's an annotation, clear it
            self.annotation.set_visible(False)
            self.fig.canvas.draw_idle()  

    #plot the probability of being chosen in time of node with the highest number of links
    def plot_probability_top_links(self):
        sorted_nodes = sorted(self.nodes, key=lambda node: node.links)
        self.plot_probability_in_time(sorted_nodes[-1].id)

    #Plot probability of being chosen of the top node
    def plot_probability_in_time(self, number_node):
        f, ax = plt.subplots(1, figsize=(5, 5))
        node_prob = [prob["probability"] for prob in self.probabilities_nodes if prob["node"]==number_node]
        ax.set_title("Probability in time, node id:"+str(number_node))
        ax.set_ylabel('Probability of chosen node')
        ax.set_xlabel("new links")
         
        plt.plot(node_prob)
        return node_prob


    #Shows the probability that each chosen node had when it was chosen
    def plot_probability_of_chosen_nodes(self):
        f, ax = plt.subplots(1, figsize=(5, 5))
        ax.set_title("Probability of nodes at the time of choice")
        ax.set_ylabel('Probability of Node')
        ax.set_xlabel("new nodes")
        x = np.linspace(0,len(self.chosen_nodes), len(self.chosen_nodes))
        plt.scatter(x,self.chosen_nodes, s = 20)

    #visualizes the network
    def plot_network(self):
        self.update_graph_new_node(None)

    #plots all the graphs
    def plot_graphs(self):
        self.plot_probability_top_links()
        self.plot_probability_of_chosen_nodes()
        self.plot_clust_coeff_on_fit()
        # plt.show()

    #generates a list of all couples of nodes from a list
    #ex. from [1,2,3] it generates [(1,2), (1,3), (2,3)]
    #Note that (i,j) and (j,i) are considered the same edge, and only one of them will be included in the list
    def node_couples(self,nodes):
        couples = []
        if(len(nodes) == 2):
            return([(nodes[0],nodes[1])])
        for i in range(0,len(nodes)-1,1):
            for j in range(i+1, len(nodes),1):
                couples.append((nodes[i],nodes[j]))
        return couples

    #calculate the slice of the self.edges array of the edges starting from node k
    #self.edges list is a list of couples ordered based n the first node of the couple
    #ex [(1,0), (2,0), (2,1), (3,0), (3,1), (3,2)]
    #Every node generates self.connections_number new edges, except for the fist self.connections_number nodes, as there are
    #not enough ndoes in the noetwork yet to generate enough edges. The boundaries of the slice must be calculated accordingly
    def boundaries(self,i):
        # print("boundaries for ",i)
        if i>= self.connections_number:
            offset = (self.connections_number*(self.connections_number+1))/2
            # print("offset:",offset)
            start = self.connections_number*i - offset
            finish = start + self.connections_number
            return (int(start),int(finish))
        else:
            start=0
            for j in range(0,i,1):
                start+=j
            finish = start+i
            return(start,finish)

    #calculates the local clustering coefficient of all nodes in the network
    def calculate_clustering_coefficient(self):
        local_clust_coeff=[]
        for node in self.nodes:#iterate over all nodes to calcualte local coefficient for each
            couples = self.node_couples(node.neighbours) #generates all the possible edges between its neighbours
            #if there are no couples it means it's connected to only 1 other node in the network
            if len(couples)==0: 
                local_clust_coeff.append(None)
                continue
            connected=0
            # print(self.edges)
            for (i,j) in couples:#iterates over all couples
                #retrieve boundaries of slices of self.edges containing the edges made by i and j
                #this is done to avoid iterating over all edges list for increased performance
                (start1, finish1) = self.boundaries(i)
                (start2, finish2) = self.boundaries(j)
                #checks wether edge exists
                if (i,j) in  self.edges[start1:finish1] or (j,i) in self.edges[start2:finish2]:
                    connected+=1
            local_clust_coeff.append(connected/len(couples))
        return local_clust_coeff           
    
    #plots local clustering coefficient of nodes with regard to their fitness
    def plot_clust_coeff_on_fit(self):
        coeffs = self.calculate_clustering_coefficient()
        f, ax = plt.subplots(1, figsize=(5, 5))
        fits = [n.fitness[0] for n in self.nodes]
        ax.set_title("Clustering coefficient of nodes with respect ot their fitnesses")
        ax.set_xlabel("node fitness")
        ax.set_ylabel("clustering coefficient")
        plt.scatter(fits,coeffs, s = 15)

        try:
            average = sum(coeffs)/len(coeffs)
            def func(x,a,b,c):
                return a*np.pow(np.e,b*x)+c
            popt, _ = curve_fit(func,fits, coeffs,[0.5,-1.0,0.5])

            xx = np.linspace(min(fits),max(fits),10000)
            plt.plot(xx, func(xx, *popt), "r", label="exponential fit of local cluster coeffs.")
            plt.axhline(y = average, color = 'g', linestyle = '-', linewidth=0.4, label="average of local cluster coeffs.") 
            plt.legend()
            # plt.show()
        except:
            print("Impossible to calulate average local cluster coefficients as some of them are None")