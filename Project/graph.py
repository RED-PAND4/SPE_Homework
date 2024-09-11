from node import Node
from scipy.stats import uniform

class Graph:
    def __init__(self,m, distribution):
        self.nodes=set()
        self.edges=set()
        self.distribution=distribution

        for i in range(0,m,1):
            self.nodes.add(Node(i,self.distribution.rvs(size=1)))
            if( ((i+1)%m, i) not in self.edges):
                self.edges.add((i,(i+1)%m))
                
    def print_all(self):
        for n in self.nodes:
            n.print_node()
        print(self.edges)
    

g = Graph(7,uniform())
g.print_all()