import scipy.stats as stats

class Node:
    def __init__(self,id,fitness):
        self.id=id
        self.fitness = fitness
        self.connections=0

    def add_connection(self):
          self.connection+=1

    def print_node(self):
         print("Node ",self.id,", fitness:",self.fitness)

