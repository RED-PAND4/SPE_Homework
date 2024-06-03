from enum import Enum
from queue import PriorityQueue
import numpy as np



EventType = Enum("Type", "start reached speed trasmission stop")

class Event:
    def __init__(self, type, time, id):
        self.type = type
        self.time = time
        self.id = id

    def __str__(self) -> str:
        if self.type == EventType.arrival or self.type == EventType.departure:
            return f"Event: {self.type} of node {self.id} at {self.time}"
        return f"Event: {self.type} at {self.time}"



class EventQueue:
    def __init__(self, simulation_time):
        self.queue = PriorityQueue()
        self.queue.put((0.0, Event(EventType.start, 0, 0)))
        self.queue.put((simulation_time, Event(EventType.stop, simulation_time, 99999)))

    def __str__(self) -> str:
        string = ""
        for event in self.queue.queue:
            string += " " + event.__str__()
        return string
    
