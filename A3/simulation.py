import logging
import os
import pandas as pd
import numpy as np
from queue import Queue
from utility.compute_theoretical_statistic import *
from utility.pareto import pareto_dist
from utility.event import *



def simulation(sim_time, l, mu, ser=None):
    try:
        os.remove("simulation.log")
    except OSError:
        pass
    # State Variables during simulation
    status_server = 0 # 0: idle, 1: busy

    # time variables
    system_time = 0  #time advance to current_event.time 
    arrival_time = 0  # variable for calculate arrival time of next event
    service_time = 0  # variable for calculate service time for current event -> plan the departure event    
     
    packets = [] #list of all packet
    queue_occupation = [] # list of all events in queue currently

    # queue for events
    my_queue = EventQueue(sim_time)
    current_Event: Event = my_queue.queue.get()[1] # first event that is a start event created by EventQueue when inizialized
    system_time = current_Event.time
    server_queue= Queue() # queue for server

    logging.basicConfig(
        level=logging.INFO,
        filename="simulation.log",
        filemode="a",
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    #
    logging.disable(logging.INFO)

    while True:
        match current_Event.type:
            case EventType.start:
                print("STARTING SIMULATION")
                logging.info(f"Simulation started")
                try:
                    os.remove("packets.csv")
                    os.remove("queue_occupation.csv")
                except OSError:
                    pass
                """
                Generate the first arrival event and put it in the queue.
                """
                # first arrival
                arrival_time = np.random.exponential(1 / l)
                my_queue.queue.put((system_time + arrival_time, (Event(EventType.arrival, system_time + arrival_time, 0))))
                queue_occupation.append({'time':system_time, 'packets_in_system': server_queue.qsize(), 'width':arrival_time})


            case EventType.arrival:
                #print(f"ARRIVAL: Packet {current_Event.id} arrived at time {current_Event.time}")
                logging.info(f"ARRIVAL: Packet {current_Event.id} arrived at time {current_Event.time}")

                # collect statistic
                packet = {'id': current_Event.id,
                          'arrival_time': current_Event.time, 
                          'server_time': None, 
                          'departure_time': None, 
                          'waiting_time': None, 
                          'total_time': None}
                packets.append(packet)

                #server free?
                if status_server == 0: # free
                    status_server = 1 
                    service_time = np.random.exponential(1 / mu)
                    my_queue.queue.put((current_Event.time + service_time, (Event(EventType.departure, current_Event.time + service_time, current_Event.id))))  

                    #packet_in_queue = server_queue.qsize()
                    event = {'time': system_time, 'packets_in_system':server_queue.qsize()+1, 'width':None}
                    queue_occupation.append(event)

                    packets[current_Event.id]['server_time'] = system_time
                    packets[current_Event.id]['departure_time'] = system_time + service_time
                    logging.info(f"Serving packet {current_Event.id}")

                else: # busy
                    #packet_in_queue += 1
                    server_queue.put((current_Event))
                    event = {'time': system_time, 'packets_in_system':server_queue.qsize()+1, 'width' : None}
                    queue_occupation.append(event)
                    
                    logging.info(
                            f"Server busy, packet {current_Event.id} added to queue at time {current_Event.time}"
                    )
                
                #schedule next arrival
                arrival_time = np.random.exponential(1 / l)
                my_queue.queue.put((current_Event.time + arrival_time, (Event(EventType.arrival, current_Event.time + arrival_time, current_Event.id + 1))))

            case EventType.departure:
                #print(f"DEPARTURE : Event {current_Event.id} departed at time {current_Event.time}")
                logging.info(f"DEPARTURE : Event {current_Event.id} departed at time {current_Event.time}")
                
                #collect statistic

                packets[current_Event.id]['departure_time']= system_time
                #packets[current_Event.id]['waiting_time'] = waiting_time
                #is queue empty?
                if server_queue.empty():
                    status_server = 0
                    event = {'time': system_time, 'packets_in_system':server_queue.qsize(), 'width' : None}
                    queue_occupation.append(event)                  
                else:
                    status_server = 1
                    #packet_in_queue -= 1
                    pending_packet = server_queue.get()
                    event = {'time': system_time, 'packets_in_system':server_queue.qsize()+1, 'width': None}
                    queue_occupation.append(event)

                    if ser == "pareto":
                        service_time = (np.random.pareto(1.5, 1) +1)*0.5
                        #service_time = pareto_dist(1.5, 0.5)
                    else :
                        service_time = np.random.exponential(1 / mu)

                    packets[pending_packet.id]['server_time'] = system_time
                    packets[pending_packet.id]['departure_time'] = system_time + service_time

                    my_queue.queue.put((system_time + service_time, (Event(EventType.departure, current_Event.time + service_time, pending_packet.id))))

                    logging.info(
                        f"Picking package {pending_packet.id} from queue and serving it at time {system_time}"
                    )

                
            case EventType.stop:
                print("COMPLETE SIMULATION")
                logging.info(f"Simulation completed")
                break

            case _:
                logging.info(
                    f"Unknown event type {current_Event.type} {current_Event.time}"
                )
                break 

        current_Event = my_queue.queue.get()[1]
        system_time = float(current_Event.time )

    packets_save = pd.DataFrame(packets)
    queue_occupation_save = pd.DataFrame(queue_occupation)

    packets_save["waiting_time"] = packets_save["server_time"] - packets_save["arrival_time"]
    packets_save["total_time"] = packets_save["departure_time"] - packets_save["arrival_time"]

    # Compute width of intervals in queue occupation
    queue_occupation_save["width"] = (
        queue_occupation_save["time"].shift(-1) - queue_occupation_save["time"]
    )



    #Drop NaN values, packet not served
    packets_save.dropna(inplace=True)
    queue_occupation_save.dropna(inplace=True)

    queue_occupation_save['packets_in_system'] = queue_occupation_save['packets_in_system'].tolist()
    
    # packets_save.to_csv("packets.csv", index=False)
    # queue_occupation_save.to_csv("queue_occupation.csv", index=False)

    return packets_save, queue_occupation_save

