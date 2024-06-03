import matplotlib.pyplot as plt
import random
import logging
import math
import os
import pandas as pd
import numpy as np
from queue import Queue
from event import *
from batch_means import compute_batch_means_statistics
from plotting import *


#random.seed(42)  # for reproducibility

def trasmitted_byte():
    mu = 1150
    sigma = 400
    byte = random.uniform(1,9)
    return byte

def calculate_dist(point1, point2):
    return math.dist(point1, point2)

def get_bearing(lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)
    #print(brng)
    return brng

def random_position():
    return (random.uniform(0, 1000), random.uniform(0, 1000))

def time_calculation(point1, point2, speed):
    d = math.dist(point1, point2)
    t = d/speed
    return t

def simulation(sim_time, num_nodes, v_min, v_max):
    try:
        os.remove("simulation.log")
    except OSError:
        pass
    # State Variables during simulation
    # location = (0,0)
    # time_last_reached = 0
    # next_waypoint = (0,0)
    # current_speed = 0

    nodes_pos = []
    nodes_next_pos = []
    nodes_speed = []
    nodes_current_pos = []
    nodes_bearing = []
    nodes_time_last_reached = []

    # time variables
    system_time = 0  #time advance to current_event.time 
    arrival_time = 0  # variable for calculate arrival time of next event
     
    speeds = [] #list of all speeds
    log_events = [] # list of all events in queue currently
    trasmission = []

    # queue for events
    my_queue = EventQueue(sim_time)
    current_Event: Event = my_queue.queue.get()[1] # first event that is a start event created by EventQueue when inizialized
    system_time = current_Event.time

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
                    os.remove("speeds.csv")
                    os.remove("log_events.csv")
                    os.remove("trasmission.csv")
                except OSError:
                    pass
                """
                Generate the first arrival event and put it in the queue.
                """
                # first arrival
                nodes_pos = [random_position() for n in range(num_nodes)]
                nodes_next_pos = [random_position() for n in range(num_nodes)]
                nodes_speed = [random.uniform(v_min,v_max) for n in range(num_nodes)]
                nodes_current_pos = [0 for n in range(num_nodes)]
                nodes_bearing = [0 for n in range(num_nodes)]
                nodes_time_last_reached = [0 for n in range(num_nodes)]

                for n in range(num_nodes):
                    arrival_time = np.linalg.norm(np.array(nodes_next_pos[n]) - np.array(nodes_pos[n])) / nodes_speed[n]
                    #arrival_time = time_calculation(nodes_next_pos[n], nodes_pos[n], nodes_speed[n])
                    my_queue.queue.put((system_time + arrival_time, (Event(EventType.reached, system_time + arrival_time, n))))
                    log_events.append({'id': n, 'time_last_way': system_time, 'current point': nodes_pos[n], 'next point': nodes_next_pos[n], 'speed': nodes_speed[n]})

                #events for speed
                for i in range(5,sim_time, 10):
                    my_queue.queue.put((i, Event(EventType.speed, i, -99)))
                
                #events for distance
                for i in range(10,sim_time,20):
                    my_queue.queue.put((i, Event(EventType.distance, i, -88)))

            case EventType.reached:
                #print(f"ARRIVAL: Packet {current_Event.id} arrived at time {current_Event.time}")
                logging.info(f"ARRIVAL: Node {current_Event.id} reached at time {current_Event.time}")
                nodes_pos[current_Event.id] = nodes_next_pos[current_Event.id]
                nodes_next_pos[current_Event.id] = random_position()
                nodes_speed[current_Event.id] = random.uniform(v_min,v_max)
                arrival_time = np.linalg.norm(np.array(nodes_next_pos[current_Event.id]) - np.array(nodes_pos[current_Event.id])) / nodes_speed[current_Event.id]
                my_queue.queue.put((system_time + arrival_time, (Event(EventType.reached, system_time + arrival_time, current_Event.id))))
                log_events.append({'id': current_Event.id, 'time_last_way': system_time, 'current point': nodes_pos[current_Event.id],'next point': nodes_next_pos[current_Event.id], 'speed': nodes_speed[current_Event.id]})
                
            
            case EventType.speed:
                #print(f"DEPARTURE : Event {current_Event.id} departed at time {current_Event.time}")
                logging.info(f"SPEED")
                for n in range(num_nodes):
                    speeds.append({'time':system_time,'id': n, 'speed': nodes_speed[n] })
            
            case EventType.distance:
                logging.info(f"Distance")
                for n in range(num_nodes):
                    nodes_bearing[n] = get_bearing(nodes_pos[n][0], nodes_pos[n][1], nodes_next_pos[n][0], nodes_next_pos[n][1])
                    nodes_current_pos[n] = (nodes_pos[n][0] + nodes_speed[n]*math.cos(nodes_bearing[n]), nodes_pos[n][1] + nodes_speed[n]*math.sin(nodes_bearing[n]))
                    nodes_time_last_reached[n] -= 1
                for n in range(num_nodes):
                    for m in range(num_nodes):
                        if n != m:
                            dx = calculate_dist(nodes_current_pos[n], nodes_current_pos[m])
                            #print("DISTANCE",dx)
                            if dx<=50 and nodes_time_last_reached[n]<0:
                                print(f"communaction {n} to {m}, distance {dx}")
                                logging.info(f"COMMUNICATION: Node {n} to Node {m}, distance {dx}")
                                byte = trasmitted_byte()
                                trasmission.append({'id': n, 'to': m, 'Byte': byte})
                                nodes_time_last_reached[n] = 10
                

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

    speeds_save = pd.DataFrame(speeds)
    log_events_save = pd.DataFrame(log_events)
    trasmission_save = pd.DataFrame(trasmission)

    #Drop NaN values, packet not served
    # speeds_save.dropna(inplace=True)
    # log_events_save.dropna(inplace=True)


    speeds_save.to_csv("speeds.csv", index=False)
    log_events_save.to_csv("log_events.csv", index=False)
    trasmission_save.to_csv("trasmission.csv", index=False)
    print("returned")
    return speeds_save, log_events_save, trasmission_save



num_nodes = 20
sim_time = 10000

# VELOCITY 0-10
speeds, log_events, trasmission = simulation(sim_time, num_nodes, 0, 10)

plots = Plotting(sim_time, speeds)
# Plot autocorrelation to decide batch size for batch means
#plots.plot_auto_correlation(50)
# batch variables
# (
#     grand_mean,
#     ci_amplitude,
#     batch_means,
#     intervals,
# ) = compute_batch_means_statistics(Statistics.PACKET_IN_SYSTEM, speeds, 50, 20, 0.95)
#---
#plots.plot_batch_means(batch_means, intervals )


speeds_mean = speeds.groupby(['time']).mean()
node_speed_mean = speeds.groupby(['id']).mean()
total_mean = speeds.mean()
print(node_speed_mean)

print("Total Mean:", total_mean['speed'])
# print(f"""Mean of the speed with batch means(simulation): {grand_mean} +- {ci_amplitude}"""
#     ) 

fig, ax = plt.subplots()
plt.plot(speeds_mean['speed'])
ax.axhline(total_mean['speed'], label="speed mean", color="b")


trasmission_rate = trasmission.groupby(['id']).mean()
print(trasmission_rate)

# VELOCITY 1-10
# speeds, log_events = simulation(sim_time, num_nodes, 1, 10)


# plots = Plotting(sim_time, speeds)
# # Plot autocorrelation to decide batch size for batch means
# #plots.plot_auto_correlation(50)
# # batch variables
# # (
# #     grand_mean,
# #     ci_amplitude,
# #     batch_means,
# #     intervals,
# # ) = compute_batch_means_statistics(Statistics.PACKET_IN_SYSTEM, speeds, 50, 20, 0.95)
# # #---
# # #plots.plot_batch_means(batch_means, intervals )

# speeds_mean = speeds.groupby(['time']).mean()
# speed_mean = speeds.mean()
# #print(speeds_mean)
# print("Total Mean:", speed_mean['speed'])
# # print(f"""Mean of the speed with batch means(simulation): {grand_mean} +- {ci_amplitude}"""
# #     ) 
# fig, ax = plt.subplots()
# plt.plot(speeds_mean['speed'])
# ax.axhline(speed_mean['speed'], label="speed mean", color="b")


#plt.show()