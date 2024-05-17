import matplotlib.pyplot as plt
import matplotlib.animation as animation
from celluloid import Camera
from matplotlib import cm
import random
import logging
import math
import os
import pandas as pd
import numpy as np
from queue import Queue
from event import *

random.seed(42)  # for reproducibility

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

    # time variables
    system_time = 0  #time advance to current_event.time 
    arrival_time = 0  # variable for calculate arrival time of next event
     
    speeds = [] #list of all speeds
    log_events = [] # list of all events in queue currently

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
                    os.remove("packets.csv")
                    os.remove("log_events.csv")
                except OSError:
                    pass
                """
                Generate the first arrival event and put it in the queue.
                """
                # first arrival
                nodes_pos = [random_position() for n in range(num_nodes)]
                nodes_next_pos = [random_position() for n in range(num_nodes)]
                nodes_speed = [random.uniform(v_min,v_max) for n in range(num_nodes)]

                for n in range(num_nodes):
                    arrival_time = np.linalg.norm(np.array(nodes_next_pos[n]) - np.array(nodes_pos[n])) / nodes_speed[n]
                    #arrival_time = time_calculation(nodes_next_pos[n], nodes_pos[n], nodes_speed[n])
                    my_queue.queue.put((system_time + arrival_time, (Event(EventType.reached, system_time + arrival_time, n))))
                    log_events.append({'id': n, 'time_last_way': system_time, 'current point': nodes_pos[n], 'next point': nodes_next_pos[n], 'speed': nodes_speed[n]})

                #events for speed
                for i in range(5,sim_time, int(sim_time/100)):
                    my_queue.queue.put((i, Event(EventType.speed, i, -99)))

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

    #Drop NaN values, packet not served
    # speeds_save.dropna(inplace=True)
    # log_events_save.dropna(inplace=True)


    speeds_save.to_csv("speeds.csv", index=False)
    log_events_save.to_csv("log_events.csv", index=False)
    print("returned")
    return speeds_save, log_events_save



num_nodes = 200
sim_time = 10000
v_min = 0
v_max = 10

speeds, log_events = simulation(sim_time, num_nodes, v_min, v_max)


speeds_mean = speeds.groupby(['time']).mean()
speed_mean = speeds.mean()
print(speeds_mean)
print("Grand Mean:", speed_mean['speed'])
fig, ax = plt.subplots()
plt.plot(speeds_mean['speed'])
ax.axhline(speed_mean['speed'], label="speed mean", color="b")



# frames =100
# fig, ax = plt.subplots()

# def update(i):
#     ax.clear()
#     ax.scatter(log_events['current point'][i][0], log_events['current point'][i][1], c='b')
#     ax.set_xlim(-2, 1000)
#     ax.set_ylim(0, 1000)

# ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)
# ani.save('clear.gif', writer='pillow')


# numpoints = 200
# ti = 0
# points = log_events['current point']#np.random.random((2, numpoints))
# colors = cm.rainbow(np.linspace(0, 1000, numpoints))
# camera = Camera(plt.figure())
# for _ in range(100):
#     log_events = log_events.loc[log_events['time_last_way'] > ti]
#     points = log_events['current point']
#     plt.scatter(*points, c=colors, s=100)
#     camera.snap()
#     ti +=100
# anim = camera.animate(blit=True)
# anim.save('scatter.mp4')


#Initialize the figure and the scatter plot
# fig = plt.figure()
# scat = plt.scatter([], [], c='r')

# # Initialize the time array
# t = np.arange(0, 10, 0.1)

# # Initialize the x and y arrays
# x = np.zeros_like(t)
# y = np.zeros_like(t)

# # Define the update function
# def update(data):
#     x[:] = data[0]
#     y[:] = data[1]
#     scat.set_offsets(np.c_[x, y])
#     return scat,

# # Initialize the animation
# ani = animation.FuncAnimation(fig, update(), frames=np.c_[x, y], interval=100, blit=True)


# fig, ax = plt.subplots(figsize=(16,8))
# ax.set(xlim=(0,1000), ylim=(0,1000))

# def animate(i):
#     if i == 1:

#     if i == 2:




#print(queue)
# Plot the samples
# plt.scatter([x[0] for x in nodes_pos], [x[1] for x in nodes_pos], s=1)
# plt.xlabel('X1 - x coordinate [m]')
# plt.ylabel('X2 - y coordinate [m]')
# ax =  plt.gca()
# ax.set_aspect('equal')
plt.show()

# fig, ax = plt.subplots()
# t = np.linspace(0, 3, 40)
# g = -9.81
# v0 = 12
# z = g * t**2 / 2 + v0 * t

# v02 = 5
# z2 = g * t**2 / 2 + v02 * t

# scat = ax.scatter(log_events['current point'][0], log_events['current point'][1], c="b", s=5, label=f'v0 = {v0} m/s')
# #line2 = ax.plot(t[0], z2[0], label=f'v0 = {v02} m/s')[0]
# ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
# ax.legend()


# def update(frame):
#     # for each frame, update the data stored on each artist.
#     x = log_events['current point'][:frame]
#     y = log_events['current point'][:frame]
#     # update the scatter plot:
#     data = np.stack([x, y]).T
#     scat.set_offsets(data)
#     # update the line plot:
#     # line2.set_xdata(t[:frame])
#     # line2.set_ydata(z2[:frame])
#     return (scat)


# ani = animation.FuncAnimation(fig=fig, func=update, interval=30)
# plt.show()