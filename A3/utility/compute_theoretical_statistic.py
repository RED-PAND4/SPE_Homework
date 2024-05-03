import numpy as np
import math

# Definition
# λ = Arrival Rate ( lamb)
# μ = Service Rate (mu)
# ρ = λ / μ (rho)
# C = Number of Service Channels = 1
# M = Random Arrival/Service rate (Poisson)

# M/M/C in which C =1 so it is M/M/1
# l < mu

def comp_pi_zero(rho): #The probability of having zero vehicles in the systems
    return 1 - rho

def comp_pi_n(rho, n): #The probability of having n vehicles in the systems
    return (1 - rho) * (rho ** n)

def avg_queue_length(rho):
    return rho / (1 - rho)

def avg_total_time(rho, l):
    return 1 / (l * (1 - rho))

def avg_waiting_time(rho, l, mu):
    return avg_total_time(rho, l) - (1 / mu)

def avg_packet_in_sys(rho):
    return rho/(1 - rho)

def avg_packet_in_queue(rho):
    return rho**2/(1 - rho)