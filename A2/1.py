import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import rayleigh
from scipy.stats import expon
from scipy.special import i0
import random
from random import uniform
import matplotlib.pyplot as plt
import math
import numpy as np

sz = 20000
bin_default = 80

#-------------------- COMMON FUNCTIONS --------------------

#calculated the value of the exponent function
def calculate_expon(x, sc, lambd):
    return sc*lambd*math.e**(-lambd*x)


#rejection sampling, samples = x values, y1 = bounding funtion, y2 = function to sample
def accept(samples,y1,y2):
    accepted = []
    for a,b,c in zip(samples, y1, y2):
        ran = random.uniform(0,b)
        if ran <=c:
            accepted.append(a)
    return accepted

#plots the function and the histogram
def plot_comparison(accepted, x, y, name, ylim1, ylim2, bin_num=bin_default):
    fig, ax1 = plt.subplots()
    ax1.hist(accepted, bins=bin_num, alpha=0.8)
    plt.ylim(0,ylim1)
    ax2 = ax1.twinx()
    ax2.plot(x, y,'r-', lw=2, alpha=1, label='rice pdf')
    plt.ylim([0,ylim2])
    plt.title(name+" pdf compared to sampling histogram")

# function to calculate Rayleigh pdf at point x
def calculate_ray(x,sigma):
    return (x/(sigma**2))*math.e**(-x**2/(2*sigma**2))

# function to calculate Rice pdf at point x
def calculate_rice(x):
    sigma=2
    v=2
    return (x/sigma**2)*math.e**(-(x**2 + v**2)/(2*sigma**2))*i0(x*v/sigma**2)

# ------------------------- RAYLEIGH -------------------------
# generating rayleigh and exponential pdf
sc=2.05
scale_factor=1.45
lambd=1/scale_factor

fig, ax = plt.subplots(1, 1)
x = np.linspace(0,7, 100)
ray = rayleigh()
y_ray = [calculate_ray(i,1) for i in x]
y_expon = [calculate_expon(i, sc, lambd) for i in x]




# plotting ray and exponential
exp = expon(scale=scale_factor)
ax.plot(x, y_expon,'b-', lw=1, alpha=0.6, label='expon pdf')
ax.plot(x, y_ray, 'k-', lw=1, alpha=0.6, label='rayleigh pdf')
plt.title("Rayleigh pdf and bounding exponential pdf")


# generating samples according to exponential pdf
samples=exp.rvs(size=sz)
#calculating exp and ray values for the samples
exp_y_samples = [calculate_expon(i, sc, lambd) for i in samples]
ray_y_samples = [calculate_ray(i,1) for i in samples]

#accepting values
accepted = accept(samples, exp_y_samples, ray_y_samples)
#plotting comparison between rayleigh pdf and accepted values histogram
plot_comparison(accepted, x, y_ray, "Rayleigh", 1000, 1.45, bin_num = 60)
print("Samples per accept Ray:",sz/len(accepted))

# ------------------------- RICE -------------------------

# generating and plotting exponential bounding pdf and rice pdf
sc=2.15
scale_factor=4
lambd=1/scale_factor

fig, ax = plt.subplots(1, 1)
x = np.linspace(0,10,200)
exp_2 = expon(scale = scale_factor)

y_rice = [calculate_rice(i) for i in x]
y_expon = [calculate_expon(i,sc,lambd) for i in x]
ax.plot(x, y_rice,'k-', lw=1, alpha=0.6, label='rice pdf')
ax.plot(x, y_expon,'b-', lw=1, alpha=0.6, label='exponential pdf' )
plt.title("Rice pdf and bounding exponential pdf")

# sampling exponential bounding pdf
samples=exp_2.rvs(size=sz)

# caculating rice and exp. values for the samples
y_expon_samples = [calculate_expon(i,sc,lambd) for i in samples]
y_rice_samples = [calculate_rice(i) for i in samples]

#accepting values
accepted=[]
accepted = accept(samples, y_expon_samples, y_rice_samples)

#plotting comparison between rice pdf and accepted value histogram
plot_comparison(accepted, x, y_rice, "Rice", 510, 0.57, bin_num=100)
print("Samples per accept Rice:",sz/len(accepted))
