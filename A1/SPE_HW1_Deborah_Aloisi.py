import numpy as np
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt

## EXERCISE 1 ----------------------------------------------------------------------

l = 15 #1 
sample_size = 100000
mu = l
sigma = np.sqrt(l)

#create a poisson distribution
x = st.poisson.rvs(mu=l, size=sample_size)

#calculate P(|X-μ| >= kσ) with different k
ks =  np.arange(1, 20, 0.5).tolist() #or ks = [1,2,3,4,5,6,7,8,9,10]
probs = []# for each k
for k in ks: 
    # start count
    c = 0
    # for each data sample
    for i in x:
        # count if far from mean in k standard deviation
        if abs(i - mu) >= k * sigma :
            c += 1
    # count divided by number of sample
    probs.append(c/sample_size)

#graph1
fig, ax = plt.subplots()
ax.plot(ks, probs, 'tab:green', marker='o', markerfacecolor='orange', linewidth= 2, linestyle='-')
ax.set(xlabel='K', ylabel='P(|X-μ| >= kσ)',
       title='Graph to represent the variation of P based on K', facecolor="white")
ax.grid(color="grey", alpha=0.3)
#fig.savefig("P.png")
plt.show()

#Considerations1
print("Probability of a sample far from mean more than k standard deviation:")
for i, res in enumerate(probs):
    if not i>15:
        print("k:" + str(ks[i]) + ", probability: " \
              + str(probs[i]) + \
              " | in theory, probability should less than: " \
              + str(1/ks[i]**2)[0:5] + '--> it is :' + str(bool(probs[i]<=(1/ks[i]**2))))


#graph2
kp = []
m=0
for k in ks:
    kp.append((1/np.power(k, 2))-probs[m])
    m +=1

fig, ax = plt.subplots()
ax.plot(kp, 'tab:green', marker='o', markerfacecolor='orange')
ax.set(xlabel='(1/(k^2))-P',
       title='Graph to represent (1/(k^2))-p is never below zero', facecolor="white")
ax.grid(color="grey", alpha=0.3)
x = np.linspace(0, 40, sample_size) 
sns.lineplot(x=x, y=0, ax=ax, color="black")
#fig.savefig("not_0.png")
plt.show()

#Consideration2
print("(1/k**2)-p:")
for i, res in enumerate(probs):
    if not i>15:
        print("k:" + str(ks[i]) + ", probability: " \
              + str(probs[i]) + \
              " | in theory, (1/k**2)-p should be >=0 and it is: " \
              +str((1/ks[i]**2)-probs[i])+ '--> it is :' + str(bool((1/ks[i]**2)-probs[i]>=0)))


#graph3
k2 = []
for k in ks:
    k2.append((1/np.power(k, 2)))

fig, ax = plt.subplots()
ax.plot(k2, probs, 'tab:green', marker='o', markerfacecolor='orange')
ax.set(xlabel='1/(k^2)', ylabel='P(|X-μ| >= kσ)',
       title='Graph to represent that the line is always under the bisector', facecolor="white")
ax.grid(color="grey", alpha=0.3)
x = np.linspace(0, 1, sample_size) 
sns.lineplot(x=x, y=x, ax=ax, color="black")
#fig.savefig("not_below.png")
plt.show()



## EXERCISE 2 ----------------------------------------------------------------------

gen = np.random.default_rng(seed=42)
sample_size=100000
samples35 = gen.uniform(low=3, high=5, size=sample_size)
samples24 = gen.uniform(low=2, high=4, size=sample_size)
samples24.sort()
samples35.sort()


#Variables being in interval [3,4] from samples [3,5]
var_in_interval35 = []
for i in samples35:
    if i>=3 and i<=4:
        var_in_interval35.append(i)

#Variables being in interval [3,4] from samples [2,4]
var_in_interval24 = []
for i in samples24:
    if i>=3 and i<=4:
        var_in_interval24.append(i)


#probability in interval [3,4]
count = 0
all = np.power(sample_size, 2)
for k in var_in_interval35:
    for j in var_in_interval24:
        if j>k:
            count += 1

prob3524 = (all-count)/all
prob2435 = (count)/all

print("Probability of P{[3,5]>[2,4]} is :  "+str(prob3524))
print("Probability of P{[3,5]<[2,4]} is :  "+str(prob2435))