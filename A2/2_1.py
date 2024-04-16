import matplotlib.pyplot as plt
import random

def sample_vector():
    while True:
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        u = random.uniform(0, 10)
        if u <= abs((-1)*x1 +10 - x2):
            if u <= abs(x1-x2):
                break
    return (x1, x2)

num_samples = 5000
samples = [sample_vector() for n in range(num_samples)]

# Plot the samples
plt.scatter([x[0] for x in samples], [x[1] for x in samples], s=1)
plt.xlabel('X1 - x coordinate [m]')
plt.ylabel('X2 - y coordinate [m]')
plt.show()