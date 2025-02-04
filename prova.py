import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# Example data (x, y)
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.3, 100)  # Sine wave with noise

# Perform LOWESS smoothing
# frac: Proportion of data used for each local regression (smoothing factor)
lowess_smoothed = lowess(y, x, frac=0.3)

# Extract smoothed x and y
x_smoothed = lowess_smoothed[:, 0]
y_smoothed = lowess_smoothed[:, 1]

# Plot original scatter data
plt.scatter(x, y, color="lightblue", label="Original Data", alpha=0.6)

# Plot smoothed curve
plt.plot(x_smoothed, y_smoothed, color="red", label="LOWESS Smoothed Curve", linewidth=2)

# Add labels and legend
plt.title("Scatterplot Smoothing with LOWESS")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
