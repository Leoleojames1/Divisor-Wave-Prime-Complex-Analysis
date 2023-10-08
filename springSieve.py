import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Function to check if a number is prime
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


# Generate values of x from 0 to 14
x = np.linspace(0, 14, 300)

# Different scaling factors (k values)
scaling_factors = np.arange(2, 14)  # From 2 to 13

# Create a 3D plot with a black background
fig = plt.figure(figsize=(10, 6))  # Adjust the figsize parameter
ax = fig.add_subplot(111, projection='3d')

# Set background color to black
ax.set_facecolor('black')

# Plot each curve in 3D with different scaling factors and colors
for i, k in enumerate(scaling_factors):
    y = np.sin(x * np.pi / k)
    z = np.cos(x * np.pi / k)

    # Determine color based on whether k is prime or composite
    if is_prime(k):
        color = (1, 0.5 - 0.1 * i, 0.5 - 0.1 * i)  # Shades of red for prime k
    else:
        color = (0.5 - 0.05 * i, 1, 0.5 - 0.05 * i)  # Shades of green for composite k

    ax.plot(x, y, z, color=color)

ax.set_xlabel('X')
ax.set_ylabel('sin(x)')
ax.set_zlabel('cos(x)')
ax.set_title('3D Plot of sin(x*k) and cos(x*k)')

# Manually set the x-axis limits for a wider plot appearance
ax.set_xlim(0, 14)  # Adjust the limits as needed

plt.show()