import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrow

import sys

import dubins

# Define start and end configurations (x, y, heading)
q0 = (0.0, 0.0, math.radians(90))
q1 = (5.0, 3.0, math.radians(80))
rho = 1.0

# Compute the shortest path
path = dubins.shortest_path(q0, q1, rho)
configs, _ = path.sample_many(0.1)

# Create the plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1, 7)
ax.set_ylim(-2, 6)
ax.grid(True)

# Plot static start and end configurations
ax.plot(q0[0], q0[1], 'go', label="Start")
ax.plot(q1[0], q1[1], 'ro', label="End")

# Draw heading arrows at start and end
arrow_length = 0.5
ax.add_patch(FancyArrow(q0[0], q0[1], arrow_length*math.cos(q0[2]), arrow_length*math.sin(q0[2]), width=0.05, color='green'))
ax.add_patch(FancyArrow(q1[0], q1[1], arrow_length*math.cos(q1[2]), arrow_length*math.sin(q1[2]), width=0.05, color='red'))

(line,) = ax.plot([], [], 'b-', linewidth=2, label="Dubins path")

# Animation function
def animate(i):
    xs = [p[0] for p in configs[:i+1]]
    ys = [p[1] for p in configs[:i+1]]
    line.set_data(xs, ys)
    return (line,)

ani = animation.FuncAnimation(fig, animate, frames=len(configs), interval=50, blit=True, repeat=False)

plt.legend()
plt.title("Live Dubins Path Visualization")
plt.show()
