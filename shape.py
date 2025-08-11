from shapely.geometry import LineString
import matplotlib.pyplot as plt
from shapely.geometry import CAP_STYLE, JOIN_STYLE

# Define the line
line = LineString([(0, 0), (10, 2)])

# Create a rectangular buffer (no rounded caps or joins)
buffer = line.buffer(
    distance=0.1,  # half the thickness
    cap_style=CAP_STYLE.flat,
    join_style=JOIN_STYLE.mitre
)

# Plotting
x, y = line.xy
plt.plot(x, y, 'k--', label='Line')

# Plot buffer as polygon
xb, yb = buffer.exterior.xy
plt.fill(xb, yb, color='lightgreen', alpha=0.5, label='Rectangular Buffer')

plt.axis('equal')
plt.legend()
plt.title("Line with Rectangular Buffer")
plt.show()

print(buffer)