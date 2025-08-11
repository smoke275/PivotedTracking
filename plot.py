import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_environment():
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # VISIBILITY_CIRCLE
    circle = patches.Circle((752.59, 435.19), 200.0, fill=False, 
                           edgecolor='blue', linewidth=2, linestyle='--', 
                           alpha=0.7, label='Visibility Circle')
    ax.add_patch(circle)
    
    # Mark the center point
    ax.plot(752.59, 435.19, 'bo', markersize=8, label='Analysis Position')
    
    # ORIGINAL_WALLS
    original_walls = [
        [(274.00, 360.00), (731.00, 360.00)],
        [(731.00, 360.00), (731.00, 367.00)],
        [(731.00, 367.00), (274.00, 367.00)],
        [(274.00, 367.00), (274.00, 360.00)],
        [(868.00, 36.00), (877.00, 36.00)],
        [(877.00, 36.00), (877.00, 360.00)],
        [(877.00, 360.00), (868.00, 360.00)],
        [(868.00, 360.00), (868.00, 36.00)],
        [(868.00, 432.00), (877.00, 432.00)],
        [(877.00, 432.00), (877.00, 676.00)],
        [(877.00, 676.00), (868.00, 676.00)],
        [(868.00, 676.00), (868.00, 432.00)],
        [(868.00, 360.00), (1233.00, 360.00)],
        [(1233.00, 360.00), (1233.00, 367.00)],
        [(1233.00, 367.00), (868.00, 367.00)],
        [(868.00, 367.00), (868.00, 360.00)],
        [(868.00, 504.00), (1233.00, 504.00)],
        [(1233.00, 504.00), (1233.00, 511.00)],
        [(1233.00, 511.00), (868.00, 511.00)],
        [(868.00, 511.00), (868.00, 504.00)]
    ]
    
    for wall in original_walls:
        x_coords = [wall[0][0], wall[1][0]]
        y_coords = [wall[0][1], wall[1][1]]
        ax.plot(x_coords, y_coords, 'k-', linewidth=3, alpha=0.8)
    
    # Add label for original walls (just once)
    ax.plot([], [], 'k-', linewidth=3, alpha=0.8, label='Original Walls')
    
    # ORIGINAL_DOORS
    original_doors = [
        [(594.00, 360.00), (658.00, 360.00)],
        [(658.00, 360.00), (658.00, 367.00)],
        [(658.00, 367.00), (594.00, 367.00)],
        [(594.00, 367.00), (594.00, 360.00)],
        [(868.00, 288.00), (877.00, 288.00)],
        [(877.00, 288.00), (877.00, 338.00)],
        [(877.00, 338.00), (868.00, 338.00)],
        [(868.00, 338.00), (868.00, 288.00)],
        [(868.00, 576.00), (877.00, 576.00)],
        [(877.00, 576.00), (877.00, 626.00)],
        [(877.00, 626.00), (868.00, 626.00)],
        [(868.00, 626.00), (868.00, 576.00)]
    ]
    
    for door in original_doors:
        x_coords = [door[0][0], door[1][0]]
        y_coords = [door[0][1], door[1][1]]
        ax.plot(x_coords, y_coords, 'g-', linewidth=2, alpha=0.7)
    
    # Add label for doors (just once)
    ax.plot([], [], 'g-', linewidth=2, alpha=0.7, label='Original Doors')
    
    # RAY_WALLS (from visibility rays)
    ray_walls = [
        [(752.59, 435.19), (868.00, 435.19)],
        [(752.59, 435.19), (868.00, 442.45)],
        [(752.59, 435.19), (868.00, 449.77)],
        [(752.59, 435.19), (868.00, 457.21)],
        [(752.59, 435.19), (868.00, 464.83)],
        [(752.59, 435.19), (868.00, 472.69)],
        [(752.59, 435.19), (868.00, 480.89)],
        [(752.59, 435.19), (868.00, 489.50)],
        [(752.59, 435.19), (868.00, 498.64)],
        [(752.59, 435.19), (868.00, 508.43)],
        [(752.59, 435.19), (868.00, 519.04)],
        [(752.59, 435.19), (868.00, 530.67)],
        [(752.59, 435.19), (868.00, 543.57)],
        [(752.59, 435.19), (868.00, 558.09)],
        [(752.59, 435.19), (880.08, 589.30)],
        [(752.59, 435.19), (562.38, 373.39)],
        [(752.59, 435.19), (580.36, 367.00)],
        [(752.59, 435.19), (571.63, 350.04)],
        [(752.59, 435.19), (590.79, 317.64)],
        [(752.59, 435.19), (670.16, 367.00)],
        [(752.59, 435.19), (679.97, 367.00)],
        [(752.59, 435.19), (688.55, 367.00)],
        [(752.59, 435.19), (696.18, 367.00)],
        [(752.59, 435.19), (703.05, 367.00)],
        [(752.59, 435.19), (709.32, 367.00)],
        [(752.59, 435.19), (715.10, 367.00)],
        [(752.59, 435.19), (720.50, 367.00)],
        [(752.59, 435.19), (725.59, 367.00)],
        [(752.59, 435.19), (730.43, 367.00)],
        [(752.59, 435.19), (702.85, 241.48)],
        [(752.59, 435.19), (859.76, 266.33)],
        [(752.59, 435.19), (868.00, 276.35)],
        [(752.59, 435.19), (877.00, 284.81)],
        [(752.59, 435.19), (889.50, 289.40)],
        [(752.59, 435.19), (906.69, 307.71)],
        [(752.59, 435.19), (868.00, 351.34)],
        [(752.59, 435.19), (868.00, 361.95)],
        [(752.59, 435.19), (876.63, 367.00)],
        [(752.59, 435.19), (897.51, 367.00)],
        [(752.59, 435.19), (924.83, 367.00)],
        [(752.59, 435.19), (942.80, 373.39)],
        [(752.59, 435.19), (952.20, 422.64)]
    ]
    
    for ray in ray_walls:
        x_coords = [ray[0][0], ray[1][0]]
        y_coords = [ray[0][1], ray[1][1]]
        ax.plot(x_coords, y_coords, 'r-', linewidth=0.8, alpha=0.6)
    
    # Add label for ray walls (just once)
    ax.plot([], [], 'r-', linewidth=0.8, alpha=0.6, label='Ray Walls')
    
    # BREAKOFF_WALLS (from breakoff lines)
    breakoff_walls = [
        [(868.00, 558.09), (880.08, 589.30)],
        [(590.79, 317.64), (670.16, 367.00)],
        [(730.43, 367.00), (702.85, 241.48)],
        [(906.69, 307.71), (868.00, 351.34)],
        [(952.20, 422.64), (868.00, 435.19)]
    ]
    
    for breakoff in breakoff_walls:
        x_coords = [breakoff[0][0], breakoff[1][0]]
        y_coords = [breakoff[0][1], breakoff[1][1]]
        ax.plot(x_coords, y_coords, 'm-', linewidth=2, alpha=0.8)
    
    # Add label for breakoff walls (just once)
    ax.plot([], [], 'm-', linewidth=2, alpha=0.8, label='Breakoff Walls')
    
    # Set up the plot
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title('Clipped Environment Analysis\nPosition: (752.6, 435.2), Visibility Range: 200.0 pixels', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Set axis limits to show the full environment with some padding
    all_x = []
    all_y = []
    
    # Collect all coordinates to determine bounds
    for wall_list in [original_walls, original_doors, ray_walls, breakoff_walls]:
        for wall in wall_list:
            all_x.extend([wall[0][0], wall[1][0]])
            all_y.extend([wall[0][1], wall[1][1]])
    
    # Add circle bounds
    all_x.extend([752.59 - 200, 752.59 + 200])
    all_y.extend([435.19 - 200, 435.19 + 200])
    
    margin = 50
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    # Invert y-axis if needed (common in computer graphics coordinates)
    # ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

def print_summary():
    """Print a summary of the environment data."""
    print("Environment Analysis Summary:")
    print("-" * 40)
    print(f"Analysis Position: (752.6, 435.2)")
    print(f"Visibility Range: 200.0 pixels")
    print(f"Original Walls: 20 lines")
    print(f"Original Doors: 12 lines") 
    print(f"Ray Walls: 42 lines")
    print(f"Breakoff Walls: 5 lines")
    print(f"Circle Walls: 5012 lines (not plotted individually)")
    print(f"Total Lines: 5091")
    print(f"Visibility Circles: 1")

if __name__ == "__main__":
    print_summary()
    plot_environment()