import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

try:
    import dubins
    DUBINS_AVAILABLE = True
    print("✅ pydubins is available")
except ImportError:
    DUBINS_AVAILABLE = False
    print("❌ pydubins not found. Install with: pip install dubins")

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle

def get_dubins_path(start_pos, start_angle, end_pos, end_angle, turn_radius, step_size=0.1):
    """
    Generate Dubins path using pydubins library
    
    Args:
        start_pos: (x, y) starting position
        start_angle: starting orientation in radians
        end_pos: (x, y) ending position  
        end_angle: ending orientation in radians
        turn_radius: minimum turning radius
        step_size: distance between path points
        
    Returns:
        path_points: array of (x, y) coordinates
        path_length: total path length
        path_type: string describing the path type
    """
    if not DUBINS_AVAILABLE:
        # Fallback to straight line
        path_points = np.array([start_pos, end_pos])
        path_length = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        return path_points, path_length, "straight (no pydubins)"
    
    # Create start and end configurations: (x, y, theta)
    start_config = (start_pos[0], start_pos[1], start_angle)
    end_config = (end_pos[0], end_pos[1], end_angle)
    
    try:
        # Generate Dubins path
        path = dubins.shortest_path(start_config, end_config, turn_radius)
        
        # Get path length
        path_length = path.path_length()
        
        # Get path type
        path_type = path.path_type()
        
        # Sample points along the path
        configurations, _ = path.sample_many(step_size)
        
        # Extract x, y coordinates
        path_points = np.array([[config[0], config[1]] for config in configurations])
        
        return path_points, path_length, path_type
        
    except Exception as e:
        print(f"Error generating Dubins path: {e}")
        # Fallback to straight line
        path_points = np.array([start_pos, end_pos])
        path_length = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        return path_points, path_length, "error_fallback"

class InteractiveDubinsPlotter:
    def __init__(self, start_pos, end_pos, turn_radius):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.turn_radius = turn_radius
        
        # Initial orientations
        self.start_angle = 0.0
        self.end_angle = 0.0
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.25)
        
        # Create sliders
        ax_start_angle = plt.axes([0.2, 0.15, 0.5, 0.03])
        ax_end_angle = plt.axes([0.2, 0.10, 0.5, 0.03])
        ax_turn_radius = plt.axes([0.2, 0.05, 0.5, 0.03])
        
        self.slider_start = Slider(ax_start_angle, 'Start Angle', -180, 180, 
                                  valinit=np.degrees(self.start_angle), valfmt='%0.0f°')
        self.slider_end = Slider(ax_end_angle, 'End Angle', -180, 180, 
                                valinit=np.degrees(self.end_angle), valfmt='%0.0f°')
        self.slider_radius = Slider(ax_turn_radius, 'Turn Radius', 0.5, 3.0, 
                                   valinit=self.turn_radius, valfmt='%0.1f m')
        
        # Connect sliders to update function
        self.slider_start.on_changed(self.update_plot)
        self.slider_end.on_changed(self.update_plot)
        self.slider_radius.on_changed(self.update_plot)
        
        # Add reset button
        ax_reset = plt.axes([0.8, 0.15, 0.1, 0.04])
        self.button_reset = Button(ax_reset, 'Reset')
        self.button_reset.on_clicked(self.reset_angles)
        
        # Initial plot
        self.update_plot(None)
        
    def update_plot(self, val):
        """Update the plot when sliders change"""
        # Get current values
        self.start_angle = np.radians(self.slider_start.val)
        self.end_angle = np.radians(self.slider_end.val)
        self.turn_radius = self.slider_radius.val
        
        # Clear the axis
        self.ax.clear()
        
        # Generate Dubins path
        try:
            path_points, path_length, path_type = get_dubins_path(
                self.start_pos, self.start_angle, 
                self.end_pos, self.end_angle, 
                self.turn_radius, step_size=0.05
            )
            
            # Plot path
            self.ax.plot(path_points[:, 0], path_points[:, 1], 'b-', linewidth=3, 
                        label=f'Dubins Path ({path_type})')
            
            # Plot start and end points
            self.ax.plot(self.start_pos[0], self.start_pos[1], 'go', markersize=15, label='Start')
            self.ax.plot(self.end_pos[0], self.end_pos[1], 'ro', markersize=15, label='End')
            
            # Plot orientation arrows at start and end
            arrow_length = self.turn_radius * 0.8
            
            # Start arrow
            dx_start = arrow_length * np.cos(self.start_angle)
            dy_start = arrow_length * np.sin(self.start_angle)
            self.ax.arrow(self.start_pos[0], self.start_pos[1], dx_start, dy_start, 
                         head_width=self.turn_radius*0.2, head_length=self.turn_radius*0.15, 
                         fc='green', ec='green', linewidth=2)
            
            # End arrow
            dx_end = arrow_length * np.cos(self.end_angle)
            dy_end = arrow_length * np.sin(self.end_angle)
            self.ax.arrow(self.end_pos[0], self.end_pos[1], dx_end, dy_end, 
                         head_width=self.turn_radius*0.2, head_length=self.turn_radius*0.15, 
                         fc='red', ec='red', linewidth=2)
            
            # Plot turning circles (optional)
            if DUBINS_AVAILABLE:
                self.plot_turning_circles()
            
            # Add path info as text
            info_text = f"Path Type: {path_type}\n"
            info_text += f"Length: {path_length:.2f}m\n"
            info_text += f"Turn Radius: {self.turn_radius:.1f}m\n"
            info_text += f"Start: {np.degrees(self.start_angle):.0f}°\n"
            info_text += f"End: {np.degrees(self.end_angle):.0f}°\n"
            info_text += f"Points: {len(path_points)}"
            
            if not DUBINS_AVAILABLE:
                info_text += "\n\n⚠️ Install pydubins for\nproper Dubins paths:\npip install dubins"
            
            self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                        facecolor="lightblue", alpha=0.9))
            
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Error: {str(e)}", transform=self.ax.transAxes, 
                        ha='center', va='center', fontsize=12, color='red')
        
        # Set plot properties
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        title = 'Interactive Dubins Path Explorer'
        if DUBINS_AVAILABLE:
            title += ' (using pydubins)'
        else:
            title += ' (pydubins not installed)'
        self.ax.set_title(title)
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.axis('equal')
        
        # Set axis limits with some margin
        margin = max(2.0, self.turn_radius * 1.5)
        all_x = [self.start_pos[0], self.end_pos[1]] + path_points[:, 0].tolist()
        all_y = [self.start_pos[1], self.end_pos[1]] + path_points[:, 1].tolist()
        
        self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        plt.draw()
    
    def plot_turning_circles(self):
        """Plot the turning circles at start and end positions"""
        # Start position circles
        circle_start_L = plt.Circle((self.start_pos[0] - self.turn_radius * np.sin(self.start_angle), 
                                    self.start_pos[1] + self.turn_radius * np.cos(self.start_angle)), 
                                   self.turn_radius, fill=False, linestyle='--', alpha=0.3, color='green')
        circle_start_R = plt.Circle((self.start_pos[0] + self.turn_radius * np.sin(self.start_angle), 
                                    self.start_pos[1] - self.turn_radius * np.cos(self.start_angle)), 
                                   self.turn_radius, fill=False, linestyle='--', alpha=0.3, color='green')
        
        # End position circles
        circle_end_L = plt.Circle((self.end_pos[0] - self.turn_radius * np.sin(self.end_angle), 
                                  self.end_pos[1] + self.turn_radius * np.cos(self.end_angle)), 
                                 self.turn_radius, fill=False, linestyle='--', alpha=0.3, color='red')
        circle_end_R = plt.Circle((self.end_pos[0] + self.turn_radius * np.sin(self.end_angle), 
                                  self.end_pos[1] - self.turn_radius * np.cos(self.end_angle)), 
                                 self.turn_radius, fill=False, linestyle='--', alpha=0.3, color='red')
        
        self.ax.add_patch(circle_start_L)
        self.ax.add_patch(circle_start_R)
        self.ax.add_patch(circle_end_L)
        self.ax.add_patch(circle_end_R)
    
    def reset_angles(self, event):
        """Reset both angles to 0"""
        self.slider_start.reset()
        self.slider_end.reset()

def create_interactive_dubins(start_pos=(0, 0), end_pos=(4, 3), turn_radius=1.0):
    """Create an interactive Dubins path explorer using pydubins"""
    print("Interactive Dubins Path Explorer (pydubins)")
    print("="*50)
    
    if DUBINS_AVAILABLE:
        print("✅ Using pydubins library for proper Dubins paths")
    else:
        print("❌ pydubins not installed!")
        print("Install with: pip install dubins")
        print("Falling back to straight line paths")
    
    print()
    print("Controls:")
    print("• Start Angle: Controls the starting orientation (-180° to +180°)")
    print("• End Angle: Controls the ending orientation (-180° to +180°)")  
    print("• Turn Radius: Controls the minimum turning radius (0.5m to 3.0m)")
    print("• Reset: Reset both angles to 0°")
    print()
    
    if DUBINS_AVAILABLE:
        print("With pydubins, you'll see:")
        print("- Proper path types (LSL, RSR, LSR, RSL, RLR, LRL)")
        print("- Exact Dubins geometry")
        print("- Optimal path selection")
        print("- Accurate path lengths")
    
    plotter = InteractiveDubinsPlotter(start_pos, end_pos, turn_radius)
    plt.show()
    return plotter

def test_dubins_simple():
    """Simple test of pydubins functionality"""
    if not DUBINS_AVAILABLE:
        print("Cannot test - pydubins not installed")
        return
    
    print("\nTesting pydubins with simple examples:")
    print("-" * 40)
    
    test_cases = [
        ((0, 0), 0, (3, 0), 0, "Straight ahead"),
        ((0, 0), 0, (3, 0), np.pi, "U-turn"),
        ((0, 0), 0, (0, 3), np.pi/2, "90° left turn"),
        ((0, 0), 0, (3, 3), np.pi/2, "Diagonal with turn"),
    ]
    
    turn_radius = 1.0
    
    for start_pos, start_ang, end_pos, end_ang, description in test_cases:
        path_points, length, path_type = get_dubins_path(
            start_pos, start_ang, end_pos, end_ang, turn_radius
        )
        print(f"{description}:")
        print(f"  Path type: {path_type}")
        print(f"  Length: {length:.2f}m")
        print(f"  Points: {len(path_points)}")
        print()

# Example usage
if __name__ == "__main__":
    # Test basic functionality
    test_dubins_simple()
    
    # Create interactive explorer
    plotter = create_interactive_dubins(
        start_pos=(0, 0), 
        end_pos=(5, 3), 
        turn_radius=1.0
    )