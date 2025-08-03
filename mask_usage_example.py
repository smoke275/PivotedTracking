import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as patches
from scipy.ndimage import rotate
import risk_calculator

class InteractiveAgentWithMask:
    def __init__(self, mask_data, world_size=500):
        self.mask_data = mask_data
        self.world_size = world_size
        
        # Agent state
        self.agent_x = 0.0
        self.agent_y = 0.0
        self.agent_theta = 0.0  # orientation in radians
        
        # Movement parameters
        self.move_speed = 10.0  # pixels per keypress
        self.turn_speed = 0.2   # radians per keypress
        
        # Setup figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.fig.suptitle('Interactive Agent with Fixed Grid and Reachability Probabilities\nUse Arrow Keys: ←→ turn, ↑↓ move', fontsize=14)
        
        # World view
        self.ax.set_xlim(-world_size//2, world_size//2)
        self.ax.set_ylim(-world_size//2, world_size//2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('World View (Agent in Blue)')
        self.ax.set_xlabel('X Position (pixels)')
        self.ax.set_ylabel('Y Position (pixels)')
        
        # Agent visualization elements
        self.agent_circle = Circle((self.agent_x, self.agent_y), 5, color='blue', zorder=10)
        self.agent_arrow = patches.FancyArrowPatch((0, 0), (0, 0), 
                                                   arrowstyle='->', mutation_scale=20, 
                                                   color='red', zorder=11)
        self.ax.add_patch(self.agent_circle)
        self.ax.add_patch(self.agent_arrow)
        
        # Reachability mask overlay on world view
        self.mask_overlay = None
        self.colorbar = None  # Store colorbar for probability scale
        self.grid_lines = []  # Store grid line objects
        self.grid_visible = True  # Control grid visibility
        self.mask_visible = True  # Control mask overlay visibility
        
        # Mouse inspection elements
        self.inspection_point = None  # Circle showing clicked point
        self.hover_point = None  # Circle showing hover position
        self.last_click_x = None
        self.last_click_y = None
        self.last_hover_x = None
        self.last_hover_y = None
        self.current_probabilities = None  # Store current probability grid
        self.hover_update_throttle = 0  # Throttle hover updates for performance
        
        # Connect keyboard and mouse events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_hover)
        
        # Initial update
        self.update_display()
        
        plt.tight_layout()
    
    def create_grid_overlay(self):
        """Create fixed grid lines that only translate with the agent (no rotation)."""
        # Clear existing grid lines
        for line in self.grid_lines:
            line.remove()
        self.grid_lines.clear()
        
        # Only create grid if it's visible
        if not self.grid_visible:
            return
        
        # Grid parameters
        cell_size = self.mask_data['cell_size_px']
        grid_size = self.mask_data['grid_size']
        half_extent = (grid_size * cell_size) / 2
        
        # Create fixed grid lines (no rotation, only translation)
        num_lines = 21  # Reduced number for better visibility
        step = cell_size * (grid_size // num_lines)
        
        # Vertical lines (always vertical in world coordinates)
        for i in range(num_lines):
            x_offset = -half_extent + i * step
            x_world = self.agent_x + x_offset
            y_start_world = self.agent_y - half_extent
            y_end_world = self.agent_y + half_extent
            
            line = self.ax.plot([x_world, x_world], [y_start_world, y_end_world], 
                               'g-', alpha=0.3, linewidth=0.5, zorder=1)[0]
            self.grid_lines.append(line)
        
        # Horizontal lines (always horizontal in world coordinates)
        for i in range(num_lines):
            y_offset = -half_extent + i * step
            y_world = self.agent_y + y_offset
            x_start_world = self.agent_x - half_extent
            x_end_world = self.agent_x + half_extent
            
            line = self.ax.plot([x_start_world, x_end_world], [y_world, y_world], 
                               'g-', alpha=0.3, linewidth=0.5, zorder=1)[0]
            self.grid_lines.append(line)
    
    def create_mask_overlay(self):
        """Create probability mask overlay on the world view with fixed grid."""
        # Remove existing mask overlay and colorbar safely
        if self.mask_overlay is not None:
            try:
                self.mask_overlay.remove()
            except (ValueError, KeyError):
                pass  # Already removed
            self.mask_overlay = None
            
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except (ValueError, KeyError):
                pass  # Already removed
            self.colorbar = None
        
        # Only create mask if it's visible
        if not self.mask_visible:
            return
        
        # Get reachability probabilities for the fixed grid using API
        fixed_grid_probabilities = risk_calculator.get_reachability_probabilities_for_fixed_grid(
            self.agent_x, self.agent_y, self.agent_theta, self.mask_data)
        
        # Store current probabilities for mouse inspection
        self.current_probabilities = fixed_grid_probabilities
        
        # Calculate extent in world coordinates centered on agent
        cell_size = self.mask_data['cell_size_px']
        grid_size = self.mask_data['grid_size']
        half_extent_px = (grid_size * cell_size) / 2
        
        # The mask extent in world coordinates (agent-centered, fixed orientation)
        extent = [self.agent_x - half_extent_px, self.agent_x + half_extent_px,
                  self.agent_y - half_extent_px, self.agent_y + half_extent_px]
        
        # Display the fixed grid probabilities
        self.mask_overlay = self.ax.imshow(fixed_grid_probabilities, extent=extent, origin='lower',
                                          cmap='hot', alpha=0.6, zorder=2)
        
        # Add colorbar for probability scale
        if self.mask_overlay is not None:
            self.colorbar = self.fig.colorbar(self.mask_overlay, ax=self.ax, shrink=0.8, pad=0.02)
            self.colorbar.set_label('Reachability Probability', rotation=270, labelpad=15)
    
    def update_display(self):
        """Update all visual elements."""
        # Update agent position in world view
        self.agent_circle.center = (self.agent_x, self.agent_y)
        
        # Update agent orientation arrow
        arrow_length = 15
        end_x = self.agent_x + arrow_length * np.cos(self.agent_theta)
        end_y = self.agent_y + arrow_length * np.sin(self.agent_theta)
        self.agent_arrow.set_positions((self.agent_x, self.agent_y), (end_x, end_y))
        
        # Update grid overlay that moves with the agent
        self.create_grid_overlay()
        
        # Update mask overlay that moves with the agent
        self.create_mask_overlay()
        
        # Update title with current position and orientation
        grid_status = "Grid: ON" if self.grid_visible else "Grid: OFF"
        mask_status = "Mask: ON" if self.mask_visible else "Mask: OFF"
        self.ax.set_title(f'World View - Agent at ({self.agent_x:.1f}, {self.agent_y:.1f}), θ={np.degrees(self.agent_theta):.1f}° | {grid_status} | {mask_status}')
        
        # Show reachability at current position using API
        prob_at_center = risk_calculator.get_reachability_at_position(
            self.agent_x, self.agent_y, self.agent_theta, self.mask_data)
        
        # Create title with probability info
        click_info = ""
        if self.last_click_x is not None and self.last_click_y is not None:
            click_prob = self.get_probability_at_world_position(self.last_click_x, self.last_click_y)
            if click_prob is not None:
                click_info = f" | Click: {click_prob:.3f}"
        
        hover_info = ""
        if self.last_hover_x is not None and self.last_hover_y is not None:
            hover_prob = self.get_probability_at_world_position(self.last_hover_x, self.last_hover_y)
            if hover_prob is not None:
                hover_info = f" | Hover: {hover_prob:.3f}"
        
        self.fig.suptitle(f'Interactive Agent with Fixed Grid and Reachability Probabilities\nUse Arrow Keys: ←→ turn, ↑↓ move | G: Grid | M: Mask | Center: {prob_at_center:.3f}{click_info}{hover_info}', 
                         fontsize=14)
        
        self.fig.canvas.draw()
    
    def get_probability_at_world_position(self, world_x, world_y):
        """Get probability value at a world position."""
        if self.current_probabilities is None:
            return None
        
        # Convert world coordinates to grid indices
        cell_size = self.mask_data['cell_size_px']
        grid_size = self.mask_data['grid_size']
        half_extent_px = (grid_size * cell_size) / 2
        
        # Position relative to agent
        rel_x = world_x - self.agent_x
        rel_y = world_y - self.agent_y
        
        # Convert to grid indices
        grid_x = int((rel_x + half_extent_px) / cell_size)
        grid_y = int((rel_y + half_extent_px) / cell_size)
        
        # Check bounds
        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
            return self.current_probabilities[grid_y, grid_x]
        return None
    
    def on_mouse_click(self, event):
        """Handle mouse clicks for probability inspection."""
        if event.inaxes != self.ax:
            return
        
        # Remove previous inspection point
        if self.inspection_point is not None:
            self.inspection_point.remove()
            self.inspection_point = None
        
        # Store click coordinates
        self.last_click_x = event.xdata
        self.last_click_y = event.ydata
        
        if self.last_click_x is None or self.last_click_y is None:
            return
        
        # Get probability at clicked position
        prob_value = self.get_probability_at_world_position(self.last_click_x, self.last_click_y)
        
        if prob_value is not None:
            # Add inspection point marker
            self.inspection_point = Circle((self.last_click_x, self.last_click_y), 3, 
                                         color='white', fill=True, edgecolor='black', 
                                         linewidth=2, zorder=12)
            self.ax.add_patch(self.inspection_point)
            
            # Print detailed info to console
            print(f"\nClicked at world coordinates: ({self.last_click_x:.2f}, {self.last_click_y:.2f})")
            print(f"Reachability probability: {prob_value:.6f}")
            
            # Calculate relative position from agent
            rel_x = self.last_click_x - self.agent_x
            rel_y = self.last_click_y - self.agent_y
            distance = np.sqrt(rel_x**2 + rel_y**2)
            angle = np.degrees(np.arctan2(rel_y, rel_x))
            print(f"Relative to agent: ({rel_x:.2f}, {rel_y:.2f}), distance: {distance:.2f}px, angle: {angle:.1f}°")
        
        # Update display to show new click info
        self.update_display()
    
    def on_mouse_hover(self, event):
        """Handle mouse hover for real-time probability inspection."""
        # Throttle hover updates to reduce frequency and prevent errors
        self.hover_update_throttle += 1
        if self.hover_update_throttle % 3 != 0:  # Only update every 3rd hover event
            return
            
        if event.inaxes != self.ax:
            # Clear hover point if mouse leaves the plot area
            if self.hover_point is not None:
                try:
                    self.hover_point.remove()
                except (ValueError, KeyError):
                    pass  # Already removed
                self.hover_point = None
                self.last_hover_x = None
                self.last_hover_y = None
                self.update_hover_display_only()
            return
        
        # Update hover coordinates
        self.last_hover_x = event.xdata
        self.last_hover_y = event.ydata
        
        if self.last_hover_x is None or self.last_hover_y is None:
            return
        
        # Remove previous hover point
        if self.hover_point is not None:
            try:
                self.hover_point.remove()
            except (ValueError, KeyError):
                pass  # Already removed
            self.hover_point = None
        
        # Get probability at hover position
        prob_value = self.get_probability_at_world_position(self.last_hover_x, self.last_hover_y)
        
        if prob_value is not None:
            # Add hover point marker (smaller and more transparent than click marker)
            self.hover_point = Circle((self.last_hover_x, self.last_hover_y), 2, 
                                    color='cyan', fill=True, alpha=0.7, 
                                    edgecolor='blue', linewidth=1, zorder=11)
            self.ax.add_patch(self.hover_point)
        
        # Update display to show new hover info (lightweight version)
        self.update_hover_display_only()
    
    def update_hover_display_only(self):
        """Lightweight update for hover info only (no mask recreation)."""
        # Show reachability at current position using API
        prob_at_center = risk_calculator.get_reachability_at_position(
            self.agent_x, self.agent_y, self.agent_theta, self.mask_data)
        
        # Create title with probability info
        click_info = ""
        if self.last_click_x is not None and self.last_click_y is not None:
            click_prob = self.get_probability_at_world_position(self.last_click_x, self.last_click_y)
            if click_prob is not None:
                click_info = f" | Click: {click_prob:.3f}"
        
        hover_info = ""
        if self.last_hover_x is not None and self.last_hover_y is not None:
            hover_prob = self.get_probability_at_world_position(self.last_hover_x, self.last_hover_y)
            if hover_prob is not None:
                hover_info = f" | Hover: {hover_prob:.3f}"
        
        self.fig.suptitle(f'Interactive Agent with Fixed Grid and Reachability Probabilities\nUse Arrow Keys: ←→ turn, ↑↓ move | G: Grid | M: Mask | Center: {prob_at_center:.3f}{click_info}{hover_info}', 
                         fontsize=14)
        
        self.fig.canvas.draw_idle()  # Use draw_idle for better performance
    
    def on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == 'up':
            # Move forward
            self.agent_x += self.move_speed * np.cos(self.agent_theta)
            self.agent_y += self.move_speed * np.sin(self.agent_theta)
        elif event.key == 'down':
            # Move backward
            self.agent_x -= self.move_speed * np.cos(self.agent_theta)
            self.agent_y -= self.move_speed * np.sin(self.agent_theta)
        elif event.key == 'left':
            # Turn left (counter-clockwise)
            self.agent_theta += self.turn_speed
        elif event.key == 'right':
            # Turn right (clockwise)
            self.agent_theta -= self.turn_speed
        elif event.key == 'g':
            # Toggle grid visibility
            self.grid_visible = not self.grid_visible
        elif event.key == 'm':
            # Toggle mask overlay visibility
            self.mask_visible = not self.mask_visible
        elif event.key == 'r':
            # Reset position
            self.agent_x = 0.0
            self.agent_y = 0.0
            self.agent_theta = 0.0
        elif event.key == 'q' or event.key == 'escape':
            # Quit
            plt.close(self.fig)
            return
        
        # Keep agent within world bounds
        max_pos = self.world_size // 2 - 20
        self.agent_x = np.clip(self.agent_x, -max_pos, max_pos)
        self.agent_y = np.clip(self.agent_y, -max_pos, max_pos)
        
        # Normalize angle
        self.agent_theta = self.agent_theta % (2 * np.pi)
        
        self.update_display()

def example_usage():
    """Interactive example with agent control."""
    print("Loading probability mask...")
    
    try:
        # Load the mask using API
        data = risk_calculator.load_reachability_mask("unicycle_grid")
        
        if data is None:
            print("Failed to load reachability mask!")
            return
        
        print(f"Loaded grid: {data['grid'].shape}")
        print(f"Grid covers: ±{data['world_extent_px']/2:.1f} pixels")
        print(f"Cell size: {data['cell_size_px']:.3f} px/cell")
        print()
        print("CONTROLS:")
        print("  ↑ ↓  - Move forward/backward")
        print("  ← →  - Turn left/right") 
        print("  G    - Toggle grid overlay")
        print("  M    - Toggle mask overlay")
        print("  R    - Reset to origin")
        print("  Q/Esc - Quit")
        print("  Click - Inspect probability at point")
        print("  Hover - Real-time probability inspection")
        print()
        print("The visualization shows reachable areas from the agent's current position.")
        print("The mask represents where the agent can reach using its motion model.")
        print("The green grid shows the agent's fixed coordinate frame (no rotation).")
        print("The hot colormap overlay shows the reachability probabilities in world coordinates.")
        print("Hover over any location to see real-time probability values.")
        print("Click to pin an inspection point with detailed console output.")
        
        # Create interactive visualization
        interactive_app = InteractiveAgentWithMask(data, world_size=400)
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    example_usage()
