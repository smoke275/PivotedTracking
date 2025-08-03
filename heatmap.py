import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
import time
from matplotlib.colors import LinearSegmentedColormap

class UnicycleGridSimulator:
    def __init__(self, world_size=200, grid_resolution=400, max_linear_vel=50.0, max_angular_vel=3.0, time_horizon=4.0):
        """
        Unicycle model Monte Carlo grid simulator.
        
        Args:
            world_size (float): World size in pixels
            grid_resolution (int): Internal resolution for computation
            max_linear_vel (float): Maximum linear velocity in px/s
            max_angular_vel (float): Maximum angular velocity in rad/s
            time_horizon (float): Time horizon for trajectories in seconds
        """
        self.world_size = world_size
        self.grid_resolution = grid_resolution
        self.grid = np.zeros((grid_resolution, grid_resolution))
        
        # Colormap for visualization (black for 0 probability, progressing to red for high probability)
        colors = ['#000000', '#330033', '#660066', '#990099', '#CC00CC', '#FF0099', '#FF3366', '#FF6633', '#FF9900', '#FFCC00', '#FFFF00']
        self.cmap = LinearSegmentedColormap.from_list('heatmap', colors, N=256)
        
        # Motion constraints
        self.MAX_LINEAR_VEL = max_linear_vel   # px/s
        self.MAX_ANGULAR_VEL = max_angular_vel   # rad/s
        self.TIME_HORIZON = time_horizon      # seconds
        
    def generate_trajectory(self, linear_velocity, angular_velocity, dt=0.05):
        """Generate single trajectory using unicycle model."""
        x, y, theta = 0.0, 0.0, 0.0
        trajectory = [(x, y)]
        
        for t in np.arange(dt, self.TIME_HORIZON + dt, dt):
            x += linear_velocity * np.cos(theta) * dt
            y += linear_velocity * np.sin(theta) * dt
            theta += angular_velocity * dt
            trajectory.append((x, y))
            
        return trajectory
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices."""
        grid_x = int((x + self.world_size/2) / self.world_size * self.grid_resolution)
        grid_y = int((-y + self.world_size/2) / self.world_size * self.grid_resolution)
        return grid_x, grid_y
    
    def sample_controls(self):
        """Sample realistic control inputs - mostly curved with small exploration."""
        strategy = np.random.choice(['curved', 'exploration'], 
                                  p=[0.9, 0.1])  # 90% curved, 10% exploration
        
        if strategy == 'curved':
            # Curved paths with consistent forward motion
            linear_vel = np.random.uniform(0.4 * self.MAX_LINEAR_VEL, 0.8 * self.MAX_LINEAR_VEL)
            angular_vel = np.random.uniform(-0.8 * self.MAX_ANGULAR_VEL, 0.8 * self.MAX_ANGULAR_VEL)
        else:  # exploration
            # Slower exploration with more turning variation
            linear_vel = np.random.uniform(0.2 * self.MAX_LINEAR_VEL, 0.5 * self.MAX_LINEAR_VEL)
            angular_vel = np.random.uniform(-self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL)
        
        # Add small amount of noise
        linear_vel *= np.random.uniform(0.95, 1.05)
        angular_vel *= np.random.uniform(0.95, 1.05)
        
        # Enforce constraints (ensure minimum forward motion)
        linear_vel = np.clip(linear_vel, 0.1 * self.MAX_LINEAR_VEL, self.MAX_LINEAR_VEL)
        angular_vel = np.clip(angular_vel, -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL)
        
        return linear_vel, angular_vel
    
    def generate_probability_grid(self, num_samples=10000, smoothing_sigma=8.0, output_grid_size=500):
        """
        Generate probability grid using Monte Carlo simulation.
        
        Args:
            num_samples (int): Number of trajectory samples
            smoothing_sigma (float): Gaussian smoothing
            output_grid_size (int): Output grid resolution
            
        Returns:
            tuple: (probability_grid, grid_metadata)
        """
        print(f"Monte Carlo simulation: {num_samples} samples...")
        start_time = time.time()
        
        # Reset accumulation grid
        self.grid = np.zeros((self.grid_resolution, self.grid_resolution))
        max_reach = 0
        
        # Monte Carlo sampling with better accumulation for high-res grids
        for i in range(num_samples):
            # Sample random controls
            linear_vel, angular_vel = self.sample_controls()
            
            # Generate trajectory
            trajectory = self.generate_trajectory(linear_vel, angular_vel)
            
            # Accumulate in grid with spread for high-resolution grids
            for x, y in trajectory:
                grid_x, grid_y = self.world_to_grid(x, y)
                if 0 <= grid_x < self.grid_resolution and 0 <= grid_y < self.grid_resolution:
                    # For high-res grids, add to neighboring cells too
                    weight = 1.0 / len(trajectory)  # Normalize by trajectory length
                    
                    # Add to center cell
                    self.grid[grid_y, grid_x] += weight
                    
                    # Add to immediate neighbors with lower weight for smoother distribution
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = grid_y + dy, grid_x + dx
                            if 0 <= ny < self.grid_resolution and 0 <= nx < self.grid_resolution:
                                self.grid[ny, nx] += weight * 0.3
                
                max_reach = max(max_reach, np.sqrt(x*x + y*y))
            
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{num_samples} completed")
        
        # Apply smoothing
        if smoothing_sigma > 0:
            self.grid = gaussian_filter(self.grid, sigma=smoothing_sigma)
        
        # Normalize to probability
        if self.grid.max() > 0:
            heatmap = self.grid / self.grid.max()
        else:
            heatmap = self.grid
        
        # Crop to relevant area and resize
        # Calculate dynamic zoom limit based on time horizon and max velocity
        max_theoretical_reach = self.MAX_LINEAR_VEL * self.TIME_HORIZON
        zoom_limit = min(max_reach * 1.1, max_theoretical_reach * 0.8)  # Allow 80% of theoretical max reach
        center = self.grid_resolution // 2
        zoom_pixels = int(zoom_limit / self.world_size * self.grid_resolution)
        
        if zoom_pixels < center:
            heatmap_cropped = heatmap[center-zoom_pixels:center+zoom_pixels, 
                                    center-zoom_pixels:center+zoom_pixels]
        else:
            heatmap_cropped = heatmap
        
        # Resize to target output grid size
        zoom_factor = output_grid_size / heatmap_cropped.shape[0]
        probability_grid = zoom(heatmap_cropped, zoom_factor, order=1)
        
        # Grid metadata
        world_extent = zoom_limit * 2
        pixel_size = world_extent / output_grid_size
        
        metadata = {
            'grid_size': output_grid_size,
            'world_extent_px': world_extent,
            'cell_size_px': pixel_size,
            'center_idx': output_grid_size // 2,
            'max_probability': probability_grid.max(),
            'max_reach_px': max_reach,
            'num_samples': num_samples,
            'compute_time_s': time.time() - start_time
        }
        
        print(f"Grid generated: {output_grid_size}x{output_grid_size}")
        print(f"Coverage: ±{zoom_limit:.0f}px, {pixel_size:.1f}px per cell")
        print(f"Completed in {metadata['compute_time_s']:.1f}s")
        
        return probability_grid, metadata
    
    def save_grid(self, probability_grid, metadata, filename_base="probability_grid"):
        """Save grid and metadata to a single pickle file for use as mask."""
        import pickle
        
        # Create comprehensive data package
        grid_data = {
            'grid': probability_grid,
            'grid_size': metadata['grid_size'],
            'world_extent_px': metadata['world_extent_px'],
            'cell_size_px': metadata['cell_size_px'],
            'center_idx': metadata['center_idx'],
            'max_probability': metadata['max_probability'],
            'max_reach_px': metadata['max_reach_px'],
            'num_samples': metadata['num_samples'],
            'compute_time_s': metadata['compute_time_s'],
            
            # Motion parameters
            'max_linear_vel': self.MAX_LINEAR_VEL,
            'max_angular_vel': self.MAX_ANGULAR_VEL,
            'time_horizon': self.TIME_HORIZON,
            
            # Coordinate bounds
            'world_min_x': -metadata['world_extent_px']/2,
            'world_max_x': metadata['world_extent_px']/2,
            'world_min_y': -metadata['world_extent_px']/2,
            'world_max_y': metadata['world_extent_px']/2
        }
        
        # Save as pickle
        with open(f"{filename_base}.pkl", 'wb') as f:
            pickle.dump(grid_data, f)
        
        print(f"\nSaved: {filename_base}.pkl")
        print(f"Usage: data = pickle.load(open('{filename_base}.pkl', 'rb'))")
    
    def print_grid_info(self, probability_grid, metadata):
        """Print grid statistics."""
        print("\n" + "="*50)
        print("PROBABILITY GRID STATISTICS")
        print("="*50)
        print(f"Dimensions: {metadata['grid_size']}x{metadata['grid_size']}")
        print(f"Coverage: ±{metadata['world_extent_px']/2:.0f} pixels")
        print(f"Resolution: {metadata['cell_size_px']:.1f}px per cell")
        print(f"Max probability: {metadata['max_probability']:.4f}")
        print(f"Non-zero cells: {np.sum(probability_grid > 0.001)}")
        print(f"High probability cells (>10%): {np.sum(probability_grid > 0.1)}")
        print(f"Medium probability cells (1-10%): {np.sum((probability_grid > 0.01) & (probability_grid <= 0.1))}")
        
        # Find peak location
        max_idx = np.unravel_index(np.argmax(probability_grid), probability_grid.shape)
        center = metadata['center_idx']
        max_x = (max_idx[1] - center) * metadata['cell_size_px']
        max_y = (center - max_idx[0]) * metadata['cell_size_px']
        
        print(f"Peak at grid[{max_idx[0]}, {max_idx[1]}] = ({max_x:.1f}, {max_y:.1f})px")
        print(f"Peak probability: {probability_grid[max_idx]:.4f}")
        print("="*50)
    
    def visualize_grid(self, probability_grid, metadata):
        """Visualize the probability grid."""
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        extent = [-metadata['world_extent_px']/2, metadata['world_extent_px']/2,
                 -metadata['world_extent_px']/2, metadata['world_extent_px']/2]
        
        im = ax.imshow(probability_grid, extent=extent, origin='lower', 
                      cmap=self.cmap, alpha=0.9)
        
        # Add grid lines
        grid_size = metadata['grid_size']
        cell_size = metadata['cell_size_px']
        step = max(1, grid_size // 10)
        
        for i in range(0, grid_size + 1, step):
            pos = -metadata['world_extent_px']/2 + i * cell_size
            ax.axhline(pos, color='white', alpha=0.3, linewidth=0.5)
            ax.axvline(pos, color='white', alpha=0.3, linewidth=0.5)
        
        # Start position
        ax.plot(0, 0, 'o', markersize=8, markerfacecolor='cyan', 
                markeredgecolor='white', markeredgewidth=2, zorder=10)
        
        ax.set_xlabel('X Position (pixels)', color='white', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', color='white', fontsize=12)
        ax.set_title(f'Probability Grid\n{grid_size}x{grid_size}, {cell_size:.1f}px/cell', 
                    color='white', fontsize=14)
        ax.tick_params(colors='white')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', color='white', fontsize=12)
        cbar.ax.tick_params(colors='white')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to generate probability grid."""
    print("=== Unicycle Monte Carlo Probability Grid ===")
    
    # Motion constraint parameters (can be customized)
    max_linear_vel = 50.0    # px/s
    max_angular_vel = 3.0    # rad/s
    time_horizon = 4.0       # seconds

    print("Parameters:")
    print(f"- Linear velocity: 0-{max_linear_vel} px/s")
    print(f"- Angular velocity: ±{max_angular_vel} rad/s") 
    print(f"- Time horizon: {time_horizon} seconds")
    print("- Forward-biased motion (minimal backward reach)")
    print("- Behavior mix: 90% curved paths, 10% exploration")
    print()
    
    # Create simulator with custom motion constraints
    simulator = UnicycleGridSimulator(
        world_size=200,
        grid_resolution=400,
        max_linear_vel=max_linear_vel,
        max_angular_vel=max_angular_vel,
        time_horizon=time_horizon
    )
    
    # Generate probability grid
    grid, metadata = simulator.generate_probability_grid(
        num_samples=10000,      # Monte Carlo samples
        smoothing_sigma=8.0,    # Higher smoothing for 500x500 grid
        output_grid_size=500    # Output grid resolution (500x500)
    )
    
    # Print statistics
    simulator.print_grid_info(grid, metadata)
    
    # Save the grid and metadata for use as mask
    simulator.save_grid(grid, metadata, "unicycle_grid")
    
    # Visualize
    simulator.visualize_grid(grid, metadata)
    
    print("\nGrid generated successfully!")
    print(f"Shape: {grid.shape}")
    print(f"Max probability: {grid.max():.4f}")
    print(f"Center probability: {grid[metadata['center_idx'], metadata['center_idx']]:.4f}")
    
    return grid, metadata

if __name__ == "__main__":
    main()