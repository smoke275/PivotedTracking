import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle, FancyArrow, Rectangle
from shapely.geometry import Polygon, Point, LineString
from collections import deque, defaultdict
import time

class ProgressiveStateExpansion:
    """
    Progressive node-wise state expansion for reachability
    Like breadth-first search building outward from current state
    """
    
    def __init__(self, v_max, omega_max, time_horizon, obstacles, 
                 grid_resolution=0.3, theta_resolution=8):
        """
        Args:
            v_max: max linear velocity
            omega_max: max angular velocity  
            time_horizon: planning horizon
            obstacles: list of Shapely Polygons
            grid_resolution: spatial discretization (meters)
            theta_resolution: number of heading bins (e.g., 8 = 45° bins)
        """
        self.v_max = v_max
        self.omega_max = omega_max
        self.T = time_horizon
        self.obstacles = obstacles
        
        self.grid_res = grid_resolution
        self.theta_bins = theta_resolution
        self.theta_res = 2 * np.pi / theta_resolution
        
        # Time step for expansion
        self.dt = 0.2  # seconds per expansion step
        self.max_steps = int(time_horizon / self.dt)
        
        # Motion primitives (precomputed)
        self.primitives = self.generate_motion_primitives()
        
        print(f"Expansion parameters:")
        print(f"  Grid resolution: {grid_resolution}m")
        print(f"  Heading bins: {theta_resolution} ({np.degrees(self.theta_res):.1f}° per bin)")
        print(f"  Time step: {self.dt}s")
        print(f"  Max expansion steps: {self.max_steps}")
        print(f"  Motion primitives: {len(self.primitives)}")
    
    def generate_motion_primitives(self):
        """
        Generate motion primitives for unicycle
        Each primitive: (v, omega, duration) -> resulting state change
        """
        primitives = []
        
        # Forward motions
        for v in [self.v_max, self.v_max * 0.7, self.v_max * 0.4]:
            # Straight
            primitives.append(('forward', v, 0, self.dt))
            
            # Arcs
            for omega in [self.omega_max, self.omega_max * 0.5, -self.omega_max * 0.5, -self.omega_max]:
                primitives.append(('arc', v, omega, self.dt))
        
        # Turn in place
        for omega in [self.omega_max, -self.omega_max]:
            primitives.append(('turn', 0, omega, self.dt))
        
        return primitives
    
    def discretize_state(self, x, y, theta):
        """Convert continuous state to discrete grid cell"""
        ix = int(round(x / self.grid_res))
        iy = int(round(y / self.grid_res))
        itheta = int(round(theta / self.theta_res)) % self.theta_bins
        return (ix, iy, itheta)
    
    def continuous_state(self, ix, iy, itheta):
        """Convert discrete state back to continuous"""
        x = ix * self.grid_res
        y = iy * self.grid_res
        theta = itheta * self.theta_res
        return (x, y, theta)
    
    def apply_primitive(self, state, primitive):
        """
        Apply motion primitive to state
        Returns: new_state or None if invalid
        """
        x, y, theta = state
        prim_type, v, omega, dt = primitive
        
        # Add small noise for stochasticity
        v_noisy = v + np.random.normal(0, 0.05)
        omega_noisy = omega + np.random.normal(0, 0.02)
        
        # Forward simulate
        if prim_type == 'turn':
            # Turn in place
            new_x = x
            new_y = y
            new_theta = theta + omega_noisy * dt
        else:
            # Move forward (arc or straight)
            if abs(omega_noisy) < 0.01:
                # Straight line
                new_x = x + v_noisy * np.cos(theta) * dt
                new_y = y + v_noisy * np.sin(theta) * dt
                new_theta = theta
            else:
                # Arc (circular motion)
                radius = v_noisy / omega_noisy
                new_theta = theta + omega_noisy * dt
                new_x = x + radius * (np.sin(new_theta) - np.sin(theta))
                new_y = y - radius * (np.cos(new_theta) - np.cos(theta))
        
        # Normalize angle
        new_theta = self.normalize_angle(new_theta)
        
        return (new_x, new_y, new_theta)
    
    def is_valid_state(self, state):
        """Check if state collides with obstacles"""
        x, y, _ = state
        point = Point(x, y)
        
        for obstacle in self.obstacles:
            if obstacle.contains(point) or obstacle.distance(point) < 0.2:
                return False
        return True
    
    def compute_reachability(self, start_state):
        """
        Progressive expansion to compute reachability
        
        Algorithm:
        1. Start with initial state
        2. Expand using motion primitives  
        3. Propagate probability through graph
        4. Build outward level by level
        
        Returns:
            reachability_scores: dict mapping (x,y) -> probability
            expansion_tree: dict with full expansion info
        """
        print("\n" + "="*60)
        print("PROGRESSIVE STATE EXPANSION")
        print("="*60)
        
        # Initialize
        start_discrete = self.discretize_state(*start_state)
        
        # Priority queue for expansion (time-ordered)
        # Each entry: (time_step, discrete_state, continuous_state, probability)
        queue = deque([(0, start_discrete, start_state, 1.0)])
        
        # Track visited states and their probabilities
        visited = {start_discrete: 1.0}
        state_info = {start_discrete: {
            'continuous': start_state,
            'probability': 1.0,
            'time_step': 0,
            'parent': None
        }}
        
        # Statistics
        expansion_count = 0
        nodes_per_level = defaultdict(int)
        
        print(f"\nStarting expansion from state: {start_state}")
        print(f"Discrete state: {start_discrete}")
        
        t_start = time.time()
        
        # Progressive expansion
        while queue:
            time_step, discrete_state, cont_state, probability = queue.popleft()
            
            # Stop if exceeded time horizon
            if time_step >= self.max_steps:
                continue
            
            nodes_per_level[time_step] += 1
            
            # Expand this state with all motion primitives
            for primitive in self.primitives:
                # Apply primitive
                new_cont_state = self.apply_primitive(cont_state, primitive)
                
                # Check validity
                if not self.is_valid_state(new_cont_state):
                    continue
                
                # Discretize new state
                new_discrete = self.discretize_state(*new_cont_state)
                
                # Compute transition probability (based on primitive cost)
                transition_prob = self.compute_transition_probability(
                    cont_state, new_cont_state, primitive
                )
                
                new_probability = probability * transition_prob
                
                # Update if better or new
                if new_discrete not in visited or new_probability > visited[new_discrete]:
                    visited[new_discrete] = new_probability
                    
                    state_info[new_discrete] = {
                        'continuous': new_cont_state,
                        'probability': new_probability,
                        'time_step': time_step + 1,
                        'parent': discrete_state,
                        'primitive': primitive
                    }
                    
                    # Add to queue for further expansion
                    queue.append((time_step + 1, new_discrete, new_cont_state, new_probability))
                    expansion_count += 1
        
        t_end = time.time()
        
        # Convert to (x, y) reachability scores
        reachability_scores = {}
        for discrete_state, info in state_info.items():
            x, y, _ = info['continuous']
            key = (round(x, 2), round(y, 2))
            
            # Aggregate over different headings
            if key not in reachability_scores:
                reachability_scores[key] = 0.0
            reachability_scores[key] += info['probability']
        
        # Normalize
        total = sum(reachability_scores.values())
        if total > 0:
            reachability_scores = {k: v/total for k, v in reachability_scores.items()}
        
        # Print statistics
        print(f"\n" + "-"*60)
        print(f"EXPANSION COMPLETE")
        print(f"-"*60)
        print(f"Total time: {(t_end - t_start)*1000:.1f}ms")
        print(f"Total expansions: {expansion_count}")
        print(f"Unique states visited: {len(visited)}")
        print(f"Unique locations: {len(reachability_scores)}")
        print(f"\nNodes expanded per level:")
        for level in sorted(nodes_per_level.keys()):
            print(f"  Level {level} (t={level*self.dt:.1f}s): {nodes_per_level[level]} nodes")
        print("="*60)
        
        return reachability_scores, state_info
    
    def compute_transition_probability(self, from_state, to_state, primitive):
        """
        Compute probability of transition based on cost
        Lower cost = higher probability
        """
        prim_type, v, omega, dt = primitive
        
        # Cost components
        control_cost = 0.1 * (v**2 + omega**2) * dt
        
        # Deviation from straight ahead
        _, _, theta = from_state
        x1, y1, _ = from_state
        x2, y2, theta2 = to_state
        
        direction = np.arctan2(y2 - y1, x2 - x1)
        angle_diff = abs(self.normalize_angle(direction - theta))
        heading_cost = angle_diff / np.pi
        
        total_cost = control_cost + heading_cost
        
        # Convert to probability
        probability = np.exp(-total_cost)
        
        return probability
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def visualize_progressive_expansion(start_state, reachability_scores, state_info,
                                    obstacles, planner):
    """
    Visualize the progressive expansion
    Shows wavefront propagation
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Color map for time steps
    max_time = max(info['time_step'] for info in state_info.values())
    
    for idx, ax in enumerate(axes.flat):
        # Plot obstacles
        for obs in obstacles:
            x_coords, y_coords = obs.exterior.xy
            ax.fill(x_coords, y_coords, color='gray', alpha=0.7,
                   edgecolor='black', linewidth=2)
        
        # Plot start
        x, y, theta = start_state
        ax.plot(x, y, 'o', color='green', markersize=20,
               markeredgewidth=3, markeredgecolor='darkgreen', zorder=10)
        arrow = FancyArrow(x, y, 0.5*np.cos(theta), 0.5*np.sin(theta),
                          width=0.15, head_width=0.3, head_length=0.2,
                          fc='green', ec='darkgreen', linewidth=2, zorder=10)
        ax.add_patch(arrow)
        
        if idx == 0:
            # Plot 0: All reachable states colored by time
            title = "Expansion Wavefront (colored by time)"
            
            for discrete_state, info in state_info.items():
                x, y, theta = info['continuous']
                time_step = info['time_step']
                color_val = time_step / max_time
                
                ax.plot(x, y, 'o', markersize=4, 
                       color=plt.cm.viridis(color_val), alpha=0.6)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                       norm=plt.Normalize(0, max_time * planner.dt))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Time (seconds)')
            
        elif idx == 1:
            # Plot 1: Probability heatmap
            title = "Reachability Probability"
            
            if reachability_scores:
                xs = [p[0] for p in reachability_scores.keys()]
                ys = [p[1] for p in reachability_scores.keys()]
                scores = list(reachability_scores.values())
                
                scatter = ax.scatter(xs, ys, c=scores, s=80, cmap='hot',
                                   alpha=0.7, vmin=0, vmax=max(scores),
                                   edgecolors='black', linewidths=0.5)
                cbar = plt.colorbar(scatter, ax=ax, label='Probability')
        
        elif idx == 2:
            # Plot 2: Show specific time slices
            title = "Time Slices"
            
            time_slices = [0, max_time//3, 2*max_time//3, max_time]
            colors = ['red', 'orange', 'yellow', 'cyan']
            
            for t_slice, color in zip(time_slices, colors):
                states_at_time = [info['continuous'] for info in state_info.values()
                                 if info['time_step'] == t_slice]
                if states_at_time:
                    xs = [s[0] for s in states_at_time]
                    ys = [s[1] for s in states_at_time]
                    ax.scatter(xs, ys, c=color, s=60, alpha=0.7,
                             label=f't={t_slice*planner.dt:.1f}s',
                             edgecolors='black', linewidths=0.5)
            ax.legend(loc='upper right', fontsize=9)
        
        else:
            # Plot 3: Show expansion tree (sample)
            title = "Expansion Tree (sample paths)"
            
            # Draw sample trajectories
            sample_size = min(50, len(state_info))
            state_keys = list(state_info.keys())
            sampled_indices = np.random.choice(len(state_keys), 
                                            size=sample_size, replace=False)
            sampled_states = [state_keys[i] for i in sampled_indices]
            
            for discrete_state in sampled_states:
                # Trace back to start
                path = []
                current = discrete_state
                
                while current is not None:
                    info = state_info[current]
                    path.append(info['continuous'])
                    current = info.get('parent')
                
                if len(path) > 1:
                    path = path[::-1]  # Reverse
                    xs = [s[0] for s in path]
                    ys = [s[1] for s in path]
                    ax.plot(xs, ys, 'b-', alpha=0.2, linewidth=1)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('X (meters)', fontsize=11)
        ax.set_ylabel('Y (meters)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)
    
    plt.tight_layout()
    plt.savefig('progressive_expansion.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'progressive_expansion.png'")
    plt.show()


def create_animated_expansion(start_state, state_info, obstacles, planner, output_file='expansion.gif'):
    """
    Create animated GIF showing progressive expansion
    """
    import matplotlib.animation as animation
    
    print("\nCreating animation...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot obstacles
    for obs in obstacles:
        x_coords, y_coords = obs.exterior.xy
        ax.fill(x_coords, y_coords, color='gray', alpha=0.7,
               edgecolor='black', linewidth=2)
    
    # Plot start
    x, y, theta = start_state
    ax.plot(x, y, 'o', color='green', markersize=20,
           markeredgewidth=3, markeredgecolor='darkgreen', zorder=10)
    arrow = FancyArrow(x, y, 0.5*np.cos(theta), 0.5*np.sin(theta),
                      width=0.15, head_width=0.3, head_length=0.2,
                      fc='green', ec='darkgreen', linewidth=2, zorder=10)
    ax.add_patch(arrow)
    
    ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    
    # Organize states by time
    max_time = max(info['time_step'] for info in state_info.values())
    states_by_time = defaultdict(list)
    for discrete_state, info in state_info.items():
        states_by_time[info['time_step']].append(info['continuous'])
    
    scatter = ax.scatter([], [], c=[], s=60, cmap='hot', alpha=0.7, vmin=0, vmax=1)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=14, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scatter, time_text
    
    def animate(frame):
        # Accumulate states up to current frame
        all_states = []
        for t in range(frame + 1):
            all_states.extend(states_by_time[t])
        
        if all_states:
            positions = np.array([(s[0], s[1]) for s in all_states])
            colors = np.linspace(0, 1, len(all_states))
            
            scatter.set_offsets(positions)
            scatter.set_array(colors)
        
        time_text.set_text(f'Time: {frame * planner.dt:.1f}s\nStates: {len(all_states)}')
        return scatter, time_text
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=max_time + 1, interval=200,
                                  blit=True, repeat=True)
    
    # Save as GIF
    try:
        anim.save(output_file, writer='pillow', fps=5)
        print(f"Animation saved as '{output_file}'")
    except Exception as e:
        print(f"Could not save animation: {e}")
        print("Showing animation instead...")
        plt.show()


def analyze_expansion(reachability_scores, state_info, start_state):
    """Detailed analysis of expansion"""
    print("\n" + "="*60)
    print("EXPANSION ANALYSIS")
    print("="*60)
    
    if not reachability_scores:
        print("No reachable points!")
        return
    
    # Basic stats
    scores = np.array(list(reachability_scores.values()))
    points = np.array(list(reachability_scores.keys()))
    
    print(f"Total unique locations: {len(reachability_scores)}")
    print(f"Total unique states: {len(state_info)}")
    print(f"Mean probability: {np.mean(scores):.6f}")
    print(f"Max probability: {np.max(scores):.6f}")
    print(f"Min probability: {np.min(scores):.6f}")
    
    # Distance analysis
    x0, y0, _ = start_state
    distances = np.sqrt((points[:, 0] - x0)**2 + (points[:, 1] - y0)**2)
    print(f"\nReachable distance range:")
    print(f"  Min: {np.min(distances):.2f}m")
    print(f"  Max: {np.max(distances):.2f}m")
    print(f"  Mean: {np.mean(distances):.2f}m")
    
    # Time analysis
    time_steps = [info['time_step'] for info in state_info.values()]
    print(f"\nExpansion depth:")
    print(f"  Max time step: {max(time_steps)}")
    print(f"  States at final step: {sum(1 for t in time_steps if t == max(time_steps))}")
    
    # Best locations
    top_k = min(5, len(scores))
    top_indices = np.argsort(scores)[-top_k:][::-1]
    print(f"\nTop {top_k} reachable locations:")
    for i, idx in enumerate(top_indices):
        pt = points[idx]
        prob = scores[idx]
        dist = distances[idx]
        print(f"  {i+1}. ({pt[0]:.2f}, {pt[1]:.2f}) - prob: {prob:.6f}, dist: {dist:.2f}m")
    
    print("="*60)


def create_environment():
    """Create example environment"""
    obstacles = [
        Polygon([(3, 2), (6, 2), (6, 6), (3, 6)]),
        Polygon([(8, 4), (10, 5), (9, 7)]),
        Polygon([(1, 8), (3, 8), (3, 9), (2, 9), (2, 10), (1, 10)]),
        Polygon([(7, 0.5), (8, 0.5), (8, 1.5), (7, 1.5)]),
    ]
    return obstacles


def main():
    """Main execution"""
    print("="*60)
    print("PROGRESSIVE STATE EXPANSION REACHABILITY")
    print("="*60)
    
    # Environment
    print("\nSetting up environment...")
    obstacles = create_environment()
    
    # Robot parameters
    v_max = 2.0
    omega_max = 1.0
    time_horizon = 3.0
    start_state = (0.5, 0.5, np.pi/4)
    
    print(f"Robot: v_max={v_max}m/s, ω_max={omega_max}rad/s")
    print(f"Time horizon: {time_horizon}s")
    print(f"Start state: ({start_state[0]}, {start_state[1]}, {np.degrees(start_state[2]):.1f}°)")
    
    # Create planner
    planner = ProgressiveStateExpansion(
        v_max=v_max,
        omega_max=omega_max,
        time_horizon=time_horizon,
        obstacles=obstacles,
        grid_resolution=0.3,
        theta_resolution=8
    )
    
    # Compute reachability
    reachability_scores, state_info = planner.compute_reachability(start_state)
    
    # Analysis
    analyze_expansion(reachability_scores, state_info, start_state)
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_progressive_expansion(start_state, reachability_scores, 
                                   state_info, obstacles, planner)
    
    # Create animation
    create_animated_expansion(start_state, state_info, obstacles, planner)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()