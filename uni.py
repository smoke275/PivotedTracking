import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle, FancyArrow
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import time

class UnicycleReachability:
    """Fast reachability computation for unicycle in polygonal environment"""
    
    def __init__(self, v_max, omega_max, time_horizon, obstacles):
        """
        Args:
            v_max: maximum linear velocity
            omega_max: maximum angular velocity
            time_horizon: planning horizon
            obstacles: list of Shapely Polygon objects
        """
        self.v_max = v_max
        self.omega_max = omega_max
        self.T = time_horizon
        self.obstacles = obstacles
        self.obstacle_union = unary_union(obstacles)
        
    def compute_reachability(self, state, resolution=0.15):
        """
        Compute reachability distribution
        
        Args:
            state: (x, y, theta) current state
            resolution: grid resolution for sampling
            
        Returns:
            dict mapping (x, y) -> reachability score
        """
        x, y, theta = state
        
        print("Step 1: Computing max reachability circle...")
        max_reach = self.v_max * self.T
        
        print("Step 2: Computing visibility polygon...")
        t_start = time.time()
        visibility_poly = self.compute_visibility_polygon(x, y, max_reach)
        t_vis = time.time() - t_start
        print(f"  Visibility computed in {t_vis*1000:.1f}ms")
        
        print("Step 3: Sampling candidate points...")
        candidates = self.sample_polar_grid(x, y, theta, max_reach, resolution)
        print(f"  Generated {len(candidates)} candidate points")
        
        print("Step 4: Computing reachability scores...")
        t_start = time.time()
        reachability_scores = {}
        
        for (px, py, angle, radius) in candidates:
            # Check visibility (fast point-in-polygon test)
            point = Point(px, py)
            
            if not visibility_poly.contains(point):
                continue  # Not visible = not reachable
            
            # Compute unicycle kinematics
            angle_diff = self.normalize_angle(angle - theta)
            turn_time = abs(angle_diff) / self.omega_max
            drive_time = self.T - turn_time
            
            # Check if kinematically reachable
            if drive_time < 0:
                continue  # Can't even turn to face it
            
            max_distance = self.v_max * drive_time
            if radius > max_distance:
                continue  # Too far to reach
            
            # Compute reachability score
            cost = (radius / max_distance) + abs(angle_diff) / np.pi
            score = np.exp(-cost)
            
            if score > 0.01:  # Threshold to avoid clutter
                reachability_scores[(px, py)] = score
        
        t_score = time.time() - t_start
        print(f"  Scores computed in {t_score*1000:.1f}ms")
        
        # Normalize to probability distribution
        total = sum(reachability_scores.values())
        if total > 0:
            reachability_scores = {k: v/total for k, v in reachability_scores.items()}
        
        print(f"Total reachable points: {len(reachability_scores)}")
        
        return reachability_scores, visibility_poly
    
    def compute_visibility_polygon(self, x, y, max_radius):
        """
        Compute visibility polygon using ray casting
        
        Returns: Shapely Polygon representing visible area
        """
        num_rays = 360  # 1 degree resolution
        angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
        
        hit_points = []
        
        for angle in angles:
            # Ray direction
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Default hit point (max reach if no obstacle)
            hit_x = x + max_radius * dx
            hit_y = y + max_radius * dy
            min_dist = max_radius
            
            # Check intersection with all obstacle edges
            for obstacle in self.obstacles:
                coords = list(obstacle.exterior.coords)
                
                for i in range(len(coords) - 1):
                    edge_start = coords[i]
                    edge_end = coords[i + 1]
                    
                    # Compute ray-segment intersection
                    intersection = self.ray_segment_intersection(
                        x, y, dx, dy, edge_start, edge_end
                    )
                    
                    if intersection is not None:
                        dist = np.sqrt((intersection[0] - x)**2 + (intersection[1] - y)**2)
                        if dist < min_dist:
                            min_dist = dist
                            hit_x, hit_y = intersection
            
            hit_points.append((hit_x, hit_y))
        
        return Polygon(hit_points)
    
    def ray_segment_intersection(self, px, py, dx, dy, seg_start, seg_end):
        """
        Compute intersection between ray and line segment
        
        Ray: P + t*D where t >= 0
        Segment: S + s*(E-S) where 0 <= s <= 1
        """
        x1, y1 = seg_start
        x2, y2 = seg_end
        
        denom = dx * (y2 - y1) - dy * (x2 - x1)
        
        if abs(denom) < 1e-10:  # Parallel
            return None
        
        t = ((x1 - px) * (y2 - y1) - (y1 - py) * (x2 - x1)) / denom
        s = ((x1 - px) * dy - (y1 - py) * dx) / denom
        
        if t >= 0.001 and 0 <= s <= 1:  # Small epsilon to avoid self-intersection
            return (px + t * dx, py + t * dy)
        
        return None
    
    def sample_polar_grid(self, x, y, theta, max_radius, resolution):
        """
        Sample points in polar coordinates around agent
        
        Returns: list of (px, py, angle, radius)
        """
        candidates = []
        
        # Radial samples
        r_samples = np.arange(0.1, max_radius, resolution)
        
        # Angular samples (finer resolution close to heading)
        angles = []
        
        # Dense sampling in front (±60 degrees)
        front_angles = np.linspace(-np.pi/3, np.pi/3, 40)
        angles.extend(front_angles)
        
        # Sparse sampling to the sides and back
        side_angles = np.linspace(np.pi/3, 5*np.pi/3, 30)
        angles.extend(side_angles)
        
        angles = np.array(angles)
        
        for r in r_samples:
            for angle in angles:
                px = x + r * np.cos(angle)
                py = y + r * np.sin(angle)
                candidates.append((px, py, angle, r))
        
        return candidates
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def create_environment():
    """Create example polygonal environment"""
    obstacles = [
        # Large building
        Polygon([(3, 2), (6, 2), (6, 6), (3, 6)]),
        
        # Triangle obstacle
        Polygon([(8, 4), (10, 5), (9, 7)]),
        
        # L-shaped obstacle
        Polygon([(1, 8), (3, 8), (3, 9), (2, 9), (2, 10), (1, 10)]),
        
        # Small square
        Polygon([(7, 0.5), (8, 0.5), (8, 1.5), (7, 1.5)]),
    ]
    return obstacles


def visualize_reachability(state, reachability_scores, visibility_poly, obstacles, v_max, T):
    """Create comprehensive visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    for ax in [ax1, ax2]:
        # Plot obstacles
        for obs in obstacles:
            x_coords, y_coords = obs.exterior.xy
            ax.fill(x_coords, y_coords, color='gray', alpha=0.7, 
                   edgecolor='black', linewidth=2, label='_nolegend_')
        
        # Plot agent
        x, y, theta = state
        ax.plot(x, y, 'o', color='green', markersize=18, 
               markeredgewidth=2, markeredgecolor='darkgreen', 
               label='Robot', zorder=10)
        
        # Draw heading arrow
        arrow_length = 0.8
        arrow = FancyArrow(x, y, 
                          arrow_length * np.cos(theta),
                          arrow_length * np.sin(theta),
                          width=0.15, head_width=0.4, head_length=0.3,
                          fc='green', ec='darkgreen', linewidth=2, zorder=10)
        ax.add_patch(arrow)
        
        # Draw max reachability circle
        max_reach = v_max * T
        circle = Circle((x, y), max_reach, color='blue', fill=False,
                       linestyle='--', linewidth=2, alpha=0.5, 
                       label='Max Reach Circle')
        ax.add_patch(circle)
        
        ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axis('equal')
        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)
    
    # Left plot: Visibility polygon
    if visibility_poly is not None:
        vis_x, vis_y = visibility_poly.exterior.xy
        ax1.fill(vis_x, vis_y, color='yellow', alpha=0.2, 
                edgecolor='orange', linewidth=2, linestyle='--',
                label='Visible Region')
    
    ax1.set_title('Visibility Polygon\n(Geometric Reachability)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    
    # Right plot: Reachability heatmap
    if reachability_scores:
        xs = [p[0] for p in reachability_scores.keys()]
        ys = [p[1] for p in reachability_scores.keys()]
        scores = list(reachability_scores.values())
        
        scatter = ax2.scatter(xs, ys, c=scores, s=60, cmap='hot',
                            alpha=0.7, vmin=0, vmax=max(scores), 
                            edgecolors='black', linewidths=0.5, zorder=5)
        cbar = plt.colorbar(scatter, ax=ax2, label='Reachability Probability')
        cbar.set_label('Reachability Probability', fontsize=11, fontweight='bold')
    
    ax2.set_title('Reachability Distribution\n(Kinematic + Visibility)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('unicycle_reachability.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'unicycle_reachability.png'")
    plt.show()


def plot_reachability_3d(reachability_scores, state, obstacles):
    """Create 3D surface plot of reachability"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    if reachability_scores:
        xs = np.array([p[0] for p in reachability_scores.keys()])
        ys = np.array([p[1] for p in reachability_scores.keys()])
        scores = np.array(list(reachability_scores.values()))
        
        # Create surface
        scatter = ax.scatter(xs, ys, scores, c=scores, cmap='hot', 
                           s=20, alpha=0.6)
        
        # Add obstacles as vertical bars
        for obs in obstacles:
            x_coords, y_coords = obs.exterior.xy
            x_coords = list(x_coords)
            y_coords = list(y_coords)
            for i in range(len(x_coords) - 1):
                ax.plot([x_coords[i], x_coords[i]], 
                       [y_coords[i], y_coords[i]], 
                       [0, max(scores) * 1.1], 
                       'k-', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('X (meters)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Reachability Probability', fontsize=11, fontweight='bold')
        ax.set_title('3D Reachability Distribution', fontsize=14, fontweight='bold')
        
        plt.colorbar(scatter, ax=ax, label='Probability', shrink=0.5)
    
    plt.tight_layout()
    plt.savefig('unicycle_reachability_3d.png', dpi=150, bbox_inches='tight')
    print("3D visualization saved as 'unicycle_reachability_3d.png'")
    plt.show()


def analyze_reachability(reachability_scores, state):
    """Print statistics about reachability"""
    if not reachability_scores:
        print("No reachable points found!")
        return
    
    scores = np.array(list(reachability_scores.values()))
    points = np.array(list(reachability_scores.keys()))
    
    print("\n" + "="*60)
    print("REACHABILITY ANALYSIS")
    print("="*60)
    print(f"Total reachable points: {len(reachability_scores)}")
    print(f"Mean probability: {np.mean(scores):.6f}")
    print(f"Max probability: {np.max(scores):.6f}")
    print(f"Min probability: {np.min(scores):.6f}")
    print(f"Std deviation: {np.std(scores):.6f}")
    
    # Find highest probability point
    max_idx = np.argmax(scores)
    best_point = points[max_idx]
    print(f"\nHighest probability point: ({best_point[0]:.2f}, {best_point[1]:.2f})")
    print(f"  Probability: {scores[max_idx]:.6f}")
    
    # Analyze by sectors
    x, y, theta = state
    angles = np.arctan2(points[:, 1] - y, points[:, 0] - x)
    angle_diffs = np.abs(angles - theta)
    angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
    
    front = angle_diffs < np.pi/4
    side = (angle_diffs >= np.pi/4) & (angle_diffs < 3*np.pi/4)
    back = angle_diffs >= 3*np.pi/4
    
    print(f"\nReachability by sector:")
    print(f"  Front (±45°): {np.sum(front)} points, avg prob: {np.mean(scores[front]):.6f}")
    print(f"  Sides: {np.sum(side)} points, avg prob: {np.mean(scores[side]) if np.sum(side) > 0 else 0:.6f}")
    print(f"  Back: {np.sum(back)} points, avg prob: {np.mean(scores[back]) if np.sum(back) > 0 else 0:.6f}")
    print("="*60)


def main():
    """Main execution"""
    print("="*60)
    print("UNICYCLE REACHABILITY IN POLYGONAL ENVIRONMENT")
    print("="*60)
    
    # Setup environment
    print("\nCreating environment...")
    obstacles = create_environment()
    print(f"Environment has {len(obstacles)} obstacles")
    
    # Robot parameters
    v_max = 2.0  # m/s
    omega_max = 1.0  # rad/s
    time_horizon = 3.0  # seconds
    
    # Initial state
    state = (0.5, 0.5, np.pi/4)  # (x, y, theta)
    
    print(f"\nRobot parameters:")
    print(f"  Max linear velocity: {v_max} m/s")
    print(f"  Max angular velocity: {omega_max} rad/s")
    print(f"  Time horizon: {time_horizon} s")
    print(f"  Max reach distance: {v_max * time_horizon} m")
    print(f"\nInitial state: x={state[0]}, y={state[1]}, θ={state[2]:.2f} rad ({np.degrees(state[2]):.1f}°)")
    
    # Create planner
    planner = UnicycleReachability(v_max, omega_max, time_horizon, obstacles)
    
    # Compute reachability
    print("\n" + "-"*60)
    print("COMPUTING REACHABILITY...")
    print("-"*60)
    total_start = time.time()
    
    reachability_scores, visibility_poly = planner.compute_reachability(
        state, resolution=0.15
    )
    
    total_time = time.time() - total_start
    print("-"*60)
    print(f"TOTAL COMPUTATION TIME: {total_time*1000:.1f}ms")
    print("-"*60)
    
    # Analysis
    analyze_reachability(reachability_scores, state)
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_reachability(state, reachability_scores, visibility_poly, 
                          obstacles, v_max, time_horizon)
    plot_reachability_3d(reachability_scores, state, obstacles)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()