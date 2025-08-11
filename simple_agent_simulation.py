#!/usr/bin/env python3
"""
Simple Agent Simulation
A standalone simulation that loads environment and agents for basic movement control.
Independent of the inspection tools - creates its own simulation environment.

Key Functionality Integration:

- Simple agent movement and position tracking
- Distance calculation between agents
- Basic environment visualization
- Reachability mask visualization for evader agent
"""

import pygame
import sys
import os
import math
import pickle
import time
import numpy as np

# Make sure the project directory is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import required modules
from multitrack.models.simulation_environment import SimulationEnvironment
from multitrack.models.agents.visitor_agent import UnicycleModel
from multitrack.utils.map_graph import MapGraph
from multitrack.utils.config import (MAP_GRAPH_INSPECTION_CACHE_FILE, MAP_GRAPH_CACHE_ENABLED,
                                   MAP_GRAPH_VISIBILITY_CACHE_FILE, LEADER_LINEAR_VEL, LEADER_ANGULAR_VEL,
                                   MAP_GRAPH_NODE_COLOR, MAP_GRAPH_EDGE_COLOR)
from position_evaluator import (global_position_evaluator, update_position, get_distance, get_stats, 
                              set_environment_data, find_closest_node, get_agent_rrt_tree, 
                              update_agent_rrt_tree, update_all_rrt_trees, set_rrt_parameters,
                              get_path_to_node, find_node_at_position, clear_closest_node_cache,
                              set_closest_node_cache_parameters, get_closest_node_cache_stats,
                              force_rebuild_spatial_index, map_rrt_nodes_to_graph, 
                              map_all_rrt_nodes_to_graph, get_rrt_to_graph_mapping_stats,
                              calculate_pursuit_evasion_advantages, get_pursuit_evasion_stats)

# Import trajectory optimization system
from path_trajectory_optimizer import (initialize_trajectory_integrator, generate_trajectory_for_path,
                                     get_current_trajectory, get_trajectory_info, clear_trajectory,
                                     initialize_trajectory_calculator, get_trajectory_calculator,
                                     calculate_all_travel_times)

# Import risk calculator for reachability analysis and visibility calculations
from risk_calculator import (load_reachability_mask, get_reachability_probabilities_for_fixed_grid,
                            calculate_evader_visibility, get_visibility_statistics, calculate_visibility_sectors,
                            detect_visibility_breakoff_points)

def draw_evader_visibility(screen, agent_x, agent_y, visibility_data, show_rays=True, show_visibility_area=True, agent_theta=None):
    """
    Draw the evader's visibility on screen with orientation-based breakoff point coloring.
    
    Args:
        screen: Pygame screen to draw on
        agent_x, agent_y: Evader position
        visibility_data: List from calculate_evader_visibility()
        show_rays: Whether to show individual rays
        show_visibility_area: Whether to show the visibility polygon
        agent_theta: Agent orientation in radians (for breakoff point orientation coloring)
    """
    if not visibility_data:
        return
    
    # Colors for visualization
    ray_color = (0, 255, 255, 100)      # Cyan, semi-transparent
    blocked_ray_color = (255, 100, 100, 150)  # Red, more visible
    visibility_color = (0, 255, 255, 30)     # Light cyan, very transparent
    breakoff_color = (255, 255, 0, 200)     # Yellow, highly visible for breakoff points
    breakoff_line_color = (255, 165, 0, 180) # Orange for breakoff lines
    
    # Colors for 4 categories of breakoff points based on orientation and distance transition
    category_colors = {
        # Clockwise transitions
        'clockwise_near_far': {
            'point': (255, 50, 50),      # Bright red for clockwise near-to-far
            'line': (255, 100, 100, 180), # Light red for lines
            'middle': (200, 40, 40)       # Darker red for middle circle
        },
        'clockwise_far_near': {
            'point': (255, 150, 50),     # Orange-red for clockwise far-to-near  
            'line': (255, 180, 100, 180), # Light orange-red for lines
            'middle': (200, 120, 40)      # Darker orange-red for middle circle
        },
        # Counterclockwise transitions
        'counterclockwise_near_far': {
            'point': (50, 255, 50),      # Bright green for counterclockwise near-to-far
            'line': (100, 255, 100, 180), # Light green for lines
            'middle': (40, 200, 40)       # Darker green for middle circle
        },
        'counterclockwise_far_near': {
            'point': (50, 255, 150),     # Blue-green for counterclockwise far-to-near
            'line': (100, 255, 180, 180), # Light blue-green for lines
            'middle': (40, 200, 120)      # Darker blue-green for middle circle
        },
        # Fallback colors
        'unknown_near_far': {
            'point': (255, 255, 0),      # Yellow fallback
            'line': (255, 255, 100, 180), # Light yellow for lines
            'middle': (200, 200, 0)       # Darker yellow for middle circle
        },
        'unknown_far_near': {
            'point': (255, 200, 0),      # Orange-yellow fallback
            'line': (255, 220, 100, 180), # Light orange-yellow for lines
            'middle': (200, 160, 0)       # Darker orange-yellow for middle circle
        }
    }
    
    # Detect visibility breakoff points using the API from risk_calculator with orientation
    breakoff_points, breakoff_lines = detect_visibility_breakoff_points(
        visibility_data, 
        min_gap_distance=30,
        agent_x=agent_x,
        agent_y=agent_y, 
        agent_theta=agent_theta
    )
    
    # Draw visibility polygon
    if show_visibility_area and len(visibility_data) > 2:
        # Create a surface for transparent drawing
        visibility_surface = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
        
        # Create polygon points from ray endpoints
        polygon_points = [(int(agent_x), int(agent_y))]  # Start at agent
        for angle, endpoint, distance, blocked in visibility_data:
            polygon_points.append((int(endpoint[0]), int(endpoint[1])))
        
        # Draw filled polygon
        if len(polygon_points) > 3:
            pygame.draw.polygon(visibility_surface, visibility_color, polygon_points)
            screen.blit(visibility_surface, (0, 0))
    
    # Draw individual rays
    if show_rays:
        for angle, endpoint, distance, blocked in visibility_data:
            color = blocked_ray_color if blocked else ray_color
            
            # Create surface for transparent line
            line_surface = pygame.Surface((abs(int(endpoint[0] - agent_x)) + 2, 
                                         abs(int(endpoint[1] - agent_y)) + 2), pygame.SRCALPHA)
            
            # Calculate line position on surface
            start_pos = (max(1, int(agent_x - min(agent_x, endpoint[0]))), 
                        max(1, int(agent_y - min(agent_y, endpoint[1]))))
            end_pos = (max(1, int(endpoint[0] - min(agent_x, endpoint[0]))), 
                      max(1, int(endpoint[1] - min(agent_y, endpoint[1]))))
            
            # Draw line
            pygame.draw.line(line_surface, color, start_pos, end_pos, 1)
            
            # Blit to main screen
            screen.blit(line_surface, (min(int(agent_x), int(endpoint[0])), 
                                     min(int(agent_y), int(endpoint[1]))))
    
    # Draw breakoff/gap lines connecting points where visibility changes abruptly with category-based coloring
    for line_data in breakoff_lines:
        if len(line_data) == 4:  # New format with category
            start_point, end_point, gap_size, category = line_data
            # Choose color based on category
            if category in category_colors:
                line_color = category_colors[category]['line']
            else:
                line_color = breakoff_line_color  # Fallback to default orange
        else:  # Old format without category (fallback compatibility)
            start_point, end_point, gap_size = line_data[:3]
            line_color = breakoff_line_color
        
        # Create surface for transparent gap line
        line_width = min(int(gap_size / 10), 5) + 2  # Thicker lines for larger gaps
        gap_surface = pygame.Surface((abs(int(end_point[0] - start_point[0])) + line_width * 2, 
                                     abs(int(end_point[1] - start_point[1])) + line_width * 2), pygame.SRCALPHA)
        
        # Calculate line position on surface
        start_pos = (max(line_width, int(start_point[0] - min(start_point[0], end_point[0]))), 
                    max(line_width, int(start_point[1] - min(start_point[1], end_point[1]))))
        end_pos = (max(line_width, int(end_point[0] - min(start_point[0], end_point[0]))), 
                  max(line_width, int(end_point[1] - min(start_point[1], end_point[1]))))
        
        # Draw gap line with category-based color
        pygame.draw.line(gap_surface, line_color, start_pos, end_pos, line_width)
        
        # Blit to main screen
        screen.blit(gap_surface, (min(int(start_point[0]), int(end_point[0])) - line_width, 
                                 min(int(start_point[1]), int(end_point[1])) - line_width))
    
    # Draw breakoff points as highlighted circles with category-based coloring
    for point_data in breakoff_points:
        if len(point_data) == 3:  # New format with category
            x, y, category = point_data
            # Choose colors based on category
            if category in category_colors:
                outer_color = category_colors[category]['point']
                middle_color = category_colors[category]['middle']
            else:
                outer_color = (255, 255, 0)   # Yellow fallback
                middle_color = (255, 165, 0)  # Orange fallback
        else:  # Old format without category (fallback compatibility)
            x, y = point_data[:2]
            outer_color = (255, 255, 0)   # Yellow
            middle_color = (255, 165, 0)  # Orange
        
        # Draw a small circle at each breakoff point with category-based coloring
        pygame.draw.circle(screen, outer_color, (int(x), int(y)), 6)      # Colored outer
        pygame.draw.circle(screen, middle_color, (int(x), int(y)), 4)     # Darker middle
        pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 2) # White center

def draw_reachability_grid_overlay(screen, mask_data, agent_x, agent_y, agent_theta, alpha=128):
    """Draw the reachability grid overlay around the agent (optimized for performance)."""
    if mask_data is None:
        return
    
    # Get grid properties
    grid_size = mask_data.get('grid_size', 0)
    cell_size = mask_data.get('cell_size_px', 1)
    
    if grid_size == 0:
        return
    
    # Get reachability probabilities for the fixed grid using the API
    fixed_grid_probabilities = get_reachability_probabilities_for_fixed_grid(
        agent_x, agent_y, agent_theta, mask_data)
    
    # Calculate extent in world coordinates centered on agent
    half_extent_px = (grid_size * cell_size) / 2
    
    # Determine display scaling to fit nicely on screen
    max_display_size = 300  # Maximum pixel size for the overlay
    display_cell_size = max(1, min(max_display_size // grid_size, int(cell_size)))
    
    # Create a surface for the grid visualization
    surface_size = grid_size * display_cell_size
    grid_surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
    
    # Find max intensity for normalization
    max_intensity = fixed_grid_probabilities.max() if fixed_grid_probabilities.max() > 0 else 1.0
    
    # Flatten the probability grid for vectorized processing
    flat_probabilities = fixed_grid_probabilities.flatten()
    norm_intensities = flat_probabilities / max_intensity
    
    # Initialize color arrays for all cells
    total_cells = grid_size * grid_size
    r_values = np.zeros(total_cells, dtype=np.uint8)
    g_values = np.zeros(total_cells, dtype=np.uint8)
    b_values = np.zeros(total_cells, dtype=np.uint8)
    
    # Vectorized 'hot' colormap calculation for transparent-to-red-to-yellow-to-white
    # Transparent to red (norm_intensity < 0.33) - red increases with intensity
    mask1 = norm_intensities < 0.33
    r_values[mask1] = 255  # Always full red for consistency with previous implementation
    
    # Red to yellow (0.33 <= norm_intensity < 0.66)
    mask2 = (norm_intensities >= 0.33) & (norm_intensities < 0.66)
    r_values[mask2] = 255
    g_values[mask2] = ((norm_intensities[mask2] - 0.33) * 3 * 255).astype(np.uint8)
    
    # Yellow to white (norm_intensity >= 0.66)
    mask3 = norm_intensities >= 0.66
    r_values[mask3] = 255
    g_values[mask3] = 255
    b_values[mask3] = ((norm_intensities[mask3] - 0.66) * 3 * 255).astype(np.uint8)
    
    # Vectorized alpha calculation for transparent-to-red transition
    alpha_values = np.full(total_cells, alpha, dtype=np.uint8)
    low_intensity_mask = norm_intensities < 0.33
    alpha_values[low_intensity_mask] = (norm_intensities[low_intensity_mask] * 3 * alpha).astype(np.uint8)
    
    # Optimized drawing using direct array indexing
    if display_cell_size == 1:
        # For single-pixel cells, use direct pixel access (fastest method)
        try:
            pixel_array = pygame.surfarray.pixels_alpha(grid_surface)
            color_array = pygame.surfarray.pixels3d(grid_surface)
            
            # Set colors and alpha values directly using reshaped arrays
            color_array[:, :, 0] = r_values.reshape(grid_size, grid_size).T
            color_array[:, :, 1] = g_values.reshape(grid_size, grid_size).T
            color_array[:, :, 2] = b_values.reshape(grid_size, grid_size).T
            pixel_array[:, :] = alpha_values.reshape(grid_size, grid_size).T
            
            # Clean up array references
            del pixel_array, color_array
        except:
            # Fallback to rect drawing if pixel array access fails
            for i in range(total_cells):
                row = i // grid_size
                col = i % grid_size
                display_x = col * display_cell_size
                display_y = row * display_cell_size
                color = (r_values[i], g_values[i], b_values[i], alpha_values[i])
                cell_rect = pygame.Rect(display_x, display_y, display_cell_size, display_cell_size)
                pygame.draw.rect(grid_surface, color, cell_rect)
    else:
        # For larger cells, batch the rect operations for better performance
        rects_and_colors = []
        for i in range(total_cells):
            row = i // grid_size
            col = i % grid_size
            display_x = col * display_cell_size
            display_y = row * display_cell_size
            color = (r_values[i], g_values[i], b_values[i], alpha_values[i])
            cell_rect = pygame.Rect(display_x, display_y, display_cell_size, display_cell_size)
            rects_and_colors.append((cell_rect, color))
        
        # Draw all rects in batches to reduce function call overhead
        batch_size = 100  # Draw in batches of 100 for optimal performance
        for batch_start in range(0, len(rects_and_colors), batch_size):
            batch_end = min(batch_start + batch_size, len(rects_and_colors))
            for cell_rect, color in rects_and_colors[batch_start:batch_end]:
                pygame.draw.rect(grid_surface, color, cell_rect)
    
    # Calculate position to center the grid on the agent in screen coordinates
    screen_x = int(agent_x - surface_size // 2)
    screen_y = int(agent_y - surface_size // 2)
    
    # Blit the grid surface to the main screen centered on agent
    screen.blit(grid_surface, (screen_x, screen_y))

# Node visualization thresholds
DISTANCE_THRESHOLD = 50.0  # Distance threshold in pixels - nodes closer than this are more transparent
TIME_THRESHOLD = 100.0       # Time threshold in seconds - nodes taking longer than this are marked invalid and more transparent

# Global variables to cache strategic analysis results
cached_worst_nodes = {}

def get_worst_pursuit_nodes():
    """
    Get the RRT nodes with the worst (lowest) time advantages for Agent 1 as the pursuer.
    Uses cached results from when strategic analysis was last run.
    Agent 1 is designated as the pursuer, Agent 2 as the evader.
    
    Returns:
        dict: {agent_id: {'worst': (node_index, time_advantage, node_object), 
                         'worst_10': [(node_index, time_advantage, node_object), ...]}} 
              for the worst nodes of the pursuer
    """
    global cached_worst_nodes
    
    # Simply return the cached results - don't run any analysis here
    return cached_worst_nodes.copy()

def update_worst_pursuit_nodes():
    """
    Update the cached worst pursuit nodes by running the analysis.
    This should only be called when the P key is pressed.
    Only Agent 1 is considered the pursuer, so only Agent 1's worst positions are highlighted.
    Caches both the single worst node and the worst 10 nodes.
    """
    global cached_worst_nodes
    cached_worst_nodes = {}
    
    try:
        # Check if strategic analysis has been run by checking if stats are available
        agent1_stats = get_pursuit_evasion_stats("agent1", "agent2")
        
        # Only proceed if we have active analysis results for Agent 1 as pursuer
        if agent1_stats.get("status") != "Active":
            return  # No analysis has been run yet
        
        # Get travel times for Agent 1 to filter out nodes that exceed time threshold
        travel_times_data = get_trajectory_calculator().get_travel_times_async("agent1")
        valid_node_indices = set()
        
        if travel_times_data:
            # Get the actual RRT tree for node index mapping
            agent1_tree = get_agent_rrt_tree("agent1")
            if agent1_tree:
                # Build a set of valid node indices (those below time threshold)
                for node, travel_time, _ in travel_times_data:
                    if travel_time <= TIME_THRESHOLD:
                        # Find the node index in the tree
                        for idx, tree_node in enumerate(agent1_tree):
                            if tree_node.x == node.x and tree_node.y == node.y:
                                valid_node_indices.add(idx)
                                break
        
        # Agent1 as pursuer vs Agent2 as evader (only calculate for the pursuer)
        agent1_advantages = calculate_pursuit_evasion_advantages("agent1", "agent2")
        if agent1_advantages:
            # Filter advantages to only include nodes below time threshold
            if valid_node_indices:
                filtered_advantages = {
                    node_idx: advantage for node_idx, advantage in agent1_advantages.items()
                    if node_idx in valid_node_indices
                }
                print(f"Strategic analysis: Filtered {len(agent1_advantages)} total nodes to {len(filtered_advantages)} valid nodes (≤{TIME_THRESHOLD:.1f}s)")
            else:
                # If no travel time data available, use all nodes (fallback behavior)
                filtered_advantages = agent1_advantages
                print(f"Strategic analysis: No travel time data available, using all {len(agent1_advantages)} nodes")
            
            if not filtered_advantages:
                print("Strategic analysis: No valid nodes found below time threshold")
                return
            
            # Sort filtered nodes by time advantage (lowest first = worst for pursuer)
            sorted_nodes = sorted(filtered_advantages.items(), key=lambda x: x[1])
            
            # Get the actual RRT node objects
            agent1_tree = get_agent_rrt_tree("agent1")
            if agent1_tree:
                # Cache the single worst node from valid nodes
                worst_node_idx, worst_advantage = sorted_nodes[0]
                if worst_node_idx < len(agent1_tree):
                    worst_node = agent1_tree[worst_node_idx]
                    
                    # Cache the worst 10 nodes (or fewer if less than 10 valid nodes available)
                    worst_10_nodes = []
                    for i, (node_idx, advantage) in enumerate(sorted_nodes[:10]):
                        if node_idx < len(agent1_tree):
                            node = agent1_tree[node_idx]
                            worst_10_nodes.append((node_idx, advantage, node))
                    
                    cached_worst_nodes["agent1"] = {
                        'worst': (worst_node_idx, worst_advantage, worst_node),
                        'worst_10': worst_10_nodes
                    }
        
        worst_count = len(cached_worst_nodes.get("agent1", {}).get('worst_10', []))
        valid_count = len(valid_node_indices) if valid_node_indices else "all"
        print(f"Updated worst pursuit nodes cache: Agent 1 (pursuer) has 1 worst node + {worst_count} worst nodes cached from {valid_count} valid nodes")
        
    except Exception as e:
        print(f"Error updating worst pursuit nodes: {e}")

def main():
    global trajectory_calculator  # Make it accessible globally
    
    # Initialize Pygame
    pygame.init()
    
    # Screen dimensions (matching the original inspection tool)
    SIDEBAR_WIDTH = 250  # Width of the information sidebar
    ENVIRONMENT_WIDTH = 1280  # Width of the environment area
    ENVIRONMENT_HEIGHT = 720  # Height of the environment area
    WINDOW_WIDTH = ENVIRONMENT_WIDTH + SIDEBAR_WIDTH  # Total window width including sidebar
    WINDOW_HEIGHT = ENVIRONMENT_HEIGHT  # Window height
    
    # Set up the display
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Simple Agent Simulation")
    
    # Initialize font
    font = pygame.font.SysFont('Arial', 12)  # Smaller font size
    
    # Create the environment
    print("Creating simulation environment...")
    environment = SimulationEnvironment(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT)
    
    # Initialize map graph and try to load from cache
    print("Initializing map graph...")
    map_graph = MapGraph(
        ENVIRONMENT_WIDTH, 
        ENVIRONMENT_HEIGHT, 
        environment.get_all_walls(), 
        environment.get_doors(),
        cache_file=MAP_GRAPH_INSPECTION_CACHE_FILE  # Use inspection-specific cache file
    )
    
    # Try to load map graph from cache
    map_graph_loaded = False
    if MAP_GRAPH_CACHE_ENABLED:
        print("Attempting to load map graph from cache...")
        map_graph_loaded = map_graph.load_from_cache()
        if map_graph_loaded:
            print(f"Successfully loaded map graph from cache with {len(map_graph.nodes)} nodes and {len(map_graph.edges)} edges.")
        else:
            print("No cached map graph found. Map graph will be empty (use inspect_environment.py to generate one).")
    else:
        print("Map graph caching is disabled. Map graph will be empty.")
    
    # Set up position evaluator with environment data
    print("Configuring position evaluator with environment data...")
    set_environment_data(environment, map_graph if map_graph_loaded else None)
    print("Position evaluator configured with environment and map graph data")
    
    # Configure closest node cache parameters for optimal performance
    print("Configuring closest node cache parameters...")
    set_closest_node_cache_parameters(
        movement_threshold=12.0,  # Recalculate when agent moves 12 pixels
        time_threshold=1.5        # Recalculate after 1.5 seconds max
    )
    print("Closest node cache configured: 12px movement threshold, 1.5s time threshold")
    
    # Configure RRT* parameters for 500 nodes with forward bias
    print("Setting RRT* parameters with forward bias...")
    set_rrt_parameters(
        max_nodes=200, 
        step_size=10.0, 
        search_radius=25.0,
        forward_bias=0.7,  # 70% of samples biased forward
        forward_cone_angle=math.pi / 2  # 60 degree forward cone
    )
    print("RRT* parameters configured: 200 nodes, 10.0 step size, 25.0 search radius, 70% forward bias, 60° cone")

    # Initialize trajectory optimization system
    print("Initializing trajectory optimization system...")
    initialize_trajectory_integrator(
        max_velocity=LEADER_LINEAR_VEL,    # Use actual agent max velocity (50.0 pixels/second)
        max_acceleration=50.0,             # pixels/second² - reasonable acceleration limit
        max_turning_rate=LEADER_ANGULAR_VEL # Use actual agent max angular velocity (1.0 rad/s)
    )
    print(f"Trajectory optimizer initialized with max_vel={LEADER_LINEAR_VEL}, max_accel=50.0, max_turn_rate={LEADER_ANGULAR_VEL}")

    # Initialize threaded trajectory calculator for high-performance async calculations with multiprocessing
    print("Initializing threaded trajectory calculator with multiprocessing support...")
    trajectory_calculator = initialize_trajectory_calculator(max_workers=10)
    print("Threaded trajectory calculator initialized with 4 worker threads")

    # Agent configuration
    AGENT_LINEAR_VEL = LEADER_LINEAR_VEL
    AGENT_ANGULAR_VEL = LEADER_ANGULAR_VEL
    AGENT_COLOR = (255, 0, 255)  # Magenta for agent 1
    AGENT2_COLOR = (0, 255, 255)  # Cyan for agent 2
    AGENT_RADIUS = 16
    
    # Create agents
    print("Creating agents...")
    
    # Agent 1 - controllable with arrow keys
    agent1 = UnicycleModel(
        initial_position=(100, 100),
        walls=environment.get_all_walls(), 
        doors=environment.get_doors()
    )
    
    # Agent 2 - controllable with WASD keys  
    agent2 = UnicycleModel(
        initial_position=(200, 200),
        walls=environment.get_all_walls(), 
        doors=environment.get_doors()
    )
    
    # Try to load agent states if they exist
    AGENT_STATE_FILE = 'agent_state.pkl'
    AGENT2_STATE_FILE = 'agent2_state.pkl'
    
    if os.path.exists(AGENT_STATE_FILE):
        try:
            with open(AGENT_STATE_FILE, 'rb') as f:
                agent_state = pickle.load(f)
                agent1.state = agent_state.copy()
                print("Loaded agent 1 state from file")
        except Exception as e:
            print(f"Could not load agent 1 state: {e}")
    
    if os.path.exists(AGENT2_STATE_FILE):
        try:
            with open(AGENT2_STATE_FILE, 'rb') as f:
                agent2_state = pickle.load(f)
                agent2.state = agent2_state.copy()
                print("Loaded agent 2 state from file")
        except Exception as e:
            print(f"Could not load agent 2 state: {e}")
    
    # Clock for controlling frame rate
    clock = pygame.time.Clock()
    
    # Position evaluator update timing
    position_update_interval = 0.5  # Update position evaluator every 0.5 seconds
    last_position_update = time.time()
    
    # Travel time calculation timing
    travel_time_update_interval = 2.0  # Update travel times every 2 seconds
    last_travel_time_update = time.time()
    
    # Main simulation loop
    running = True
    show_info = True
    show_map_graph = False  # Start with map graph hidden by default
    show_rrt_trees = True   # Show RRT* trees by default
    show_trajectory = True  # Show optimized trajectory by default
    
    # Reachability mask variables
    show_reachability_mask = False  # Start with reachability mask hidden
    mask_data = None   # Will store loaded mask data
    
    # Visibility system variables
    show_evader_visibility = False  # Start with visibility hidden
    visibility_range = 200.0        # Maximum visibility distance
    visibility_data = []            # Will store visibility ray data
    visibility_update_interval = 0.1  # Update visibility every 0.1 seconds
    last_visibility_update = time.time()
    show_visibility_rays = True     # Show individual rays
    show_visibility_area = True     # Show visibility polygon
    
    # Load reachability mask
    print("Loading reachability mask...")
    mask_data = load_reachability_mask("unicycle_grid")
    if mask_data is not None:
        grid_size = mask_data.get('grid_size', 'unknown')
        world_extent = mask_data.get('world_extent_px', 0)
        cell_size = mask_data.get('cell_size_px', 0)
        print(f"Reachability mask loaded: {grid_size}x{grid_size} grid")
        print(f"Mask covers: ±{world_extent/2:.1f} pixels")
        print(f"Cell size: {cell_size:.3f} px/cell")
    else:
        print("Reachability mask not available - run heatmap.py first")
    
    # Path visualization variables
    selected_node = None
    selected_agent_id = None
    path_to_selected = []
    
    # Scrollable info panel variables
    info_scroll_offset = 0
    max_visible_lines = (WINDOW_HEIGHT - 40) // 15  # How many lines fit in the sidebar
    
    print("Starting simulation...")
    print("Controls:")
    print("  Arrow Keys: Control magenta agent (Agent 1)")
    print("  WASD Keys: Control cyan agent (Agent 2)")
    print("  G: Toggle map graph display")
    print("  R: Toggle RRT* trees display")
    print("  T: Regenerate RRT* trees + auto-map to graph")
    print("  U: Force update travel times + auto-map to graph")
    print("  P: Strategic pursuit-evasion analysis (Agent 1 pursues Agent 2)")
    print("  M: Toggle reachability grid overlay for evader (Agent 2)")
    print("  V: Toggle evader visibility system (360° rays)")
    print("  I: Toggle info display")
    print("  S: Save agent states")
    print("  Mouse Click: Select RRT node (auto-generates trajectory)")
    print("  C: Clear selected path and trajectory")
    print("  X: Clear closest node cache")
    print("  Z: Force rebuild spatial index")
    print("  ESC: Quit")
    print("")
    if mask_data is not None:
        print("Reachability Grid Overlay:")
        print("  M: Show/hide Agent 2's reachability grid")
        print("  Grid follows Agent 2 position and orientation")
        print("  Shows probability-based coloring (hot colormap)")
        print("")
    print("Evader Visibility System:")
    print("  V: Show/hide Agent 2's 360° visibility rays")
    print(f"  Current range: {visibility_range:.0f} pixels")
    print("  Using 100 rays (3.6° increments) for optimal accuracy/performance")
    print("  Rays blocked by walls, pass through doors")
    print("  Breakoff Point Categories (relative to Agent 2's orientation):")
    print("    Red circles/lines = clockwise near-to-far transitions")
    print("    Orange-red circles/lines = clockwise far-to-near transitions")
    print("    Green circles/lines = counterclockwise near-to-far transitions")
    print("    Blue-green circles/lines = counterclockwise far-to-near transitions")
    print("  Updates in real-time as Agent 2 moves and rotates")
    print("")
    print("Node Visualization:")
    print(f"  Distance threshold: {DISTANCE_THRESHOLD:.1f}px (nodes closer = more transparent)")
    print(f"  Time threshold: {TIME_THRESHOLD:.1f}s (nodes slower = invalid/transparent)")
    print("  Invalid nodes are marked with 'X' and high transparency")
    print("  Strategic highlighting: Worst node = red, Worst 10 nodes = orange")
    print(f"  Strategic analysis only considers nodes ≤{TIME_THRESHOLD:.1f}s (valid nodes)")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_i:
                    show_info = not show_info
                    print(f"Info display: {'ON' if show_info else 'OFF'}")
                elif event.key == pygame.K_g:
                    show_map_graph = not show_map_graph
                    print(f"Map graph display: {'ON' if show_map_graph else 'OFF'}")
                elif event.key == pygame.K_r:
                    show_rrt_trees = not show_rrt_trees
                    print(f"RRT* trees display: {'ON' if show_rrt_trees else 'OFF'}")
                elif event.key == pygame.K_m:
                    if mask_data is not None:
                        show_reachability_mask = not show_reachability_mask
                        print(f"🎯 Reachability grid overlay: {'ON' if show_reachability_mask else 'OFF'}")
                        if show_reachability_mask:
                            print("🟢 Showing reachability grid around Agent 2 (evader)")
                            print("📍 Use WASD to move Agent 2 and see the grid follow")
                            print("🔄 Grid rotates with Agent 2's orientation")
                    else:
                        print("❌ Reachability mask not available - run heatmap.py first")
                elif event.key == pygame.K_v:
                    show_evader_visibility = not show_evader_visibility
                    print(f"👁️  Evader visibility system: {'ON' if show_evader_visibility else 'OFF'}")
                    if show_evader_visibility:
                        print("🔍 Showing Agent 2 (evader) 360° visibility")
                        print("🚫 Red rays = blocked by walls")
                        print("✅ Cyan rays = clear line of sight")
                        print("🔄 Use WASD to move Agent 2 and see visibility change")
                        print(f"📏 Visibility range: {visibility_range:.0f} pixels")
                elif event.key == pygame.K_t:
                    # Clear cached worst nodes since trees will change
                    global cached_worst_nodes
                    cached_worst_nodes = {}
                    
                    update_all_rrt_trees()
                    print("RRT* trees regenerated for all agents")
                    # Clear selected path and trajectory since trees changed
                    selected_node = None
                    selected_agent_id = None
                    path_to_selected = []
                    clear_trajectory()
                    # Invalidate trajectory calculation caches
                    get_trajectory_calculator().invalidate_cache()
                    print("Selected path, trajectory, and calculation caches cleared due to tree regeneration")
                    
                    # Immediately start calculating travel times for the new trees
                    stats = get_stats()
                    if stats.get('rrt_enabled', False):
                        agent1_nodes = stats.get('rrt_nodes_agent1', 0)
                        agent2_nodes = stats.get('rrt_nodes_agent2', 0)
                        
                        if agent1_nodes > 0:
                            print(f"Starting travel time calculation for Agent 1 ({agent1_nodes} nodes)...")
                            get_trajectory_calculator().get_travel_times_async("agent1")
                        
                        if agent2_nodes > 0:
                            print(f"Starting travel time calculation for Agent 2 ({agent2_nodes} nodes)...")
                            get_trajectory_calculator().get_travel_times_async("agent2")
                        
                        # Automatically perform RRT-to-map-graph mapping after regenerating trees
                        if map_graph_loaded and (agent1_nodes > 0 or agent2_nodes > 0):
                            print("\nAutomatically mapping RRT nodes to closest map graph nodes...")
                            all_mappings = map_all_rrt_nodes_to_graph()
                            
                            for agent_id, mapping in all_mappings.items():
                                print(f"  {agent_id}: Mapped {len(mapping)} RRT nodes to map graph nodes")
                            
                            # Show mapping statistics
                            mapping_stats = get_rrt_to_graph_mapping_stats()
                            print(f"  Mapping Statistics:")
                            for agent_id, agent_stats in mapping_stats['mappings_per_agent'].items():
                                success_rate = agent_stats['mapping_success_rate'] * 100
                                print(f"    {agent_id}: {agent_stats['mapped_nodes']}/{agent_stats['rrt_nodes']} nodes mapped ({success_rate:.1f}% success)")
                elif event.key == pygame.K_u:
                    # Force update travel times by invalidating cache and recalculating
                    get_trajectory_calculator().invalidate_cache()
                    print("Travel time cache cleared - forcing recalculation...")
                    
                    stats = get_stats()
                    if stats.get('rrt_enabled', False):
                        agent1_nodes = stats.get('rrt_nodes_agent1', 0)
                        agent2_nodes = stats.get('rrt_nodes_agent2', 0)
                        
                        if agent1_nodes > 0:
                            print(f"Forcing travel time calculation for Agent 1 ({agent1_nodes} nodes)...")
                            get_trajectory_calculator().get_travel_times_async("agent1")
                        
                        if agent2_nodes > 0:
                            print(f"Forcing travel time calculation for Agent 2 ({agent2_nodes} nodes)...")
                            get_trajectory_calculator().get_travel_times_async("agent2")
                        
                        # Automatically perform RRT-to-map-graph mapping after travel time calculation
                        if map_graph_loaded and (agent1_nodes > 0 or agent2_nodes > 0):
                            print("\nAutomatically mapping RRT nodes to closest map graph nodes...")
                            all_mappings = map_all_rrt_nodes_to_graph()
                            
                            for agent_id, mapping in all_mappings.items():
                                print(f"  {agent_id}: Mapped {len(mapping)} RRT nodes to map graph nodes")
                            
                            # Show mapping statistics
                            mapping_stats = get_rrt_to_graph_mapping_stats()
                            print(f"  Mapping Statistics:")
                            for agent_id, agent_stats in mapping_stats['mappings_per_agent'].items():
                                success_rate = agent_stats['mapping_success_rate'] * 100
                                print(f"    {agent_id}: {agent_stats['mapped_nodes']}/{agent_stats['rrt_nodes']} nodes mapped ({success_rate:.1f}% success)")
                        
                    else:
                        print("No RRT trees available for travel time calculation")
                elif event.key == pygame.K_p:
                    # Perform strategic pursuit-evasion analysis (separate from travel time updates)
                    print("Starting strategic pursuit-evasion analysis...")
                    
                    stats = get_stats()
                    if stats.get('rrt_enabled', False):
                        agent1_nodes = stats.get('rrt_nodes_agent1', 0)
                        agent2_nodes = stats.get('rrt_nodes_agent2', 0)
                        
                        if agent1_nodes > 0 and agent2_nodes > 0:
                            # Ensure RRT-to-map-graph mapping is done first
                            if map_graph_loaded:
                                print("Ensuring RRT-to-map-graph mapping is complete...")
                                all_mappings = map_all_rrt_nodes_to_graph()
                                
                                for agent_id, mapping in all_mappings.items():
                                    print(f"  {agent_id}: Mapped {len(mapping)} RRT nodes to map graph nodes")
                            
                            print("\nPerforming strategic pursuit-evasion analysis...")
                            print("  Agent 1 (pursuer) vs Agent 2 (evader):")
                            
                            # Agent1 as pursuer, Agent2 as evader
                            agent1_advantages = calculate_pursuit_evasion_advantages("agent1", "agent2")
                            agent1_stats = get_pursuit_evasion_stats("agent1", "agent2")
                            
                            if agent1_stats.get("status") == "Active":
                                print(f"    Analyzed {agent1_stats['pursuer_nodes_analyzed']} pursuer positions")
                                print(f"    Time advantage range: {agent1_stats['min_time_advantage']:.2f}s to {agent1_stats['max_time_advantage']:.2f}s")
                                print(f"    Average advantage: {agent1_stats['avg_time_advantage']:.2f}s")
                                print(f"    Positions with advantage: {agent1_stats['positive_advantages']}/{agent1_stats['pursuer_nodes_analyzed']}")
                            
                            print("Strategic analysis completed!")
                            
                            # Update the cached worst pursuit nodes for highlighting
                            update_worst_pursuit_nodes()
                        else:
                            print("Strategic analysis requires both agents to have RRT trees. Press T to generate trees first.")
                    else:
                        print("No RRT trees available for strategic analysis. Press T to generate trees first.")
                elif event.key == pygame.K_c:
                    # Clear selected path and trajectory
                    selected_node = None
                    selected_agent_id = None
                    path_to_selected = []
                    clear_trajectory()
                    print("Selected path and trajectory cleared")
                elif event.key == pygame.K_s:
                    # Save agent states
                    try:
                        with open(AGENT_STATE_FILE, 'wb') as f:
                            pickle.dump(agent1.state, f)
                        with open(AGENT2_STATE_FILE, 'wb') as f:
                            pickle.dump(agent2.state, f)
                        print("Agent states saved")
                    except Exception as e:
                        print(f"Error saving agent states: {e}")
                elif event.key == pygame.K_x:
                    # Clear closest node cache
                    clear_closest_node_cache()
                    print("Closest node cache cleared")
                elif event.key == pygame.K_z:
                    # Force rebuild spatial index
                    force_rebuild_spatial_index()
                    print("Spatial index (KD-tree) rebuilt")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_x, mouse_y = event.pos
                    
                    # Only search for nodes in the environment area (not sidebar)
                    if mouse_x < ENVIRONMENT_WIDTH:
                        # Try to find a node near the click position for each agent
                        for agent_id in ["agent1", "agent2"]:
                            node = find_node_at_position(agent_id, mouse_x, mouse_y, tolerance=15.0)
                            if node:
                                selected_node = node
                                selected_agent_id = agent_id
                                path_to_selected = get_path_to_node(agent_id, node)
                                print(f"Selected node in {agent_id} tree at ({node.x:.1f}, {node.y:.1f})")
                                print(f"Path has {len(path_to_selected)} nodes, cost: {node.cost:.2f}")
                                
                                # Automatically generate optimized trajectory asynchronously
                                print(f"Starting async trajectory generation for {agent_id} with {len(path_to_selected)} nodes...")
                                
                                def trajectory_callback(result, agent_id):
                                    if result:
                                        print(f"Async trajectory generation completed for {agent_id}!")
                                    else:
                                        print(f"Async trajectory generation failed for {agent_id}")
                                
                                trajectory_calculator = get_trajectory_calculator()
                                trajectory_calculator.generate_trajectory_async(
                                    path_to_selected, agent_id, num_points=150, callback=trajectory_callback
                                )
                                break
                        else:
                            # No node found
                            if selected_node:
                                selected_node = None
                                selected_agent_id = None
                                path_to_selected = []
                                clear_trajectory()
                                print("No node found at click position - path and trajectory cleared")
            elif event.type == pygame.MOUSEWHEEL:
                # Handle mouse wheel scrolling in the info panel
                if show_info:
                    info_scroll_offset -= event.y * 3  # Scroll 3 lines at a time
                    # We'll clamp the scroll offset later when we know the total lines
        
        # Get current key states for continuous movement
        keys = pygame.key.get_pressed()
        
        # Agent 1 control (arrow keys)
        linear_vel1 = 0
        angular_vel1 = 0
        if keys[pygame.K_UP]:
            linear_vel1 = AGENT_LINEAR_VEL
        if keys[pygame.K_DOWN]:
            linear_vel1 = -AGENT_LINEAR_VEL
        if keys[pygame.K_LEFT]:
            angular_vel1 = -AGENT_ANGULAR_VEL
        if keys[pygame.K_RIGHT]:
            angular_vel1 = AGENT_ANGULAR_VEL
        
        # Agent 2 control (WASD keys)
        linear_vel2 = 0
        angular_vel2 = 0
        if keys[pygame.K_w]:
            linear_vel2 = AGENT_LINEAR_VEL
        if keys[pygame.K_s]:
            linear_vel2 = -AGENT_LINEAR_VEL
        if keys[pygame.K_a]:
            angular_vel2 = -AGENT_ANGULAR_VEL
        if keys[pygame.K_d]:
            angular_vel2 = AGENT_ANGULAR_VEL
        
        # Update agents
        agent1.set_controls(linear_vel1, angular_vel1)
        agent1.update(dt=0.1, walls=environment.get_all_walls(), doors=environment.get_doors())
        
        agent2.set_controls(linear_vel2, angular_vel2)
        agent2.update(dt=0.1, walls=environment.get_all_walls(), doors=environment.get_doors())
        
        # Update evader visibility system periodically
        if show_evader_visibility and current_time - last_visibility_update >= visibility_update_interval:
            visibility_data = calculate_evader_visibility(
                agent2.state[0], agent2.state[1], 
                visibility_range, 
                environment.get_all_walls(), 
                environment.get_doors(),
                num_rays=100  # 5-degree increments for good balance of accuracy vs performance
            )
            last_visibility_update = current_time
        
        # Update position evaluator periodically
        current_time = time.time()
        if current_time - last_position_update >= position_update_interval:
            # Update agent positions in the evaluator
            update_position("agent1", agent1.state[0], agent1.state[1], agent1.state[2], (linear_vel1, angular_vel1))
            update_position("agent2", agent2.state[0], agent2.state[1], agent2.state[2], (linear_vel2, angular_vel2))
            
            # Get distance between agents
            distance = get_distance("agent1", "agent2")
            if distance is not None:
                print(f"Position Evaluator - Distance between agents: {distance:.2f} pixels")
            
            # Get evaluator statistics
            stats = get_stats()
            print(f"Position Evaluator - Tracking {stats.get('agent_count', 0)} agents")
            print(f"Position Evaluator - Environment data: {'Available' if stats.get('has_environment', False) else 'Not available'}")
            
            # Test closest node functionality if map graph is available
            if map_graph_loaded:
                closest_node_agent1 = find_closest_node("agent1")
                closest_node_agent2 = find_closest_node("agent2")
                if closest_node_agent1 is not None:
                    print(f"Position Evaluator - Agent 1 closest to node {closest_node_agent1}")
                if closest_node_agent2 is not None:
                    print(f"Position Evaluator - Agent 2 closest to node {closest_node_agent2}")
            
            last_position_update = current_time
        
        # Update travel time calculations periodically
        if current_time - last_travel_time_update >= travel_time_update_interval:
            # Trigger travel time calculations for both agents
            stats = get_stats()
            if stats.get('rrt_enabled', False):
                agent1_nodes = stats.get('rrt_nodes_agent1', 0)
                agent2_nodes = stats.get('rrt_nodes_agent2', 0)
                
                if agent1_nodes > 0:
                    # Trigger async calculation for agent1 (this will use cache if already calculated)
                    get_trajectory_calculator().get_travel_times_async("agent1")
                
                if agent2_nodes > 0:
                    # Trigger async calculation for agent2 (this will use cache if already calculated)
                    get_trajectory_calculator().get_travel_times_async("agent2")
            
            last_travel_time_update = current_time
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw environment
        environment.draw(screen, font)
        
        # Draw evader visibility system
        if show_evader_visibility and visibility_data:
            draw_evader_visibility(
                screen, 
                agent2.state[0], agent2.state[1], 
                visibility_data,
                show_rays=show_visibility_rays,
                show_visibility_area=show_visibility_area,
                agent_theta=agent2.state[2]  # Pass agent's orientation for breakoff point coloring
            )
        
        # Draw map graph if enabled and loaded
        if show_map_graph and map_graph_loaded:
            # Draw the graph edges first
            for edge in map_graph.edges:
                i, j = edge
                start = map_graph.nodes[i]
                end = map_graph.nodes[j]
                pygame.draw.line(screen, MAP_GRAPH_EDGE_COLOR, start, end, 1)
            
            # Draw the graph nodes
            for i, node in enumerate(map_graph.nodes):
                pygame.draw.circle(screen, MAP_GRAPH_NODE_COLOR, node, 4)
        
        # Draw RRT* trees if enabled
        if show_rrt_trees:
            # Colors for different agents' trees
            rrt_colors = {
                "agent1": (100, 255, 100),  # Light green for agent 1
                "agent2": (100, 100, 255),  # Light blue for agent 2
            }
            
            # Get worst pursuit nodes for highlighting
            worst_pursuit_nodes = get_worst_pursuit_nodes()
            
            # Get current agent positions for distance calculations
            agent1_pos = (agent1.state[0], agent1.state[1])
            agent2_pos = (agent2.state[0], agent2.state[1])
            agent_positions = {
                "agent1": agent1_pos,
                "agent2": agent2_pos
            }
            
            for agent_id in ["agent1", "agent2"]:
                tree = get_agent_rrt_tree(agent_id)
                if tree:
                    color = rrt_colors.get(agent_id, (128, 128, 128))
                    current_agent_pos = agent_positions[agent_id]
                    
                    # Get travel times for this agent if available
                    travel_times_data = get_trajectory_calculator().get_travel_times_async(agent_id)
                    travel_times_dict = {}
                    if travel_times_data:
                        # Convert travel times list to dictionary indexed by node index
                        for node, travel_time, path_len in travel_times_data:
                            # Find the node index in the tree
                            for idx, tree_node in enumerate(tree):
                                if tree_node.x == node.x and tree_node.y == node.y:
                                    travel_times_dict[idx] = travel_time
                                    break
                    
                    # Draw tree edges with transparency based on node conditions
                    for node in tree:
                        if node.parent:
                            # Calculate edge transparency based on child node conditions
                            node_pos = (node.x, node.y)
                            distance_to_agent = math.sqrt((node_pos[0] - current_agent_pos[0])**2 + 
                                                        (node_pos[1] - current_agent_pos[1])**2)
                            
                            # Check if node index exists and get timing data
                            node_idx = None
                            for idx, tree_node in enumerate(tree):
                                if tree_node.x == node.x and tree_node.y == node.y:
                                    node_idx = idx
                                    break
                            
                            # Determine transparency based on thresholds
                            edge_alpha = 255  # Default full opacity
                            if distance_to_agent < DISTANCE_THRESHOLD:
                                edge_alpha = 100  # More transparent for close nodes
                            elif node_idx is not None and node_idx in travel_times_dict:
                                travel_time = travel_times_dict[node_idx]
                                if travel_time > TIME_THRESHOLD:
                                    edge_alpha = 80  # Most transparent for invalid (slow) nodes
                            
                            # Create surface for transparent drawing
                            if edge_alpha < 255:
                                edge_surface = pygame.Surface((abs(int(node.x - node.parent.x)) + 2, 
                                                             abs(int(node.y - node.parent.y)) + 2), pygame.SRCALPHA)
                                start_pos = (max(0, int(node.parent.x - min(node.x, node.parent.x))), 
                                           max(0, int(node.parent.y - min(node.y, node.parent.y))))
                                end_pos = (max(0, int(node.x - min(node.x, node.parent.x))), 
                                         max(0, int(node.y - min(node.y, node.parent.y))))
                                edge_color_with_alpha = (*color, edge_alpha)
                                pygame.draw.line(edge_surface, edge_color_with_alpha, start_pos, end_pos, 1)
                                screen.blit(edge_surface, (min(int(node.x), int(node.parent.x)), 
                                                         min(int(node.y), int(node.parent.y))))
                            else:
                                start_pos = (int(node.parent.x), int(node.parent.y))
                                end_pos = (int(node.x), int(node.y))
                                pygame.draw.line(screen, color, start_pos, end_pos, 1)
                    
                    # Draw tree nodes with transparency based on distance and timing thresholds
                    for i, node in enumerate(tree):
                        pos = (int(node.x), int(node.y))
                        node_pos = (node.x, node.y)
                        
                        # Calculate distance to current agent
                        distance_to_agent = math.sqrt((node_pos[0] - current_agent_pos[0])**2 + 
                                                    (node_pos[1] - current_agent_pos[1])**2)
                        
                        # Check if this is the worst pursuit node or in the worst 10
                        is_worst_node = False
                        is_worst_10_node = False
                        
                        if agent_id in worst_pursuit_nodes:
                            # Check if this is the single worst node
                            worst_data = worst_pursuit_nodes[agent_id].get('worst')
                            if worst_data and worst_data[0] == i:
                                is_worst_node = True
                            
                            # Check if this is in the worst 10 nodes
                            worst_10_data = worst_pursuit_nodes[agent_id].get('worst_10', [])
                            for node_idx, _, _ in worst_10_data:
                                if node_idx == i:
                                    is_worst_10_node = True
                                    break
                        
                        # Determine node transparency and validity
                        node_alpha = 255  # Default full opacity
                        is_invalid = False
                        
                        # Apply distance threshold
                        if distance_to_agent < DISTANCE_THRESHOLD:
                            node_alpha = 120  # More transparent for close nodes
                        
                        # Apply timing threshold if timing data is available
                        if i in travel_times_dict:
                            travel_time = travel_times_dict[i]
                            if travel_time > TIME_THRESHOLD:
                                node_alpha = 60  # Most transparent for invalid (slow) nodes
                                is_invalid = True
                        
                        # Draw node with appropriate transparency and highlighting
                        if is_worst_node:
                            # Highlight the single worst pursuit node with red color and larger size (always full opacity)
                            pygame.draw.circle(screen, (255, 50, 50), pos, 8)  # Red highlight
                            pygame.draw.circle(screen, (255, 255, 255), pos, 6)  # White inner circle
                            pygame.draw.circle(screen, color, pos, 4)  # Original color center
                        elif is_worst_10_node:
                            # Highlight nodes in worst 10 with orange color and medium size (always full opacity)
                            pygame.draw.circle(screen, (255, 140, 0), pos, 6)  # Orange highlight
                            pygame.draw.circle(screen, (255, 255, 255), pos, 4)  # White inner circle
                            pygame.draw.circle(screen, color, pos, 2)  # Original color center
                        elif node_alpha < 255:
                            # Draw transparent node
                            node_size = 4 if node.parent is None else 2
                            if is_invalid:
                                # Draw invalid nodes with a different visual indicator
                                node_surface = pygame.Surface((node_size * 2 + 2, node_size * 2 + 2), pygame.SRCALPHA)
                                transparent_color = (*color, node_alpha)
                                pygame.draw.circle(node_surface, transparent_color, 
                                                 (node_size + 1, node_size + 1), node_size)
                                # Add a small 'X' to indicate invalid
                                pygame.draw.line(node_surface, (255, 255, 255, node_alpha + 50), 
                                               (1, 1), (node_size * 2 + 1, node_size * 2 + 1), 1)
                                pygame.draw.line(node_surface, (255, 255, 255, node_alpha + 50), 
                                               (node_size * 2 + 1, 1), (1, node_size * 2 + 1), 1)
                                screen.blit(node_surface, (pos[0] - node_size - 1, pos[1] - node_size - 1))
                            else:
                                # Draw normal transparent node
                                node_surface = pygame.Surface((node_size * 2 + 2, node_size * 2 + 2), pygame.SRCALPHA)
                                transparent_color = (*color, node_alpha)
                                pygame.draw.circle(node_surface, transparent_color, 
                                                 (node_size + 1, node_size + 1), node_size)
                                screen.blit(node_surface, (pos[0] - node_size - 1, pos[1] - node_size - 1))
                        else:
                            # Draw normal opaque node
                            if node.parent is None:  # Root node
                                pygame.draw.circle(screen, color, pos, 4)
                            else:
                                pygame.draw.circle(screen, color, pos, 2)
        
        # Draw optimized trajectory
        if show_trajectory:
            trajectory = get_current_trajectory()
            if trajectory:
                path = trajectory['path']
                if len(path) >= 2:
                    # Draw the optimized path as a smooth magenta curve
                    trajectory_color = (255, 0, 255)  # Magenta
                    for i in range(len(path) - 1):
                        start_pos = (int(path[i][0]), int(path[i][1]))
                        end_pos = (int(path[i+1][0]), int(path[i+1][1]))
                        pygame.draw.line(screen, trajectory_color, start_pos, end_pos, 3)
        
        # Draw info panel if enabled
        if show_info:
            info_lines = [
                "Simple Agent Simulation",
                "",
                "Controls:",
                "Arrow Keys: Control Agent 1 (Magenta)",
                "WASD Keys: Control Agent 2 (Cyan)",
                "G: Toggle map graph display",
                "R: Toggle RRT* trees display",
                "M: Toggle reachability grid overlay (Agent 2)",
                "V: Toggle evader visibility (Agent 2 360° rays)",
                "T: Regenerate RRT* trees + auto-map to graph",
                "U: Force update travel times + auto-map to graph",
                "P: Strategic analysis (Agent 1 pursues Agent 2)",
                "I: Toggle this info display",
                "S: Save agent states to file",
                "Mouse Click: Select RRT node (auto-gen traj)",
                "C: Clear selected path & trajectory",
                "X: Clear closest node cache",
                "Z: Force rebuild spatial index",
                "ESC: Quit simulation",
                "",
                f"Agent 1 Position: ({agent1.state[0]:.1f}, {agent1.state[1]:.1f})",
                f"Agent 1 Heading: {math.degrees(agent1.state[2]):.1f}°",
                f"Agent 2 Position: ({agent2.state[0]:.1f}, {agent2.state[1]:.1f})",
                f"Agent 2 Heading: {math.degrees(agent2.state[2]):.1f}°",
                "",
                f"FPS: {int(clock.get_fps())}",
                "",
                "Position Evaluator:",
                f"Evaluator Distance: {get_distance('agent1', 'agent2'):.1f}" if get_distance('agent1', 'agent2') else "Evaluator Distance: N/A",
                f"Tracked Agents: {get_stats().get('agent_count', 0)}",
                f"Environment Data: {'Yes' if get_stats().get('has_environment', False) else 'No'}",
                f"Update Interval: {position_update_interval}s",
                "",
                f"Map Graph: {'ON' if show_map_graph else 'OFF'}",
                f"RRT* Trees: {'ON' if show_rrt_trees else 'OFF'}",
                f"Reachability Grid: {'ON' if show_reachability_mask else 'OFF'}" + (" (Agent 2)" if show_reachability_mask else ""),
                f"Evader Visibility: {'ON' if show_evader_visibility else 'OFF'}" + (f" (Range: {visibility_range:.0f}px)" if show_evader_visibility else ""),
            ]
            
            # Add visibility system details if active
            if show_evader_visibility:
                visibility_stats = get_visibility_statistics(visibility_data)
                
                # Get breakoff point statistics
                breakoff_points, breakoff_lines = detect_visibility_breakoff_points(
                    visibility_data, 
                    min_gap_distance=30,
                    agent_x=agent2.state[0],
                    agent_y=agent2.state[1], 
                    agent_theta=agent2.state[2]
                )
                
                # Count breakoff points by category
                category_counts = {
                    'clockwise_near_far': 0,
                    'clockwise_far_near': 0,
                    'counterclockwise_near_far': 0,
                    'counterclockwise_far_near': 0,
                    'unknown': 0
                }
                
                for point in breakoff_points:
                    if len(point) >= 3:
                        category = point[2]
                        if category in category_counts:
                            category_counts[category] += 1
                        else:
                            category_counts['unknown'] += 1
                
                info_lines.extend([
                    "",
                    "Visibility System:",
                    f"Total rays: {visibility_stats['total_rays']}",
                    f"Clear rays: {visibility_stats['clear_rays']}",
                    f"Blocked rays: {visibility_stats['blocked_rays']}",
                    f"Visibility %: {visibility_stats['visibility_percentage']:.1f}%",
                    f"Avg distance: {visibility_stats['average_visibility_distance']:.1f}px",
                    f"Max distance: {visibility_stats['max_visibility_distance']:.1f}px",
                    f"Breakoff points: {len(breakoff_points)} total",
                    f"  Clockwise near→far (red): {category_counts['clockwise_near_far']}",
                    f"  Clockwise far→near (orange-red): {category_counts['clockwise_far_near']}",
                    f"  Counter near→far (green): {category_counts['counterclockwise_near_far']}",
                    f"  Counter far→near (blue-green): {category_counts['counterclockwise_far_near']}",
                    f"Show rays: {'ON' if show_visibility_rays else 'OFF'}",
                    f"Show area: {'ON' if show_visibility_area else 'OFF'}",
                    f"Update rate: {1/visibility_update_interval:.1f} Hz",
                ])
            
            # Add selected path information
            if selected_node and selected_agent_id:
                info_lines.extend([
                    "",
                    "Selected Path:",
                    f"Agent: {selected_agent_id}",
                    f"Target: ({selected_node.x:.1f}, {selected_node.y:.1f})",
                    f"Path Length: {len(path_to_selected)} nodes",
                    f"Path Cost: {selected_node.cost:.2f}",
                    f"Path Distance: {sum(math.sqrt((path_to_selected[i+1].x - path_to_selected[i].x)**2 + (path_to_selected[i+1].y - path_to_selected[i].y)**2) for i in range(len(path_to_selected)-1)):.1f}" if len(path_to_selected) > 1 else "0.0",
                ])
                
                # Add travel time for selected node if available
                travel_times_data = get_trajectory_calculator().get_travel_times_async(selected_agent_id)
                if travel_times_data:
                    # Find the travel time for the selected node
                    selected_travel_time = None
                    for node, travel_time, path_len in travel_times_data:
                        if abs(node.x - selected_node.x) < 0.1 and abs(node.y - selected_node.y) < 0.1:
                            selected_travel_time = travel_time
                            break
                    
                    if selected_travel_time is not None:
                        validity_status = "Valid" if selected_travel_time <= TIME_THRESHOLD else "Invalid (too slow)"
                        info_lines.append(f"Travel Time: {selected_travel_time:.2f}s ({validity_status})")
                    else:
                        info_lines.append("Travel Time: Calculating...")
                elif get_trajectory_calculator().is_calculating(selected_agent_id):
                    info_lines.append("Travel Time: Calculating in background...")
                else:
                    info_lines.append("Travel Time: Not calculated yet")
            else:
                info_lines.extend([
                    "",
                    "Selected Path: None",
                    "(Click on RRT node to auto-generate)",
                ])
            
            # Add trajectory optimization information
            traj_info = get_trajectory_info()
            info_lines.extend([
                "",
                "Optimized Trajectory:",
                f"Status: {traj_info.get('status', 'Unknown')}",
            ])
            
            if traj_info.get('status') == 'Active':
                info_lines.extend([
                    f"Agent: {traj_info.get('agent_id', 'N/A')}",
                    f"Travel Time: {traj_info.get('total_time', 'N/A')}",
                    f"Path Distance: {traj_info.get('total_distance', 'N/A')}",
                    f"Peak Velocity: {traj_info.get('peak_velocity', 'N/A')}",
                    f"Avg Velocity: {traj_info.get('avg_velocity', 'N/A')}",
                    f"Max Constraints:",
                    f"  Velocity: {LEADER_LINEAR_VEL:.1f} px/s",
                    f"  Turn Rate: {LEADER_ANGULAR_VEL:.1f} rad/s",
                    f"  Acceleration: 50.0 px/s²",
                ])
            else:
                info_lines.extend([
                    "(Click on RRT node for auto-generation)",
                    f"Agent Max Velocity: {LEADER_LINEAR_VEL:.1f} px/s",
                    f"Agent Max Turn Rate: {LEADER_ANGULAR_VEL:.1f} rad/s",
                ])
            
            # Add RRT* tree information
            stats = get_stats()
            if stats.get('rrt_enabled', False):
                info_lines.extend([
                    "",
                    "RRT* Trees:",
                    f"Max nodes per tree: {stats.get('rrt_max_nodes', 0)}",
                    f"Step size: {stats.get('rrt_step_size', 0):.1f}",
                    f"Search radius: {stats.get('rrt_search_radius', 0):.1f}",
                    f"Forward bias: {stats.get('rrt_forward_bias', 0):.1%}",
                    f"Forward cone: {stats.get('rrt_forward_cone_angle', 0):.0f}°",
                    "",
                    "Node Visualization Thresholds:",
                    f"Distance threshold: {DISTANCE_THRESHOLD:.1f} px",
                    f"Time threshold: {TIME_THRESHOLD:.1f}s",
                    "(Nodes closer than distance threshold = transparent)",
                    "(Nodes slower than time threshold = invalid/transparent)",
                ])
                
                # Add per-agent tree info
                agent1_nodes = stats.get('rrt_nodes_agent1', 0)
                agent2_nodes = stats.get('rrt_nodes_agent2', 0)
                if agent1_nodes > 0:
                    info_lines.append(f"Agent 1 tree: {agent1_nodes} nodes")
                if agent2_nodes > 0:
                    info_lines.append(f"Agent 2 tree: {agent2_nodes} nodes")
                
                # Add longest travel times for each agent (async calculation)
                info_lines.extend([
                    "",
                    "Longest Travel Times:",
                ])
                
                # Calculate and display for Agent 1 (async)
                if agent1_nodes > 0:
                    travel_times_1 = get_trajectory_calculator().get_travel_times_async("agent1")
                    if travel_times_1:
                        info_lines.append("Agent 1:")
                        for i, (node, travel_time, path_len) in enumerate(travel_times_1[:2]):  # Top 2
                            info_lines.append(f"  #{i+1}: {travel_time:.2f}s ({path_len} nodes)")
                            info_lines.append(f"      to ({node.x:.0f}, {node.y:.0f})")
                    elif get_trajectory_calculator().is_calculating("agent1"):
                        info_lines.append("Agent 1: Calculating in background...")
                    else:
                        info_lines.append("Agent 1: No data yet")
                
                # Calculate and display for Agent 2 (async)
                if agent2_nodes > 0:
                    travel_times_2 = get_trajectory_calculator().get_travel_times_async("agent2")
                    if travel_times_2:
                        info_lines.append("Agent 2:")
                        for i, (node, travel_time, path_len) in enumerate(travel_times_2[:2]):  # Top 2
                            info_lines.append(f"  #{i+1}: {travel_time:.2f}s ({path_len} nodes)")
                            info_lines.append(f"      to ({node.x:.0f}, {node.y:.0f})")
                    elif get_trajectory_calculator().is_calculating("agent2"):
                        info_lines.append("Agent 2: Calculating in background...")
                    else:
                        info_lines.append("Agent 2: No data yet")
                
                # Add average travel times for all nodes
                info_lines.extend([
                    "",
                    "Average Travel Times (All Nodes):",
                ])
                
                # Calculate average for Agent 1
                if agent1_nodes > 0:
                    travel_times_1 = get_trajectory_calculator().get_travel_times_async("agent1")
                    if travel_times_1:
                        travel_time_values = [travel_time for _, travel_time, _ in travel_times_1]
                        total_time = sum(travel_time_values)
                        avg_time = total_time / len(travel_time_values)
                        min_time = min(travel_time_values)
                        max_time = max(travel_time_values)
                        
                        # Count nodes by time brackets
                        fast_nodes = sum(1 for t in travel_time_values if t <= TIME_THRESHOLD)
                        slow_nodes = len(travel_time_values) - fast_nodes
                        
                        info_lines.extend([
                            f"Agent 1: {avg_time:.2f}s avg ({len(travel_times_1)} nodes)",
                            f"  Range: {min_time:.2f}s - {max_time:.2f}s",
                            f"  Fast (≤{TIME_THRESHOLD:.1f}s): {fast_nodes}, Slow (>{TIME_THRESHOLD:.1f}s): {slow_nodes}"
                        ])
                    elif get_trajectory_calculator().is_calculating("agent1"):
                        info_lines.append("Agent 1: Calculating...")
                    else:
                        info_lines.append("Agent 1: No timing data")
                
                # Calculate average for Agent 2
                if agent2_nodes > 0:
                    travel_times_2 = get_trajectory_calculator().get_travel_times_async("agent2")
                    if travel_times_2:
                        travel_time_values = [travel_time for _, travel_time, _ in travel_times_2]
                        total_time = sum(travel_time_values)
                        avg_time = total_time / len(travel_time_values)
                        min_time = min(travel_time_values)
                        max_time = max(travel_time_values)
                        
                        # Count nodes by time brackets
                        fast_nodes = sum(1 for t in travel_time_values if t <= TIME_THRESHOLD)
                        slow_nodes = len(travel_time_values) - fast_nodes
                        
                        info_lines.extend([
                            f"Agent 2: {avg_time:.2f}s avg ({len(travel_times_2)} nodes)",
                            f"  Range: {min_time:.2f}s - {max_time:.2f}s",
                            f"  Fast (≤{TIME_THRESHOLD:.1f}s): {fast_nodes}, Slow (>{TIME_THRESHOLD:.1f}s): {slow_nodes}"
                        ])
                    elif get_trajectory_calculator().is_calculating("agent2"):
                        info_lines.append("Agent 2: Calculating...")
                    else:
                        info_lines.append("Agent 2: No timing data")
            
            # Add map graph info if loaded
            if map_graph_loaded:
                stats = get_stats()
                cache_stats = get_closest_node_cache_stats()
                info_lines.extend([
                    f"Nodes: {stats.get('map_nodes', 0)}",
                    f"Edges: {stats.get('map_edges', 0)}"
                ])
                
                # Add closest node optimization info
                info_lines.extend([
                    "",
                    "Closest Node Optimization:",
                    f"Method: {cache_stats.get('optimization_method', 'Unknown')}",
                    f"Cache Size: {cache_stats.get('cache_size', 0)} agents",
                    f"Cache Thresholds:",
                    f"  Movement: {cache_stats.get('movement_threshold', 0):.1f} px",
                    f"  Time: {cache_stats.get('time_threshold', 0):.1f}s",
                    f"SciPy Available: {'Yes' if cache_stats.get('scipy_available', False) else 'No'}",
                    f"KD-tree Active: {'Yes' if cache_stats.get('kdtree_available', False) else 'No'}",
                ])
                
                # Add RRT-to-Map-Graph mapping info
                if map_graph_loaded:
                    mapping_stats = get_rrt_to_graph_mapping_stats()
                    if mapping_stats['total_agents'] > 0:
                        info_lines.extend([
                            "",
                            "RRT to Map Graph Mapping:",
                        ])
                        
                        for agent_id, agent_stats in mapping_stats['mappings_per_agent'].items():
                            success_rate = agent_stats['mapping_success_rate'] * 100
                            info_lines.append(f"{agent_id}: {agent_stats['mapped_nodes']}/{agent_stats['rrt_nodes']} ({success_rate:.1f}%)")
                    else:
                        info_lines.extend([
                            "",
                            "RRT to Map Graph Mapping:",
                            "No mappings yet (press T/U to generate)",
                        ])
                    
                    # Add strategic pursuit-evasion analysis results
                    agent1_nodes = stats.get('rrt_nodes_agent1', 0)
                    agent2_nodes = stats.get('rrt_nodes_agent2', 0)
                    
                    if agent1_nodes > 0 and agent2_nodes > 0:
                        info_lines.extend([
                            "",
                            "Strategic Analysis:",
                            "Agent 1 = Pursuer, Agent 2 = Evader",
                            "Press P to run strategic analysis",
                            "(Results will appear here after running)",
                        ])
                        
                        # Add worst pursuit node information
                        worst_nodes = get_worst_pursuit_nodes()
                        if worst_nodes:
                            info_lines.extend([
                                "",
                                "Worst Pursuit Positions:",
                                f"(Only nodes ≤{TIME_THRESHOLD:.1f}s considered)",
                            ])
                            
                            for agent_id, data in worst_nodes.items():
                                # Only show for Agent 1 (the pursuer)
                                if agent_id == "agent1":
                                    # Show the single worst node (red highlight)
                                    worst_data = data.get('worst')
                                    if worst_data:
                                        node_idx, advantage, node = worst_data
                                        info_lines.extend([
                                            f"Worst Node (red): #{node_idx}",
                                            f"  Position: ({node.x:.0f}, {node.y:.0f})",
                                            f"  Time disadvantage: {-advantage:.2f}s",
                                        ])
                                    
                                    # Show summary of worst 10 nodes (orange highlights)
                                    worst_10_data = data.get('worst_10', [])
                                    if worst_10_data:
                                        avg_disadvantage = sum(-adv for _, adv, _ in worst_10_data) / len(worst_10_data)
                                        worst_disadvantage = max(-adv for _, adv, _ in worst_10_data)
                                        info_lines.extend([
                                            f"Worst 10 Nodes (orange): {len(worst_10_data)} nodes",
                                            f"  Avg disadvantage: {avg_disadvantage:.2f}s",
                                            f"  Max disadvantage: {worst_disadvantage:.2f}s",
                                        ])
                    else:
                        info_lines.extend([
                            "",
                            "Strategic Analysis:",
                            "Need both agents' RRT trees (press T/U)",
                        ])
                
                # Add closest node info
                closest_node_agent1 = find_closest_node("agent1")
                closest_node_agent2 = find_closest_node("agent2")
                if closest_node_agent1 is not None:
                    info_lines.append(f"Agent 1 → Node {closest_node_agent1}")
                if closest_node_agent2 is not None:
                    info_lines.append(f"Agent 2 → Node {closest_node_agent2}")
            else:
                info_lines.append("Map Graph: Not loaded")
                info_lines.append("(Run inspect_environment.py first)")
            
            # Clamp scroll offset to valid range
            max_scroll = max(0, len(info_lines) - max_visible_lines)
            info_scroll_offset = max(0, min(info_scroll_offset, max_scroll))
            
            # Calculate visible lines
            visible_lines = info_lines[info_scroll_offset:info_scroll_offset + max_visible_lines]
            
            # Create info background (fixed size)
            info_bg = pygame.Surface((300, WINDOW_HEIGHT - 40))
            info_bg.fill((0, 0, 0))
            info_bg.set_alpha(180)
            
            # Position info panel on the right side (in the sidebar area)
            info_x = ENVIRONMENT_WIDTH + 10  # Position in sidebar
            info_y = 20
            screen.blit(info_bg, (info_x, info_y))
            
            # Draw scroll indicator if needed
            if max_scroll > 0:
                scroll_bar_height = max(20, (max_visible_lines / len(info_lines)) * (WINDOW_HEIGHT - 60))
                scroll_bar_y = info_y + 10 + (info_scroll_offset / max_scroll) * (WINDOW_HEIGHT - 80 - scroll_bar_height)
                pygame.draw.rect(screen, (100, 100, 100), (info_x + 280, scroll_bar_y, 8, scroll_bar_height))
            
            # Draw visible info text
            for i, line in enumerate(visible_lines):
                if line:  # Skip empty lines
                    text_surface = font.render(line, True, (255, 255, 255))
                    screen.blit(text_surface, (info_x + 10, info_y + 10 + i * 15))  # Smaller line spacing
            
            # Draw scroll hint at bottom if there's more content
            if info_scroll_offset + max_visible_lines < len(info_lines):
                hint_surface = font.render("(Scroll down for more...)", True, (150, 150, 150))
                screen.blit(hint_surface, (info_x + 10, info_y + WINDOW_HEIGHT - 60))
            elif info_scroll_offset > 0:
                hint_surface = font.render("(Scroll up for more...)", True, (150, 150, 150))
                screen.blit(hint_surface, (info_x + 10, info_y + WINDOW_HEIGHT - 60))
        
        # Draw reachability mask ON TOP of everything (Agent 2 evader)
        if show_reachability_mask and mask_data is not None:
            # Get Agent 2's current state (position and orientation)
            agent2_x, agent2_y, agent2_theta = agent2.state[0], agent2.state[1], agent2.state[2]
            
            # Draw the reachability grid overlay centered on Agent 2's position
            draw_reachability_grid_overlay(screen, mask_data, agent2_x, agent2_y, agent2_theta, alpha=180)
        elif show_reachability_mask:
            # Only print warning occasionally, not every frame
            if pygame.time.get_ticks() % 3000 < 50:  # Every 3 seconds for ~50ms
                print("❌ Reachability mask enabled but mask data not available")
        
        # Draw agents ON TOP of everything including reachability grid
        def draw_agent(agent, color, label):
            x, y, theta = agent.state[0], agent.state[1], agent.state[2]
            
            # Draw agent body (circle)
            pygame.draw.circle(screen, color, (int(x), int(y)), AGENT_RADIUS)
            
            # Draw direction indicator (line showing heading)
            end_x = x + AGENT_RADIUS * 1.5 * math.cos(theta)
            end_y = y + AGENT_RADIUS * 1.5 * math.sin(theta)
            pygame.draw.line(screen, (255, 255, 255), (int(x), int(y)), (int(end_x), int(end_y)), 3)
            
            # Draw agent label
            label_surface = font.render(label, True, (255, 255, 255))
            screen.blit(label_surface, (int(x) - 20, int(y) - AGENT_RADIUS - 25))
        
        # Draw both agents on top of all other visual elements
        draw_agent(agent1, AGENT_COLOR, "Agent 1")
        draw_agent(agent2, AGENT2_COLOR, "Agent 2")
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        clock.tick(60)  # 60 FPS
    
    # Save agent states on exit
    try:
        with open(AGENT_STATE_FILE, 'wb') as f:
            pickle.dump(agent1.state, f)
        with open(AGENT2_STATE_FILE, 'wb') as f:
            pickle.dump(agent2.state, f)
        print("Agent states saved on exit")
    except Exception as e:
        print(f"Error saving agent states on exit: {e}")
    
    # Cleanup threaded trajectory calculator
    print("Shutting down trajectory calculation threads...")
    get_trajectory_calculator().shutdown()
    
    # Cleanup
    pygame.quit()
    print("Simulation ended")

if __name__ == "__main__":
    main()
