"""
Environment Inspection Simulation
This script renders the simulation environment without any agents for inspection purposes.
"""

import pygame
import sys
import os
import multiprocessing
import math
import time
import csv
import numpy as np
import pickle
from datetime import datetime

from multitrack.utils.config import *
from multitrack.models.simulation_environment import SimulationEnvironment
from multitrack.utils.map_graph import MapGraph
from multitrack.utils.pathfinding import find_shortest_path, calculate_path_length, smooth_path, create_dynamically_feasible_path, find_closest_node
from multitrack.utils.optimize_path import optimize_path_with_visibility

# Import probability overlay for in-game visualization
try:
    import probability_overlay
    OVERLAY_AVAILABLE = True
except ImportError:
    OVERLAY_AVAILABLE = False
    print("Warning: probability_overlay module not available. Overlay functionality disabled.")

# PHASE 4B OPTIMIZATION: Agent2ComputationOptimizer for selective computation
class Agent2ComputationOptimizer:
    """
    Phase 4B optimization: Selective computation system to avoid unnecessary recomputations
    when agent state hasn't changed significantly.
    """
    def __init__(self):
        self.last_position = None
        self.last_angle = None
        self.last_result = None
        self.last_gap_probabilities = None
        
        # Configurable thresholds for detecting significant changes
        self.position_threshold = 5.0  # pixels - recompute if agent moves >5px
        self.angle_threshold = 0.1     # radians (~5.7 degrees) - recompute if angle changes significantly
        
        # Performance tracking
        self.computation_skipped = 0
        self.computation_performed = 0
        
        # Cache invalidation tracking
        self.frames_since_last_computation = 0
        self.max_frames_without_computation = 30  # Force recomputation every 30 frames max
    
    def needs_recomputation(self, current_pos, current_angle):
        """
        Determine if Agent2 computation needs to be performed based on state changes.
        
        Args:
            current_pos: Current agent position (x, y)
            current_angle: Current agent angle in radians
            
        Returns:
            bool: True if recomputation is needed, False if cached result can be used
        """
        # Always compute on first run
        if self.last_position is None:
            return True
            
        # Force recomputation after max frames to prevent stale data
        if self.frames_since_last_computation >= self.max_frames_without_computation:
            return True
        
        # Check position change
        pos_delta = math.sqrt((current_pos[0] - self.last_position[0])**2 + 
                             (current_pos[1] - self.last_position[1])**2)
        
        # Check angle change
        angle_delta = abs(current_angle - self.last_angle)
        # Handle angle wraparound (e.g., from 359° to 1°)
        angle_delta = min(angle_delta, 2*math.pi - angle_delta)
        
        return (pos_delta > self.position_threshold or 
                angle_delta > self.angle_threshold)
    
    def update_state(self, position, angle, gap_probabilities):
        """Update cached state after computation"""
        self.last_position = position
        self.last_angle = angle
        self.last_gap_probabilities = gap_probabilities.copy() if gap_probabilities else {}
        self.computation_performed += 1
        self.frames_since_last_computation = 0
    
    def skip_computation(self):
        """Record that computation was skipped and return cached result"""
        self.computation_skipped += 1
        self.frames_since_last_computation += 1
        return self.last_gap_probabilities if self.last_gap_probabilities else {}
    
    def get_performance_stats(self):
        """Get performance statistics for monitoring"""
        total_calls = self.computation_performed + self.computation_skipped
        skip_rate = self.computation_skipped / total_calls if total_calls > 0 else 0
        return {
            'total_calls': total_calls,
            'computations_performed': self.computation_performed,
            'computations_skipped': self.computation_skipped,
            'skip_rate_percent': skip_rate * 100,
            'frames_since_last': self.frames_since_last_computation
        }

# Global instance for Agent2 computation optimization
agent2_optimizer = Agent2ComputationOptimizer()

# PHASE 4B OPTIMIZATION: Mathematical shortcuts and fast approximations
class FastMathOptimizations:
    """
    Phase 4B optimization: Fast mathematical operations for improved performance.
    Provides approximations and optimized calculations that maintain visual quality.
    """
    
    @staticmethod
    def fast_distance_squared(p1, p2):
        """Fast squared distance calculation (avoids sqrt when only comparison needed)"""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy
    
    @staticmethod
    def fast_distance_check(p1, p2, threshold):
        """Fast distance check using squared distance to avoid sqrt"""
        return FastMathOptimizations.fast_distance_squared(p1, p2) < (threshold * threshold)
    
    @staticmethod
    def fast_normalize_angle(angle):
        """Fast angle normalization to [0, 2π] range"""
        while angle < 0:
            angle += 2 * math.pi
        while angle >= 2 * math.pi:
            angle -= 2 * math.pi
        return angle
    
    @staticmethod
    def fast_angle_difference(angle1, angle2):
        """Fast calculation of minimum angle difference (handles wraparound)"""
        diff = abs(angle1 - angle2)
        return min(diff, 2 * math.pi - diff)

class TrigLookupTable:
    """
    Phase 4B optimization: Pre-computed trigonometric lookup table for common angles.
    Trades memory for computation speed.
    """
    
    def __init__(self, precision=1000):
        self.precision = precision
        self.angle_step = 2 * math.pi / precision
        
        # Pre-compute sin/cos for common angles
        self.angles = np.linspace(0, 2 * math.pi, precision, endpoint=False)
        self.cos_table = np.cos(self.angles)
        self.sin_table = np.sin(self.angles)
    
    def fast_cos(self, angle):
        """Fast cosine lookup with linear interpolation"""
        # Normalize angle to [0, 2π]
        angle = angle % (2 * math.pi)
        
        # Find nearest table index
        index = int(angle / self.angle_step)
        index = min(index, self.precision - 1)
        
        return self.cos_table[index]
    
    def fast_sin(self, angle):
        """Fast sine lookup with linear interpolation"""
        # Normalize angle to [0, 2π]
        angle = angle % (2 * math.pi)
        
        # Find nearest table index
        index = int(angle / self.angle_step)
        index = min(index, self.precision - 1)
        
        return self.sin_table[index]

# Global instances for mathematical optimizations
fast_math = FastMathOptimizations()
trig_lookup = TrigLookupTable(precision=3600)  # 0.1 degree precision

def point_to_line_distance(point, line_start, line_end):
    """
    Calculate the shortest distance from a point to a line segment.
    
    Args:
        point: Tuple (x, y) representing the point
        line_start: Tuple (x, y) representing the start of the line segment
        line_end: Tuple (x, y) representing the end of the line segment
    
    Returns:
        float: The shortest distance from the point to the line segment
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    
    if len_sq != 0:
        param = dot / len_sq
    
    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    
    dx = px - xx
    dy = py - yy
    return math.sqrt(dx * dx + dy * dy)

def run_environment_inspection(multicore=True, num_cores=None, auto_analyze=False, load_visibility=False, visibility_cache_file=None):
    pygame.init()

    # Import dimensions from the simulation module
    from multitrack.simulation.unicycle_reachability_simulation import (
        ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, WIDTH, HEIGHT, SIDEBAR_WIDTH,
        MAP_GRAPH_GRID_SIZE, MAP_GRAPH_MAX_EDGE_DISTANCE, MAP_GRAPH_MAX_CONNECTIONS
    )
    
    # Import cache settings
    from multitrack.utils.config import (
        MAP_GRAPH_CACHE_ENABLED, 
        MAP_GRAPH_CACHE_FILE,
        MAP_GRAPH_INSPECTION_CACHE_FILE,
        MAP_GRAPH_VISIBILITY_CACHE_FILE,
        MAP_GRAPH_VISIBILITY_RANGE
    )

    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Environment Inspection")

    # Initialize the environment
    environment = SimulationEnvironment(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT)

    # --- AGENT SETUP ---
    from multitrack.models.agents.visitor_agent import UnicycleModel
    
    AGENT_STATE_FILE = os.path.join(os.path.dirname(__file__), 'agent_state.pkl')
    AGENT2_STATE_FILE = os.path.join(os.path.dirname(__file__), 'agent2_state.pkl')
    
    # Try to load first agent state from file
    agent_state = None
    if os.path.exists(AGENT_STATE_FILE):
        try:
            with open(AGENT_STATE_FILE, 'rb') as f:
                agent_state = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load agent state: {e}")
    if agent_state is not None:
        agent = UnicycleModel(initial_position=agent_state[:2], walls=environment.get_all_walls(), doors=environment.get_doors())
        agent.state = agent_state.copy()
    else:
        agent = UnicycleModel(walls=environment.get_all_walls(), doors=environment.get_doors())
    
    # Try to load second agent state from file
    agent2_state = None
    if os.path.exists(AGENT2_STATE_FILE):
        try:
            with open(AGENT2_STATE_FILE, 'rb') as f:
                agent2_state = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load agent2 state: {e}")
    if agent2_state is not None:
        agent2 = UnicycleModel(initial_position=agent2_state[:2], walls=environment.get_all_walls(), doors=environment.get_doors())
        agent2.state = agent2_state.copy()
    else:
        agent2 = UnicycleModel(walls=environment.get_all_walls(), doors=environment.get_doors())
    
    AGENT_LINEAR_VEL = LEADER_LINEAR_VEL
    AGENT_ANGULAR_VEL = LEADER_ANGULAR_VEL
    AGENT_COLOR = (255, 0, 255)  # Bright magenta for maximum visibility (first agent)
    AGENT2_COLOR = (0, 255, 255)  # Bright cyan for agent 2 (contrasting with magenta visibility indicators)
    AGENT_RADIUS = 16

    # Initialize font for display
    font = pygame.font.SysFont('Arial', 14)

    # Clock for controlling the frame rate
    clock = pygame.time.Clock()
    
    # Function to draw path between nodes
    def draw_path(path, color=(255, 215, 0), width=4, start_color=None, end_color=None, dashed=False):
        """
        Draw a path on the screen
        
        Args:
            path: List of points [(x, y)] forming the path
            color: RGB color tuple for the path
            width: Width of the path line
            start_color: Color for start point (defaults to green if None)
            end_color: Color for end point (defaults to red if None)
            dashed: Whether to draw a dashed line instead of solid
        """
        if not path or len(path) < 2:
            return
        
        # Create a surface for the path
        path_surface = pygame.Surface((ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT), pygame.SRCALPHA)
        
        # Draw lines connecting the path points
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            
            if dashed:
                # Draw dashed line
                dash_length = 5
                space_length = 5
                
                # Calculate direction vector
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Normalize direction vector
                if dist > 0:
                    dx, dy = dx/dist, dy/dist
                else:
                    continue  # Skip zero-length segments
                
                # Draw dash segments
                pos = start
                step = 0
                while step < dist:
                    # Calculate dash start and end
                    dash_start = (pos[0], pos[1])
                    dash_end_dist = min(step + dash_length, dist)
                    dash_end = (start[0] + dx * dash_end_dist, start[1] + dy * dash_end_dist)
                    
                    # Draw dash
                    pygame.draw.line(path_surface, color, dash_start, dash_end, width)
                    
                    # Move to start of next dash
                    step += dash_length + space_length
                    pos = (start[0] + dx * step, start[1] + dy * step)
            else:
                # Draw solid line
                pygame.draw.line(path_surface, color, start, end, width)
        
        # Draw circles at each node in the path
        for point in path:
            pygame.draw.circle(path_surface, color, point, width + 1)
        
        # Use provided colors or defaults for start and end
        start_color = start_color or (0, 255, 0)  # Default: green start
        end_color = end_color or (255, 0, 0)      # Default: red end
        
        # Highlight start and end points
        pygame.draw.circle(path_surface, start_color, path[0], width + 2)  # Start point
        pygame.draw.circle(path_surface, end_color, path[-1], width + 2)  # End point
        
        # Blit the path surface to the screen
        screen.blit(path_surface, (0, 0))
    
    # Loading status display function
    def update_loading_status(message, progress):
        screen.fill((0, 0, 0))
        
        # Create loading text
        loading_font = pygame.font.SysFont('Arial', 24)
        loading_text = loading_font.render(f"{message} {progress*100:.1f}%", True, (255, 255, 255))
        
        # Position text in center of screen
        text_rect = loading_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(loading_text, text_rect)
        
        # Draw progress bar
        bar_width = WIDTH // 2
        bar_height = 20
        bar_x = (WIDTH - bar_width) // 2
        bar_y = HEIGHT // 2 + 30
        
        # Outline
        pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height), 2)
        # Fill
        pygame.draw.rect(screen, (0, 255, 0), (bar_x, bar_y, int(bar_width * progress), bar_height))
        
        pygame.display.flip()
    
    # Function to display an animated loading screen with more details
    def render_animated_loading_screen(message, progress, additional_text=None):
        screen.fill((20, 20, 30))  # Dark blue background
        
        # Create title
        title_font = pygame.font.SysFont('Arial', 30)
        title_text = title_font.render("Environment Inspection Tool", True, (200, 200, 255))
        title_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 4))
        screen.blit(title_text, title_rect)
        
        # Create loading text
        loading_font = pygame.font.SysFont('Arial', 24)
        loading_text = loading_font.render(f"{message} {progress*100:.1f}%", True, (255, 255, 255))
        text_rect = loading_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(loading_text, text_rect)
        
        # Draw progress bar
        bar_width = WIDTH // 2
        bar_height = 20
        bar_x = (WIDTH - bar_width) // 2
        bar_y = HEIGHT // 2 + 30
        
        # Outline
        pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height), 2)
        # Fill with gradient
        for i in range(int(bar_width * progress)):
            # Create a gradient from green to blue
            color_r = int(max(0, 100 - (i / bar_width) * 100))
            color_g = int(min(255, 100 + (i / bar_width) * 155))
            color_b = int(min(255, 100 + (i / bar_width) * 155))
            pygame.draw.line(screen, (color_r, color_g, color_b), (bar_x + i, bar_y), (bar_x + i, bar_y + bar_height))
        
        # Display additional text if provided
        if additional_text:
            info_font = pygame.font.SysFont('Arial', 18)
            y_offset = HEIGHT // 2 + 70
            for line in additional_text:
                info_text = info_font.render(line, True, (200, 200, 200))
                info_rect = info_text.get_rect(center=(WIDTH // 2, y_offset))
                screen.blit(info_text, info_rect)
                y_offset += 25
        
        # Add animated spinner
        current_time = pygame.time.get_ticks()
        spinner_radius = 15
        spinner_pos = (WIDTH // 2, HEIGHT // 2 + 100)
        spinner_angle = (current_time / 30) % 360  # Rotate based on time
        
        # Draw spinner segments
        for i in range(8):
            angle = spinner_angle + i * 45
            color_intensity = 100 + (i * 20)
            end_pos = (
                spinner_pos[0] + spinner_radius * math.cos(math.radians(angle)),
                spinner_pos[1] + spinner_radius * math.sin(math.radians(angle))
            )
            pygame.draw.line(
                screen, 
                (min(255, color_intensity), min(255, color_intensity), 255), 
                spinner_pos, 
                end_pos, 
                3
            )
        
        pygame.display.flip()
        pygame.event.pump()  # Process events to prevent "not responding"
    
    # Check if we can load from cache first
    update_loading_status("Checking for cached map graph...", 0.0)
    cache_loaded = False
    
    # Initialize map graph with inspection-specific cache file
    map_graph = MapGraph(
        ENVIRONMENT_WIDTH, 
        ENVIRONMENT_HEIGHT, 
        environment.get_all_walls(), 
        environment.get_doors(),
        cache_file=MAP_GRAPH_INSPECTION_CACHE_FILE  # Use inspection-specific cache file
    )
    
    # Initialize visibility map as empty dict
    visibility_map = {}
    selected_node_index = None
    
    # Agent-following functionality
    follow_agent_mode = False  # Toggle for agent-following mode
    agent_last_position = None  # Track agent's last position
    agent_movement_threshold = 20.0  # Minimum distance to consider agent has moved
    agent_following_node_index = None  # Current node closest to agent when following
    
    # Initialize path visualization variables
    show_path = False
    current_path = None
    dynamic_path = None  # Added for dynamic path
    path_end_index = None
    path_length = 0.0
    dynamic_path_length = 0.0  # Added for dynamic path
    
    # Initialize simple probability overlay variable
    show_probability_overlay = False
    
    # Initialize visibility gaps display variable
    show_visibility_gaps = False
    
    # Initialize visibility gaps display and probability overlay for second agent
    show_agent2_visibility_gaps = False
    show_agent2_probability_overlay = False
    
    # Initialize agent 2 rotating rods display variable
    show_agent2_rods = False
    
    # Initialize extended probability set (gap arcs) display variable
    show_extended_probability_set = False
    
    # Initialize rotating rods display variable
    show_rotating_rods = False
    
    # Initialize combined probability mode (multiply Agent 1 and Agent 2 probabilities)
    show_combined_probability_mode = False
    
    # Initialize threat classification mode
    show_threat_classification = False
    threat_threshold = 0.1  # Same threshold as combined probability visualization (was 0.3)
    
    # Time horizon parameters for probability overlay
    time_horizon = 4.0  # Look-ahead time in seconds
    min_time_horizon = 0.5  # Minimum time horizon
    max_time_horizon = 10.0  # Maximum time horizon
    time_horizon_step = 0.5  # Step size for adjusting time horizon
    
    # Try to load from cache if enabled
    if MAP_GRAPH_CACHE_ENABLED:
        print("Attempting to load inspection map graph from cache...")
        cache_loaded = map_graph.load_from_cache()
        if cache_loaded:
            print(f"Successfully loaded map graph from cache with {len(map_graph.nodes)} nodes and {len(map_graph.edges)} edges.")
        else:
            print("Cache loading failed or no valid cache found.")
    
    # Generate map graph if not loaded from cache
    if not cache_loaded:
        print("Generating new map graph...")
        if multicore:
            cores_to_use = num_cores if num_cores else multiprocessing.cpu_count()
            print(f"Generating map graph using {cores_to_use} CPU cores...")
            map_graph.generate_parallel(update_loading_status, cores_to_use)
        else:
            print("Generating map graph using single core...")
            map_graph.generate(update_loading_status)
        
        print(f"Map graph generated with {len(map_graph.nodes)} nodes and {len(map_graph.edges)} edges.")
        
        # Save to cache for future use if caching is enabled
        if MAP_GRAPH_CACHE_ENABLED:
            update_loading_status("Saving inspection map graph to cache...", 0.95)
            map_graph.save_to_cache()

    running = True
    map_graph_enabled = True  # Map graph functionality always enabled
    show_map_graph_visuals = True  # Start with map graph visuals visible
    
    # FPS tracking variables
    show_fps = True
    frame_times = []
    fps = 0
    avg_frame_time = 0
    frame_start_time = 0
    
    # Agent 2 computation timing
    agent2_computation_ms = 0
    
    # Handle automatic loading and analysis of visibility data if requested
    if load_visibility:
        print("Auto-loading visibility data requested...")
        visibility_map = map_graph.load_visibility_data(
            render_animated_loading_screen,
            visibility_cache_file or MAP_GRAPH_VISIBILITY_CACHE_FILE
        )
        if visibility_map:
            total_connections = sum(len(nodes) for nodes in visibility_map.values())
            print(f"Loaded visibility data: {total_connections:,} total sight lines")
            # Set initial selected node
            if selected_node_index is None and map_graph.nodes:
                selected_node_index = 0
        else:
            print("Could not load visibility data - continuing without it")
            
    # Handle automatic visibility analysis if requested
    if auto_analyze and not visibility_map:
        print("Auto-analyzing visibility requested...")
        # Use the animated loading screen for better feedback
        cores_to_use = num_cores if num_cores else None
        visibility_map = map_graph.analyze_node_visibility(
            MAP_GRAPH_VISIBILITY_RANGE,
            render_animated_loading_screen,
            cores_to_use,
            visibility_cache_file or MAP_GRAPH_VISIBILITY_CACHE_FILE
        )
        if visibility_map:
            # Set initial selected node after analysis
            if selected_node_index is None and map_graph.nodes:
                selected_node_index = 0
    
    # Function to calculate and display detailed stats during visibility analysis
    def calculate_visibility_stats(visibility_map, total_nodes):
        """Calculate statistics about the visibility data"""
        if not visibility_map:
            return []
            
        # Calculate statistics
        total_connections = sum(len(nodes) for nodes in visibility_map.values())
        avg_connections = total_connections / len(visibility_map) if visibility_map else 0
        max_connections = max(len(nodes) for nodes in visibility_map.values()) if visibility_map else 0
        visibility_pct = 100 * avg_connections / total_nodes if total_nodes > 0 else 0
        
        # Max visibility node
        max_visibility_node = max(visibility_map.items(), key=lambda x: len(x[1]))[0] if visibility_map else None
        
        return [
            f"Visibility Statistics:",
            f"Total sight lines: {total_connections:,}",
            f"Average visible nodes: {avg_connections:.1f}",
            f"Visibility percentage: {visibility_pct:.1f}%",
            f"Maximum connections: {max_connections}",
            f"Most visible node: {max_visibility_node}"
        ]
    
    try:
        while running:
            frame_start_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_q:
                        # Toggle FPS display
                        show_fps = not show_fps
                    elif event.key == pygame.K_g:
                        # Toggle map graph visual display (functionality remains enabled)
                        show_map_graph_visuals = not show_map_graph_visuals
                        print(f"Map graph visuals: {'On' if show_map_graph_visuals else 'Off'}")
                    elif event.key == pygame.K_r and map_graph_enabled:
                        # Regenerate map graph
                        print("Regenerating map graph...")
                        # Create new map graph with the current configuration parameters
                        map_graph = MapGraph(
                            ENVIRONMENT_WIDTH, 
                            ENVIRONMENT_HEIGHT, 
                            environment.get_all_walls(), 
                            environment.get_doors(),
                            cache_file=MAP_GRAPH_INSPECTION_CACHE_FILE  # Use inspection-specific cache file
                        )
                        if multicore:
                            cores_to_use = num_cores if num_cores else multiprocessing.cpu_count()
                            map_graph.generate_parallel(update_loading_status, cores_to_use)
                        else:
                            map_graph.generate(update_loading_status)
                        print(f"Map graph regenerated with {len(map_graph.nodes)} nodes and {len(map_graph.edges)} edges.")
                        
                        # Save to cache if enabled
                        if MAP_GRAPH_CACHE_ENABLED:
                            print(f"Saving regenerated map graph to inspection cache file")
                            try:
                                map_graph.save_to_cache()
                                print("Inspection map graph successfully saved to cache.")
                            except Exception as e:
                                print(f"Error saving inspection map graph to cache: {e}")
                    elif event.key == pygame.K_v:
                        # Generate and save visibility information
                        print(f"Analyzing node visibility (range: {MAP_GRAPH_VISIBILITY_RANGE} pixels)...")
                        # Show a message on screen
                        message_font = pygame.font.SysFont('Arial', 30)
                        message = message_font.render("Preparing visibility analysis...", True, (255, 255, 255))
                        screen.blit(message, (WIDTH // 2 - message.get_width() // 2, HEIGHT // 2 - message.get_height() // 2))
                        pygame.display.flip()
                        
                        # Give the UI a moment to update before starting intensive processing
                        pygame.event.pump()
                        time.sleep(0.1)
                        
                        # Start the visibility analysis with the improved animated loading screen
                        # Pass num_cores to allow command-line control of parallelism
                        cores_to_use = num_cores if num_cores else None
                        visibility_map = map_graph.analyze_node_visibility(
                            MAP_GRAPH_VISIBILITY_RANGE,
                            render_animated_loading_screen,
                            cores_to_use,
                            visibility_cache_file or MAP_GRAPH_VISIBILITY_CACHE_FILE
                        )
                        if visibility_map:
                            total_connections = sum(len(nodes) for nodes in visibility_map.values())
                            print(f"Visibility analysis complete: {total_connections} total sight lines")
                            
                            # Calculate and display visibility statistics
                            visibility_stats = calculate_visibility_stats(visibility_map, len(map_graph.nodes))
                            for stat in visibility_stats:
                                print(stat)
                            
                            # Select first node if none selected
                            if selected_node_index is None and map_graph.nodes:
                                selected_node_index = 0
                    
                    elif event.key == pygame.K_l:
                        # Load visibility data from cache
                        print("Loading visibility data from cache...")
                        message_font = pygame.font.SysFont('Arial', 30)
                        message = message_font.render("Loading visibility data...", True, (255, 255, 255))
                        screen.blit(message, (WIDTH // 2 - message.get_width() // 2, HEIGHT // 2 - message.get_height() // 2))
                        pygame.display.flip()
                        
                        # Pass the animated loading screen function for better feedback
                        visibility_map = map_graph.load_visibility_data(
                            render_animated_loading_screen,
                            visibility_cache_file or MAP_GRAPH_VISIBILITY_CACHE_FILE
                        )
                        if visibility_map:
                            total_connections = sum(len(nodes) for nodes in visibility_map.values())
                            print(f"Loaded visibility data: {total_connections} total sight lines")
                            
                            # Calculate and display visibility statistics
                            visibility_stats = calculate_visibility_stats(visibility_map, len(map_graph.nodes))
                            for stat in visibility_stats:
                                print(stat)
                            
                            # Select first node if none selected
                            if selected_node_index is None and map_graph.nodes:
                                selected_node_index = 0
                        else:
                            print("No visibility data found. Press V to analyze visibility.")
                        
                    elif event.key == pygame.K_n:
                        # Go to next node in map graph
                        if map_graph.nodes:
                            if selected_node_index is None:
                                selected_node_index = 0
                            else:
                                selected_node_index = (selected_node_index + 1) % len(map_graph.nodes)
                            
                            if visibility_map and selected_node_index in visibility_map:
                                visible_count = len(visibility_map[selected_node_index])
                                print(f"Selected node {selected_node_index} with {visible_count} visible nodes")
                            else:
                                print(f"Selected node {selected_node_index} (no visibility data)")
                    
                    elif event.key == pygame.K_p:
                        # Go to previous node in map graph
                        if map_graph.nodes:
                            if selected_node_index is None:
                                selected_node_index = len(map_graph.nodes) - 1
                            else:
                                selected_node_index = (selected_node_index - 1) % len(map_graph.nodes)
                            
                            if visibility_map and selected_node_index in visibility_map:
                                visible_count = len(visibility_map[selected_node_index])
                                print(f"Selected node {selected_node_index} with {visible_count} visible nodes")
                            else:
                                print(f"Selected node {selected_node_index} (no visibility data)")
                    elif event.key == pygame.K_t:
                        # Toggle path display
                        if current_path:
                            show_path = not show_path
                            print(f"Path display: {'On' if show_path else 'Off'}")
                        else:
                            print("No path to toggle. Right-click on a node to set a destination.")
                    elif event.key == pygame.K_c:
                        # Clear the current path
                        if current_path:
                            current_path = None
                            dynamic_path = None
                            path_end_index = None
                            path_length = 0.0
                            dynamic_path_length = 0.0
                            show_path = False
                            print("Path cleared")
                        else:
                            print("No path to clear")
                    elif event.key == pygame.K_f:
                        # Toggle agent-following mode
                        follow_agent_mode = not follow_agent_mode
                        if follow_agent_mode:
                            print("Agent-following mode: ON - Visibility will track agent movement")
                            # Initialize agent position tracking
                            agent_last_position = (agent.state[0], agent.state[1])
                            # Find closest node to current agent position
                            agent_pos = (agent.state[0], agent.state[1])
                            agent_following_node_index = find_closest_node(map_graph.nodes, agent_pos)
                            if agent_following_node_index is not None and visibility_map:
                                selected_node_index = agent_following_node_index
                                print(f"Following agent at node {selected_node_index}")
                        else:
                            print("Agent-following mode: OFF - Manual node selection enabled")
                    elif event.key == pygame.K_o:
                        # Toggle probability overlay
                        show_probability_overlay = not show_probability_overlay
                        if show_probability_overlay:
                            print("Agent 1 probability: ON - Showing distance-based probability")
                        else:
                            print("Agent 1 probability: OFF")
                    elif event.key == pygame.K_b:
                        # Toggle visibility gaps display for first agent
                        show_visibility_gaps = not show_visibility_gaps
                        if show_visibility_gaps:
                            print("Agent 1 visibility: ON - Blue gaps for discontinuities")
                        else:
                            print("Agent 1 visibility: OFF")
                    elif event.key == pygame.K_j:
                        # Toggle probability overlay for second agent
                        show_agent2_probability_overlay = not show_agent2_probability_overlay
                        if show_agent2_probability_overlay:
                            print("Agent 2 probability: ON - Pink-green blend, dynamic visibility range")
                            # Automatically turn off visibility gaps when enabling probability overlay
                            show_agent2_visibility_gaps = False
                            # Automatically show rotating rods for agent 2
                            show_agent2_rods = True
                            print("Agent 2 rotating rods: ON")
                        else:
                            print("Agent 2 probability: OFF")
                            # Turn off agent 2's rods when disabling probability overlay
                            show_agent2_rods = False
                    elif event.key == pygame.K_k:
                        # Toggle visibility gaps display for second agent
                        show_agent2_visibility_gaps = not show_agent2_visibility_gaps
                        if show_agent2_visibility_gaps:
                            print("Agent 2 visibility: ON - Cyan gaps and dynamic range circle")
                            # Automatically turn off probability overlay when enabling visibility gaps
                            show_agent2_probability_overlay = False
                            # Turn off agent 2's rods when enabling visibility gaps
                            show_agent2_rods = False
                        else:
                            print("Agent 2 visibility: OFF")
                    elif event.key == pygame.K_h:
                        # Toggle extended probability set (gap arcs) display for agent 1 only
                        show_extended_probability_set = not show_extended_probability_set
                        if show_extended_probability_set:
                            print("Extended probability: ON - Showing gap arcs for agent 1")
                        else:
                            print("Extended probability: OFF - Showing green visibility lines for agent 1")
                    elif event.key == pygame.K_y:
                        # Toggle rotating rods display
                        show_rotating_rods = not show_rotating_rods
                        if show_rotating_rods:
                            print("Rotating rods: ON - Showing temporal coverage during rotation")
                            # Reset agent2_rods when global rods are on
                            show_agent2_rods = False
                            # Check if prerequisites are met
                            if not show_probability_overlay:
                                print("  Warning: Probability overlay OFF. Press O first.")
                            elif not visibility_map:
                                print("  Warning: No visibility data. Press V or L.")
                            elif selected_node_index is None or selected_node_index not in visibility_map:
                                print("  Warning: No valid node selected. Click a node.")
                            else:
                                print("  Ready - rods visible when gaps detected.")
                        else:
                            print("Rotating rods: OFF")
                    elif event.key == pygame.K_z:
                        # Z key: Auto-enable agent 1 features (F+O+B+Y) + H (extended probability set)
                        print("Z: Enabling agent 1 features (F+O+B+Y+H)...")
                        
                        # 1. Enable agent-following mode (F key)
                        if not follow_agent_mode:
                            follow_agent_mode = True
                            print("✓ Agent-following: ON")
                            # Initialize agent position tracking
                            agent_last_position = (agent.state[0], agent.state[1])
                            # Find closest node to current agent position
                            agent_pos = (agent.state[0], agent.state[1])
                            agent_following_node_index = find_closest_node(map_graph.nodes, agent_pos)
                            if agent_following_node_index is not None and visibility_map:
                                selected_node_index = agent_following_node_index
                                print(f"  Following at node {selected_node_index}")
                        else:
                            print("✓ Agent-following: Already ON")
                        
                        # 2. Enable probability overlay (O key)
                        if not show_probability_overlay:
                            show_probability_overlay = True
                            print("✓ Probability overlay: ON")
                        else:
                            print("✓ Probability overlay: Already ON")
                        
                        # 3. Enable visibility gaps display (B key)
                        if not show_visibility_gaps:
                            show_visibility_gaps = True
                            print("✓ Visibility gaps: ON")
                        else:
                            print("✓ Visibility gaps: Already ON")
                        
                        # 4. Enable rotating rods display (Y key)
                        if not show_rotating_rods:
                            show_rotating_rods = True
                            print("✓ Rotating rods: ON")
                        else:
                            print("✓ Rotating rods: Already ON")
                        
                        # 5. Enable extended probability set (H key)
                        if not show_extended_probability_set:
                            show_extended_probability_set = True
                            print("✓ Extended probability set: ON")
                        else:
                            print("✓ Extended probability set: Already ON")
                        
                        # Check for prerequisites and warn if any are missing
                        if not visibility_map:
                            print("⚠ Warning: No visibility data. Press V to analyze or L to load.")
                        elif selected_node_index is None or selected_node_index not in visibility_map:
                            print("⚠ Warning: No valid node selected. Move agent or click on a node.")
                        else:
                            print("✓ All agent 1 features enabled successfully! System ready.")
                        
                        print("Complete agent 1 system activated. Use arrow keys to move and observe the enhanced visualization.")
                    elif event.key == pygame.K_m:
                        # M key: Toggle combined probability mode (multiply Agent 1 and Agent 2 probabilities)
                        show_combined_probability_mode = not show_combined_probability_mode
                        if show_combined_probability_mode:
                            print("Combined probability mode: ON")
                            print("  Multiplying Agent 1 and Agent 2 probabilities")
                            print("  Purple-yellow color scheme: low to high combined probability")
                            print("  Note: Individual probability overlays will be hidden")
                            # Auto-enable required modes for both agents
                            if not show_probability_overlay:
                                show_probability_overlay = True
                                print("  Auto-enabled Agent 1 probability overlay")
                            if not show_agent2_probability_overlay:
                                show_agent2_probability_overlay = True
                                show_agent2_rods = True
                                print("  Auto-enabled Agent 2 probability overlay")
                        else:
                            print("Combined probability mode: OFF")
                            print("  Individual probability overlays restored")
                    elif event.key == pygame.K_u:
                        # U key: Toggle threat classification mode
                        show_threat_classification = not show_threat_classification if 'show_threat_classification' in locals() else True
                        if show_threat_classification:
                            print("Threat classification mode: ON")
                            print("  Rod-based threat analysis with color coding")
                            print(f"  Threat threshold: {threat_threshold:.2f}")
                            print("  Rod colors: Red, Green, Blue, Orange, Magenta, Cyan, Yellow, Purple")
                            print("  White nodes: Visibility-only threats (no rod involvement)")
                            print("  Multiple small dots: Overlapping rod contributions")
                            # Auto-enable required modes
                            if not show_combined_probability_mode:
                                show_combined_probability_mode = True
                                print("  Auto-enabled combined probability mode")
                            if not show_agent2_probability_overlay:
                                show_agent2_probability_overlay = True
                                show_agent2_rods = True
                                print("  Auto-enabled Agent 2 probability overlay and rods")
                        else:
                            show_threat_classification = False
                            print("Threat classification mode: OFF")
                            print("  Restored standard combined probability visualization")
                    elif event.key == pygame.K_x:
                        # X key: Combined Z+G+M functionality (Auto-enable complete dual-agent system)
                        print("X: Activating complete dual-agent visualization system (Z+G+M)...")
                        
                        # 1. Execute Z key functionality: Auto-enable agent 1 features (F+O+B+Y+H)
                        print("Step 1/3: Enabling agent 1 features (F+O+B+Y+H)...")
                        
                        # Enable agent-following mode (F key)
                        if not follow_agent_mode:
                            follow_agent_mode = True
                            print("  ✓ Agent-following: ON")
                            # Initialize agent position tracking
                            agent_last_position = (agent.state[0], agent.state[1])
                            # Find closest node to current agent position
                            agent_pos = (agent.state[0], agent.state[1])
                            agent_following_node_index = find_closest_node(map_graph.nodes, agent_pos)
                            if agent_following_node_index is not None and visibility_map:
                                selected_node_index = agent_following_node_index
                                print(f"    Following at node {selected_node_index}")
                        else:
                            print("  ✓ Agent-following: Already ON")
                        
                        # Enable probability overlay (O key)
                        if not show_probability_overlay:
                            show_probability_overlay = True
                            print("  ✓ Probability overlay: ON")
                        else:
                            print("  ✓ Probability overlay: Already ON")
                        
                        # Enable visibility gaps display (B key)
                        if not show_visibility_gaps:
                            show_visibility_gaps = True
                            print("  ✓ Visibility gaps: ON")
                        else:
                            print("  ✓ Visibility gaps: Already ON")
                        
                        # Enable rotating rods display (Y key)
                        if not show_rotating_rods:
                            show_rotating_rods = True
                            print("  ✓ Rotating rods: ON")
                        else:
                            print("  ✓ Rotating rods: Already ON")
                        
                        # Enable extended probability set (H key)
                        if not show_extended_probability_set:
                            show_extended_probability_set = True
                            print("  ✓ Extended probability set: ON")
                        else:
                            print("  ✓ Extended probability set: Already ON")
                        
                        # 2. Execute G key functionality: Enable map graph visuals
                        print("Step 2/3: Enabling map graph display...")
                        if not show_map_graph_visuals:
                            show_map_graph_visuals = True
                            print("  ✓ Map graph visuals: ON")
                        else:
                            print("  ✓ Map graph visuals: Already ON")
                        
                        # 3. Execute M key functionality: Enable combined probability mode
                        print("Step 3/3: Enabling combined probability mode...")
                        if not show_combined_probability_mode:
                            show_combined_probability_mode = True
                            print("  ✓ Combined probability mode: ON")
                            print("    Multiplying Agent 1 and Agent 2 probabilities")
                            print("    Purple-yellow color scheme: low to high combined probability")
                            # Auto-enable required modes for both agents
                            if not show_agent2_probability_overlay:
                                show_agent2_probability_overlay = True
                                show_agent2_rods = True
                                print("    Auto-enabled Agent 2 probability overlay")
                        else:
                            print("  ✓ Combined probability mode: Already ON")
                        
                        # Final status check
                        print("\n🎯 COMPLETE DUAL-AGENT SYSTEM ACTIVATED! 🎯")
                        if not visibility_map:
                            print("⚠ Warning: No visibility data. Press V to analyze or L to load.")
                        elif selected_node_index is None or selected_node_index not in visibility_map:
                            print("⚠ Warning: No valid node selected. Move agent or click on a node.")
                        else:
                            print("✅ All systems operational! Use arrow keys to move and observe the complete visualization.")
                        
                        print("Features active: Agent following + Probability overlays + Visibility gaps + Rotating rods + Map graph + Combined probability mode")
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        # Increase time horizon
                        if show_probability_overlay:
                            time_horizon = min(max_time_horizon, time_horizon + time_horizon_step)
                            print(f"Time horizon: {time_horizon:.1f}s (max reach: {time_horizon * LEADER_LINEAR_VEL:.0f} pixels)")
                        else:
                            print("Enable probability overlay (O) to adjust time horizon")
                    elif event.key == pygame.K_MINUS:
                        # Decrease time horizon
                        if show_probability_overlay:
                            time_horizon = max(min_time_horizon, time_horizon - time_horizon_step)
                            print(f"Time horizon: {time_horizon:.1f}s (max reach: {time_horizon * LEADER_LINEAR_VEL:.0f} pixels)")
                        else:
                            print("Enable probability overlay (O) to adjust time horizon")
                    elif event.key == pygame.K_i:
                        # I key: Toggle probability overlay
                        if OVERLAY_AVAILABLE:
                            if show_combined_probability_mode:
                                overlay_enabled = probability_overlay.toggle_probability_overlay()
                                print(f"Probability overlay: {'ON' if overlay_enabled else 'OFF'}")
                                if overlay_enabled:
                                    print("Overlay will show combined probability distribution in real-time")
                            else:
                                print("Enable combined probability mode (M key) first to show overlay.")
                        else:
                            print("Overlay functionality not available.")

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and map_graph_enabled:  # Left mouse button
                        # Get mouse position
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        
                        # Find the closest node to the mouse click
                        closest_node_index = None
                        closest_distance = float('inf')
                        
                        # Only search within a limited area for better performance
                        click_search_radius = 20  # pixels
                        
                        for i, node in enumerate(map_graph.nodes):
                            node_x, node_y = node
                            # Quick bounds check before calculating exact distance
                            if abs(node_x - mouse_x) > click_search_radius or abs(node_y - mouse_y) > click_search_radius:
                                continue
                                
                            # Calculate exact distance
                            distance = ((node_x - mouse_x) ** 2 + (node_y - mouse_y) ** 2) ** 0.5
                            
                            # Only consider nodes within the click radius
                            if distance < click_search_radius and distance < closest_distance:
                                closest_distance = distance
                                closest_node_index = i
                        
                        # If a node was clicked, select it
                        if closest_node_index is not None:
                            # Store previous selection to show changes
                            previous_node_index = selected_node_index
                            selected_node_index = closest_node_index
                            
                            # Get node coordinates for better feedback
                            node_x, node_y = map_graph.nodes[selected_node_index]
                            
                            # Get visibility information
                            visible_nodes = []
                            if visibility_map and selected_node_index in visibility_map:
                                visible_nodes = visibility_map[selected_node_index]
                                visible_count = len(visible_nodes)
                                
                                # Calculate percentage of nodes visible
                                visibility_percentage = (visible_count / len(map_graph.nodes)) * 100
                                
                                print(f"Selected node {selected_node_index} at ({int(node_x)}, {int(node_y)})")
                                print(f"Visibility: {visible_count:,} nodes ({visibility_percentage:.1f}% of all nodes)")
                                
                                # If this is a new selection, play a subtle sound effect for feedback
                                if previous_node_index != selected_node_index:
                                    # Simple feedback with pygame.mixer if available
                                    try:
                                        pygame.mixer.init()
                                        pygame.mixer.music.stop()  # Stop any playing sound
                                        
                                        # Generate a quick beep
                                        frequency = 1000  # Hz
                                        duration = 100  # ms
                                        sample_rate = 44100  # Hz
                                        
                                        # Generate square wave (simple beep)
                                        buf = np.sin(2 * np.pi * np.arange(sample_rate * duration / 1000) * frequency / sample_rate)
                                        sound = pygame.sndarray.make_sound((buf * 32767).astype(np.int16))
                                        sound.set_volume(0.2)  # Keep volume low
                                        sound.play()
                                    except:
                                        # If sound fails, just continue silently
                                        pass
                            else:
                                print(f"Selected node {selected_node_index} at ({int(node_x)}, {int(node_y)})")
                                print("No visibility data available for this node.")
                                print("Press V to analyze visibility or L to load cached data.")
                        else:
                            print("No node found near the click position")
                    elif event.button == 3 and map_graph_enabled:  # Right mouse button
                        # Get mouse position
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        
                        # Find the closest node to the mouse click
                        target_node_index = None
                        closest_distance = float('inf')
                        click_search_radius = 20  # pixels
                        
                        for i, node in enumerate(map_graph.nodes):
                            node_x, node_y = node
                            if abs(node_x - mouse_x) > click_search_radius or abs(node_y - mouse_y) > click_search_radius:
                                continue
                                
                            distance = ((node_x - mouse_x) ** 2 + (node_y - mouse_y) ** 2) ** 0.5
                            
                            if distance < click_search_radius and distance < closest_distance:
                                closest_distance = distance
                                target_node_index = i
                        
                        # If a target node was clicked, find path from agent to target
                        if target_node_index is not None:
                            # Get agent's current position
                            agent_pos = (agent.state[0], agent.state[1])
                            
                            # Set the path end point
                            path_end_index = target_node_index
                            
                            # Find shortest path between agent and target node
                            current_path = find_shortest_path(
                                map_graph, 
                                agent_pos, 
                                map_graph.nodes[path_end_index]
                            )
                            
                            if current_path:
                                # Store the original path for comparison
                                original_path = current_path.copy()
                                original_path_length = calculate_path_length(original_path)
                                
                                # Optimize path using visibility data if available
                                if visibility_map:
                                    # Use visibility data to create shortcuts between nodes that can see each other
                                    current_path = optimize_path_with_visibility(current_path, visibility_map, map_graph)
                                    print(f"Path optimized with visibility data: {len(original_path)} → {len(current_path)} nodes")
                                
                                # Smooth the path to remove unnecessary points
                                current_path = smooth_path(current_path, max_points=15)
                                
                                # Calculate path length
                                path_length = calculate_path_length(current_path)
                                
                                # Calculate optimization gain
                                path_optimization_gain = original_path_length - path_length
                                
                                # Create dynamically feasible path that respects agent's motion constraints
                                dynamic_path = create_dynamically_feasible_path(
                                    current_path,
                                    agent.state,  # Pass the current agent state
                                    max_speed=AGENT_LINEAR_VEL,
                                    max_angular_vel=AGENT_ANGULAR_VEL,
                                    dt=0.1,
                                    sim_steps=1000  # Increase steps for longer paths
                                )
                                
                                # Calculate dynamic path length
                                dynamic_path_length = calculate_path_length(dynamic_path)
                                
                                show_path = True
                                print(f"Path found from agent to node {path_end_index}")
                                print(f"Original path: {len(current_path)} nodes, {path_length:.1f} pixels")
                                print(f"Dynamic path: {len(dynamic_path)} nodes, {dynamic_path_length:.1f} pixels")
                            else:
                                print("No path found from agent to the selected node")
                        else:
                            print("No node found near the right-click position")

            # --- AGENT CONTROL (after event handling, before drawing) ---
            keys = pygame.key.get_pressed()
            
            # First agent control (arrow keys)
            linear_vel = 0
            angular_vel = 0
            if keys[pygame.K_UP]:
                linear_vel = AGENT_LINEAR_VEL
            if keys[pygame.K_DOWN]:
                linear_vel = -AGENT_LINEAR_VEL
            if keys[pygame.K_LEFT]:
                angular_vel = -AGENT_ANGULAR_VEL
            if keys[pygame.K_RIGHT]:
                angular_vel = AGENT_ANGULAR_VEL
            agent.set_controls(linear_vel, angular_vel)
            agent.update(dt=0.1, walls=environment.get_all_walls(), doors=environment.get_doors())

            # Second agent control (WASD keys)
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
            agent2.set_controls(linear_vel2, angular_vel2)
            agent2.update(dt=0.1, walls=environment.get_all_walls(), doors=environment.get_doors())

            # Agent-following functionality
            if follow_agent_mode and visibility_map:
                current_agent_position = (agent.state[0], agent.state[1])
                
                # Check if agent has moved significantly
                agent_moved = False
                if agent_last_position is None:
                    agent_moved = True
                else:
                    distance_moved = ((current_agent_position[0] - agent_last_position[0]) ** 2 + 
                                    (current_agent_position[1] - agent_last_position[1]) ** 2) ** 0.5
                    if distance_moved >= agent_movement_threshold:
                        agent_moved = True
                
                # Update visibility display if agent moved significantly
                if agent_moved:
                    # Find the closest node to the agent's current position
                    new_closest_node = find_closest_node(map_graph.nodes, current_agent_position)
                    
                    # Only update if we found a node and it's different from current
                    if (new_closest_node is not None and 
                        new_closest_node != agent_following_node_index and
                        new_closest_node in visibility_map):
                        
                        agent_following_node_index = new_closest_node
                        selected_node_index = agent_following_node_index
                        agent_last_position = current_agent_position
                        
                        # Optional: Print debug info (can be removed later)
                        visible_count = len(visibility_map[selected_node_index])
                        print(f"Agent moved - now following node {selected_node_index} with {visible_count} visible nodes")

            # Debug: print agent state
            # print(f"Agent state: {agent.state}")

            # Clear the screen
            screen.fill((0, 0, 0))

            # Draw the environment
            environment.draw(screen, font)
            
            # Calculate node probabilities if probability overlay is enabled (INDEPENDENT of map graph visuals)
            node_probabilities = {}
            if show_probability_overlay and visibility_map and selected_node_index in visibility_map:
                    agent_x, agent_y = agent.state[0], agent.state[1]
                    agent_theta = agent.state[2]  # Agent's heading angle
                    
                    # Calculate maximum reachable distance based on time horizon and agent speed
                    agent_speed = LEADER_LINEAR_VEL  # Use leader speed from config
                    max_reachable_distance = time_horizon * agent_speed
                    
                    # Use the reachable distance directly for probability calculation
                    max_distance = max_reachable_distance
                    
                    # Get the list of nodes visible from the selected node
                    visible_node_indices = set(visibility_map[selected_node_index])
                    
                    # Check if time horizon is too restrictive or too permissive
                    reachable_count = 0
                    total_visible_count = len(visible_node_indices)
                    
                    # First pass: count reachable nodes
                    for i, node in enumerate(map_graph.nodes):
                        if i in visible_node_indices:
                            node_x, node_y = node
                            distance = math.sqrt((node_x - agent_x)**2 + (node_y - agent_y)**2)
                            if distance <= max_distance:
                                reachable_count += 1
                    
                    # Handle edge cases
                    edge_case_handled = False
                    if reachable_count == 0 and total_visible_count > 0:
                        # Time horizon too low - no nodes are reachable
                        # Set very small probabilities for closest visible nodes
                        closest_distances = []
                        for i, node in enumerate(map_graph.nodes):
                            if i in visible_node_indices:
                                node_x, node_y = node
                                distance = math.sqrt((node_x - agent_x)**2 + (node_y - agent_y)**2)
                                closest_distances.append((distance, i))
                        
                        # Sort by distance and assign small probabilities to closest 3 nodes
                        closest_distances.sort()
                        for idx, (dist, node_idx) in enumerate(closest_distances[:3]):
                            node_probabilities[node_idx] = 0.1 * (1.0 - idx * 0.3)  # 0.1, 0.07, 0.04
                        edge_case_handled = True
                    
                    elif reachable_count == total_visible_count and total_visible_count > 1:
                        # Time horizon too high - all visible nodes are reachable
                        # Use the current max_distance but ensure probabilities are well distributed
                        # Don't change max_distance here - let it use the time horizon-based value
                        edge_case_handled = False  # Let normal calculation proceed with time horizon distance
                    
                    # Normal probability calculation (or edge case for "all reachable")
                    if not edge_case_handled:
                        for i, node in enumerate(map_graph.nodes):
                            # Only calculate probability for nodes that are visible from the selected node
                            if i in visible_node_indices:
                                node_x, node_y = node
                                distance = math.sqrt((node_x - agent_x)**2 + (node_y - agent_y)**2)
                                
                                if distance <= max_distance:
                                    # Calculate distance-based probability (1.0 at agent position, 0.0 at max_distance)
                                    distance_prob = max(0, 1.0 - (distance / max_distance))
                                    
                                    # Calculate heading angle bias
                                    if distance > 1.0:  # Avoid division by zero for nodes very close to agent
                                        # Calculate angle from agent to node
                                        node_angle = math.atan2(node_y - agent_y, node_x - agent_x)
                                        
                                        # Calculate angular difference between agent heading and direction to node
                                        angle_diff = abs(agent_theta - node_angle)
                                        # Normalize to [0, π] (shortest angular distance)
                                        if angle_diff > math.pi:
                                            angle_diff = 2 * math.pi - angle_diff
                                        
                                        # Convert angle difference to heading bias factor
                                        heading_bias = max(0, 1.0 - (angle_diff / math.pi))
                                        
                                        # Weight the heading bias (0.3 = 30% distance, 70% heading)
                                        probability = distance_prob * (0.3 + 0.7 * heading_bias)
                                    else:
                                        # For very close nodes, use distance-only probability
                                        probability = distance_prob
                                    
                                    if probability > 0.05:  # Only store significant probabilities
                                        node_probabilities[i] = probability
            
            # Draw map graph visuals if enabled
            if show_map_graph_visuals:
                # Draw the graph edges first
                for edge in map_graph.edges:
                    i, j = edge
                    start = map_graph.nodes[i]
                    end = map_graph.nodes[j]
                    pygame.draw.line(screen, MAP_GRAPH_EDGE_COLOR, start, end, 1)
                
                # Draw regular nodes first (probability overlay nodes will be drawn later after visibility)
                for i, node in enumerate(map_graph.nodes):
                    # Determine if this is the node under the mouse for hover effect
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    distance = ((node[0] - mouse_x) ** 2 + (node[1] - mouse_y) ** 2) ** 0.5
                    
                    if distance < 15:  # Mouse hover highlight
                        pygame.draw.circle(screen, (255, 165, 0), node, 6)  # Orange highlight for hover
                    else:
                        # Draw regular node (probability overlay nodes will be drawn later)
                        pygame.draw.circle(screen, MAP_GRAPH_NODE_COLOR, node, 4)
                
                # Display map graph stats
                stats_font = pygame.font.SysFont('Arial', 20)
                stats_text = [
                    f"Map Graph Parameters:",
                    f"Grid Size: {MAP_GRAPH_GRID_SIZE}",
                    f"Nodes: {len(map_graph.nodes)}",
                    f"Edges: {len(map_graph.edges)}",
                    f"Max Edge Distance: {MAP_GRAPH_MAX_EDGE_DISTANCE}",
                    f"Max Connections: {MAP_GRAPH_MAX_CONNECTIONS}"
                ]
                
                # Draw stats background
                stats_bg = pygame.Surface((300, len(stats_text) * 25 + 10))
                stats_bg.fill((0, 0, 0))
                stats_bg.set_alpha(180)
                
                # Position at bottom right of screen
                stats_x = WIDTH - 310
                stats_y = HEIGHT - len(stats_text) * 25 - 20
                
                # Draw background and text
                screen.blit(stats_bg, (stats_x, stats_y))
                for i, text in enumerate(stats_text):
                    text_surf = stats_font.render(text, True, (200, 200, 255))
                    screen.blit(text_surf, (stats_x + 10, stats_y + 10 + i * 25))

            # Calculate FPS for this frame
            if frame_start_time:  # Only calculate if frame_start_time exists
                frame_end_time = pygame.time.get_ticks()
                frame_time = frame_end_time - frame_start_time
                frame_times.append(frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                avg_frame_time = sum(frame_times) / len(frame_times)
                fps = int(1000 / max(1, avg_frame_time))

            # Display info text
            info_text = [
                "Environment Inspection Mode",
                f"Size: {ENVIRONMENT_WIDTH}x{ENVIRONMENT_HEIGHT}",
                "Agent Controls:",
                "Arrow Keys: Control magenta agent (1st)",
                "WASD Keys: Control cyan agent (2nd)",
                "",
                "Other Controls:",
                "Q: Toggle FPS display",
                "G: Toggle map graph display",
                "R: Regenerate map graph (when visible)",
                "V: Analyze node visibility",
                "L: Load visibility data",
                "N: Next node (visibility mode)",
                "P: Previous node (visibility mode)",
                "F: Toggle agent-following mode",
                f"O: Toggle prob overlay {'(ON)' if show_probability_overlay else '(OFF)'} - needs visibility, light blue-red blend",
                f"B: Toggle 1st agent gaps {'(ON)' if show_visibility_gaps else '(OFF)'} - blue/violet",
                f"J: 2nd agent prob {'(ON)' if show_agent2_probability_overlay else '(OFF)'} - dynamic visibility range, pink-green blend",
                f"K: 2nd agent visibility {'(ON)' if show_agent2_visibility_gaps else '(OFF)'} - cyan/green, dynamic range",
                f"Y: Rotating rods {'(ON)' if show_rotating_rods else '(OFF)'} - shows gap arcs",
                f"M: Combined mode {'(ON)' if show_combined_probability_mode else '(OFF)'} - multiply probabilities",
                f"U: Threat classification {'(ON)' if show_threat_classification else '(OFF)'} - rod-based threat analysis",
            ]
            
            # Add time horizon info when probability overlay is enabled
            if show_probability_overlay:
                max_reachable_distance = time_horizon * LEADER_LINEAR_VEL
                info_text.extend([
                    f"Time: {time_horizon:.1f}s (range: {max_reachable_distance:.0f}px) | +/-: Adjust"
                ])
            
            info_text.extend([
                "Mouse: Left-click to select | Right-click for paths",
                "T: Toggle path | C: Clear path | ESC: Exit",
                "Yellow path: Graph-based | Cyan: Dynamically feasible"
            ])
            
            # Add cache info
            if MAP_GRAPH_CACHE_ENABLED:
                cache_status = "Enabled" if MAP_GRAPH_CACHE_ENABLED else "Disabled"
                cache_info = f"Map cache: {cache_status}"
                
                # For visibility cache, just indicate if custom or default without showing the full path
                if visibility_cache_file and visibility_cache_file != MAP_GRAPH_VISIBILITY_CACHE_FILE:
                    visibility_cache = "Visibility cache: Custom"
                else:
                    visibility_cache = "Visibility cache: Default"
                    
                info_text.append(cache_info)
                info_text.append(visibility_cache)
            
            # Add FPS display if enabled
            if show_fps:
                info_text.append(f"FPS: {fps} (Avg frame time: {avg_frame_time:.1f}ms)")
                # Add agent 2 computation time if available and show detailed breakdown
                if agent2_computation_ms > 0:
                    info_text.append(f"Agent 2 computation: {agent2_computation_ms}ms")
                    # Show percentage of frame time consumed by agent 2 computation
                    if avg_frame_time > 0:
                        computation_percentage = (agent2_computation_ms / avg_frame_time) * 100
                        info_text.append(f"  ({computation_percentage:.1f}% of frame time)")
            
            # Display the info on the screen
            info_bg = pygame.Surface((SIDEBAR_WIDTH - 20, len(info_text) * 20 + 10))
            info_bg.fill((0, 0, 0))
            info_bg.set_alpha(180)
            
            # Position at top right in sidebar area
            screen.blit(info_bg, (ENVIRONMENT_WIDTH + 10, 10))
            
            for i, text in enumerate(info_text):
                text_surface = font.render(text, True, (255, 255, 255))
                screen.blit(text_surface, (ENVIRONMENT_WIDTH + 20, 20 + i * 20))

            # Draw visibility data if selected node and visibility map exist
            if map_graph_enabled and selected_node_index is not None:
                if selected_node_index < len(map_graph.nodes):
                    # Get selected node's position
                    selected_node = map_graph.nodes[selected_node_index]
                    
                    # If visibility data is available, draw the connections
                    if visibility_map and selected_node_index in visibility_map:
                        # Draw a large circle showing the visibility range
                        # Surface needs to be larger than the range to fit the entire circle
                        range_size = MAP_GRAPH_VISIBILITY_RANGE * 2
                        range_surface = pygame.Surface((range_size, range_size), pygame.SRCALPHA)
                        
                        # Create pulsing effect for the range indicator
                        pulse_intensity = (math.sin(pygame.time.get_ticks() / 500) + 1) / 4 + 0.25  # Value between 0.25 and 0.75
                        range_alpha = int(40 * pulse_intensity)  # Pulsing alpha between 10 and 30
                        
                        # Draw the visibility range with pulsing effect
                        pygame.draw.circle(range_surface, (0, 255, 100, range_alpha), (range_size//2, range_size//2), MAP_GRAPH_VISIBILITY_RANGE)
                        
                        # Draw an additional thin circle at the exact visibility range
                        pygame.draw.circle(range_surface, (0, 255, 120, 80), (range_size//2, range_size//2), MAP_GRAPH_VISIBILITY_RANGE, 1)
                        
                        # Position the surface so the circle is centered on the selected node
                        range_x = selected_node[0] - range_size//2
                        range_y = selected_node[1] - range_size//2
                        screen.blit(range_surface, (range_x, range_y))
                        
                        # Get visible nodes and prepare for visualization
                        visible_nodes = visibility_map[selected_node_index]
                        visible_count = len(visible_nodes)
                        
                        # Group nodes by distance for better visualization (close, medium, far)
                        node_distance_groups = {
                            'close': [],
                            'medium': [],
                            'far': []
                        }
                        
                        # Calculate distance thresholds based on visibility range
                        close_threshold = MAP_GRAPH_VISIBILITY_RANGE * 0.33
                        medium_threshold = MAP_GRAPH_VISIBILITY_RANGE * 0.66
                        
                        # Categorize visible nodes by distance
                        for visible_index in visible_nodes:
                            if visible_index < len(map_graph.nodes):  # Safety check
                                visible_node = map_graph.nodes[visible_index]
                                node_distance = math.dist(selected_node, visible_node)
                                
                                if node_distance <= close_threshold:
                                    node_distance_groups['close'].append(visible_node)
                                elif node_distance <= medium_threshold:
                                    node_distance_groups['medium'].append(visible_node)
                                else:
                                    node_distance_groups['far'].append(visible_node)
                        
                        # Conditionally draw either green visibility lines OR gap arcs based on 'H' key toggle
                        if show_extended_probability_set:
                            # Show extended probability set (gap arcs) instead of green visibility lines
                            # Import vision utilities
                            from multitrack.utils.vision import cast_vision_ray
                            
                            # Cast rays in all directions to find visibility discontinuities
                            num_rays = 360  # Every 1 degree for much finer resolution
                            angle_step = (2 * math.pi) / num_rays
                            ray_endpoints = []
                            
                            # Cast rays in all directions
                            for i in range(num_rays):
                                angle = i * angle_step
                                endpoint = cast_vision_ray(
                                    selected_node[0], 
                                    selected_node[1], 
                                    angle, 
                                    MAP_GRAPH_VISIBILITY_RANGE,
                                    environment.get_all_walls(),
                                    environment.get_doors()  # Doors allow vision through
                                )
                                ray_endpoints.append(endpoint)
                            
                            # Find discontinuities in ray distances and connect successive rays with abrupt changes
                            min_gap_distance = 30  # Minimum distance difference to consider a gap
                            gap_lines = []
                            
                            for i in range(num_rays):
                                current_endpoint = ray_endpoints[i]
                                next_endpoint = ray_endpoints[(i + 1) % num_rays]  # Wrap around
                                
                                # Calculate distances from selected node
                                current_dist = math.dist(selected_node, current_endpoint)
                                next_dist = math.dist(selected_node, next_endpoint)
                                
                                # Check for significant distance change (gap) between successive rays
                                distance_diff = abs(current_dist - next_dist)
                                if distance_diff > min_gap_distance:
                                    # Record this gap line connecting the two successive ray endpoints
                                    gap_lines.append((current_endpoint, next_endpoint, distance_diff))
                            
                            # Draw all the gap lines with orientation-based coloring
                            for start_point, end_point, gap_size in gap_lines:
                                # Determine gap orientation relative to clockwise ray casting
                                start_dist = math.dist(selected_node, start_point)
                                end_dist = math.dist(selected_node, end_point)
                                
                                # Classify gap type based on distance progression
                                is_near_to_far = start_dist < end_dist  # Near point first, far point second
                                is_far_to_near = start_dist > end_dist  # Far point first, near point second
                                
                                # Choose base color based on gap orientation
                                if is_near_to_far:
                                    # Blue for near-to-far transitions (expanding gaps)
                                    base_color = (0, 100, 255) if gap_size > 150 else (50, 150, 255) if gap_size > 80 else (100, 200, 255)
                                elif is_far_to_near:
                                    # Violet for far-to-near transitions (contracting gaps)
                                    base_color = (150, 0, 255) if gap_size > 150 else (180, 50, 255) if gap_size > 80 else (200, 100, 255)
                                else:
                                    # Fallback color for equal distances (rare case)
                                    base_color = (100, 100, 100)
                                
                                # Determine line width based on gap size
                                if gap_size > 150:
                                    line_width = 3
                                elif gap_size > 80:
                                    line_width = 2
                                else:
                                    line_width = 1
                                
                                # Draw the gap line connecting successive ray endpoints
                                pygame.draw.line(screen, base_color, start_point, end_point, line_width)
                                
                                # Draw small circles at the gap endpoints to highlight them
                                circle_size = max(2, min(5, int(gap_size / 30)))
                                pygame.draw.circle(screen, base_color, start_point, circle_size)
                                pygame.draw.circle(screen, base_color, end_point, circle_size)
                        else:
                            # Show original green visibility lines
                            # Draw lines to close nodes (bright green)
                            for visible_node in node_distance_groups['close']:
                                pygame.draw.line(screen, (50, 255, 50), selected_node, visible_node, 2)
                                pygame.draw.circle(screen, (50, 255, 50), visible_node, 5)
                            
                            # Draw lines to medium-range nodes (yellow-green)
                            for visible_node in node_distance_groups['medium']:
                                pygame.draw.line(screen, (150, 255, 30), selected_node, visible_node, 1)
                                pygame.draw.circle(screen, (150, 255, 30), visible_node, 4)
                            
                            # Draw lines to far nodes (faded green)
                            for visible_node in node_distance_groups['far']:
                                pygame.draw.line(screen, (100, 180, 30, 150), selected_node, visible_node, 1)
                                pygame.draw.circle(screen, (100, 180, 30), visible_node, 3)
                            
                        # VISIBILITY GAPS VISUALIZATION: Draw ray casting discontinuities (abrupt changes in visibility)
                        if show_visibility_gaps:
                            # Import vision utilities
                            from multitrack.utils.vision import cast_vision_ray
                            
                            # Cast rays in all directions to find visibility discontinuities
                            num_rays = 360  # Every 1 degree for much finer resolution
                            angle_step = (2 * math.pi) / num_rays
                            ray_endpoints = []
                            
                            # Cast rays in all directions
                            for i in range(num_rays):
                                angle = i * angle_step
                                endpoint = cast_vision_ray(
                                    selected_node[0], 
                                    selected_node[1], 
                                    angle, 
                                    MAP_GRAPH_VISIBILITY_RANGE,
                                    environment.get_all_walls(),
                                    environment.get_doors()  # Doors allow vision through
                                )
                                ray_endpoints.append(endpoint)
                            
                            # Find discontinuities in ray distances and connect successive rays with abrupt changes
                            min_gap_distance = 30  # Minimum distance difference to consider a gap (reduced for finer detection)
                            gap_lines = []
                            
                            for i in range(num_rays):
                                current_endpoint = ray_endpoints[i]
                                next_endpoint = ray_endpoints[(i + 1) % num_rays]  # Wrap around
                                
                                # Calculate distances from selected node
                                current_dist = math.dist(selected_node, current_endpoint)
                                next_dist = math.dist(selected_node, next_endpoint)
                                
                                # Check for significant distance change (gap) between successive rays
                                distance_diff = abs(current_dist - next_dist)
                                if distance_diff > min_gap_distance:
                                    # Record this gap line connecting the two successive ray endpoints
                                    gap_lines.append((current_endpoint, next_endpoint, distance_diff))
                            
                            # Draw all the gap lines with orientation-based coloring
                            for start_point, end_point, gap_size in gap_lines:
                                # Determine gap orientation relative to clockwise ray casting
                                start_dist = math.dist(selected_node, start_point)
                                end_dist = math.dist(selected_node, end_point)
                                
                                # Classify gap type based on distance progression
                                is_near_to_far = start_dist < end_dist  # Near point first, far point second
                                is_far_to_near = start_dist > end_dist  # Far point first, near point second
                                
                                # Choose base color based on gap orientation
                                if is_near_to_far:
                                    # Blue for near-to-far transitions (expanding gaps)
                                    base_color = (0, 100, 255) if gap_size > 150 else (50, 150, 255) if gap_size > 80 else (100, 200, 255)
                                elif is_far_to_near:
                                    # Violet for far-to-near transitions (contracting gaps)
                                    base_color = (150, 0, 255) if gap_size > 150 else (180, 50, 255) if gap_size > 80 else (200, 100, 255)
                                else:
                                    # Fallback color for equal distances (rare case)
                                    base_color = (100, 100, 100)
                                
                                # Determine line width based on gap size
                                if gap_size > 150:
                                    line_width = 3
                                elif gap_size > 80:
                                    line_width = 2
                                else:
                                    line_width = 1
                                
                                # Draw the gap line connecting successive ray endpoints
                                pygame.draw.line(screen, base_color, start_point, end_point, line_width)
                                
                                # Draw small circles at the gap endpoints to highlight them
                                circle_size = max(2, min(5, int(gap_size / 30)))
                                pygame.draw.circle(screen, base_color, start_point, circle_size)
                                pygame.draw.circle(screen, base_color, end_point, circle_size)
                            
                            # EXTENDED PROBABILITIES COMPUTATION: Always compute when conditions are met
                            propagated_probabilities = {}
                            if show_probability_overlay and node_probabilities:
                                # Process gaps to compute extended probabilities in the background
                                for start_point, end_point, gap_size in gap_lines:
                                    # Only process significant gaps
                                    if gap_size < 50:
                                        continue
                                    
                                    # Determine gap orientation and near/far points
                                    start_dist = math.dist(selected_node, start_point)
                                    end_dist = math.dist(selected_node, end_point)
                                    
                                    # Determine near point (pivot point) and far point
                                    if start_dist < end_dist:
                                        near_point = start_point
                                        far_point = end_point
                                        is_blue_gap = True  # Near-to-far (expanding)
                                    else:
                                        near_point = end_point
                                        far_point = start_point
                                        is_blue_gap = False  # Far-to-near (contracting)
                                    
                                    # Calculate initial rod angle (along the gap line from near to far point)
                                    initial_rod_angle = math.atan2(far_point[1] - near_point[1], far_point[0] - near_point[0])
                                    
                                    # Calculate rod length: should extend from near point to the edge of reachability circle
                                    # Get agent position and max reachable distance
                                    agent_x, agent_y = agent.state[0], agent.state[1]
                                    max_reachable_distance = time_horizon * LEADER_LINEAR_VEL
                                    
                                    # Calculate distance from agent to rod base (near point)
                                    distance_to_rod_base = math.sqrt((near_point[0] - agent_x)**2 + (near_point[1] - agent_y)**2)
                                    
                                    # Calculate maximum rod length that stays within reachability circle
                                    if distance_to_rod_base >= max_reachable_distance:
                                        # Rod base is outside reachability circle, use minimal rod
                                        rod_length = 10
                                    else:
                                        # Rod length = remaining distance from base to circle edge
                                        remaining_distance_to_circle = max_reachable_distance - distance_to_rod_base
                                        
                                        # Also consider the original gap size as a constraint
                                        original_gap_rod_length = math.dist(near_point, far_point)
                                        
                                        # Use the smaller of the two: gap size or remaining circle distance
                                        rod_length = min(remaining_distance_to_circle, original_gap_rod_length)
                                        
                                        # Ensure minimum rod length for visibility
                                        rod_length = max(20, rod_length)
                                    
                                    max_rotation = math.pi / 4  # Maximum 45 degrees rotation
                                    
                                    # Determine rotation direction based on gap color
                                    if is_blue_gap:
                                        rotation_direction = -1  # Anticlockwise (counterclockwise)
                                    else:
                                        rotation_direction = 1   # Clockwise
                                    
                                    # SINGLE DIRECTION ROTATION: Rod pivots at near point, rotates in one direction only
                                    rod_base = near_point
                                    
                                    # Calculate the swept arc range
                                    sweep_start_angle = initial_rod_angle
                                    sweep_end_angle = initial_rod_angle + max_rotation * rotation_direction
                                    
                                    
                                    # OPTIMIZED GRADIENT-BASED PROBABILITY PROPAGATION
                                    # Step 1: Record probability values at gap endpoints
                                    
                                    # Find probability at near point (rod base)
                                    near_point_prob = 0.0
                                    min_distance_near = float('inf')
                                    for node_idx, prob in node_probabilities.items():
                                        node_pos = map_graph.nodes[node_idx]
                                        dist_to_near = math.dist(node_pos, near_point)
                                        if dist_to_near < min_distance_near and dist_to_near < 50:  # Within 50px
                                            min_distance_near = dist_to_near
                                            near_point_prob = prob
                                    
                                    # Find probability at far point (gap end)
                                    far_point_prob = 0.0
                                    min_distance_far = float('inf')
                                    far_point_actual = (
                                        rod_base[0] + rod_length * math.cos(initial_rod_angle),
                                        rod_base[1] + rod_length * math.sin(initial_rod_angle)
                                    )
                                    for node_idx, prob in node_probabilities.items():
                                        node_pos = map_graph.nodes[node_idx]
                                        dist_to_far = math.dist(node_pos, far_point_actual)
                                        if dist_to_far < min_distance_far and dist_to_far < 50:  # Within 50px
                                            min_distance_far = dist_to_far
                                            far_point_prob = prob
                                    
                                    # If no probabilities found nearby, use default values
                                    if near_point_prob == 0.0 and far_point_prob == 0.0:
                                        near_point_prob = 0.3  # Default probability at near point
                                        far_point_prob = 0.1   # Lower probability at far point
                                    
                                    # Step 2: FAST GRID PROCESSING - optimized for speed
                                    
                                    # REDUCED grid density for better performance
                                    angle_steps = 15  # Reduced from 40
                                    radius_steps = 8  # Reduced from 20
                                    
                                    # Calculate sweep bounds
                                    total_sweep_angle = abs(sweep_end_angle - sweep_start_angle)
                                    
                                    # PRE-FILTER: Only consider nodes that are NOT already probabilized and in general area
                                    candidate_nodes = []
                                    arc_center_x = rod_base[0] + (rod_length / 2) * math.cos(initial_rod_angle)
                                    arc_center_y = rod_base[1] + (rod_length / 2) * math.sin(initial_rod_angle)
                                    filter_radius = rod_length + 50  # General area around the arc
                                    
                                    for j, node in enumerate(map_graph.nodes):
                                        if j not in node_probabilities:  # Skip nodes with existing probabilities
                                            # Quick distance check to arc center
                                            dx = node[0] - arc_center_x
                                            dy = node[1] - arc_center_y
                                            if dx*dx + dy*dy <= filter_radius*filter_radius:  # Avoid sqrt
                                                candidate_nodes.append((j, node))
                                    
                                    # Process grid points with optimized node search
                                    search_radius = 25
                                    search_radius_sq = search_radius * search_radius  # Avoid sqrt in distance calc
                                    
                                    for a in range(angle_steps + 1):
                                        for r in range(1, radius_steps + 1):
                                            angle_progress = a / angle_steps
                                            radius_progress = r / radius_steps
                                            current_angle = sweep_start_angle + angle_progress * (sweep_end_angle - sweep_start_angle)
                                            current_radius = radius_progress * rod_length
                                            
                                            sweep_x = rod_base[0] + current_radius * math.cos(current_angle)
                                            sweep_y = rod_base[1] + current_radius * math.sin(current_angle)
                                            
                                            # Boundary check (avoid sqrt)
                                            dx_agent = sweep_x - agent_x
                                            dy_agent = sweep_y - agent_y
                                            dist_sq_agent = dx_agent*dx_agent + dy_agent*dy_agent
                                            
                                            if dist_sq_agent <= max_reachable_distance*max_reachable_distance:
                                                # Calculate probability with stronger distance-based decay
                                                base_probability = (1 - radius_progress) * near_point_prob + radius_progress * far_point_prob
                                                
                                                # Stronger angular decay - harder to reach at wider angles
                                                angular_decay = max(0.1, 1.0 - (angle_progress * 0.8))
                                                
                                                # Distance-based decay - exponential decay with distance from rod base
                                                distance_decay = max(0.2, 1.0 - (radius_progress ** 1.5) * 0.7)
                                                
                                                # Overall propagation decay
                                                propagation_decay = 0.7
                                                
                                                final_probability = base_probability * angular_decay * distance_decay * propagation_decay
                                                
                                                if final_probability > 0.03:  # Only store significant probabilities
                                                    # FAST node search - only through pre-filtered candidates
                                                    closest_node_idx = None
                                                    closest_distance_sq = search_radius_sq
                                                    
                                                    for node_idx, node in candidate_nodes:
                                                        dx = node[0] - sweep_x
                                                        dy = node[1] - sweep_y
                                                        dist_sq = dx*dx + dy*dy  # Avoid sqrt
                                                        
                                                        if dist_sq < closest_distance_sq:
                                                            closest_distance_sq = dist_sq
                                                            closest_node_idx = node_idx
                                                    
                                                    # Assign probability
                                                    if closest_node_idx is not None:
                                                        if closest_node_idx in propagated_probabilities:
                                                            propagated_probabilities[closest_node_idx] = max(
                                                                propagated_probabilities[closest_node_idx], final_probability)
                                                        else:
                                                            propagated_probabilities[closest_node_idx] = final_probability
            
            # Draw probability overlay nodes (INDEPENDENT of map graph visuals)
            if show_probability_overlay and node_probabilities and not show_combined_probability_mode:
                # Merge base probabilities with extended probabilities from rotating rods
                merged_probabilities = {}
                
                # Start with base reachability probabilities
                for node_idx, base_prob in node_probabilities.items():
                    merged_probabilities[node_idx] = base_prob
                
                # Add/merge extended probabilities (computed earlier in background)
                if 'propagated_probabilities' in locals() and propagated_probabilities:
                    for node_idx, extended_prob in propagated_probabilities.items():
                        if node_idx in merged_probabilities:
                            # Use maximum of base and extended probability (union of reachable sets)
                            merged_probabilities[node_idx] = max(merged_probabilities[node_idx], extended_prob)
                        else:
                            # Add new extended probability locations
                            merged_probabilities[node_idx] = extended_prob
                
                # Draw merged probability overlay nodes
                for i, node in enumerate(map_graph.nodes):
                    if i in merged_probabilities:
                        probability = merged_probabilities[i]
                        
                        # Color blending scheme for Agent 1: probability determines mix between light blue and red
                        # Low probability (e.g., 0.1): mostly light blue (0.9) + little red (0.1)
                        # High probability (e.g., 0.9): mostly red (0.9) + little light blue (0.1)
                        
                        # Define base colors for light blue-red blending
                        light_blue_color = (173, 216, 230)  # Light blue for low probability
                        red_color = (255, 0, 0)    # Pure red for high probability
                        
                        # Use probability directly for color blending (0.0 to 1.0)
                        red_weight = probability  # How much red (high probability)
                        light_blue_weight = 1.0 - probability  # How much light blue (low probability)
                        
                        # Blend the colors
                        red = int(light_blue_color[0] * light_blue_weight + red_color[0] * red_weight)
                        green = int(light_blue_color[1] * light_blue_weight + red_color[1] * red_weight)
                        blue = int(light_blue_color[2] * light_blue_weight + red_color[2] * red_weight)
                        
                        color = (red, green, blue)
                        
                        # Keep original circle sizes (4-8 pixels based on probability)
                        node_size = int(4 + probability * 4)  # Size 4-8 based on probability
                        pygame.draw.circle(screen, color, node, node_size)
                        
                        # Add subtle glow effect for high probability nodes
                        if probability > 0.7:
                            # Glow uses red tint for high probability
                            glow_color = (255, 0, 0)  # Red glow
                            pygame.draw.circle(screen, glow_color, node, node_size + 2)

            # Draw mouse hover effects and selected node highlight (INDEPENDENT of map graph visuals)
            if map_graph and map_graph.nodes:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for i, node in enumerate(map_graph.nodes):
                    distance = ((node[0] - mouse_x) ** 2 + (node[1] - mouse_y) ** 2) ** 0.5
                    if distance < 15:  # Mouse hover highlight
                        pygame.draw.circle(screen, (255, 165, 0), node, 6)  # Orange highlight for hover
                
                # Always draw the selected node with a different color and larger size (on top of everything)
                # Add pulsing effect to selected node
                pulse = (math.sin(pygame.time.get_ticks() / 300) + 1) / 4 + 0.75  # Value between 0.75 and 1.25
                
                # Draw outer glow/halo
                glow_size = int(12 * pulse)
                pygame.draw.circle(screen, (255, 255, 100, 100), selected_node, glow_size)
                
                # Draw the selected node itself
                pygame.draw.circle(screen, (255, 255, 0), selected_node, 8)
                pygame.draw.circle(screen, (255, 200, 0), selected_node, 4)
                
                # Draw selection information with enhanced details (INDEPENDENT of map graph visuals)
                visibility_info = [
                    f"Selected Node: {selected_node_index}",
                    f"Position: ({int(selected_node[0])}, {int(selected_node[1])})"
                ]
                
                # Add visibility info if available
                if visibility_map and selected_node_index in visibility_map:
                    visible_nodes = visibility_map[selected_node_index]
                    visible_count = len(visible_nodes)
                    visibility_percentage = (visible_count / len(map_graph.nodes)) * 100 if map_graph.nodes else 0
                    
                    # Count nodes by distance category (need to recalculate here)
                    close_threshold = MAP_GRAPH_VISIBILITY_RANGE * 0.3
                    medium_threshold = MAP_GRAPH_VISIBILITY_RANGE * 0.6
                    
                    node_distance_groups = {'close': [], 'medium': [], 'far': []}
                    
                    for node_idx in visible_nodes:
                        if node_idx < len(map_graph.nodes):
                            node = map_graph.nodes[node_idx]
                            distance = math.sqrt((node[0] - selected_node[0])**2 + (node[1] - selected_node[1])**2)
                            
                            if distance <= close_threshold:
                                node_distance_groups['close'].append(node_idx)
                            elif distance <= medium_threshold:
                                node_distance_groups['medium'].append(node_idx)
                            else:
                                node_distance_groups['far'].append(node_idx)
                    
                    close_count = len(node_distance_groups['close'])
                    medium_count = len(node_distance_groups['medium'])
                    far_count = len(node_distance_groups['far'])
                    
                    visibility_info.extend([
                        f"Visible Nodes: {visible_count:,} ({visibility_percentage:.1f}% of map)",
                        f"Distance Groups:",
                        f"  Close: {close_count} nodes (≤{int(close_threshold)}px)",
                        f"  Medium: {medium_count} nodes (≤{int(medium_threshold)}px)",
                        f"  Far: {far_count} nodes (≤{MAP_GRAPH_VISIBILITY_RANGE}px)",
                        f"Visibility Range: {MAP_GRAPH_VISIBILITY_RANGE} pixels"
                    ])
                else:
                    visibility_info.extend([
                        "No visibility data available",
                        "Press V to analyze visibility",
                        "Press L to load cached data"
                    ])
                
                visibility_info.append("Click on nodes or use N/P to navigate")
                
                # Create background for visibility info
                info_width = 350
                info_bg = pygame.Surface((info_width, len(visibility_info) * 25 + 10))
                info_bg.fill((0, 0, 30))  # Dark blue background
                info_bg.set_alpha(200)
                
                # Position at top left
                info_x = 10
                info_y = 10
                
                # Draw background with rounded corners effect
                screen.blit(info_bg, (info_x, info_y))
                
                # Add a title bar effect
                pygame.draw.rect(screen, (0, 100, 200), (info_x, info_y, info_width, 30))
                pygame.draw.rect(screen, (0, 70, 150), (info_x, info_y, info_width, 30), 1)
                
                # Draw a thin border around the entire info panel
                pygame.draw.rect(screen, (100, 100, 200), (info_x, info_y, info_width, len(visibility_info) * 25 + 10), 1)
                
                # Render each line of text with appropriate styling
                title_font = pygame.font.SysFont('Arial', 22, bold=True)
                regular_font = pygame.font.SysFont('Arial', 20)
                small_font = pygame.font.SysFont('Arial', 18)
                
                # Draw title
                title_text = title_font.render("Node Visibility Analysis", True, (255, 255, 255))
                screen.blit(title_text, (info_x + (info_width - title_text.get_width()) // 2, info_y + 5))
                
                # Draw content with different colors based on type
                y_offset = info_y + 35
                
                for i, text in enumerate(visibility_info):
                    if i == 0:
                        # Draw node ID with highlighted color
                        node_text = regular_font.render(text, True, (255, 255, 100))
                        screen.blit(node_text, (info_x + 10, y_offset))
                        y_offset += 25
                    elif "Position:" in text:
                        # Position info
                        pos_text = regular_font.render(text, True, (200, 200, 255))
                        screen.blit(pos_text, (info_x + 10, y_offset))
                        y_offset += 25
                    elif "Visible Nodes:" in text:
                        # Visibility count (important info in bright color)
                        vis_text = regular_font.render(text, True, (50, 255, 50))
                        screen.blit(vis_text, (info_x + 10, y_offset))
                        y_offset += 25
                    elif "Distance Groups:" in text:
                        # Section header
                        header_text = regular_font.render(text, True, (255, 200, 100))
                        screen.blit(header_text, (info_x + 10, y_offset))
                        y_offset += 25
                    elif text.startswith("  Close:"):
                        # Close distance nodes (bright green)
                        close_text = small_font.render(text, True, (50, 255, 50))
                        screen.blit(close_text, (info_x + 10, y_offset))
                        y_offset += 20
                    elif text.startswith("  Medium:"):
                        # Medium distance nodes (yellow-green)
                        medium_text = small_font.render(text, True, (150, 255, 30))
                        screen.blit(medium_text, (info_x + 10, y_offset))
                        y_offset += 20
                    elif text.startswith("  Far:"):
                        # Far distance nodes (faded green)
                        far_text = small_font.render(text, True, (100, 180, 30))
                        screen.blit(far_text, (info_x + 10, y_offset))
                        y_offset += 20
                    elif "No visibility data" in text:
                        # Warning message
                        warning_text = regular_font.render(text, True, (255, 100, 100))
                        screen.blit(warning_text, (info_x + 10, y_offset))
                        y_offset += 25
                    elif "Press V" in text or "Press L" in text:
                        # Instruction text
                        instruction_text = small_font.render(text, True, (200, 200, 200))
                        screen.blit(instruction_text, (info_x + 20, y_offset))
                        y_offset += 20
                    elif "Click on nodes" in text:
                        # Help text at the bottom
                        help_text = small_font.render(text, True, (180, 180, 220))
                        screen.blit(help_text, (info_x + 10, y_offset + 5))
                        y_offset += 25
                    else:
                        # Default styling
                        default_text = regular_font.render(text, True, (200, 200, 255))
                        screen.blit(default_text, (info_x + 10, y_offset))
                        y_offset += 25

            # Draw path if one exists and should be shown
            if show_path and current_path:
                # Draw the original path first (solid gold/yellow)
                draw_path(current_path, color=(255, 215, 0), width=4)
                
                # Draw the dynamic path if it exists (dashed cyan)
                if dynamic_path:
                    # Use a different color and dashed line for the dynamic path
                    draw_path(
                        dynamic_path, 
                        color=(0, 200, 255), 
                        width=3, 
                        start_color=(0, 180, 0),  # Slightly different green
                        end_color=(220, 0, 0),    # Slightly different red
                        dashed=True
                    )
                
                # Display path information
                path_info = [
                    f"Path Information:",
                    f"Graph Path Length: {path_length:.1f} pixels",
                    f"Graph Path Nodes: {len(current_path)}",
                ]
                
                # Add visibility optimization info if available
                if 'path_optimization_gain' in locals() and path_optimization_gain > 0:
                    optimization_percent = (path_optimization_gain / original_path_length) * 100 if original_path_length > 0 else 0
                    path_info.extend([
                        f"Visibility Optimization:",
                        f"  Original Nodes: {len(original_path)}",
                        f"  Distance Saved: {path_optimization_gain:.1f} pixels ({optimization_percent:.1f}%)",
                    ])
                
                # Add dynamic path info if it exists
                if dynamic_path:
                    path_info.extend([
                        f"Dynamic Path Length: {dynamic_path_length:.1f} pixels",
                        f"Dynamic Path Points: {len(dynamic_path)}",
                        f"Length Difference: {(dynamic_path_length - path_length):.1f} pixels",
                    ])
                
                path_info.extend([
                    f"Start: Agent Position",
                    f"End: Node {path_end_index}",
                    "T: Toggle path visibility",
                    "C: Clear path"
                ])
                
                # Create background for path info - adjust height based on content
                info_width = 300  # Increased width for more text
                info_bg = pygame.Surface((info_width, len(path_info) * 25 + 10))
                info_bg.fill((30, 30, 0))  # Dark gold background
                info_bg.set_alpha(200)
                
                # Position at bottom left
                info_x = 10
                info_y = HEIGHT - len(path_info) * 25 - 20
                
                # Draw background
                screen.blit(info_bg, (info_x, info_y))
                
                # Draw title bar
                pygame.draw.rect(screen, (180, 150, 0), (info_x, info_y, info_width, 30))
                
                # Draw border
                pygame.draw.rect(screen, (255, 215, 0), (info_x, info_y, info_width, len(path_info) * 25 + 10), 1)
                
                # Define fonts for the path info rendering
                title_font = pygame.font.SysFont('Arial', 22, bold=True)
                regular_font = pygame.font.SysFont('Arial', 20)
                small_font = pygame.font.SysFont('Arial', 18)
                
                # Render title
                title_text = title_font.render("Path Visualization", True, (0, 0, 0))
                screen.blit(title_text, (info_x + (info_width - title_text.get_width()) // 2, info_y + 5))
                
                # Render content
                y_offset = info_y + 35
                for i, text in enumerate(path_info):
                    if i == 0:  # Skip header
                        continue
                    elif "Graph Path" in text:
                        # Original path info (yellow)
                        graph_text = regular_font.render(text, True, (255, 255, 100))
                        screen.blit(graph_text, (info_x + 10, y_offset))
                        y_offset += 25
                    elif "Dynamic Path" in text:
                        # Dynamic path info (cyan)
                        dynamic_text = regular_font.render(text, True, (100, 255, 255))
                        screen.blit(dynamic_text, (info_x + 10, y_offset))
                        y_offset += 25
                    elif "Length Difference" in text:
                        # Length difference (cyan)
                        diff = float(text.split(":")[1].strip().split()[0])
                        # Choose color based on length difference
                        diff_color = (100, 255, 100) if diff <= 0 else (255, 180, 100)  # Green if shorter or equal, orange if longer
                        diff_text = regular_font.render(text, True, diff_color)
                        screen.blit(diff_text, (info_x + 10, y_offset))
                        y_offset += 25
                    elif "Start:" in text or "End:" in text:
                        # Position info
                        pos_text = regular_font.render(text, True, (200, 200, 255))
                        screen.blit(pos_text, (info_x + 10, y_offset))
                        y_offset += 25
                    elif text.startswith("T:") or text.startswith("C:"):
                        # Controls
                        control_text = small_font.render(text, True, (200, 200, 200))
                        screen.blit(control_text, (info_x + 10, y_offset))
                        y_offset += 25
                    else:
                        # Default text
                        path_text = regular_font.render(text, True, (255, 215, 0))
                        screen.blit(path_text, (info_x + 10, y_offset))
                        y_offset += 25

            # Get agent positions for later drawing (moved drawing to the end)
            x, y, theta, _ = agent.state
            x2, y2, theta2, _ = agent2.state
            
            # Store font for agent labels
            label_font = pygame.font.SysFont('Arial', 14, bold=True)
            
            # SECOND AGENT VISUALIZATION OPTIONS
            # Generate gap lines for both visibility gaps and rotating rods
            gap_lines = []
            if show_agent2_visibility_gaps or show_agent2_rods:
                # Import vision utilities
                from multitrack.utils.vision import cast_vision_ray
                
                # Cast rays in all directions to find visibility discontinuities
                num_rays = 360  # Every 1 degree for finer resolution
                angle_step = (2 * math.pi) / num_rays
                ray_endpoints = []
                
                # Cast rays in all directions from the second agent's position
                for i in range(num_rays):
                    angle = i * angle_step
                    endpoint = cast_vision_ray(
                        x2, 
                        y2, 
                        angle, 
                        MAP_GRAPH_VISIBILITY_RANGE,
                        environment.get_all_walls(),
                        environment.get_doors()  # Doors allow vision through
                    )
                    ray_endpoints.append(endpoint)
                
                # Find discontinuities in ray distances and connect successive rays with abrupt changes
                min_gap_distance = 30  # Minimum distance difference to consider a gap
                
                for i in range(num_rays):
                    current_endpoint = ray_endpoints[i]
                    next_endpoint = ray_endpoints[(i + 1) % num_rays]  # Wrap around
                    
                    # Calculate distances from agent position
                    current_dist = math.dist((x2, y2), current_endpoint)
                    next_dist = math.dist((x2, y2), next_endpoint)
                    
                    # Check for significant distance change (gap) between successive rays
                    distance_diff = abs(current_dist - next_dist)
                    if distance_diff > min_gap_distance:
                        # Record this gap line connecting the two successive ray endpoints
                        gap_lines.append((current_endpoint, next_endpoint, distance_diff))
            # First priority: Show visibility gaps if enabled
            if show_agent2_visibility_gaps:
                # First, use preprocessed visibility data to show visible map graph nodes
                if visibility_map and map_graph:
                    # Find closest node to the second agent's position
                    agent2_pos = (x2, y2)
                    agent2_node_index = find_closest_node(map_graph.nodes, agent2_pos)
                    
                    if agent2_node_index is not None:
                        agent2_node = map_graph.nodes[agent2_node_index]
                        
                        # Get visible nodes from the visibility map
                        visible_nodes = visibility_map[agent2_node_index]
                        
                        # Group nodes by distance for better visualization (close, medium, far)
                        agent2_node_distance_groups = {
                            'close': [],
                            'medium': [],
                            'far': []
                        }
                        
                        # Calculate distance thresholds based on visibility range
                        close_threshold = MAP_GRAPH_VISIBILITY_RANGE * 0.33
                        medium_threshold = MAP_GRAPH_VISIBILITY_RANGE * 0.66
                        
                        # Categorize visible nodes by distance
                        for visible_index in visible_nodes:
                            if visible_index < len(map_graph.nodes):  # Safety check
                                visible_node = map_graph.nodes[visible_index]
                                node_distance = math.dist(agent2_node, visible_node)
                                
                                if node_distance <= close_threshold:
                                    agent2_node_distance_groups['close'].append(visible_node)
                                elif node_distance <= medium_threshold:
                                    agent2_node_distance_groups['medium'].append(visible_node)
                                else:
                                    agent2_node_distance_groups['far'].append(visible_node)
                        
                        # Draw lines to close nodes (bright cyan to match agent 2's color)
                        for visible_node in agent2_node_distance_groups['close']:
                            pygame.draw.line(screen, (0, 255, 255), agent2_node, visible_node, 2)
                            pygame.draw.circle(screen, (0, 255, 255), visible_node, 5)
                        
                        # Draw lines to medium-range nodes (lighter cyan)
                        for visible_node in agent2_node_distance_groups['medium']:
                            pygame.draw.line(screen, (100, 255, 255), agent2_node, visible_node, 1)
                            pygame.draw.circle(screen, (100, 255, 255), visible_node, 4)
                        
                        # Draw lines to far nodes (faded cyan)
                        for visible_node in agent2_node_distance_groups['far']:
                            pygame.draw.line(screen, (150, 255, 255, 150), agent2_node, visible_node, 1)
                            pygame.draw.circle(screen, (150, 255, 255), visible_node, 3)
                
                # Draw all the gap lines with orientation-based coloring
                for start_point, end_point, gap_size in gap_lines:
                    # Determine gap orientation relative to clockwise ray casting
                    start_dist = math.dist((x2, y2), start_point)
                    end_dist = math.dist((x2, y2), end_point)
                    
                    # Classify gap type based on distance progression
                    is_near_to_far = start_dist < end_dist  # Near point first, far point second
                    is_far_to_near = start_dist > end_dist  # Far point first, near point second
                    
                    # Choose base color based on gap orientation - using cyan/green for second agent's gaps
                    if is_near_to_far:
                        # Cyan for near-to-far transitions (expanding gaps)
                        base_color = (0, 200, 255) if gap_size > 150 else (0, 220, 255) if gap_size > 80 else (0, 240, 255)
                    elif is_far_to_near:
                        # Green-cyan for far-to-near transitions (contracting gaps)
                        base_color = (0, 240, 180) if gap_size > 150 else (0, 255, 180) if gap_size > 80 else (0, 255, 220)
                    else:
                        # Fallback color for equal distances (rare case)
                        base_color = (0, 200, 200)
                    
                    # Determine line width based on gap size
                    if gap_size > 150:
                        line_width = 3
                    elif gap_size > 80:
                        line_width = 2
                    else:
                        line_width = 1
                    
                    # Draw the gap line connecting successive ray endpoints
                    pygame.draw.line(screen, base_color, start_point, end_point, line_width)
                    
                    # Draw small circles at the gap endpoints to highlight them
                    circle_size = max(2, min(5, int(gap_size / 30)))
                    pygame.draw.circle(screen, base_color, start_point, circle_size)
                    pygame.draw.circle(screen, base_color, end_point, circle_size)
                
                # Draw individual visibility points (only when probability overlay is OFF)
                if not show_agent2_probability_overlay:
                    for start_point, end_point, gap_size in gap_lines:
                        # Use a magenta color for agent 2 visibility points to match the agent color
                        pygame.draw.circle(screen, (255, 50, 255), start_point, 1)
                        pygame.draw.circle(screen, (255, 50, 255), end_point, 1)
            
            # Second priority: Show probability overlay for agent 2 if enabled
            elif show_agent2_probability_overlay and not show_combined_probability_mode:
                # Visibility-based probability propagation for agent 2
                agent2_x, agent2_y = agent2.state[0], agent2.state[1]
                
                # DYNAMIC RANGE CALCULATION: Agent 1 reachability + inter-agent distance (max 800px)
                agent1_x, agent1_y = x, y  # Agent 1 position
                agent1_reachability = time_horizon * LEADER_LINEAR_VEL  # Agent 1's reachable distance
                inter_agent_distance = math.dist((agent1_x, agent1_y), (agent2_x, agent2_y))
                
                # Calculate dynamic vision range: reachability + distance between agents, capped at 800
                dynamic_vision_range = min(agent1_reachability + inter_agent_distance, DEFAULT_VISION_RANGE)
                agent2_vision_range = dynamic_vision_range
                
                # Initialize agent 2 probability list/dictionary (similar to agent 1)
                agent2_node_probabilities = {}
                
                # Get visibility data for agent 2 if available
                if visibility_map and map_graph:
                    # Start timing visibility-based probability calculation
                    visibility_calculation_start = time.perf_counter()
                    
                    # Find closest node to agent 2's position
                    agent2_pos = (agent2_x, agent2_y)
                    agent2_node_index = find_closest_node(map_graph.nodes, agent2_pos)
                    
                    if agent2_node_index is not None:
                        # Get visible nodes from the visibility map
                        visible_node_indices = set(visibility_map[agent2_node_index])
                        
                        # Calculate and store probabilities for all map graph nodes within range
                        for i, node in enumerate(map_graph.nodes):
                            node_x, node_y = node
                            
                            # Calculate distance from agent 2 to this node
                            dist_to_node = math.dist((agent2_x, agent2_y), (node_x, node_y))
                            
                            # Only process nodes within the 800px range
                            if dist_to_node <= agent2_vision_range:
                                # Check if this node is actually visible to agent 2
                                if i in visible_node_indices:
                                    # Node is visible: use fixed base probability (only store if > 0)
                                    if AGENT2_BASE_PROBABILITY > 0:
                                        agent2_node_probabilities[i] = AGENT2_BASE_PROBABILITY
                                # Note: nodes not visible or with 0 probability are not stored (optimization)
                    
                    # End timing visibility calculation
                    visibility_calculation_end = time.perf_counter()
                    visibility_calculation_time = visibility_calculation_end - visibility_calculation_start
                    
                else:
                    # Fallback: if no visibility data available, show uniform probability (original behavior)
                    fallback_start = time.perf_counter()
                    
                    for i, node in enumerate(map_graph.nodes):
                        node_x, node_y = node
                        
                        # Calculate distance from agent 2 to this node
                        dist_to_node = math.dist((agent2_x, agent2_y), (node_x, node_y))
                        
                        # Simple uniform probability within range (fallback behavior)
                        if dist_to_node <= agent2_vision_range:
                            # Use configured base probability for fallback (only store if > 0)
                            if AGENT2_BASE_PROBABILITY > 0:
                                agent2_node_probabilities[i] = AGENT2_BASE_PROBABILITY
                    
                    # End timing fallback calculation
                    fallback_end = time.perf_counter()
                    visibility_calculation_time = fallback_end - fallback_start
                
                # INTEGRATE GAP PROBABILITIES: Visibility-based probabilities override gap-based ones
                # This happens after visibility calculations but before drawing
                if 'agent2_gap_probabilities' in locals() and agent2_gap_probabilities:
                    for node_index, gap_prob in agent2_gap_probabilities.items():
                        if node_index not in agent2_node_probabilities:
                            # Only gap probability exists (no visibility): use gap probability
                            agent2_node_probabilities[node_index] = gap_prob
                        # If visibility probability exists, it overrides gap probability (no action needed)
                
                # Draw the agent 2 probability nodes from the integrated probabilities
                for i, probability in agent2_node_probabilities.items():
                    # All stored probabilities are > 0 by design (optimization)
                    node_x, node_y = map_graph.nodes[i]
                    
                    # Color blending scheme: probability determines mix between pink and green
                    # Low probability (e.g., 0.1): mostly pink (0.9) + little green (0.1)
                    # High probability (e.g., 0.9): mostly green (0.9) + little pink (0.1)
                    
                    # Define base colors
                    pink_color = (255, 105, 180)  # Hot pink
                    green_color = (0, 255, 100)   # Bright green
                    
                    # Use probability directly for color blending (0.0 to 1.0)
                    green_weight = probability  # How much green
                    pink_weight = 1.0 - probability  # How much pink
                    
                    # Blend the colors
                    red = int(pink_color[0] * pink_weight + green_color[0] * green_weight)
                    green = int(pink_color[1] * pink_weight + green_color[1] * green_weight)
                    blue = int(pink_color[2] * pink_weight + green_color[2] * green_weight)
                    
                    color = (red, green, blue)
                    
                    # Make circles smaller (reduced from 3-8 to 2-4)
                    min_size, max_size = 2, 4
                    node_size = int(min_size + probability * (max_size - min_size))
                    
                    # Draw the probability node with smaller circles
                    pygame.draw.circle(screen, color, (node_x, node_y), node_size)
                    
                    # Add subtle glow effect for higher probability nodes (>0.7)
                    if probability > 0.7:
                        # Glow uses more green for high probability
                        glow_color = (0, 255, 150)  # Green-cyan glow
                        glow_size = node_size + 1  # Smaller glow
                        
                        # Draw the glow circle
                        pygame.draw.circle(screen, glow_color, (node_x, node_y), glow_size)
                
                # Draw the visibility range circle for reference (cyan to match the probability nodes)
                pygame.draw.circle(screen, (0, 200, 200), (int(agent2_x), int(agent2_y)), agent2_vision_range, 2)
            
            # COMBINED PROBABILITY MODE: Multiply Agent 1 and Agent 2 probabilities
            elif show_combined_probability_mode:
                # Calculate combined probabilities for all map graph nodes
                combined_probabilities = {}
                
                # Get Agent 1 probabilities (merged with extended probabilities if available)
                agent1_probabilities = {}
                if show_probability_overlay and node_probabilities:
                    # Start with base reachability probabilities
                    for node_idx, base_prob in node_probabilities.items():
                        agent1_probabilities[node_idx] = base_prob
                    
                    # Merge with extended probabilities from rotating rods if available
                    if 'propagated_probabilities' in locals() and propagated_probabilities:
                        for node_idx, extended_prob in propagated_probabilities.items():
                            if node_idx in agent1_probabilities:
                                agent1_probabilities[node_idx] = max(agent1_probabilities[node_idx], extended_prob)
                            else:
                                agent1_probabilities[node_idx] = extended_prob
                
                # Get Agent 2 probabilities (merged with gap probabilities if available)
                agent2_probabilities = {}
                if show_agent2_probability_overlay and map_graph:
                    # Calculate Agent 2 visibility-based probabilities
                    agent2_x, agent2_y = agent2.state[0], agent2.state[1]
                    
                    # DYNAMIC RANGE CALCULATION: Agent 1 reachability + inter-agent distance (max 800px)
                    agent1_x, agent1_y = x, y  # Agent 1 position
                    agent1_reachability = time_horizon * LEADER_LINEAR_VEL  # Agent 1's reachable distance
                    inter_agent_distance = math.dist((agent1_x, agent1_y), (agent2_x, agent2_y))
                    
                    # Calculate dynamic vision range: reachability + distance between agents, capped at 800
                    agent2_vision_range = min(agent1_reachability + inter_agent_distance, DEFAULT_VISION_RANGE)
                    
                    if visibility_map:
                        agent2_pos = (agent2_x, agent2_y)
                        agent2_node_index = find_closest_node(map_graph.nodes, agent2_pos)
                        
                        if agent2_node_index is not None:
                            visible_node_indices = set(visibility_map[agent2_node_index])
                            
                            for i, node in enumerate(map_graph.nodes):
                                node_x, node_y = node
                                dist_to_node = math.dist((agent2_x, agent2_y), (node_x, node_y))
                                
                                if dist_to_node <= agent2_vision_range:
                                    if i in visible_node_indices:
                                        if AGENT2_BASE_PROBABILITY > 0:
                                            agent2_probabilities[i] = AGENT2_BASE_PROBABILITY
                    
                    # Merge with gap probabilities if available
                    if 'agent2_gap_probabilities' in locals() and agent2_gap_probabilities:
                        for node_index, gap_prob in agent2_gap_probabilities.items():
                            if node_index not in agent2_probabilities:
                                agent2_probabilities[node_index] = gap_prob
                
                # Calculate combined probabilities (multiplication)
                all_node_indices = set(agent1_probabilities.keys()) | set(agent2_probabilities.keys())
                
                # THREAT CLASSIFICATION SYSTEM: Threshold and rod-based threat analysis
                # (Only when threat classification mode is enabled)
                threat_nodes = {}  # Store threat nodes with their rod source information
                
                if show_threat_classification:
                    rod_colors = [
                        (255, 0, 0),    # Red - Rod 1
                        (0, 255, 0),    # Green - Rod 2  
                        (0, 0, 255),    # Blue - Rod 3
                        (255, 165, 0),  # Orange - Rod 4
                        (255, 0, 255),  # Magenta - Rod 5
                        (0, 255, 255),  # Cyan - Rod 6
                        (255, 255, 0),  # Yellow - Rod 7
                        (128, 0, 128),  # Purple - Rod 8
                    ]
                
                # Track rod-based threats from Agent 2 gap probabilities
                if show_threat_classification:
                    node_rod_sources = {}  # Map node_index -> list of (rod_id, probability)
                    
                    if 'agent2_rod_node_mapping' in locals() and agent2_rod_node_mapping:
                        # Use the proper rod tracking from gap processing
                        node_rod_sources = agent2_rod_node_mapping
                    elif 'agent2_gap_probabilities' in locals() and agent2_gap_probabilities:
                        # Fallback: assign rod IDs based on gap processing order (less accurate)
                        rod_id = 0
                        for node_index, gap_prob in agent2_gap_probabilities.items():
                            if node_index not in node_rod_sources:
                                node_rod_sources[node_index] = []
                            
                            current_rod_id = rod_id % len(rod_colors)
                            node_rod_sources[node_index].append((current_rod_id, gap_prob))
                            rod_id += 1
                
                for node_idx in all_node_indices:
                    prob1 = agent1_probabilities.get(node_idx, 0.0)
                    prob2 = agent2_probabilities.get(node_idx, 0.0)
                    
                    # Multiply probabilities (both must exist and be non-zero)
                    if prob1 > 0 and prob2 > 0:
                        combined_prob = prob1 * prob2
                        
                        # THREAT CLASSIFICATION: Check if above threshold (only when enabled)
                        if show_threat_classification and combined_prob >= threat_threshold:
                            # This is a threat node - classify by rod source
                            rod_sources = node_rod_sources.get(node_idx, []) if 'node_rod_sources' in locals() else []
                            
                            if rod_sources:
                                # Node has rod-based sources - classify by dominant rod
                                dominant_rod = max(rod_sources, key=lambda x: x[1])  # Highest probability rod
                                threat_nodes[node_idx] = {
                                    'probability': combined_prob,
                                    'rod_id': dominant_rod[0],
                                    'all_rods': rod_sources,
                                    'type': 'rod_based'
                                }
                            else:
                                # Threat from visibility only (no rod involvement)
                                threat_nodes[node_idx] = {
                                    'probability': combined_prob,
                                    'rod_id': None,
                                    'all_rods': [],
                                    'type': 'visibility_only'
                                }
                        
                        # Store for regular combined probability visualization (if enabled)
                        # Use consistent threshold for both visualization and threat classification
                        if combined_prob >= threat_threshold:
                            combined_probabilities[node_idx] = combined_prob
                
                # THREAT VISUALIZATION: Draw threat nodes with rod-based colors
                for node_idx, threat_info in threat_nodes.items():
                    node_x, node_y = map_graph.nodes[node_idx]
                    
                    if threat_info['type'] == 'rod_based' and threat_info['rod_id'] is not None:
                        # Rod-based threat - use rod color
                        rod_id = threat_info['rod_id']
                        base_color = rod_colors[rod_id % len(rod_colors)]
                        
                        # Intensity based on probability (above threshold)
                        prob_normalized = min(1.0, (threat_info['probability'] - threat_threshold) / (1.0 - threat_threshold))
                        intensity = 0.5 + 0.5 * prob_normalized  # 50% to 100% intensity
                        
                        color = tuple(int(c * intensity) for c in base_color)
                        
                        # Larger size for threats
                        node_size = 6 + int(prob_normalized * 4)  # 6-10 pixels
                        
                        # Draw threat node
                        pygame.draw.circle(screen, color, (node_x, node_y), node_size)
                        
                        # Draw outline to show it's a threat
                        pygame.draw.circle(screen, (255, 255, 255), (node_x, node_y), node_size + 1, 1)
                        
                        # Handle overlapping rods - draw small indicators
                        if len(threat_info['all_rods']) > 1:
                            # Multiple rods contribute - draw small dots around the main node
                            angle_step = 2 * math.pi / len(threat_info['all_rods'])
                            for i, (rod_id, _) in enumerate(threat_info['all_rods']):
                                angle = i * angle_step
                                offset_x = int(node_x + (node_size + 3) * math.cos(angle))
                                offset_y = int(node_y + (node_size + 3) * math.sin(angle))
                                rod_color = rod_colors[rod_id % len(rod_colors)]
                                pygame.draw.circle(screen, rod_color, (offset_x, offset_y), 2)
                    
                    elif threat_info['type'] == 'visibility_only':
                        # Visibility-based threat (no rod involvement) - use white/gray
                        prob_normalized = min(1.0, (threat_info['probability'] - threat_threshold) / (1.0 - threat_threshold))
                        intensity = int(128 + 127 * prob_normalized)  # Gray to white
                        color = (intensity, intensity, intensity)
                        
                        node_size = 6 + int(prob_normalized * 4)
                        pygame.draw.circle(screen, color, (node_x, node_y), node_size)
                        pygame.draw.circle(screen, (255, 255, 255), (node_x, node_y), node_size + 1, 1)
                
                # Draw combined probability nodes with purple-yellow color scheme
                # When threat classification is enabled, all combined probability nodes become threats
                # When threat classification is disabled, show all combined probability nodes normally
                if show_threat_classification:
                    # With consistent thresholds, all combined_probabilities nodes are now classified as threats
                    # No separate visualization needed - all nodes shown as rod-based or visibility-based threats
                    pass
                else:
                    # Threat classification disabled - show all combined probabilities normally
                    for node_idx, combined_prob in combined_probabilities.items():
                        node_x, node_y = map_graph.nodes[node_idx]
                        
                        # Purple-yellow color scheme: purple (low) to yellow (high)
                        # Low probability: dark purple, High probability: bright yellow
                        purple_color = (128, 0, 128)  # Dark purple
                        yellow_color = (255, 255, 0)  # Bright yellow
                        
                        # Normalize combined probability for color blending (0.0 to 1.0)
                        # Since we multiply probabilities, the range is typically much smaller
                        # Use a scaling factor to make the visualization more visible
                        display_prob = min(1.0, combined_prob * 10)  # Scale up for better visibility
                        
                        # Color blending
                        yellow_weight = display_prob
                        purple_weight = 1.0 - display_prob
                        
                        red = int(purple_color[0] * purple_weight + yellow_color[0] * yellow_weight)
                        green = int(purple_color[1] * purple_weight + yellow_color[1] * yellow_weight)
                        blue = int(purple_color[2] * purple_weight + yellow_color[2] * yellow_weight)
                        
                        color = (red, green, blue)
                        
                        # Node size based on combined probability (larger for higher probability)
                        min_size, max_size = 3, 7  # Slightly larger than individual modes
                        node_size = int(min_size + display_prob * (max_size - min_size))
                        
                        # Draw the combined probability node
                        pygame.draw.circle(screen, color, (node_x, node_y), node_size)
                        
                        # Add glow effect for very high combined probabilities
                        if display_prob > 0.8:
                            glow_color = (255, 255, 150)  # Bright yellow glow
                            glow_size = node_size + 2
                            pygame.draw.circle(screen, glow_color, (node_x, node_y), glow_size)
                
                # Draw both agents' visibility ranges for reference
                if map_graph:
                    # Agent 1 reachability circle (faint blue)
                    max_reachable_distance = time_horizon * LEADER_LINEAR_VEL
                    agent1_circle_color = (100, 150, 255)
                    pygame.draw.circle(screen, agent1_circle_color, (int(x), int(y)), int(max_reachable_distance), 1)
                    
                    # Agent 2 dynamic visibility circle (faint cyan)
                    # Calculate dynamic range for Agent 2
                    agent1_reachability = time_horizon * LEADER_LINEAR_VEL
                    inter_agent_distance = math.dist((x, y), (agent2_x, agent2_y))
                    agent2_dynamic_range = min(agent1_reachability + inter_agent_distance, DEFAULT_VISION_RANGE)
                    
                    agent2_circle_color = (0, 150, 150)
                    pygame.draw.circle(screen, agent2_circle_color, (int(agent2_x), int(agent2_y)), int(agent2_dynamic_range), 1)
                
                # Display combined mode info
                info_text = [
                    "Combined Probability Mode",
                    f"Agent 1 nodes: {len(agent1_probabilities)}",
                    f"Agent 2 nodes: {len(agent2_probabilities)}",
                    f"Combined nodes: {len(combined_probabilities)}",
                    "Purple → Yellow: Low → High"
                ]
                
                # Create info box
                info_font = pygame.font.SysFont('Arial', 16)
                info_bg = pygame.Surface((250, len(info_text) * 20 + 10))
                info_bg.fill((40, 0, 40))  # Dark purple background
                info_bg.set_alpha(200)
                
                screen.blit(info_bg, (WIDTH - 260, 10))
                
                for i, text in enumerate(info_text):
                    color = (255, 255, 255) if i == 0 else (200, 200, 200)
                    text_surface = info_font.render(text, True, color)
                    screen.blit(text_surface, (WIDTH - 250, 20 + i * 20))
            
            # Draw reachability circle LAST (on top of everything) when probability overlay is enabled
            if show_probability_overlay and not show_combined_probability_mode:
                max_reachable_distance = time_horizon * LEADER_LINEAR_VEL
                
                # Create pulsing effect for the circle
                pulse_intensity = (math.sin(pygame.time.get_ticks() / 800) + 1) / 2  # Value between 0 and 1
                
                # Draw the main reachability circle with higher visibility
                circle_alpha = int(120 + 60 * pulse_intensity)  # Alpha between 120 and 180
                circle_color = (100, 200, 255)  # Bright cyan
                
                # Draw multiple circle outlines for better visibility
                for width in range(1, 4):
                    alpha = circle_alpha - (width - 1) * 30  # Fade outer rings
                    circle_surface = pygame.Surface((ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT), pygame.SRCALPHA)
                    pygame.draw.circle(circle_surface, (*circle_color, alpha), (int(x), int(y)), int(max_reachable_distance), width)
                    screen.blit(circle_surface, (0, 0))
                
                # Draw a very faint filled circle to show the reachable area
                filled_alpha = int(20 + 15 * pulse_intensity)  # Alpha between 20 and 35
                filled_surface = pygame.Surface((ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT), pygame.SRCALPHA)
                pygame.draw.circle(filled_surface, (*circle_color, filled_alpha), (int(x), int(y)), int(max_reachable_distance))
                screen.blit(filled_surface, (0, 0))
            
            # Draw visibility range circle for the second agent (escort)
            if show_agent2_visibility_gaps:
                # DYNAMIC RANGE CALCULATION: Agent 1 reachability + inter-agent distance (max 800px)
                agent1_x, agent1_y = x, y  # Agent 1 position
                agent1_reachability = time_horizon * LEADER_LINEAR_VEL  # Agent 1's reachable distance
                inter_agent_distance = math.dist((agent1_x, agent1_y), (x2, y2))
                
                # Calculate dynamic vision range: reachability + distance between agents, capped at 800
                escort_visibility_range = min(agent1_reachability + inter_agent_distance, DEFAULT_VISION_RANGE)
                escort_circle_color = (0, 255, 255)  # Bright cyan color for second agent
                
                # Create pulsing effect for the second agent's circle
                escort_pulse_intensity = (math.sin(pygame.time.get_ticks() / 500) + 1) / 2  # Value between 0 and 1
                
                # Draw a single clean circle with pulsing effect
                escort_circle_alpha = int(180 + 75 * escort_pulse_intensity)  # Alpha between 180 and 255
                escort_circle_surface = pygame.Surface((ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT), pygame.SRCALPHA)
                pygame.draw.circle(escort_circle_surface, (*escort_circle_color, escort_circle_alpha), 
                                  (int(x2), int(y2)), int(escort_visibility_range), 2)
                screen.blit(escort_circle_surface, (0, 0))
            
            # INSTANTANEOUS SWEEP-BASED PROBABILITY ASSIGNMENT FOR AGENT 2 (Independent section)
            # Can be enabled either globally via Y key (show_rotating_rods) or individually via J key (show_agent2_rods)
            if (show_rotating_rods or show_agent2_rods) and visibility_map and gap_lines:
                # DYNAMIC RANGE CALCULATION: Agent 1 reachability + inter-agent distance (max 800px)
                agent1_x, agent1_y = x, y  # Agent 1 position
                agent1_reachability = time_horizon * LEADER_LINEAR_VEL  # Agent 1's reachable distance
                inter_agent_distance = math.dist((agent1_x, agent1_y), (x2, y2))
                
                # Calculate dynamic vision range: reachability + distance between agents, capped at 800
                agent2_dynamic_vision_range = min(agent1_reachability + inter_agent_distance, DEFAULT_VISION_RANGE)
                
                # PHASE 4B OPTIMIZATION: Selective computation - check if recomputation is needed
                current_agent2_pos = (x2, y2)
                current_agent2_angle = math.atan2(math.sin(theta2), math.cos(theta2))  # Normalize angle
                
                # Check if we can skip computation based on agent state changes
                if not agent2_optimizer.needs_recomputation(current_agent2_pos, current_agent2_angle):
                    # Skip computation, use cached result
                    agent2_gap_probabilities = agent2_optimizer.skip_computation()
                    
                    # Still need to do visual rendering, but skip the expensive computation
                    # Set timing variables for consistency with logging
                    agent2_computation_start = time.perf_counter()
                    visibility_calculation_time = 0.0
                    gap_processing_time = 0.0  
                    node_iteration_time = 0.0
                    total_gaps_processed = 0
                    total_angles_processed = 0
                    total_nodes_checked = 0
                    total_probabilities_assigned = len(agent2_gap_probabilities)
                    total_nodes_skipped = 0
                    node_count = len(map_graph.nodes) if map_graph and map_graph.nodes else 0
                    
                    # Skip to visual rendering section (computation bypassed)
                    computation_skipped = True
                else:
                    # Perform full computation
                    computation_skipped = False
                    
                    # PERFORMANCE TIMING: High-precision timing for Phase 1 optimization
                    agent2_computation_start = time.perf_counter()
                    
                    # Detailed timing breakdown variables
                    visibility_calculation_time = 0.0
                    gap_processing_time = 0.0
                    node_iteration_time = 0.0
                    
                    # Count operations for debugging
                    total_gaps_processed = 0
                    total_angles_processed = 0
                    total_nodes_checked = 0
                total_probabilities_assigned = 0
                total_nodes_skipped = 0  # Track nodes filtered out by spatial optimization
                
                # Track node count for performance analysis
                node_count = len(map_graph.nodes) if map_graph and map_graph.nodes else 0
                
                # PHASE 4B OPTIMIZATION: Only perform expensive computation if not skipped
                if not computation_skipped:
                    # INSTANTANEOUS SWEEP: Process all angles at once to assign gap probabilities
                    agent2_gap_probabilities = {}  # Stores gap-based probabilities separate from visibility
                    agent2_rod_node_mapping = {}   # NEW: Track which rod created which node probabilities
                    
                    # Start timing gap processing
                    gap_processing_start = time.perf_counter()
                
                rod_id = 0  # Track rod ID for threat classification
                for start_point, end_point, gap_size in gap_lines:
                    # Only process significant gaps
                    if gap_size < 50:
                        continue
                    
                    total_gaps_processed += 1
                    
                    # Determine gap orientation relative to clockwise ray casting
                    start_dist = math.dist((x2, y2), start_point)
                    end_dist = math.dist((x2, y2), end_point)
                    
                    # Determine near point (pivot point) and far point
                    if start_dist < end_dist:
                        near_point = start_point
                        far_point = end_point
                        is_cyan_gap = True  # Near-to-far (expanding)
                    else:
                        near_point = end_point
                        far_point = start_point
                        is_cyan_gap = False  # Far-to-near (contracting)
                    
                    # Calculate initial rod angle (along the gap line from near to far point)
                    initial_rod_angle = math.atan2(far_point[1] - near_point[1], far_point[0] - near_point[0])
                    
                    # Calculate rod length based on the gap size
                    original_gap_rod_length = math.dist(near_point, far_point)
                    rod_length = max(20, original_gap_rod_length)
                    
                    max_rotation = math.pi / 4  # Maximum 45 degrees rotation
                    
                    # Determine rotation direction based on gap color
                    if is_cyan_gap:
                        rotation_direction = -1  # Anticlockwise (counterclockwise)
                        gap_color = (0, 200, 255)  # Cyan tone
                    else:
                        rotation_direction = 1   # Clockwise
                        gap_color = (0, 240, 180)  # Green-cyan tone
                    
                    rod_base = near_point
                    
                    # INSTANTANEOUS SWEEP: Calculate probabilities for all angles at once
                    sweep_start_angle = initial_rod_angle
                    sweep_end_angle = initial_rod_angle + max_rotation * rotation_direction
                    total_sweep_angle = abs(sweep_end_angle - sweep_start_angle)
                    
                    # PHASE 3 VECTORIZED OPTIMIZATION: Replace nested loops with NumPy operations
                    # This reduces computation from O(angles×nodes) to O(1) vectorized operations
                    
                    # Pre-filter nodes within vision range once for this gap
                    if map_graph and map_graph.nodes:
                        # Convert all nodes to NumPy arrays for vectorized operations
                        all_nodes = np.array(map_graph.nodes)
                        
                        # PHASE 4B OPTIMIZATION: Enhanced spatial filtering with multiple criteria
                        agent2_pos = np.array([x2, y2])
                        
                        # Primary filter: Distance-based (use fast squared distance)
                        distances_squared = np.sum((all_nodes - agent2_pos)**2, axis=1)
                        vision_range_squared = agent2_dynamic_vision_range * agent2_dynamic_vision_range
                        within_range_mask = distances_squared <= vision_range_squared
                        
                        # Secondary filter: Gap-relevance filter (only consider nodes near gap direction)
                        if len(all_nodes) > 100:  # Only apply for large node sets
                            # Calculate gap center direction
                            gap_center = np.array([(near_point[0] + far_point[0]) / 2, 
                                                  (near_point[1] + far_point[1]) / 2])
                            gap_direction = gap_center - agent2_pos
                            gap_direction_norm = np.linalg.norm(gap_direction)
                            
                            if gap_direction_norm > 0:
                                gap_direction = gap_direction / gap_direction_norm
                                
                                # Calculate angle between agent->node and agent->gap vectors
                                node_directions = all_nodes - agent2_pos
                                node_distances = np.linalg.norm(node_directions, axis=1)
                                
                                # Avoid division by zero
                                valid_nodes_mask = node_distances > 1e-6
                                
                                if np.any(valid_nodes_mask):
                                    node_directions_norm = node_directions[valid_nodes_mask] / node_distances[valid_nodes_mask, np.newaxis]
                                    
                                    # Dot product for angle calculation
                                    dot_products = np.dot(node_directions_norm, gap_direction)
                                    
                                    # Keep nodes within ±60 degrees of gap direction
                                    angle_threshold = math.cos(math.pi / 3)  # 60 degrees
                                    gap_relevant_mask = np.zeros(len(all_nodes), dtype=bool)
                                    gap_relevant_mask[valid_nodes_mask] = dot_products >= angle_threshold
                                    
                                    # Combine filters
                                    within_range_mask = within_range_mask & gap_relevant_mask
                        
                        within_range_indices = np.where(within_range_mask)[0]
                        within_range_nodes = all_nodes[within_range_mask]
                        
                        # Track statistics
                        total_nodes_skipped += len(all_nodes) - len(within_range_nodes)
                        total_nodes_checked += len(within_range_nodes)
                        
                        if len(within_range_nodes) > 0:
                            # Generate all sweep angles at once
                            num_sweep_angles = 20
                            total_angles_processed += num_sweep_angles + 1
                            
                            # Vectorized angle generation
                            angle_steps = np.linspace(0, 1, num_sweep_angles + 1)
                            sweep_angles = sweep_start_angle + angle_steps * (sweep_end_angle - sweep_start_angle)
                            
                            # Vectorized gap probabilities (0.8 to 1.0 range)
                            base_gap_probability = 0.8
                            gap_probabilities = base_gap_probability + (angle_steps * 0.2)
                            
                            # Vectorized rod end positions for all angles
                            rod_ends_x = rod_base[0] + rod_length * np.cos(sweep_angles)
                            rod_ends_y = rod_base[1] + rod_length * np.sin(sweep_angles)
                            
                            # For each angle, calculate distances from all nodes to the rod line
                            bar_width = 15.0  # Wider sweep for probability assignment
                            
                            # Start timing vectorized computation
                            node_iteration_start = time.perf_counter()
                            
                            # Initialize result arrays
                            node_max_probabilities = np.zeros(len(within_range_nodes))
                            
                            # Process all angles and nodes simultaneously
                            for angle_idx, (angle, gap_prob, rod_end_x, rod_end_y) in enumerate(
                                zip(sweep_angles, gap_probabilities, rod_ends_x, rod_ends_y)
                            ):
                                # Vectorized rod line calculations
                                rod_base_arr = np.array(rod_base)
                                rod_end_arr = np.array([rod_end_x, rod_end_y])
                                
                                # Rod vector
                                rod_vec = rod_end_arr - rod_base_arr
                                rod_length_sq = np.dot(rod_vec, rod_vec)
                                
                                if rod_length_sq > 0:
                                    # Vectors from rod start to all nodes
                                    node_vecs = within_range_nodes - rod_base_arr
                                    
                                    # Vectorized projection onto rod line
                                    projections = np.dot(node_vecs, rod_vec) / rod_length_sq
                                    projections = np.clip(projections, 0, 1)
                                    
                                    # Vectorized closest points on rod line
                                    closest_points = rod_base_arr + projections[:, np.newaxis] * rod_vec
                                    
                                    # Vectorized distances from nodes to rod line
                                    distance_vectors = within_range_nodes - closest_points
                                    distances_to_rod = np.linalg.norm(distance_vectors, axis=1)
                                    
                                    # Boolean mask for nodes within bar width
                                    within_bar_mask = distances_to_rod <= bar_width
                                    
                                    # Update maximum probabilities for affected nodes
                                    node_max_probabilities[within_bar_mask] = np.maximum(
                                        node_max_probabilities[within_bar_mask], gap_prob
                                    )
                            
                            # Assign final probabilities to nodes
                            significant_prob_mask = node_max_probabilities > 0
                            significant_indices = within_range_indices[significant_prob_mask]
                            significant_probs = node_max_probabilities[significant_prob_mask]
                            
                            for idx, prob in zip(significant_indices, significant_probs):
                                if idx in agent2_gap_probabilities:
                                    agent2_gap_probabilities[idx] = max(agent2_gap_probabilities[idx], prob)
                                else:
                                    agent2_gap_probabilities[idx] = prob
                                
                                # NEW: Track rod source for threat classification
                                if idx not in agent2_rod_node_mapping:
                                    agent2_rod_node_mapping[idx] = []
                                agent2_rod_node_mapping[idx].append((rod_id, prob))
                                
                                total_probabilities_assigned += 1
                            
                            # End timing vectorized computation
                            node_iteration_end = time.perf_counter()
                            node_iteration_time += (node_iteration_end - node_iteration_start)
                
                    # Increment rod ID for next gap
                    rod_id += 1
                
                # End timing gap processing
                gap_processing_end = time.perf_counter()
                gap_processing_time = gap_processing_end - gap_processing_start
                
                # PHASE 4B OPTIMIZATION: Update optimizer state after computation
                agent2_optimizer.update_state(current_agent2_pos, current_agent2_angle, agent2_gap_probabilities)
                
                # VISUAL REPRESENTATION: Draw static swept areas (no animation)
                for start_point, end_point, gap_size in gap_lines:
                    # Only process significant gaps
                    if gap_size < 50:
                        continue
                    
                    # Same gap analysis as above
                    start_dist = math.dist((x2, y2), start_point)
                    end_dist = math.dist((x2, y2), end_point)
                    
                    if start_dist < end_dist:
                        near_point = start_point
                        far_point = end_point
                        is_cyan_gap = True  # Near-to-far (expanding)
                    else:
                        near_point = end_point
                        far_point = start_point
                        is_cyan_gap = False  # Far-to-near (contracting)
                    
                    initial_rod_angle = math.atan2(far_point[1] - near_point[1], far_point[0] - near_point[0])
                    original_gap_rod_length = math.dist(near_point, far_point)
                    rod_length = max(20, original_gap_rod_length)
                    max_rotation = math.pi / 4
                    
                    if is_cyan_gap:
                        rotation_direction = -1
                        gap_color = (0, 200, 255)
                    else:
                        rotation_direction = 1
                        gap_color = (0, 240, 180)
                    
                    rod_base = near_point
                    
                    # Draw rod base indicator (pivot point)
                    pygame.draw.circle(screen, (0, 255, 255), rod_base, 8)
                    pygame.draw.circle(screen, (0, 180, 180), rod_base, 4)
                    
                    # Draw the complete swept area (static, not animated)
                    sweep_surface = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
                    sweep_start_angle = initial_rod_angle
                    sweep_end_angle = initial_rod_angle + max_rotation * rotation_direction
                    
                    # Use gap color for swept area with transparency
                    sweep_color = (*gap_color, 40)
                    
                    # Draw the swept arc area as a polygon
                    arc_points = [rod_base]
                    num_arc_points = 20
                    for i in range(num_arc_points + 1):
                        t = i / num_arc_points
                        angle = sweep_start_angle + t * (sweep_end_angle - sweep_start_angle)
                        # PHASE 4B OPTIMIZATION: Use fast trigonometric lookup
                        point = (
                            rod_base[0] + rod_length * trig_lookup.fast_cos(angle),
                            rod_base[1] + rod_length * trig_lookup.fast_sin(angle)
                        )
                        arc_points.append(point)
                    
                    pygame.draw.polygon(sweep_surface, sweep_color, arc_points)
                    screen.blit(sweep_surface, (0, 0))
                    
                    # Draw the boundary lines of the swept area
                    boundary_color = (gap_color[0]//2, gap_color[1]//2, gap_color[2]//2)
                    # PHASE 4B OPTIMIZATION: Use fast trigonometric lookup
                    start_boundary = (
                        rod_base[0] + rod_length * trig_lookup.fast_cos(sweep_start_angle),
                        rod_base[1] + rod_length * trig_lookup.fast_sin(sweep_start_angle)
                    )
                    end_boundary = (
                        rod_base[0] + rod_length * trig_lookup.fast_cos(sweep_end_angle),
                        rod_base[1] + rod_length * trig_lookup.fast_sin(sweep_end_angle)
                    )
                    pygame.draw.line(screen, boundary_color, rod_base, start_boundary, 2)
                    pygame.draw.line(screen, boundary_color, rod_base, end_boundary, 2)
                    
                    # Draw the gap line (initial rod position)
                    # PHASE 4B OPTIMIZATION: Use fast trigonometric lookup
                    initial_rod_end = (
                        rod_base[0] + rod_length * trig_lookup.fast_cos(initial_rod_angle),
                        rod_base[1] + rod_length * trig_lookup.fast_sin(initial_rod_angle)
                    )
                    pygame.draw.line(screen, gap_color, rod_base, initial_rod_end, 3)
                
                # INTEGRATE GAP PROBABILITIES: Add to agent2_node_probabilities BEFORE visibility calculations
                # This ensures gap probabilities are considered in the visibility-based system
                if 'agent2_gap_probabilities' in locals() and agent2_gap_probabilities:
                    # Integration happens after visibility calculations (see lines above)
                    # This section just ensures the gap probabilities are calculated and ready
                    pass
                
                # PERFORMANCE TIMING: Complete detailed timing analysis for Phase 1 optimization
                agent2_computation_end = time.perf_counter()
                total_computation_time = agent2_computation_end - agent2_computation_start
                
                # Convert to milliseconds for consistency with existing display
                total_time_ms = total_computation_time * 1000
                visibility_time_ms = visibility_calculation_time * 1000
                gap_time_ms = gap_processing_time * 1000
                node_iteration_time_ms = node_iteration_time * 1000
                
                # Calculate current FPS
                current_fps = clock.get_fps()
                
                # PHASE 4B: Get selective computation statistics
                optimizer_stats = agent2_optimizer.get_performance_stats()
                
                # Log performance data to CSV for Phase 4B analysis
                performance_log_file = "agent2_performance_log.csv"
                timestamp = datetime.now().isoformat()
                
                # Create CSV header if file doesn't exist
                write_header = not os.path.exists(performance_log_file)
                
                with open(performance_log_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if write_header:
                        writer.writerow([
                            'timestamp', 'total_time_ms', 'visibility_time_ms', 'gap_time_ms', 
                            'node_iteration_time_ms', 'node_count', 'fps', 'gaps_processed', 
                            'angles_processed', 'nodes_checked', 'probabilities_assigned', 'nodes_skipped',
                            'computation_skipped', 'skip_rate_percent', 'optimizer_total_calls'
                        ])
                    
                    writer.writerow([
                        timestamp, total_time_ms, visibility_time_ms, gap_time_ms,
                        node_iteration_time_ms, node_count, current_fps, total_gaps_processed,
                        total_angles_processed, total_nodes_checked, total_probabilities_assigned, total_nodes_skipped,
                        computation_skipped, optimizer_stats['skip_rate_percent'], optimizer_stats['total_calls']
                    ])
                
                # Store computation time for FPS display (maintain backward compatibility)
                agent2_computation_ms = total_time_ms
            
            # NOW DRAW AGENTS ON TOP OF ALL VISUALIZATION ELEMENTS
            # This ensures agents are clearly visible above all visibility indicators
            
            # Draw the first agent (magenta - arrow key controlled)
            pygame.draw.circle(screen, AGENT_COLOR, (int(x), int(y)), AGENT_RADIUS)
            end_x = x + AGENT_RADIUS * 1.2 * math.cos(theta)
            end_y = y + AGENT_RADIUS * 1.2 * math.sin(theta)
            pygame.draw.line(screen, (255,255,255), (x, y), (end_x, end_y), 3)
            
            # Add label for first agent
            label_text = label_font.render("↑", True, (255, 255, 255))
            label_bg = pygame.Surface((label_text.get_width() + 4, label_text.get_height() + 2))
            label_bg.fill((100, 0, 100))
            label_bg.set_alpha(180)
            screen.blit(label_bg, (int(x) - label_text.get_width()//2 - 2, int(y) - AGENT_RADIUS - 20))
            screen.blit(label_text, (int(x) - label_text.get_width()//2, int(y) - AGENT_RADIUS - 19))
            
            # Draw the second agent (cyan - WASD controlled) with enhanced visibility
            # Draw a larger highlight circle first (bright white/light cyan for contrast)
            pygame.draw.circle(screen, (200, 255, 255), (int(x2), int(y2)), AGENT_RADIUS + 4)
            # Draw the actual agent
            pygame.draw.circle(screen, AGENT2_COLOR, (int(x2), int(y2)), AGENT_RADIUS)
            end_x2 = x2 + AGENT_RADIUS * 1.2 * math.cos(theta2)
            end_y2 = y2 + AGENT_RADIUS * 1.2 * math.sin(theta2)
            pygame.draw.line(screen, (255,255,255), (x2, y2), (end_x2, end_y2), 3)
            
            # Add label for second agent with matching background color
            label_text2 = label_font.render("W", True, (255, 255, 255))
            label_bg2 = pygame.Surface((label_text2.get_width() + 4, label_text2.get_height() + 2))
            label_bg2.fill((0, 180, 180))  # Match the cyan color scheme
            label_bg2.set_alpha(180)
            screen.blit(label_bg2, (int(x2) - label_text2.get_width()//2 - 2, int(y2) - AGENT_RADIUS - 20))
            screen.blit(label_text2, (int(x2) - label_text2.get_width()//2, int(y2) - AGENT_RADIUS - 19))

            # Draw probability overlay if enabled and data is available
            if OVERLAY_AVAILABLE and show_combined_probability_mode and 'combined_probabilities' in locals() and combined_probabilities:
                probability_overlay.draw_probability_overlay(screen, combined_probabilities)

            # Update the display
            pygame.display.flip()

            # Cap the frame rate
            clock.tick(60)
    finally:
        # Save agent states on exit
        try:
            with open(AGENT_STATE_FILE, 'wb') as f:
                pickle.dump(agent.state, f)
        except Exception as e:
            print(f"Warning: Could not save agent state: {e}")
            
        try:
            with open(AGENT2_STATE_FILE, 'wb') as f:
                pickle.dump(agent2.state, f)
        except Exception as e:
            print(f"Warning: Could not save agent2 state: {e}")

    pygame.quit()
    sys.exit()

# PHASE 4B OPTIMIZATION: Agent2ComputationOptimizer for selective computation
class Agent2ComputationOptimizer:
    """
    Phase 4B optimization: Selective computation system to avoid unnecessary recomputations
    when agent state hasn't changed significantly.
    """
    def __init__(self):
        self.last_position = None
        self.last_angle = None
        self.last_result = None
        self.last_gap_probabilities = None
        
        # Configurable thresholds for detecting significant changes
        self.position_threshold = 5.0  # pixels - recompute if agent moves >5px
        self.angle_threshold = 0.1     # radians (~5.7 degrees) - recompute if angle changes significantly
        
        # Performance tracking
        self.computation_skipped = 0
        self.computation_performed = 0
        
        # Cache invalidation tracking
        self.frames_since_last_computation = 0
        self.max_frames_without_computation = 30  # Force recomputation every 30 frames max
    
    def needs_recomputation(self, current_pos, current_angle):
        """
        Determine if Agent2 computation needs to be performed based on state changes.
        
        Args:
            current_pos: Current agent position (x, y)
            current_angle: Current agent angle in radians
            
        Returns:
            bool: True if recomputation is needed, False if cached result can be used
        """
        # Always compute on first run
        if self.last_position is None:
            return True
            
        # Force recomputation after max frames to prevent stale data
        if self.frames_since_last_computation >= self.max_frames_without_computation:
            return True
        
        # Check position change
        pos_delta = math.sqrt((current_pos[0] - self.last_position[0])**2 + 
                             (current_pos[1] - self.last_position[1])**2)
        
        # Check angle change
        angle_delta = abs(current_angle - self.last_angle)
        # Handle angle wraparound (e.g., from 359° to 1°)
        angle_delta = min(angle_delta, 2*math.pi - angle_delta)
        
        return (pos_delta > self.position_threshold or 
                angle_delta > self.angle_threshold)
    
    def update_state(self, position, angle, gap_probabilities):
        """Update cached state after computation"""
        self.last_position = position
        self.last_angle = angle
        self.last_gap_probabilities = gap_probabilities.copy() if gap_probabilities else {}
        self.computation_performed += 1
        self.frames_since_last_computation = 0
    
    def skip_computation(self):
        """Record that computation was skipped and return cached result"""
        self.computation_skipped += 1
        self.frames_since_last_computation += 1
        return self.last_gap_probabilities if self.last_gap_probabilities else {}
    
    def get_performance_stats(self):
        """Get performance statistics for monitoring"""
        total_calls = self.computation_performed + self.computation_skipped
        skip_rate = self.computation_skipped / total_calls if total_calls > 0 else 0
        return {
            'total_calls': total_calls,
            'computations_performed': self.computation_performed,
            'computations_skipped': self.computation_skipped,
            'skip_rate_percent': skip_rate * 100,
            'frames_since_last': self.frames_since_last_computation
        }

# Global instance for Agent2 computation optimization
agent2_optimizer = Agent2ComputationOptimizer()
