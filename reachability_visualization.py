#!/usr/bin/env python3
"""
Evader Reachability Set Visualization with Node Probability Distribution
This script extends the environment inspection tool to show reachability set 
for the evader at its current location with probabilities for each map node.
"""

import os
import sys
import pygame
import numpy as np
import math
import time
import multiprocessing
from math import sin, cos, pi, sqrt, log

# Make sure the project directory is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from multitrack.utils.config import *
from multitrack.models.simulation_environment import SimulationEnvironment
from multitrack.utils.map_graph import MapGraph
from multitrack.models.agents.visitor_agent import UnicycleModel
from multitrack.utils.pathfinding import find_shortest_path

# Import environment dimensions
from multitrack.simulation.unicycle_reachability_simulation import (
    ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, WIDTH, HEIGHT, SIDEBAR_WIDTH
)

# Define loading status functions
def update_loading_status(message, progress):
    """Simple loading status function"""
    print(f"{message} {progress*100:.1f}%")

def render_animated_loading_screen(message, progress, additional_text=None):
    """Simple animated loading screen function"""
    update_loading_status(message, progress)
    if additional_text:
        for line in additional_text:
            print(f"  {line}")

def generate_trajectory(agent_state, agent_controls, prediction_steps, prediction_dt, walls=None, doors=None):
    """
    Generate a single trajectory sample for the agent
    
    Args:
        agent_state: Initial state (x, y, theta, v) of the agent
        agent_controls: Controls (linear_vel, angular_vel) of the agent
        prediction_steps: Number of prediction steps
        prediction_dt: Time step for prediction
        walls: List of walls to check for collisions
        doors: List of doors to check for collisions
        
    Returns:
        List of points [(x, y)] in the trajectory
    """
    agent_x, agent_y, agent_theta, agent_v = agent_state
    linear_vel, angular_vel = agent_controls
    
    # Create trajectory starting at agent's current position
    trajectory = [(agent_x, agent_y)]
    
    # Initialize with current state plus small noise
    x = agent_x + np.random.normal(0, 3)
    y = agent_y + np.random.normal(0, 3)
    theta = agent_theta + np.random.normal(0, 0.1)
    v = max(0, agent_v + np.random.normal(0, 5))
    
    # Agent radius for collision checking
    agent_radius = 16
    
    # Generate the trajectory steps
    for _ in range(prediction_steps):
        # Add control variation
        v_ctrl = v + np.random.normal(0, 10)
        omega = angular_vel + np.random.normal(0, 0.2)
        
        # Calculate new position with unicycle dynamics
        new_x = x + v_ctrl * cos(theta) * prediction_dt + np.random.normal(0, 3)
        new_y = y + v_ctrl * sin(theta) * prediction_dt + np.random.normal(0, 3)
        new_theta = theta + omega * prediction_dt + np.random.normal(0, 0.1)
        
        # Normalize angle
        new_theta = (new_theta + pi) % (2 * pi) - pi
        
        # Apply boundaries
        new_x = max(0, min(new_x, ENVIRONMENT_WIDTH))
        new_y = max(0, min(new_y, ENVIRONMENT_HEIGHT))
        
        # Check for collisions with walls if provided
        collision = False
        if walls:
            for wall in walls:
                # Check if the path from (x,y) to (new_x,new_y) intersects with the wall
                # or if the agent at new position (with radius) collides with the wall
                wx1, wy1, wx2, wy2 = wall
                
                # Simple line segment intersection check
                def ccw(A, B, C):
                    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
                
                def intersect(A, B, C, D):
                    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
                
                # Check if trajectory path intersects with wall
                if intersect((x, y), (new_x, new_y), (wx1, wy1), (wx2, wy2)):
                    collision = True
                    break
                
                # Check if agent at new position is too close to wall
                # Distance from point to line segment
                def point_to_line_dist(px, py, x1, y1, x2, y2):
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
                    return sqrt(dx * dx + dy * dy)
                
                # Check if agent is too close to wall
                if point_to_line_dist(new_x, new_y, wx1, wy1, wx2, wy2) < agent_radius:
                    collision = True
                    break
        
        # Check for door collisions if provided (doors can be passed through)
        if not collision and doors:
            for door in doors:
                # Doors are special - the agent can pass through them
                # We'll assume doors are always open for this visualization
                # In a more complex implementation, you might check door state
                pass  # Skip door collision checking
        
        # If collision detected, try to adjust the path or stop
        if collision:
            # Option 1: Stop at current position (simple approach)
            # trajectory.append((x, y))
            
            # Option 2: Try random direction to simulate bouncing/deflection
            bounce_angle = np.random.uniform(0, 2*pi)
            bounce_distance = np.random.uniform(5, 15)
            new_x = x + bounce_distance * cos(bounce_angle)
            new_y = y + bounce_distance * sin(bounce_angle)
            
            # Apply boundaries again after bounce
            new_x = max(0, min(new_x, ENVIRONMENT_WIDTH))
            new_y = max(0, min(new_y, ENVIRONMENT_HEIGHT))
        
        # Update position
        x, y, theta = new_x, new_y, new_theta
        
        # Add position to trajectory
        trajectory.append((x, y))
    
    return trajectory

def process_node_chunk(nodes_chunk, node_indices, trajectories, agent_pos, max_range):
    """
    Process a chunk of nodes to calculate their hit counts
    
    Args:
        nodes_chunk: List of nodes (x, y) to process
        node_indices: Corresponding indices of the nodes
        trajectories: List of trajectories to check against
        agent_pos: Agent's current position (x, y)
        max_range: Maximum range from agent to consider
        
    Returns:
        Dictionary mapping node indices to hit counts
    """
    chunk_hits = {}
    
    for idx, node in zip(node_indices, nodes_chunk):
        # Skip nodes that are too far from the agent
        if max_range:
            distance_to_agent = math.dist(agent_pos, node)
            if distance_to_agent > max_range:
                continue
        
        hit_count = 0
        # Count trajectories that pass near this node
        for trajectory in trajectories:
            for point in trajectory:
                # Check if trajectory point is close to this node
                distance = math.dist(point, node)
                if distance < 20:  # Detection radius (can adjust as needed)
                    hit_count += 1
                    break  # Only count each trajectory once per node
        
        if hit_count > 0:
            chunk_hits[idx] = hit_count
    
    return chunk_hits

def calculate_node_reachability(agent, map_graph, num_samples=200, prediction_steps=20, prediction_dt=0.1, max_range=None, use_multicore=True, num_cores=None, walls=None, doors=None):
    """
    Calculate the reachability set and node probabilities for an agent
    
    Args:
        agent: The agent (evader) for which to calculate the reachability set
        map_graph: The map graph with nodes
        num_samples: Number of trajectory samples to generate
        prediction_steps: Number of steps to predict into the future
        prediction_dt: Time step for prediction
        max_range: Maximum reachability range (None = unlimited)
        use_multicore: Whether to use multicore processing
        num_cores: Number of cores to use (None = auto-detect)
        walls: List of walls to check for collisions
        doors: List of doors to check for collisions
        
    Returns:
        Dictionary mapping node indices to probability values
    """
    if max_range is None:
        max_range = MAP_GRAPH_VISIBILITY_RANGE  # Use the same range as visibility analysis
    
    # Get agent state and controls
    agent_state = agent.state
    agent_pos = (agent_state[0], agent_state[1])
    agent_controls = agent.controls
    
    # Generate sample trajectories
    print(f"Generating {num_samples} trajectory samples...")
    trajectories = []
    
    # If using multicore for trajectory generation (for large numbers of samples)
    if use_multicore and num_samples > 100:
        cores_to_use = num_cores if num_cores else multiprocessing.cpu_count()
        samples_per_core = num_samples // cores_to_use
        
        # Create a pool of workers
        with multiprocessing.Pool(processes=cores_to_use) as pool:
            # Generate trajectories in parallel
            trajectory_chunks = pool.starmap(
                generate_trajectory,
                [(agent_state, agent_controls, prediction_steps, prediction_dt, walls, doors) 
                 for _ in range(num_samples)]
            )
            trajectories = trajectory_chunks
    else:
        # Sequential trajectory generation
        for _ in range(num_samples):
            trajectory = generate_trajectory(agent_state, agent_controls, prediction_steps, prediction_dt, walls, doors)
            trajectories.append(trajectory)
    
    print(f"Generated {len(trajectories)} trajectories")
    
    # Process nodes to calculate probabilities
    if use_multicore:
        cores_to_use = num_cores if num_cores else multiprocessing.cpu_count()
        print(f"Processing nodes using {cores_to_use} CPU cores...")
        
        # Split nodes into chunks for parallel processing
        nodes = map_graph.nodes
        chunk_size = max(1, len(nodes) // cores_to_use)
        node_chunks = []
        node_indices_chunks = []
        
        for i in range(0, len(nodes), chunk_size):
            end = min(i + chunk_size, len(nodes))
            node_chunks.append(nodes[i:end])
            node_indices_chunks.append(list(range(i, end)))
        
        # Process node chunks in parallel
        with multiprocessing.Pool(processes=cores_to_use) as pool:
            results = pool.starmap(
                process_node_chunk,
                [(chunk, indices, trajectories, agent_pos, max_range) 
                 for chunk, indices in zip(node_chunks, node_indices_chunks)]
            )
        
        # Combine results from all processes
        node_hits = {}
        for result in results:
            node_hits.update(result)
    else:
        # Sequential node processing
        print("Processing nodes in single-core mode...")
        node_hits = {}
        
        # For each node, count how many trajectories pass near it
        for i, node in enumerate(map_graph.nodes):
            # Skip nodes that are too far from the agent
            if max_range:
                distance_to_agent = math.dist(agent_pos, node)
                if distance_to_agent > max_range:
                    continue
            
            hit_count = 0
            # Count trajectories that pass near this node
            for trajectory in trajectories:
                for point in trajectory:
                    # Check if trajectory point is close to this node
                    distance = math.dist(point, node)
                    if distance < 20:  # Detection radius
                        hit_count += 1
                        break  # Only count each trajectory once per node
            
            if hit_count > 0:
                node_hits[i] = hit_count
    
    # Convert hits to probabilities
    node_probabilities = {}
    total_hits = sum(node_hits.values())
    if total_hits > 0:
        for i in node_hits:
            node_probabilities[i] = node_hits[i] / total_hits
    
    print(f"Reachability calculation complete: {len(node_probabilities)} nodes with non-zero probability")
    return node_probabilities

def draw_node_probabilities(screen, map_graph, probabilities, selected_node=None):
    """
    Draw nodes with color based on their probability value
    
    Args:
        screen: pygame surface to draw on
        map_graph: The map graph with nodes
        probabilities: Dictionary mapping node indices to probability values
        selected_node: Index of currently selected node, if any
    """
    # Find maximum probability for normalization
    max_prob = max(probabilities.values()) if probabilities else 1.0
    if max_prob == 0:
        max_prob = 1.0
    
    # Create a surface for the visualization
    viz_surface = pygame.Surface((ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT), pygame.SRCALPHA)
    
    # Draw connections between nodes
    for edge in map_graph.edges:
        i, j = edge
        # Only draw edges if both nodes have probability
        if i in probabilities and j in probabilities:
            start = map_graph.nodes[i]
            end = map_graph.nodes[j]
            
            # Calculate average probability for edge color
            avg_prob = (probabilities[i] + probabilities[j]) / 2
            
            # Scale the intensity based on probability
            intensity = int(255 * avg_prob / max_prob)
            edge_color = (0, intensity, intensity, 100)  # Cyan with alpha
            
            pygame.draw.line(viz_surface, edge_color, start, end, 1)
    
    # Draw the nodes
    for i, node in enumerate(map_graph.nodes):
        if i in probabilities:
            # Calculate node color based on probability
            prob = probabilities[i]
            intensity = int(255 * prob / max_prob)
            
            # Use a cyan color gradient from dark to light
            node_color = (0, intensity, intensity)
            
            # Size also based on probability (bigger = higher probability)
            size = max(3, int(5 * prob / max_prob) + 3)
            
            # Draw the node
            pygame.draw.circle(viz_surface, node_color, node, size)
            
            # Add glow effect for higher probabilities
            if prob > 0.5 * max_prob:
                glow_size = size + 4
                glow_alpha = int(150 * prob / max_prob)
                pygame.draw.circle(viz_surface, (*node_color, glow_alpha), node, glow_size)
    
    # If there's a selected node, highlight it
    if selected_node is not None and selected_node < len(map_graph.nodes):
        node = map_graph.nodes[selected_node]
        
        # Create pulsing effect
        pulse = (sin(pygame.time.get_ticks() / 300) + 1) / 4 + 0.75
        glow_size = int(12 * pulse)
        
        # Draw highlight
        pygame.draw.circle(viz_surface, (255, 255, 0, 100), node, glow_size)
        pygame.draw.circle(viz_surface, (255, 255, 0), node, 8)
        pygame.draw.circle(viz_surface, (255, 200, 0), node, 4)
    
    # Blit the visualization surface to the screen
    screen.blit(viz_surface, (0, 0))

def get_node_probability_info(map_graph, node_idx, probabilities, agent):
    """Generate info text about a node's probability"""
    if node_idx is None or node_idx >= len(map_graph.nodes):
        return ["No node selected"]
    
    node = map_graph.nodes[node_idx]
    info = [
        f"Selected Node: {node_idx}",
        f"Position: ({int(node[0])}, {int(node[1])})"
    ]
    
    if node_idx in probabilities:
        prob = probabilities[node_idx]
        # Find rank of this node's probability
        sorted_probs = sorted(probabilities.values(), reverse=True)
        rank = sorted_probs.index(prob) + 1
        
        info.extend([
            f"Probability: {prob:.4f}",
            f"Rank: {rank} of {len(probabilities)}",
            f"Percentile: {100 * (1 - rank/len(probabilities)):.1f}%"
        ])
        
        # Add distance from evader
        node_x, node_y = node
        evader_x, evader_y = agent.state[0], agent.state[1]
        distance = math.dist((node_x, node_y), (evader_x, evader_y))
        info.append(f"Distance from evader: {distance:.1f} pixels")
    else:
        info.append("No probability data for this node")
    
    return info

def draw_path(screen, path, color=(255, 215, 0), width=4):
    """
    Draw a path on the screen
    
    Args:
        screen: pygame surface to draw on
        path: List of points [(x, y)] forming the path
        color: RGB color tuple for the path
        width: Width of the path line
    """
    if not path or len(path) < 2:
        return
    
    # Create a surface for the path
    path_surface = pygame.Surface((ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT), pygame.SRCALPHA)
    
    # Draw lines connecting the path points
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        pygame.draw.line(path_surface, color, start, end, width)
    
    # Draw circles at each node in the path
    for point in path:
        pygame.draw.circle(path_surface, color, point, width + 1)
    
    # Highlight start and end points
    pygame.draw.circle(path_surface, (0, 255, 0), path[0], width + 2)  # Start is green
    pygame.draw.circle(path_surface, (255, 0, 0), path[-1], width + 2)  # End is red
    
    # Blit the path surface to the screen
    screen.blit(path_surface, (0, 0))

def calculate_path_length(path):
    """
    Calculate the total length of a path
    
    Args:
        path: List of points [(x, y)] forming the path
        
    Returns:
        Total length of the path in pixels
    """
    if not path or len(path) < 2:
        return 0
    
    total_length = 0
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        total_length += math.dist(start, end)
    
    return total_length
    
    if node_idx in probabilities:
        prob = probabilities[node_idx]
        # Find rank of this node's probability
        sorted_probs = sorted(probabilities.values(), reverse=True)
        rank = sorted_probs.index(prob) + 1
        
        info.extend([
            f"Probability: {prob:.4f}",
            f"Rank: {rank} of {len(probabilities)}",
            f"Percentile: {100 * (1 - rank/len(probabilities)):.1f}%"
        ])
        
        # Add distance from evader
        node_x, node_y = node
        evader_x, evader_y = agent.state[0], agent.state[1]
        distance = math.dist((node_x, node_y), (evader_x, evader_y))
        info.append(f"Distance from evader: {distance:.1f} pixels")
    else:
        info.append("No probability data for this node")
    
    return info

def run_reachability_visualization(multicore=True, num_cores=None):
    """
    Run the reachability set visualization
    
    Args:
        multicore: Whether to use multicore processing
        num_cores: Number of cores to use (None = auto-detect)
    """
    pygame.init()
    
    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Evader Reachability Visualization")
    
    # Initialize the environment
    environment = SimulationEnvironment(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT)
    
    # Initialize font for display
    font = pygame.font.SysFont('Arial', 18)
    
    # Clock for controlling the frame rate
    clock = pygame.time.Clock()
    
    # Initialize the agent (evader)
    agent = UnicycleModel(walls=environment.get_all_walls(), doors=environment.get_doors())
    agent_color = (255, 0, 255)  # Magenta
    agent_radius = 16
    
    # Constants for controls
    AGENT_LINEAR_VEL = 50.0
    AGENT_ANGULAR_VEL = 1.0
    
    # Load or generate map graph
    print("Setting up map graph...")
    map_graph = MapGraph(
        ENVIRONMENT_WIDTH,
        ENVIRONMENT_HEIGHT,
        environment.get_all_walls(),
        environment.get_doors(),
        cache_file=MAP_GRAPH_INSPECTION_CACHE_FILE  # Use inspection-specific cache file
    )
    
    # Try to load from cache if enabled
    cache_loaded = False
    if MAP_GRAPH_CACHE_ENABLED:
        print("Attempting to load map graph from cache...")
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
            update_loading_status("Saving map graph to cache...", 0.95)
            map_graph.save_to_cache()
    
    # Variables for reachability visualization
    show_reachability = True
    auto_update_reachability = True
    selected_node_index = None
    reachability_update_timer = 0
    reachability_update_interval = 1.0  # Update every second
    node_probabilities = {}
    
    # Path visualization variables
    show_path = False
    current_path = None
    path_destination = None
    
    # Calculate initial reachability
    print("Calculating initial reachability probabilities...")
    node_probabilities = calculate_node_reachability(
        agent, 
        map_graph,
        use_multicore=multicore,
        num_cores=num_cores,
        walls=environment.get_all_walls(),
        doors=environment.get_doors()
    )
    
    # Main loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Toggle reachability display
                    show_reachability = not show_reachability
                    print(f"Reachability display: {'On' if show_reachability else 'Off'}")
                elif event.key == pygame.K_a:
                    # Toggle auto-update
                    auto_update_reachability = not auto_update_reachability
                    print(f"Auto-update: {'On' if auto_update_reachability else 'Off'}")
                elif event.key == pygame.K_u:
                    # Manual update
                    print("Manually updating reachability probabilities...")
                    node_probabilities = calculate_node_reachability(
                        agent, 
                        map_graph, 
                        use_multicore=multicore, 
                        num_cores=num_cores,
                        walls=environment.get_all_walls(),
                        doors=environment.get_doors()
                    )
                elif event.key == pygame.K_m:
                    # Toggle multicore processing
                    multicore = not multicore
                    print(f"Multicore processing: {'On' if multicore else 'Off'}")
                elif event.key == pygame.K_n:
                    # Go to next node
                    if map_graph.nodes:
                        if selected_node_index is None:
                            selected_node_index = 0
                        else:
                            selected_node_index = (selected_node_index + 1) % len(map_graph.nodes)
                        
                        # Print info about the selected node
                        node_x, node_y = map_graph.nodes[selected_node_index]
                        if selected_node_index in node_probabilities:
                            prob = node_probabilities[selected_node_index]
                            print(f"Selected node {selected_node_index} at ({int(node_x)}, {int(node_y)}) with probability {prob:.4f}")
                        else:
                            print(f"Selected node {selected_node_index} at ({int(node_x)}, {int(node_y)}) (no probability data)")
                elif event.key == pygame.K_c:
                    # Clear the current path
                    current_path = None
                    show_path = False
                    path_destination = None
                    print("Path cleared")
                elif event.key == pygame.K_p:
                    # Go to previous node
                    if map_graph.nodes:
                        if selected_node_index is None:
                            selected_node_index = len(map_graph.nodes) - 1
                        else:
                            selected_node_index = (selected_node_index - 1) % len(map_graph.nodes)
                        
                        # Print info about the selected node
                        node_x, node_y = map_graph.nodes[selected_node_index]
                        if selected_node_index in node_probabilities:
                            prob = node_probabilities[selected_node_index]
                            print(f"Selected node {selected_node_index} at ({int(node_x)}, {int(node_y)}) with probability {prob:.4f}")
                        else:
                            print(f"Selected node {selected_node_index} at ({int(node_x)}, {int(node_y)}) (no probability data)")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get mouse position
                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                # Only process clicks in the environment area
                if mouse_x < ENVIRONMENT_WIDTH:
                    if event.button == 1:  # Left mouse button
                        # Find the closest node to the mouse click
                        closest_node_index = None
                        closest_distance = float('inf')
                        click_search_radius = 20  # pixels
                        
                        for i, node in enumerate(map_graph.nodes):
                            node_x, node_y = node
                            # Quick bounds check
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
                            selected_node_index = closest_node_index
                            
                            # Print info about the selected node
                            node_x, node_y = map_graph.nodes[selected_node_index]
                            if selected_node_index in node_probabilities:
                                prob = node_probabilities[selected_node_index]
                                print(f"Selected node {selected_node_index} at ({int(node_x)}, {int(node_y)}) with probability {prob:.4f}")
                            else:
                                print(f"Selected node {selected_node_index} at ({int(node_x)}, {int(node_y)}) (no probability data)")
                    
                    elif event.button == 3:  # Right mouse button - set path destination
                        # Set the path destination and calculate path
                        path_destination = (mouse_x, mouse_y)
                        agent_pos = (agent.state[0], agent.state[1])
                        
                        print(f"Calculating path from ({int(agent_pos[0])}, {int(agent_pos[1])}) to ({int(path_destination[0])}, {int(path_destination[1])})")
                        
                        # Find the shortest path
                        current_path = find_shortest_path(map_graph, agent_pos, path_destination)
                        
                        if current_path:
                            show_path = True
                            path_length = calculate_path_length(current_path)
                            print(f"Path found! Length: {path_length:.1f} pixels, {len(current_path)} nodes")
                        else:
                            print("No path found between these points!")
        
        # Agent control with arrow keys
        keys = pygame.key.get_pressed()
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
        
        # Update agent
        agent.set_controls(linear_vel, angular_vel)
        agent.update(dt=0.1, walls=environment.get_all_walls(), doors=environment.get_doors())
        
        # Auto-update reachability
        current_time = time.time()
        if auto_update_reachability and current_time - reachability_update_timer >= reachability_update_interval:
            node_probabilities = calculate_node_reachability(
                agent, 
                map_graph,
                use_multicore=multicore,
                num_cores=num_cores,
                walls=environment.get_all_walls(),
                doors=environment.get_doors()
            )
            reachability_update_timer = current_time
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw environment
        environment.draw(screen, font)
        
        # Draw reachability probabilities
        if show_reachability and node_probabilities:
            draw_node_probabilities(screen, map_graph, node_probabilities, selected_node_index)
        
        # Draw path if enabled
        if show_path and current_path:
            draw_path(screen, current_path)
        
        # Draw agent
        x, y = agent.state[0], agent.state[1]
        theta = agent.state[2]
        pygame.draw.circle(screen, agent_color, (int(x), int(y)), agent_radius)
        # Draw direction indicator
        end_x = x + agent_radius * cos(theta)
        end_y = y + agent_radius * sin(theta)
        pygame.draw.line(screen, (255, 255, 255), (x, y), (end_x, end_y), 2)
        
        # Display info text in sidebar
        info_text = [
            "Evader Reachability Visualization",
            f"Nodes: {len(map_graph.nodes)}",
            f"Reachability nodes: {len(node_probabilities)}",
            f"Auto-update: {'On' if auto_update_reachability else 'Off'}",
            f"Processing: {'Multicore' if multicore else 'Single-core'}",
            f"Collision checking: Walls only (doors passable)",
            "",
            "Controls:",
            "  Arrow keys: Move evader",
            "  R: Toggle reachability display",
            "  A: Toggle auto-update",
            "  M: Toggle multicore processing",
            "  U: Manual update",
            "  N/P: Next/Previous node",
            "  Click: Select node",
            "  Right-click: Set path destination",
            "  C: Clear path",
            "  ESC: Quit"
        ]
        
        # Add node-specific info if a node is selected
        if selected_node_index is not None:
            node_info = get_node_probability_info(map_graph, selected_node_index, node_probabilities, agent)
            info_text.append("")
            info_text.append("Selected Node Info:")
            info_text.extend(node_info)
            
        # Add path information if a path is displayed
        if show_path and current_path:
            path_length = calculate_path_length(current_path)
            info_text.append("")
            info_text.append("Path Information:")
            info_text.append(f"Path length: {path_length:.1f} pixels")
            info_text.append(f"Nodes in path: {len(current_path)}")
            if path_destination:
                info_text.append(f"Destination: ({int(path_destination[0])}, {int(path_destination[1])})")
        
        # Draw info text
        for i, text in enumerate(info_text):
            text_surf = font.render(text, True, (255, 255, 255))
            screen.blit(text_surf, (ENVIRONMENT_WIDTH + 10, 10 + i * 20))
        
        # Update display
        pygame.display.flip()
        
        # Control framerate
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    # Default to multicore processing with auto-detection of core count
    run_reachability_visualization(multicore=True, num_cores=None)
