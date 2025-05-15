#!/usr/bin/env python3
"""
Pathfinding utilities for the PivotedTracking project.
Implements algorithms for finding paths through the map graph.
"""

import heapq
import math
import random
import numpy as np

def find_shortest_path(map_graph, start_pos, end_pos):
    """
    Find the shortest path from start_pos to end_pos using A* algorithm
    
    Args:
        map_graph: The map graph with nodes and edges
        start_pos: Starting position (x, y) coordinates
        end_pos: Target position (x, y) coordinates
        
    Returns:
        A list of positions [(x, y)] forming the path from start to end,
        or None if no path is found
    """
    # Use the PRM approach for faster wall-respecting pathfinding
    return find_prm_path(map_graph, start_pos, end_pos)

def find_fast_path(map_graph, start_pos, end_pos):
    """
    Find a fast, reasonable path from start_pos to end_pos
    This uses a greedy best-first search approach that is faster than A*
    but may not always find the optimal (shortest) path.
    
    Args:
        map_graph: The map graph with nodes and edges
        start_pos: Starting position (x, y) coordinates
        end_pos: Target position (x, y) coordinates
        
    Returns:
        A list of positions [(x, y)] forming the path from start to end,
        or None if no path is found
    """
    # Find the closest nodes to start and end positions
    start_node_idx = find_closest_node(map_graph.nodes, start_pos)
    end_node_idx = find_closest_node(map_graph.nodes, end_pos)
    
    if start_node_idx is None or end_node_idx is None:
        return None
    
    # If start and end are the same node
    if start_node_idx == end_node_idx:
        return [map_graph.nodes[start_node_idx]]
        
    # Direct path optimization: if the nodes are very close, just return a direct line
    if distance(map_graph.nodes[start_node_idx], map_graph.nodes[end_node_idx]) < 200:
        return [map_graph.nodes[start_node_idx], map_graph.nodes[end_node_idx]]
    
    # Create adjacency list representation of the graph (faster construction)
    graph = {}
    for edge in map_graph.edges:
        if edge[0] not in graph:
            graph[edge[0]] = []
        if edge[1] not in graph:
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
    
    # Use a greedy best-first search (faster than A*)
    visited = set()
    priority_queue = [(heuristic(map_graph.nodes[start_node_idx], map_graph.nodes[end_node_idx]), start_node_idx)]
    came_from = {}
    
    # Early termination counters
    max_iterations = min(500, len(map_graph.nodes) // 2)  # Limit search iterations
    iterations = 0
    
    while priority_queue and iterations < max_iterations:
        iterations += 1
        _, current = heapq.heappop(priority_queue)
        
        if current == end_node_idx:
            # Path found, reconstruct and return it
            path = reconstruct_path(came_from, current, map_graph.nodes)
            return path
        
        if current in visited:
            continue
            
        visited.add(current)
        
        # No neighbors in extreme case
        if current not in graph:
            continue
            
        # Process neighbors
        for neighbor in graph[current]:
            if neighbor in visited:
                continue
                
            # Only update if this is a new path or a better one
            if neighbor not in came_from:
                came_from[neighbor] = current
                # Priority is just the heuristic distance to goal (greedy)
                priority = heuristic(map_graph.nodes[neighbor], map_graph.nodes[end_node_idx])
                heapq.heappush(priority_queue, (priority, neighbor))
    
    # Fall back to the optimal path finder (which respects walls)
    return find_optimal_path(map_graph, start_pos, end_pos)

def find_prm_path(map_graph, start_pos, end_pos):
    """
    Find a path using the existing map graph as a probabilistic roadmap
    This method leverages the pre-calculated graph to find wall-respecting paths quickly
    
    Args:
        map_graph: The map graph with nodes and edges
        start_pos: Starting position (x, y) coordinates
        end_pos: Target position (x, y) coordinates
        
    Returns:
        A list of positions [(x, y)] forming the path from start to end,
        or None if no path is found
    """
    # Find the closest nodes to start and end positions
    start_node_idx = find_closest_node(map_graph.nodes, start_pos)
    end_node_idx = find_closest_node(map_graph.nodes, end_pos)
    
    if start_node_idx is None or end_node_idx is None:
        return None
    
    # If start and end are the same node
    if start_node_idx == end_node_idx:
        return [map_graph.nodes[start_node_idx]]
    
    # Build graph representation (only accessible areas - through existing edges)
    graph = {}
    for edge in map_graph.edges:
        if edge[0] not in graph:
            graph[edge[0]] = []
        if edge[1] not in graph:
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
    
    # Use a bidirectional search for faster results
    # Forward search from start
    forward_visited = {start_node_idx: None}  # node -> parent
    forward_queue = [start_node_idx]
    
    # Backward search from end
    backward_visited = {end_node_idx: None}  # node -> parent
    backward_queue = [end_node_idx]
    
    # Common node that connects both searches (if found)
    meeting_node = None
    
    # Maximum iterations to prevent infinite loops
    max_iterations = min(1000, len(map_graph.nodes))
    iterations = 0
    
    # Search until both queues are empty or meeting point is found
    while forward_queue and backward_queue and iterations < max_iterations and meeting_node is None:
        iterations += 1
        
        # Expand forward search
        if forward_queue:
            current = forward_queue.pop(0)
            
            # Check if we've reached a node in the backward search
            if current in backward_visited:
                meeting_node = current
                break
            
            # Otherwise, expand neighbors
            if current in graph:
                for neighbor in graph[current]:
                    if neighbor not in forward_visited:
                        forward_visited[neighbor] = current
                        forward_queue.append(neighbor)
        
        # Expand backward search
        if backward_queue:
            current = backward_queue.pop(0)
            
            # Check if we've reached a node in the forward search
            if current in forward_visited:
                meeting_node = current
                break
            
            # Otherwise, expand neighbors
            if current in graph:
                for neighbor in graph[current]:
                    if neighbor not in backward_visited:
                        backward_visited[neighbor] = current
                        backward_queue.append(neighbor)
    
    # If a meeting point was found, reconstruct the path
    if meeting_node is not None:
        # Build path from start to meeting node
        forward_path = []
        current = meeting_node
        while current is not None:
            forward_path.append(map_graph.nodes[current])
            current = forward_visited[current]
        forward_path.reverse()
        
        # Build path from meeting node to end
        backward_path = []
        current = meeting_node
        while current is not None:
            backward_path.append(map_graph.nodes[current])
            current = backward_visited[current]
        
        # Skip first node in backward path as it's the meeting node
        combined_path = forward_path + backward_path[1:]
        
        # Add actual start and end positions to make path more accurate
        if distance(combined_path[0], start_pos) > 5:
            combined_path.insert(0, start_pos)
        if distance(combined_path[-1], end_pos) > 5:
            combined_path.append(end_pos)
        
        return combined_path
    
    # If no path found, try the optimal path finder
    return find_optimal_path(map_graph, start_pos, end_pos)
    
def find_optimal_path(map_graph, start_pos, end_pos):
    """
    Find the optimal (shortest) path using A* algorithm (original implementation)
    """
    # Find the closest nodes to start and end positions
    start_node_idx = find_closest_node(map_graph.nodes, start_pos)
    end_node_idx = find_closest_node(map_graph.nodes, end_pos)
    
    if start_node_idx is None or end_node_idx is None:
        return None
    
    # If start and end are the same node
    if start_node_idx == end_node_idx:
        return [map_graph.nodes[start_node_idx]]
    
    # Create adjacency list representation of the graph
    graph = {}
    for i, node in enumerate(map_graph.nodes):
        neighbors = []
        for edge in map_graph.edges:
            if edge[0] == i:
                neighbors.append(edge[1])
            elif edge[1] == i:
                neighbors.append(edge[0])
        graph[i] = neighbors
    
    # A* algorithm implementation
    open_set = []  # Priority queue of nodes to explore
    heapq.heappush(open_set, (0, start_node_idx))  # (priority, node_idx)
    
    # Dictionary mapping node to its parent in the path
    came_from = {}
    
    # Cost from start to node
    g_score = {i: float('inf') for i in range(len(map_graph.nodes))}
    g_score[start_node_idx] = 0
    
    # Estimated total cost from start to goal through node
    f_score = {i: float('inf') for i in range(len(map_graph.nodes))}
    f_score[start_node_idx] = heuristic(map_graph.nodes[start_node_idx], map_graph.nodes[end_node_idx])
    
    # Set of nodes already evaluated
    closed_set = set()
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == end_node_idx:
            # Path found, reconstruct and return it
            path = reconstruct_path(came_from, current, map_graph.nodes)
            return path
        
        closed_set.add(current)
        
        for neighbor in graph[current]:
            if neighbor in closed_set:
                continue
            
            # Calculate tentative g_score
            tentative_g_score = g_score[current] + distance(map_graph.nodes[current], map_graph.nodes[neighbor])
            
            if tentative_g_score < g_score[neighbor]:
                # This path is better than any previous one
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(map_graph.nodes[neighbor], map_graph.nodes[end_node_idx])
                
                # Add to open set if not already there
                if all(item[1] != neighbor for item in open_set):
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # No path found
    return None

def find_closest_node(nodes, position):
    """Find the index of the node closest to the given position"""
    if not nodes:
        return None
        
    closest_idx = None
    closest_distance = float('inf')
    
    for i, node in enumerate(nodes):
        dist = distance(node, position)
        if dist < closest_distance:
            closest_distance = dist
            closest_idx = i
    
    return closest_idx

def distance(pos1, pos2):
    """Calculate Euclidean distance between two positions"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def heuristic(pos1, pos2):
    """Heuristic function for A* (Euclidean distance)"""
    return distance(pos1, pos2)

def reconstruct_path(came_from, current, nodes):
    """Reconstruct the path from the came_from dictionary"""
    path = [nodes[current]]
    while current in came_from:
        current = came_from[current]
        path.append(nodes[current])
    
    return path[::-1]  # Reverse to get path from start to end

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
        total_length += math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    
    return total_length

def smooth_path(path, max_points=10):
    """
    Smooth a path by reducing the number of points while maintaining the general shape
    
    Args:
        path: Original path as list of points [(x, y)]
        max_points: Maximum number of points to keep in the smoothed path
        
    Returns:
        Smoothed path with fewer points
    """
    if not path or len(path) <= 2:
        return path
    
    # Always keep start and end points
    if len(path) <= max_points:
        return path
    
    # Calculate how many points to skip
    skip = max(1, len(path) // max_points)
    
    # Create smoothed path
    smoothed = [path[0]]  # Always include start
    
    for i in range(skip, len(path) - 1, skip):
        smoothed.append(path[i])
    
    smoothed.append(path[-1])  # Always include end
    
    return smoothed

def create_dynamically_feasible_path(path, agent_state, max_speed=100, max_angular_vel=2.0, dt=0.1, sim_steps=500):
    """
    Create a dynamically feasible path that the unicycle agent can actually follow
    
    Args:
        path: Original path as list of points [(x, y)]
        agent_state: Current agent state [x, y, theta, v]
        max_speed: Maximum linear speed of the agent
        max_angular_vel: Maximum angular velocity of the agent
        dt: Time step for simulation
        sim_steps: Maximum simulation steps
        
    Returns:
        A feasible trajectory [(x, y)] that respects the agent's dynamics
    """
    if not path or len(path) < 2:
        return path
    
    # Initialize simulated agent state: [x, y, theta, v]
    sim_state = np.array([agent_state[0], agent_state[1], agent_state[2], 0.0])
    
    # Waypoints from the original path
    waypoints = path
    
    # Initialize result trajectory with the initial position
    trajectory = [(sim_state[0], sim_state[1])]
    
    # Simple controller gains
    k_linear = 1.0  # Proportional gain for linear velocity
    k_angular = 2.0  # Proportional gain for angular velocity
    
    # Target threshold (how close we need to get to consider a waypoint reached)
    waypoint_threshold = 20.0
    
    current_waypoint_idx = 0
    
    # Simulate the agent following the waypoints
    for _ in range(sim_steps):
        if current_waypoint_idx >= len(waypoints):
            break
            
        # Current target waypoint
        target = waypoints[current_waypoint_idx]
        
        # Calculate distance and angle to target
        dx = target[0] - sim_state[0]
        dy = target[1] - sim_state[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if we've reached the current waypoint
        if distance < waypoint_threshold:
            current_waypoint_idx += 1
            if current_waypoint_idx >= len(waypoints):
                break
            continue
            
        # Calculate desired heading angle to target
        desired_theta = math.atan2(dy, dx)
        
        # Calculate angle difference (accounting for angle wrapping)
        angle_diff = math.atan2(math.sin(desired_theta - sim_state[2]), 
                               math.cos(desired_theta - sim_state[2]))
        
        # Simple proportional controller
        linear_vel = min(k_linear * distance, max_speed)
        angular_vel = min(max(k_angular * angle_diff, -max_angular_vel), max_angular_vel)
        
        # Update the simulated agent state using unicycle dynamics
        sim_state[0] += linear_vel * math.cos(sim_state[2]) * dt
        sim_state[1] += linear_vel * math.sin(sim_state[2]) * dt
        sim_state[2] += angular_vel * dt
        sim_state[3] = linear_vel  # Update velocity
        
        # Add the new point to the trajectory
        trajectory.append((sim_state[0], sim_state[1]))
    
    return trajectory
