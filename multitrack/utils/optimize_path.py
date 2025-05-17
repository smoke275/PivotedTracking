#!/usr/bin/env python3
"""
Path optimization utilities for using visibility data to create more efficient paths.
"""

import math
from multitrack.utils.pathfinding import find_closest_node

def optimize_path_with_visibility(path, visibility_map, map_graph=None):
    """
    Optimize a path using visibility data by creating shortcuts between nodes that can see each other.
    
    Args:
        path: The original path as a list of points [(x, y)]
        visibility_map: Dictionary mapping node indices to lists of visible node indices
        map_graph: The map graph containing nodes to look up in visibility data
        
    Returns:
        The optimized path as a list of points [(x, y)]
    """
    # If path is too short or no visibility data or no map_graph, no optimization possible
    if not path or len(path) < 3 or not visibility_map or not map_graph:
        return path
    
    # Create a copy of the path to avoid modifying the original
    optimized_path = [path[0]]
    
    # Current position in the optimization (start at the beginning)
    current_idx = 0
    
    while current_idx < len(path) - 1:
        # Start looking from the furthest possible node
        found_visible_node = False
        current_pos = path[current_idx]
        
        # Get the map_graph node index for the current position
        current_node_idx = find_closest_node(map_graph.nodes, current_pos)
        
        if current_node_idx is None:
            # If we can't find the current node in the map graph, just move to next point
            current_idx += 1
            if current_idx < len(path):
                optimized_path.append(path[current_idx])
            continue
            
        # Look ahead as far as possible for visible nodes
        for look_ahead_idx in range(len(path) - 1, current_idx, -1):
            # Skip adjacent nodes (they're always "visible" as they're directly connected)
            if look_ahead_idx <= current_idx + 1:
                continue
                
            # Get the position of the node we're looking ahead to
            look_ahead_pos = path[look_ahead_idx]
            
            # Find the corresponding node index in the map_graph
            look_ahead_node_idx = find_closest_node(map_graph.nodes, look_ahead_pos)
            
            if look_ahead_node_idx is None:
                continue  # Skip if we can't find this node in the map graph
                
            # Check if the nodes can see each other according to visibility data
            if current_node_idx in visibility_map and look_ahead_node_idx in visibility_map[current_node_idx]:
                # Found a visible node further ahead, add it to the path and skip to it
                optimized_path.append(path[look_ahead_idx])
                current_idx = look_ahead_idx
                found_visible_node = True
                break
        
        # If no visible node found ahead, just move to the next node in the original path
        if not found_visible_node:
            current_idx += 1
            if current_idx < len(path):
                optimized_path.append(path[current_idx])
    
    return optimized_path
