#!/usr/bin/env python3
"""
Geometry Utilities
Handles geometric calculations, coordinate transformations, and spatial utilities.
"""

import math
import numpy as np


def generate_rectangular_buffer_and_convert(wall_rect):
    """
    Generate a rectangular buffer around a wall using Shapely and convert back to pygame.Rect format.
    
    Args:
        wall_rect: pygame.Rect object representing the original wall
        
    Returns:
        pygame.Rect object representing the buffered wall
    """
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        import pygame
        
        # Convert pygame.Rect to shapely Polygon
        x, y, width, height = wall_rect.x, wall_rect.y, wall_rect.width, wall_rect.height
        
        # Create polygon from rectangle coordinates
        wall_polygon = Polygon([
            (x, y),                    # Top-left
            (x + width, y),            # Top-right
            (x + width, y + height),   # Bottom-right
            (x, y + height)            # Bottom-left
        ])
        
        # Apply buffer (you can adjust the buffer distance as needed)
        buffer_distance = 2.0  # Buffer distance in pixels
        buffered_polygon = wall_polygon.buffer(buffer_distance, join_style=2)  # join_style=2 for mitered joins
        
        # Convert back to pygame.Rect
        # Get bounding box of buffered polygon
        minx, miny, maxx, maxy = buffered_polygon.bounds
        
        # Create new pygame.Rect from bounding box
        buffered_rect = pygame.Rect(
            int(minx), 
            int(miny), 
            int(maxx - minx), 
            int(maxy - miny)
        )
        
        return buffered_rect
        
    except ImportError:
        print("Warning: Shapely not available, using original wall without buffering")
        return wall_rect
    except Exception as e:
        print(f"Warning: Error applying shapely buffer: {e}, using original wall")
        return wall_rect


def point_to_line_distance(px, py, line_start, line_end):
    """
    Calculate the shortest distance from a point to a line segment.
    
    Args:
        px, py: Point coordinates
        line_start: (x1, y1) start of line segment
        line_end: (x2, y2) end of line segment
    
    Returns:
        Shortest distance from point to line segment
    """
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line start to end
    dx = x2 - x1
    dy = y2 - y1
    
    # If line segment has zero length, return distance to point
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Calculate the parameter t for the closest point on the line
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    
    # Clamp t to [0, 1] to stay within the line segment
    t = max(0, min(1, t))
    
    # Calculate the closest point on the line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Return distance from point to closest point
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)


def calculate_breakoff_line_midpoints(breakoff_lines):
    """
    Calculate midpoints of breakoff lines for exploration analysis.
    
    Args:
        breakoff_lines: List of (start_point, end_point, gap_size, category) tuples
        
    Returns:
        List of (midpoint_x, midpoint_y, gap_size, category) tuples
    """
    midpoints = []
    
    if not breakoff_lines:
        return midpoints
    
    for start_point, end_point, gap_size, category in breakoff_lines:
        # Calculate midpoint
        midpoint_x = (start_point[0] + end_point[0]) / 2.0
        midpoint_y = (start_point[1] + end_point[1]) / 2.0
        
        midpoints.append((midpoint_x, midpoint_y, gap_size, category))
    
    return midpoints


def calculate_exploration_offset_points(breakoff_midpoints, breakoff_lines, offset_distance=50.0, spatial_index=None, agent_x=None, agent_y=None):
    """
    Calculate offset points from breakoff midpoints toward unexplored areas.
    
    Args:
        breakoff_midpoints: List of (midpoint_x, midpoint_y, gap_size, category) tuples
        breakoff_lines: List of (start_point, end_point, gap_size, category) tuples  
        offset_distance: Distance to offset from midpoint (default 50.0 pixels)
        spatial_index: Spatial index for finding nearest nodes (optional)
        agent_x, agent_y: Agent position for directional calculations (optional)
        
    Returns:
        List of (offset_x, offset_y, gap_size, category, direction, nearest_node_info) tuples
    """
    exploration_points = []
    
    if not breakoff_midpoints or not breakoff_lines:
        return exploration_points
    
    for i, (midpoint_x, midpoint_y, gap_size, category) in enumerate(breakoff_midpoints):
        # Get corresponding breakoff line
        if i < len(breakoff_lines):
            start_point, end_point, _, _ = breakoff_lines[i]
            
            # Calculate perpendicular direction to the breakoff line
            line_dx = end_point[0] - start_point[0]
            line_dy = end_point[1] - start_point[1]
            
            # Calculate perpendicular vector (rotate 90 degrees)
            perp_dx = -line_dy
            perp_dy = line_dx
            
            # Normalize perpendicular vector
            perp_length = math.sqrt(perp_dx**2 + perp_dy**2)
            if perp_length > 0:
                perp_dx /= perp_length
                perp_dy /= perp_length
                
                # Calculate offset point in the perpendicular direction
                offset_x = midpoint_x + perp_dx * offset_distance
                offset_y = midpoint_y + perp_dy * offset_distance
                
                # Determine exploration direction relative to agent if position is provided
                direction = 'unknown'
                if agent_x is not None and agent_y is not None:
                    # Vector from agent to offset point
                    to_offset_dx = offset_x - agent_x
                    to_offset_dy = offset_y - agent_y
                    
                    # Calculate angle
                    angle = math.atan2(to_offset_dy, to_offset_dx)
                    angle_degrees = math.degrees(angle)
                    
                    # Categorize direction
                    if -22.5 <= angle_degrees <= 22.5:
                        direction = 'east'
                    elif 22.5 < angle_degrees <= 67.5:
                        direction = 'northeast'
                    elif 67.5 < angle_degrees <= 112.5:
                        direction = 'north'
                    elif 112.5 < angle_degrees <= 157.5:
                        direction = 'northwest'
                    elif 157.5 < angle_degrees <= 180 or -180 <= angle_degrees <= -157.5:
                        direction = 'west'
                    elif -157.5 < angle_degrees <= -112.5:
                        direction = 'southwest'
                    elif -112.5 < angle_degrees <= -67.5:
                        direction = 'south'
                    elif -67.5 < angle_degrees <= -22.5:
                        direction = 'southeast'
                
                # Find nearest node if spatial index is available
                nearest_node_info = None
                if spatial_index is not None:
                    try:
                        map_graph = spatial_index.map_graph
                        if hasattr(map_graph, 'nodes') and map_graph.nodes:
                            # Find nearest node to offset point
                            min_distance = float('inf')
                            nearest_node_index = None
                            
                            for node_idx, node_pos in enumerate(map_graph.nodes):
                                node_distance = math.sqrt(
                                    (offset_x - node_pos[0])**2 + (offset_y - node_pos[1])**2
                                )
                                if node_distance < min_distance:
                                    min_distance = node_distance
                                    nearest_node_index = node_idx
                            
                            if nearest_node_index is not None:
                                nearest_node_info = (nearest_node_index, min_distance)
                    except Exception as e:
                        print(f"Warning: Could not find nearest node: {e}")
                
                exploration_points.append((
                    offset_x, offset_y, gap_size, category, direction, nearest_node_info
                ))
    
    return exploration_points


def convert_rectangles_to_lines(rectangles):
    """
    Convert rectangle objects to line segments (4 lines per rectangle).
    
    Args:
        rectangles: List of rectangle objects (pygame.Rect or similar with x, y, width, height)
        
    Returns:
        List of ((x1, y1), (x2, y2)) line tuples
    """
    lines = []
    
    for rect in rectangles:
        # Extract rectangle bounds
        if hasattr(rect, 'x'):  # pygame.Rect style
            x, y, width, height = rect.x, rect.y, rect.width, rect.height
        elif len(rect) >= 4:  # Tuple/list style
            x, y, width, height = rect[:4]
        else:
            continue  # Skip invalid rectangles
        
        # Calculate corner points
        top_left = (x, y)
        top_right = (x + width, y)
        bottom_right = (x + width, y + height)
        bottom_left = (x, y + height)
        
        # Add four sides as lines
        lines.extend([
            (top_left, top_right),      # Top side
            (top_right, bottom_right),  # Right side
            (bottom_right, bottom_left), # Bottom side
            (bottom_left, top_left)     # Left side
        ])
    
    return lines


def vectorized_point_in_polygon(x, y, polygon_points):
    """
    Vectorized point-in-polygon test using ray casting algorithm.
    
    Args:
        x, y: numpy arrays of point coordinates to test
        polygon_points: List of (x, y) tuples defining the polygon vertices
        
    Returns:
        numpy boolean array indicating which points are inside the polygon
    """
    if len(polygon_points) < 3:
        return np.zeros(len(x), dtype=bool)
    
    try:
        # Convert polygon to numpy array
        poly = np.array(polygon_points, dtype=np.float32)
        n_vertices = len(poly)
        
        # Ensure polygon is closed
        if not np.allclose(poly[0], poly[-1]):
            poly = np.vstack([poly, poly[0]])
            n_vertices += 1
        
        # Initialize result array
        inside = np.zeros(len(x), dtype=bool)
        
        # Ray casting algorithm - vectorized
        for i in range(n_vertices - 1):
            x1, y1 = poly[i]
            x2, y2 = poly[i + 1]
            
            # Check if ray crosses edge
            cond1 = (y1 > y) != (y2 > y)
            
            # Calculate intersection x-coordinate
            # Only compute where cond1 is True to avoid division by zero
            where_cond1 = np.where(cond1)
            if len(where_cond1[0]) > 0:
                y_subset = y[where_cond1]
                x_subset = x[where_cond1]
                
                # Avoid division by zero
                y_diff = y2 - y1
                if abs(y_diff) > 1e-10:  # Not horizontal line
                    x_intersect = x1 + (y_subset - y1) * (x2 - x1) / y_diff
                    cond2 = x_subset < x_intersect
                    inside[where_cond1] ^= cond2  # XOR to toggle
        
        return inside
        
    except Exception as e:
        print(f"Warning: Error in vectorized point-in-polygon test: {e}")
        # Fallback: return all True (no polygon constraint)
        return np.ones(len(x), dtype=bool)
