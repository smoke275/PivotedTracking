#!/usr/bin/env python3
"""
Visibility Utilities
Functions for calculating visibility, visibility statistics, sectors, and breakoff points.
"""

import math
import time


def calculate_evader_visibility(agent_x, agent_y, visibility_range, walls, doors, num_rays=100):
    """
    Calculate 360-degree visibility for the evader by shooting rays in all directions.
    Now automatically uses C++ optimization when available.
    
    Args:
        agent_x, agent_y: Evader position
        visibility_range: Maximum visibility distance
        walls: List of wall rectangles that block vision
        doors: List of door rectangles that allow vision through
        num_rays: Number of rays to cast (default 72 = 5-degree increments for good balance)
        
    Returns:
        List of (angle, endpoint, distance, blocked) tuples where:
        - angle: Ray angle in radians
        - endpoint: (x, y) where ray ends
        - distance: Distance from agent to endpoint
        - blocked: True if ray hit a wall, False if reached max range
        
    Ray Configuration:
        - Default: 100 rays = 3.6° angular resolution (360° / 100 = 3.6°)
        - Ray 0: 0° (East), Ray 25: 90° (North), Ray 50: 180° (West), Ray 75: 270° (South)
        - Each ray tests collision with all wall segments
        - Rays can pass through doors but are blocked by walls
        - Distance tolerance of 1 pixel to determine if ray was blocked
        
    Performance:
        - Automatically uses C++ implementation for 5-15x speedup when available
        - Falls back to Python implementation if C++ extension not built
        - Build C++ extension with: python setup.py build_ext --inplace
    """
    try:
        # Try to use the optimized implementation
        from fast_visibility_calculator import calculate_visibility_optimized
        return calculate_visibility_optimized(agent_x, agent_y, visibility_range, walls, doors, num_rays)
    except ImportError:
        # Fallback to original implementation
        from multitrack.utils.vision import cast_vision_ray
        
        visibility_data = []
        angle_step = (2 * math.pi) / num_rays
        
        for i in range(num_rays):
            angle = i * angle_step
            
            # Cast ray in this direction
            endpoint = cast_vision_ray(
                agent_x, agent_y, angle, visibility_range, walls, doors
            )
            
            # Calculate distance to endpoint
            distance = math.sqrt((endpoint[0] - agent_x)**2 + (endpoint[1] - agent_y)**2)
            
            # Check if ray was blocked (distance less than max range)
            blocked = distance < visibility_range - 1  # Small tolerance
            
            visibility_data.append((angle, endpoint, distance, blocked))
        
        return visibility_data


def get_visibility_statistics(visibility_data):
    """
    Calculate statistics from visibility data.
    
    Args:
        visibility_data: List from calculate_evader_visibility()
        
    Returns:
        Dict with visibility statistics
    """
    if not visibility_data:
        return {
            'total_rays': 0,
            'clear_rays': 0,
            'blocked_rays': 0,
            'visibility_percentage': 0.0,
            'average_visibility_distance': 0.0,
            'max_visibility_distance': 0.0,
            'min_visibility_distance': 0.0
        }
    
    total_rays = len(visibility_data)
    blocked_rays = sum(1 for _, _, _, blocked in visibility_data if blocked)
    clear_rays = total_rays - blocked_rays
    
    distances = [distance for _, _, distance, _ in visibility_data]
    
    return {
        'total_rays': total_rays,
        'clear_rays': clear_rays,
        'blocked_rays': blocked_rays,
        'visibility_percentage': (clear_rays / total_rays * 100) if total_rays > 0 else 0.0,
        'average_visibility_distance': sum(distances) / len(distances) if distances else 0.0,
        'max_visibility_distance': max(distances) if distances else 0.0,
        'min_visibility_distance': min(distances) if distances else 0.0
    }


def calculate_visibility_sectors(visibility_data, num_sectors=8):
    """
    Analyze visibility by sectors (like compass directions).
    
    Args:
        visibility_data: List from calculate_evader_visibility()
        num_sectors: Number of sectors to divide 360° into (default 8 = 45° each)
        
    Returns:
        List of sector statistics
    """
    if not visibility_data:
        return []
    
    sector_angle = 2 * math.pi / num_sectors
    sectors = [[] for _ in range(num_sectors)]
    
    # Group rays by sector
    for angle, endpoint, distance, blocked in visibility_data:
        sector_idx = int(angle / sector_angle) % num_sectors
        sectors[sector_idx].append((angle, endpoint, distance, blocked))
    
    # Calculate statistics for each sector
    sector_stats = []
    for i, sector_rays in enumerate(sectors):
        if sector_rays:
            blocked_count = sum(1 for _, _, _, blocked in sector_rays if blocked)
            total_count = len(sector_rays)
            clear_count = total_count - blocked_count
            avg_distance = sum(distance for _, _, distance, _ in sector_rays) / total_count
            
            sector_stats.append({
                'sector_index': i,
                'start_angle_deg': i * (360 / num_sectors),
                'end_angle_deg': (i + 1) * (360 / num_sectors),
                'total_rays': total_count,
                'clear_rays': clear_count,
                'blocked_rays': blocked_count,
                'visibility_percentage': (clear_count / total_count * 100) if total_count > 0 else 0.0,
                'average_distance': avg_distance
            })
        else:
            sector_stats.append({
                'sector_index': i,
                'start_angle_deg': i * (360 / num_sectors),
                'end_angle_deg': (i + 1) * (360 / num_sectors),
                'total_rays': 0,
                'clear_rays': 0,
                'blocked_rays': 0,
                'visibility_percentage': 0.0,
                'average_distance': 0.0
            })
    
    return sector_stats


def detect_visibility_breakoff_points(visibility_data, min_gap_distance=30, agent_x=None, agent_y=None, agent_theta=None):
    """
    Detect visibility breakoff points where there are abrupt changes in ray distances.
    Now includes orientation detection and distance transition classification.
    
    Args:
        visibility_data: List from calculate_evader_visibility()
        min_gap_distance: Minimum distance difference to consider a gap (default 30 pixels)
        agent_x, agent_y: Agent position (required for orientation calculation)
        agent_theta: Agent orientation in radians (required for orientation calculation)
        
    Returns:
        Tuple of (breakoff_points, breakoff_lines) where:
        - breakoff_points: List of (x, y, category) tuples where category is one of:
          'clockwise_near_far', 'clockwise_far_near', 'counterclockwise_near_far', 'counterclockwise_far_near'
        - breakoff_lines: List of (start_point, end_point, gap_size, category) tuples for connecting lines
    """
    if not visibility_data or len(visibility_data) < 2:
        return [], []
    
    breakoff_points = []
    breakoff_lines = []
    
    for i in range(len(visibility_data)):
        current_data = visibility_data[i]
        next_data = visibility_data[(i + 1) % len(visibility_data)]  # Wrap around
        
        current_distance = current_data[2]  # distance field
        next_distance = next_data[2]
        current_endpoint = current_data[1]  # endpoint field  
        next_endpoint = next_data[1]
        current_angle = current_data[0]  # angle field
        next_angle = next_data[0]
        
        # Check for significant distance change (gap) between successive rays
        distance_diff = abs(current_distance - next_distance)
        if distance_diff > min_gap_distance:
            # Determine distance transition type
            distance_transition = 'near_far' if current_distance < next_distance else 'far_near'
            
            # Determine orientation relative to agent's heading
            orientation_prefix = 'unknown'
            if agent_x is not None and agent_y is not None and agent_theta is not None:
                # Calculate the relative angle of the gap midpoint to agent's heading
                gap_midpoint_angle = (current_angle + next_angle) / 2
                
                # Handle angle wrapping for proper midpoint calculation
                angle_diff = next_angle - current_angle
                if angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                elif angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                gap_midpoint_angle = current_angle + angle_diff / 2
                
                # Normalize the gap midpoint angle
                while gap_midpoint_angle > math.pi:
                    gap_midpoint_angle -= 2 * math.pi
                while gap_midpoint_angle <= -math.pi:
                    gap_midpoint_angle += 2 * math.pi
                
                # Calculate relative angle from agent's heading
                relative_angle = gap_midpoint_angle - agent_theta
                
                # Normalize relative angle to [-π, π]
                while relative_angle > math.pi:
                    relative_angle -= 2 * math.pi
                while relative_angle <= -math.pi:
                    relative_angle += 2 * math.pi
                
                # Determine if gap is clockwise or counterclockwise relative to agent's heading
                # Positive relative angle = counterclockwise, negative = clockwise
                orientation_prefix = 'counterclockwise' if relative_angle > 0 else 'clockwise'
            
            # Combine orientation and distance transition into category
            category = f"{orientation_prefix}_{distance_transition}"
            
            # Mark both endpoints as breakoff points with category
            breakoff_points.extend([
                (current_endpoint[0], current_endpoint[1], category),
                (next_endpoint[0], next_endpoint[1], category)
            ])
            # Store line connecting the breakoff points with category
            breakoff_lines.append((current_endpoint, next_endpoint, distance_diff, category))
    
    return breakoff_points, breakoff_lines


def create_visibility_boundary_polygon(visibility_rays, agent_x, agent_y):
    """
    Create a polygon by connecting all the endpoints of visibility rays in order.
    
    Args:
        visibility_rays: List of (angle, endpoint, distance, blocked) tuples
        agent_x, agent_y: Agent position (not used in polygon, but kept for consistency)
        
    Returns:
        List of (x, y) coordinates forming the visibility boundary polygon
    """
    if not visibility_rays or len(visibility_rays) < 3:
        return []
    
    # Sort rays by angle to ensure proper polygon ordering
    sorted_rays = sorted(visibility_rays, key=lambda ray: ray[0])
    
    # Extract endpoints to form the polygon
    polygon_points = []
    for angle, endpoint, distance, blocked in sorted_rays:
        # Ensure endpoint is valid
        if endpoint and len(endpoint) >= 2:
            polygon_points.append((float(endpoint[0]), float(endpoint[1])))
    
    # Only return polygon if we have at least 3 points
    if len(polygon_points) < 3:
        return []
    
    # Note: We don't explicitly close the polygon here since pygame.draw.polygon 
    # automatically connects the last point to the first point
    
    return polygon_points


def convert_visibility_rays_to_walls(visibility_rays, agent_x, agent_y):
    """
    Convert visibility rays to simple line representations.
    Only adds walls for boundary rays of consecutive full-length ray sequences.
    
    Args:
        visibility_rays: List of (angle, endpoint, distance, blocked) tuples
        agent_x, agent_y: Agent position (ray start point)
        
    Returns:
        List of line dictionaries with 'start' and 'end' points representing ray lines
    """
    ray_walls = []
    
    if not visibility_rays:
        return ray_walls
    
    # Identify which rays to convert to walls
    rays_to_convert = []
    
    # Find consecutive sequences of full-length (unblocked) rays
    # and only add the first and last ray of each sequence
    i = 0
    while i < len(visibility_rays):
        angle, endpoint, distance, blocked = visibility_rays[i]
        
        if blocked:
            # Blocked ray - always add it
            rays_to_convert.append(i)
            i += 1
        else:
            # Start of a potential full-length sequence
            sequence_start = i
            
            # Find the end of consecutive full-length rays
            while i < len(visibility_rays) and not visibility_rays[i][3]:
                i += 1
            sequence_end = i - 1
            
            # Add first and last rays of the sequence
            if sequence_start == sequence_end:
                rays_to_convert.append(sequence_start)
            else:
                rays_to_convert.append(sequence_start)
                rays_to_convert.append(sequence_end)
    
    # Convert selected rays to simple line representations
    for ray_idx in rays_to_convert:
        angle, endpoint, distance, blocked = visibility_rays[ray_idx]
        
        # Create a simple line from agent position to ray endpoint
        start_point = (agent_x, agent_y)
        end_point = (endpoint[0], endpoint[1])
        
        # Store as a line dictionary
        ray_walls.append({
            'start': start_point,
            'end': end_point,
            'type': 'ray_line'
        })
    
    return ray_walls


def create_ray_connecting_walls(visibility_rays, breakoff_lines, agent_x, agent_y):
    """
    Create walls connecting consecutive ray endpoints that were converted to walls.
    Avoids duplicating connections that are already covered by breakoff lines.
    
    Args:
        visibility_rays: List of (angle, endpoint, distance, blocked) tuples
        breakoff_lines: List of (start_point, end_point, gap_size, category) tuples
        agent_x, agent_y: Agent position
        
    Returns:
        List of line dictionaries for connecting walls between consecutive ray endpoints
    """
    connecting_walls = []
    
    if not visibility_rays or len(visibility_rays) < 2:
        return connecting_walls
    
    # Get the rays that would be converted to walls (same logic as convert_visibility_rays_to_walls)
    rays_converted_to_walls = []
    i = 0
    while i < len(visibility_rays):
        angle, endpoint, distance, blocked = visibility_rays[i]
        
        if blocked:
            rays_converted_to_walls.append(i)
            i += 1
        else:
            # Start of a potential full-length sequence
            sequence_start = i
            
            # Find the end of consecutive full-length rays
            while i < len(visibility_rays) and not visibility_rays[i][3]:
                i += 1
            sequence_end = i - 1
            
            # Add first and last rays of the sequence
            if sequence_start == sequence_end:
                rays_converted_to_walls.append(sequence_start)
            else:
                rays_converted_to_walls.append(sequence_start)
                rays_converted_to_walls.append(sequence_end)
    
    # Create set of existing breakoff line endpoints for quick lookup
    breakoff_endpoints = set()
    for start_point, end_point, gap_size, category in breakoff_lines:
        breakoff_endpoints.add((round(start_point[0], 2), round(start_point[1], 2)))
        breakoff_endpoints.add((round(end_point[0], 2), round(end_point[1], 2)))
    
    # Connect consecutive rays that were converted to walls
    for i in range(len(rays_converted_to_walls)):
        current_ray_idx = rays_converted_to_walls[i]
        next_ray_idx = rays_converted_to_walls[(i + 1) % len(rays_converted_to_walls)]  # Wrap around
        
        current_endpoint = visibility_rays[current_ray_idx][1]  # endpoint field
        next_endpoint = visibility_rays[next_ray_idx][1]  # endpoint field
        
        # Check if this connection is already covered by a breakoff line
        current_rounded = (round(current_endpoint[0], 2), round(current_endpoint[1], 2))
        next_rounded = (round(next_endpoint[0], 2), round(next_endpoint[1], 2))
        
        # Skip if both endpoints are already in breakoff lines (would be redundant)
        if current_rounded in breakoff_endpoints and next_rounded in breakoff_endpoints:
            continue
        
        # Create connecting wall
        connecting_walls.append({
            'start': current_endpoint,
            'end': next_endpoint,
            'type': 'ray_connecting_line'
        })
    
    return connecting_walls


def create_visibility_circle_walls(agent_x, agent_y, visibility_range, wall_thickness=2, num_segments=120):
    """
    Create wall rectangles that form a circular boundary at the visibility range.
    
    Args:
        agent_x, agent_y: Agent position (center of circle)
        visibility_range: Radius of the visibility circle
        wall_thickness: Thickness of the wall segments (default 2 pixels)
        num_segments: Number of segments to approximate the circle (default 120 = 3° per segment)
        
    Returns:
        List of pygame.Rect objects representing walls along the circle perimeter
    """
    walls = []
    angle_step = (2 * math.pi) / num_segments
    
    for i in range(num_segments):
        # Calculate start and end angles for this segment
        start_angle = i * angle_step
        end_angle = (i + 1) * angle_step
        
        # Calculate segment endpoints
        start_x = agent_x + visibility_range * math.cos(start_angle)
        start_y = agent_y + visibility_range * math.sin(start_angle)
        end_x = agent_x + visibility_range * math.cos(end_angle)
        end_y = agent_y + visibility_range * math.sin(end_angle)
        
        # Calculate segment center and dimensions
        center_x = (start_x + end_x) / 2
        center_y = (start_y + end_y) / 2
        segment_length = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Create rectangle for this segment
        # Note: This is a simplified approach - in practice you might want to use
        # rotated rectangles for better accuracy
        rect_x = center_x - segment_length / 2
        rect_y = center_y - wall_thickness / 2
        
        try:
            import pygame
            wall_rect = pygame.Rect(rect_x, rect_y, segment_length, wall_thickness)
            walls.append(wall_rect)
        except ImportError:
            # Fallback representation without pygame
            walls.append({
                'x': rect_x,
                'y': rect_y,
                'width': segment_length,
                'height': wall_thickness,
                'type': 'circle_segment'
            })
    
    return walls
