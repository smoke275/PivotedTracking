"""
Vision module for agent detection and ray casting in the multitrack package.
Adapted from the original vision.py file.
"""

import math
import pygame
from multitrack.utils.config import *
from multitrack.utils.geometry import (
    get_line_intersection, normalize_angle, get_angle_difference
)

def cast_vision_ray(start_x, start_y, angle, max_distance, walls, doors):
    """
    Cast a single ray and find where it hits a wall, accounting for doors.
    
    Args:
        start_x, start_y: Starting point coordinates
        angle: The angle to cast the ray at (in radians)
        max_distance: Maximum distance the ray can travel
        walls: List of wall rectangles
        doors: List of door rectangles
        
    Returns:
        Tuple (x, y) of the point where the ray hits a wall or reaches max distance
    """
    # Calculate end point of ray at maximum distance
    end_x = start_x + math.cos(angle) * max_distance
    end_y = start_y + math.sin(angle) * max_distance
    
    # Check each wall segment for intersection
    closest_point = None
    closest_dist_squared = float('inf')
    
    # First convert all walls to line segments
    wall_segments = []
    for wall in walls:
        # Add wall segments
        wall_segments.append(((wall.x, wall.y), (wall.x + wall.width, wall.y)))  # Top
        wall_segments.append(((wall.x, wall.y), (wall.x, wall.y + wall.height)))  # Left
        wall_segments.append(((wall.x + wall.width, wall.y), (wall.x + wall.width, wall.y + wall.height)))  # Right
        wall_segments.append(((wall.x, wall.y + wall.height), (wall.x + wall.width, wall.y + wall.height)))  # Bottom
    
    # Check each segment for intersection
    for segment in wall_segments:
        p1, p2 = segment
        
        # Get intersection point with this segment
        intersection = get_line_intersection(
            start_x, start_y, end_x, end_y,
            p1[0], p1[1], p2[0], p2[1]
        )
        
        if intersection:
            int_x, int_y = intersection
            
            # Don't count if intersection is in a door
            in_door = False
            for door in doors:
                # Use a slightly expanded door rect to ensure rays pass through
                door_rect = pygame.Rect(door.x-2, door.y-2, door.width+4, door.height+4)
                if door_rect.collidepoint(int_x, int_y):
                    in_door = True
                    break
                    
            if in_door:
                continue
                
            # Calculate squared distance to this intersection
            dist_squared = (int_x - start_x)**2 + (int_y - start_y)**2
            
            # Update if this is closer than current closest
            if dist_squared < closest_dist_squared:
                closest_dist_squared = dist_squared
                closest_point = (int_x, int_y)
    
    # Return either the intersection point or the max distance point
    if closest_point:
        return closest_point
    else:
        return (end_x, end_y)

def is_agent_in_vision_cone(observer, target, vision_range, vision_angle, walls, doors):
    """
    Determine if a target agent is inside the observer's vision cone using ray casting.
    
    Args:
        observer: Observer agent with state [x, y, theta, v]
        target: Target agent with state [x, y, theta, v] and noisy_position
        vision_range: Maximum distance the observer can see
        vision_angle: Field of view angle (in radians)
        walls: List of wall rectangles
        doors: List of door rectangles
        
    Returns:
        True if target is visible to observer, False otherwise
    """
    # Extract positions
    obs_x, obs_y = observer.state[0], observer.state[1]
    obs_orientation = observer.state[2]
    
    # Use the noisy position of the target instead of the actual position
    target_x, target_y = target.noisy_position[0], target.noisy_position[1]
    target_orientation = target.noisy_position[2]
    
    # Check if target is within vision range
    distance = math.sqrt((target_x - obs_x)**2 + (target_y - obs_y)**2)
    if distance > vision_range:
        return False
        
    # Calculate angle to target
    dx = target_x - obs_x
    dy = target_y - obs_y
    angle_to_target = math.atan2(dy, dx)
    
    # Normalize angles to range [-π, π]
    obs_orientation = normalize_angle(obs_orientation)
    angle_to_target = normalize_angle(angle_to_target)
    
    # Calculate the absolute angle difference
    angle_diff = get_angle_difference(angle_to_target, obs_orientation)
    
    # Check if within vision cone angle
    if angle_diff <= vision_angle/2:
        # Define sampling points around target (treating it like a circle)
        detection_radius = AGENT_SIZE * 0.8
        num_sample_points = 8
        
        # Sample points include the center and points around the perimeter
        sample_points = [(target_x, target_y)]  # Start with center
        
        # Add points around the perimeter
        for i in range(num_sample_points):
            angle = 2 * math.pi * i / num_sample_points
            sample_x = target_x + math.cos(angle) * detection_radius
            sample_y = target_y + math.sin(angle) * detection_radius
            sample_points.append((sample_x, sample_y))
        
        # Check for direct line of sight to any sample point
        for point_x, point_y in sample_points:
            ray_angle = math.atan2(point_y - obs_y, point_x - obs_x)
            hit_point = cast_vision_ray(obs_x, obs_y, ray_angle, vision_range, walls, doors)
            
            # Calculate distances
            hit_distance = math.sqrt((hit_point[0] - obs_x)**2 + (hit_point[1] - obs_y)**2)
            sample_distance = math.sqrt((point_x - obs_x)**2 + (point_y - obs_y)**2)
            
            # If sample point is closer than wall hit, target is visible
            if sample_distance <= hit_distance:
                return True
    
    return False

def get_vision_cone_points(agent, vision_range, vision_angle, walls, doors):
    """
    Get the points that define an agent's vision cone polygon.
    
    Args:
        agent: Agent with state [x, y, theta, v]
        vision_range: Maximum vision distance
        vision_angle: Field of view angle (in radians)
        walls: List of wall rectangles
        doors: List of door rectangles
        
    Returns:
        List of points defining the vision cone polygon
    """
    center_x, center_y = agent.state[0], agent.state[1]
    orientation = agent.state[2]
    
    # Start with agent's position
    vision_points = [(center_x, center_y)]
    
    # Cast rays at different angles within the vision cone
    for i in range(NUM_VISION_RAYS + 1):
        # Calculate angle for this ray
        ray_angle = orientation - vision_angle/2 + (vision_angle * i / NUM_VISION_RAYS)
        
        # Use ray casting to find where this ray hits a wall or reaches max distance
        hit_point = cast_vision_ray(center_x, center_y, ray_angle, vision_range, walls, doors)
        vision_points.append(hit_point)
    
    return vision_points

def get_line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Calculate the intersection point of two line segments if it exists.
    
    Args:
        x1, y1, x2, y2: First line segment coordinates
        x3, y3, x4, y4: Second line segment coordinates
        
    Returns:
        (x, y) intersection point or None if lines don't intersect
    """
    # Calculate denominators
    den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    # Lines are parallel if denominator is zero
    if den == 0:
        return None
        
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
    
    # Check if intersection point is within both line segments
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        # Calculate intersection point
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)
        
    return None