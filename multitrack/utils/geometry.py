"""
Geometry utilities for the multitrack package.
Contains functions for line intersections and angle calculations.
"""

import math

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

def normalize_angle(angle):
    """
    Normalize an angle to the range [-π, π]
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def get_angle_difference(angle1, angle2):
    """
    Get the smallest difference between two angles (handles wrap-around)
    """
    angle_diff = abs(angle1 - angle2)
    if angle_diff > math.pi:
        angle_diff = 2 * math.pi - angle_diff
    return angle_diff