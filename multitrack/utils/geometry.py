"""
Geometry utilities for the multitrack package.
Contains functions for line intersections and angle calculations.
"""

import math
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

def point_segment_distance(px, py, x1, y1, x2, y2):
    """
    Calculate the minimum distance from a point (px, py) to a line segment (x1, y1) to (x2, y2)
    
    Args:
        px, py: Point coordinates
        x1, y1, x2, y2: Line segment coordinates
        
    Returns:
        Minimum distance from the point to the line segment
    """
    # Vector from line segment start to end
    dx = x2 - x1
    dy = y2 - y1
    
    # If segment is a point, return distance to that point
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Calculate projection proportion along line segment
    t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / (dx**2 + dy**2)))
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Return distance to closest point
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

def point_segment_distance_batch(points, segments):
    """
    Calculate distances from multiple points to multiple line segments using vectorized operations
    
    Args:
        points: Array of shape (n_points, 2) containing point coordinates
        segments: Array of shape (n_segments, 4) with each row containing [x1, y1, x2, y2]
        
    Returns:
        Array of shape (n_points, n_segments) with distances from each point to each segment
    """
    n_points = points.shape[0]
    n_segments = segments.shape[0]
    
    # Use numpy for CPU implementation
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    if not isinstance(segments, np.ndarray):
        segments = np.array(segments)
    
    # Extract points and segment endpoints
    px = points[:, 0].reshape(-1, 1)  # Shape: (n_points, 1)
    py = points[:, 1].reshape(-1, 1)  # Shape: (n_points, 1)
    
    x1 = segments[:, 0].reshape(1, -1)  # Shape: (1, n_segments)
    y1 = segments[:, 1].reshape(1, -1)  # Shape: (1, n_segments)
    x2 = segments[:, 2].reshape(1, -1)  # Shape: (1, n_segments)
    y2 = segments[:, 3].reshape(1, -1)  # Shape: (1, n_segments)
    
    # Vectors from segment start to end
    dx = x2 - x1  # Shape: (1, n_segments)
    dy = y2 - y1  # Shape: (1, n_segments)
    
    # Handle point segments (zero length)
    is_point = (dx == 0) & (dy == 0)
    
    # For non-zero segments, compute projection proportion
    segment_len_sq = dx**2 + dy**2  # Shape: (1, n_segments)
    
    # Calculate the projection proportion (t parameter)
    # Vectorized across all points and all segments
    dot_product = (px - x1) * dx + (py - y1) * dy  # Shape: (n_points, n_segments)
    t = np.divide(dot_product, segment_len_sq, out=np.zeros_like(dot_product), where=segment_len_sq != 0)
    
    # Clamp t to [0, 1] for valid projection onto line segment
    t = np.clip(t, 0, 1)
    
    # Compute closest points on segments
    closest_x = x1 + t * dx  # Shape: (n_points, n_segments)
    closest_y = y1 + t * dy  # Shape: (n_points, n_segments)
    
    # Calculate squared distances
    dist_sq = (px - closest_x)**2 + (py - closest_y)**2  # Shape: (n_points, n_segments)
    
    # For point segments, use direct distance to the point
    point_dist_sq = (px - x1)**2 + (py - y1)**2  # Shape: (n_points, n_segments)
    
    # Select appropriate distance based on whether segment is a point
    dist_sq = np.where(is_point, point_dist_sq, dist_sq)
    
    # Return the square root of distances
    return np.sqrt(dist_sq)

def point_segment_distance_batch_torch(points, segments, device):
    """
    Calculate distances from multiple points to multiple line segments using GPU
    
    Args:
        points: Tensor of shape (n_points, 2) containing point coordinates
        segments: Tensor of shape (n_segments, 4) with each row containing [x1, y1, x2, y2]
        device: The torch device to use
        
    Returns:
        Tensor of shape (n_points, n_segments) with distances from each point to each segment
    """
    if not TORCH_AVAILABLE:
        raise ImportError("Torch implementation called but torch is not available")
    
    # Convert to torch tensors if not already
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32, device=device)
    if not isinstance(segments, torch.Tensor):
        segments = torch.tensor(segments, dtype=torch.float32, device=device)
    
    # Extract points and segment endpoints
    px = points[:, 0].reshape(-1, 1)  # Shape: (n_points, 1)
    py = points[:, 1].reshape(-1, 1)  # Shape: (n_points, 1)
    
    x1 = segments[:, 0].reshape(1, -1)  # Shape: (1, n_segments)
    y1 = segments[:, 1].reshape(1, -1)  # Shape: (1, n_segments)
    x2 = segments[:, 2].reshape(1, -1)  # Shape: (1, n_segments)
    y2 = segments[:, 3].reshape(1, -1)  # Shape: (1, n_segments)
    
    # Vectors from segment start to end
    dx = x2 - x1  # Shape: (1, n_segments)
    dy = y2 - y1  # Shape: (1, n_segments)
    
    # Handle point segments (zero length)
    is_point = (dx == 0) & (dy == 0)
    
    # For non-zero segments, compute projection proportion
    segment_len_sq = dx**2 + dy**2  # Shape: (1, n_segments)
    
    # Calculate the projection proportion (t parameter)
    # Vectorized across all points and all segments
    dot_product = (px - x1) * dx + (py - y1) * dy  # Shape: (n_points, n_segments)
    
    # Safely divide to handle zeros
    t = torch.zeros_like(dot_product)
    mask = segment_len_sq != 0
    t = torch.where(mask, dot_product / (segment_len_sq + 1e-10), t)
    
    # Clamp t to [0, 1] for valid projection onto line segment
    t = torch.clamp(t, 0, 1)
    
    # Compute closest points on segments
    closest_x = x1 + t * dx  # Shape: (n_points, n_segments)
    closest_y = y1 + t * dy  # Shape: (n_points, n_segments)
    
    # Calculate squared distances
    dist_sq = (px - closest_x)**2 + (py - closest_y)**2  # Shape: (n_points, n_segments)
    
    # For point segments, use direct distance to the point
    point_dist_sq = (px - x1)**2 + (py - y1)**2  # Shape: (n_points, n_segments)
    
    # Select appropriate distance based on whether segment is a point
    dist_sq = torch.where(is_point, point_dist_sq, dist_sq)
    
    # Return the square root of distances
    return torch.sqrt(dist_sq)