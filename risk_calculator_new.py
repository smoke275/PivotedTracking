#!/usr/bin/env python3
"""
Risk Calculator
Handles reachability calculations, mask transformations, and visibility calculations.
"""

import pickle
import numpy as np
import math

# Import environment constants for world boundary checking
try:
    from simulation_config import ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT
except ImportError:
    # Fallback values if simulation_config is not available
    ENVIRONMENT_WIDTH = 1280
    ENVIRONMENT_HEIGHT = 720

# Import polygon exploration functionality
from polygon_exploration import calculate_polygon_exploration_paths

# Global cache for reusable arrays to avoid allocation overhead
_ARRAY_CACHE = {}

def _get_cached_array(key, shape, dtype):
    """Get a cached array or create a new one if needed."""
    cache_key = (key, shape, dtype.name)
    if cache_key not in _ARRAY_CACHE:
        _ARRAY_CACHE[cache_key] = np.empty(shape, dtype=dtype)
    return _ARRAY_CACHE[cache_key]

def load_reachability_mask(filename_base="unicycle_grid"):
    """Load reachability mask from pickle file."""
    try:
        with open(f"{filename_base}.pkl", 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"Warning: {filename_base}.pkl not found. Run heatmap.py first to generate reachability mask.")
        return None
    except Exception as e:
        print(f"Error loading reachability mask: {e}")
        return None

def world_to_grid_coords(world_x, world_y, mask_data):
    """Convert world coordinates to grid indices."""
    center = mask_data['center_idx']
    cell_size = mask_data['cell_size_px']
    
    grid_col = int(world_x / cell_size + center)
    grid_row = int(center - world_y / cell_size)
    
    return grid_row, grid_col

def get_reachability_probabilities_for_fixed_grid(agent_x, agent_y, agent_theta, mask_data):
    """Calculate reachability probabilities for a fixed grid centered on agent (ultra-optimized)."""
    grid_size = mask_data['grid_size']
    cell_size = mask_data['cell_size_px']
    original_center = mask_data['center_idx']
    original_grid = mask_data['grid']
    
    # Pre-compute constants and use float32 for better cache performance
    center_f = float(grid_size // 2)
    cos_theta = np.float32(np.cos(agent_theta))
    sin_theta = np.float32(np.sin(agent_theta))
    inv_cell_size = np.float32(1.0 / cell_size)
    original_center_f = np.float32(original_center)
    
    # Create optimized coordinate arrays
    coords = np.arange(grid_size, dtype=np.float32)
    j_coords = coords - center_f
    i_coords = center_f - coords
    
    # Use outer operations for memory-efficient broadcasting
    # This avoids creating large intermediate arrays
    j_scaled = j_coords * cell_size
    i_scaled = i_coords * cell_size
    
    # Compute rotation using efficient numpy operations
    # Create the rotated coordinate grids using outer product pattern
    j_cos = j_scaled * cos_theta  # 1D array
    j_sin = j_scaled * sin_theta  # 1D array
    i_cos = i_scaled * cos_theta  # 1D array  
    i_sin = i_scaled * sin_theta  # 1D array
    
    # Use broadcasting to create final coordinate arrays
    orig_rel_x = j_cos[np.newaxis, :] - i_sin[:, np.newaxis]  # Broadcasting
    orig_rel_y = j_sin[np.newaxis, :] + i_cos[:, np.newaxis]  # Broadcasting
    
    # Optimized index conversion with combined operations
    orig_col = np.round(orig_rel_x * inv_cell_size + original_center_f).astype(np.int32)
    orig_row = np.round(original_center_f - orig_rel_y * inv_cell_size).astype(np.int32)
    
    # Efficient bounds checking using numpy's logical operations
    valid_mask = ((orig_row >= 0) & (orig_row < grid_size) & 
                 (orig_col >= 0) & (orig_col < grid_size))
    
    # Initialize result grid with correct dtype
    result_grid = np.zeros((grid_size, grid_size), dtype=original_grid.dtype)
    
    # Use the most efficient indexing pattern
    if np.any(valid_mask):
        # Extract valid indices in one operation
        valid_indices = np.where(valid_mask)
        valid_orig_rows = orig_row[valid_indices]
        valid_orig_cols = orig_col[valid_indices]
        
        # Single assignment operation
        result_grid[valid_indices] = original_grid[valid_orig_rows, valid_orig_cols]
    
    return result_grid

def get_reachability_at_position(agent_x, agent_y, agent_theta, mask_data):
    """Get reachability probability at agent's current position."""
    fixed_grid_probs = get_reachability_probabilities_for_fixed_grid(agent_x, agent_y, agent_theta, mask_data)
    center_idx = mask_data['center_idx']
    return fixed_grid_probs[center_idx, center_idx]

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


class EvaderAnalysis:
    """
    Unified container for all evader analysis data.
    This makes it easy to add new analysis types as attributes.
    """
    def __init__(self):
        # Core analysis data
        self.reachability_grid = None
        self.reachability_at_position = None
        self.visibility_rays = None
        self.visibility_statistics = None
        self.visibility_sectors = None
        self.breakoff_points = None
        self.breakoff_lines = None
        self.breakoff_midpoints = None  # Midpoints of breakoff lines
        self.exploration_offset_points = None  # Offset points toward unexplored areas
        self.exploration_nearest_nodes = None  # Nearest map graph nodes to exploration points
        self.polygon_exploration_paths = None  # Polygon exploration paths for breakpoints
        
        # Environment clipping data for performance optimization
        self.clipped_walls = None  # Walls within visibility range
        self.clipped_doors = None  # Doors within visibility range
        self.visibility_bounding_box = None  # (min_x, min_y, max_x, max_y)
        self.clipping_statistics = None  # Stats about clipping efficiency
        self.visibility_circle = None  # Visibility circle as simple circle parameters
        
        # Future extensibility - easily add new analysis types
        self.rods = None  # For future rod analysis
        self.threat_assessment = None  # For future threat analysis
        self.escape_routes = None  # For future escape route analysis
        self.pursuit_advantage = None  # For future pursuit analysis
        
        # Metadata
        self.agent_position = None
        self.agent_orientation = None
        self.visibility_range = None
        self.analysis_timestamp = None
        self.spatial_index = None  # Spatial index for fast spatial queries

def clip_environment_to_visibility_range(agent_x, agent_y, visibility_range, walls, doors):
    """
    Clip walls and doors to only include those within or intersecting the visibility circle.
    This significantly improves performance by reducing the number of collision checks.
    
    Args:
        agent_x, agent_y: Evader position (center of visibility circle)
        visibility_range: Visibility radius
        walls: List of wall rectangles
        doors: List of door rectangles
        
    Returns:
        Tuple of (clipped_walls, clipped_doors, bounding_box, clipping_stats)
        - clipped_walls: Walls within/intersecting visibility circle
        - clipped_doors: Doors within/intersecting visibility circle  
        - bounding_box: (min_x, min_y, max_x, max_y) of visibility circle
        - clipping_stats: Dictionary with clipping efficiency statistics
    """
    # Calculate visibility circle bounding box
    min_x = agent_x - visibility_range
    max_x = agent_x + visibility_range
    min_y = agent_y - visibility_range
    max_y = agent_y + visibility_range
    bounding_box = (min_x, min_y, max_x, max_y)
    
    # Helper function to check if rectangle intersects with visibility circle
    def rectangle_intersects_circle(rect, center_x, center_y, radius):
        """Check if a rectangle intersects with a circle."""
        # Rectangle bounds
        rect_left, rect_top, rect_width, rect_height = rect
        rect_right = rect_left + rect_width
        rect_bottom = rect_top + rect_height
        
        # Find closest point on rectangle to circle center
        closest_x = max(rect_left, min(center_x, rect_right))
        closest_y = max(rect_top, min(center_y, rect_bottom))
        
        # Calculate distance from circle center to closest point
        distance_sq = (closest_x - center_x) ** 2 + (closest_y - center_y) ** 2
        
        # Check if distance is within radius
        return distance_sq <= radius ** 2
    
    # Clip walls to visibility range
    clipped_walls = []
    for wall in walls:
        if rectangle_intersects_circle(wall, agent_x, agent_y, visibility_range):
            clipped_walls.append(wall)
    
    # Clip doors to visibility range
    clipped_doors = []
    for door in doors:
        if rectangle_intersects_circle(door, agent_x, agent_y, visibility_range):
            clipped_doors.append(door)
    
    # Calculate clipping statistics
    original_wall_count = len(walls)
    original_door_count = len(doors)
    clipped_wall_count = len(clipped_walls)
    clipped_door_count = len(clipped_doors)
    
    wall_reduction_percent = ((original_wall_count - clipped_wall_count) / original_wall_count * 100) if original_wall_count > 0 else 0
    door_reduction_percent = ((original_door_count - clipped_door_count) / original_door_count * 100) if original_door_count > 0 else 0
    
    clipping_stats = {
        'original_walls': original_wall_count,
        'clipped_walls': clipped_wall_count,
        'walls_removed': original_wall_count - clipped_wall_count,
        'wall_reduction_percent': wall_reduction_percent,
        'original_doors': original_door_count,
        'clipped_doors': clipped_door_count,
        'doors_removed': original_door_count - clipped_door_count,
        'door_reduction_percent': door_reduction_percent,
        'total_original_objects': original_wall_count + original_door_count,
        'total_clipped_objects': clipped_wall_count + clipped_door_count,
        'total_reduction_percent': ((original_wall_count + original_door_count - clipped_wall_count - clipped_door_count) / 
                                  (original_wall_count + original_door_count) * 100) if (original_wall_count + original_door_count) > 0 else 0
    }
    
    return clipped_walls, clipped_doors, bounding_box, clipping_stats

def calculate_evader_analysis(agent_x, agent_y, agent_theta, visibility_range, walls, 
                            mask_data=None, num_rays=100, num_sectors=8, min_gap_distance=30, spatial_index=None, save_lines_to_file=None):
    """
    Unified API that calculates all evader analysis data and returns it as a single object.
    
    Args:
        agent_x, agent_y: Evader position
        agent_theta: Evader orientation in radians
        visibility_range: Maximum visibility distance
        walls: List of wall rectangles that block vision
        mask_data: Reachability mask data (optional)
        num_rays: Number of visibility rays to cast (default 100)
        num_sectors: Number of sectors for visibility analysis (default 8)
        min_gap_distance: Minimum distance for breakoff point detection (default 30)
        spatial_index: Spatial index for fast spatial queries (optional)
        save_lines_to_file: If provided, save all lines from clipped environment to this filename (optional)
        
    Returns:
        EvaderAnalysis object with all computed data as attributes
    """
    import time
    
    analysis = EvaderAnalysis()
    
    # Store metadata
    analysis.agent_position = (agent_x, agent_y)
    analysis.agent_orientation = agent_theta
    analysis.visibility_range = visibility_range
    analysis.analysis_timestamp = time.time()
    analysis.spatial_index = spatial_index  # Store spatial index for use within analysis
    
    # Clip environment to visibility range for performance optimization
    analysis.clipped_walls, analysis.clipped_doors, analysis.visibility_bounding_box, analysis.clipping_statistics = \
        clip_environment_to_visibility_range(agent_x, agent_y, visibility_range, walls, [])
    
    # Calculate reachability analysis if mask data is available
    if mask_data is not None:
        analysis.reachability_grid = get_reachability_probabilities_for_fixed_grid(
            agent_x, agent_y, agent_theta, mask_data
        )
        analysis.reachability_at_position = get_reachability_at_position(
            agent_x, agent_y, agent_theta, mask_data
        )
    
    # Calculate visibility analysis using clipped environment for better performance
    analysis.visibility_rays = calculate_evader_visibility(
        agent_x, agent_y, visibility_range, analysis.clipped_walls, [], num_rays
    )
    
    # Calculate visibility statistics
    analysis.visibility_statistics = get_visibility_statistics(analysis.visibility_rays)
    
    # Calculate visibility sectors
    analysis.visibility_sectors = calculate_visibility_sectors(analysis.visibility_rays, num_sectors)
    
    # Calculate breakoff points
    analysis.breakoff_points, analysis.breakoff_lines = detect_visibility_breakoff_points(
        analysis.visibility_rays, min_gap_distance, agent_x, agent_y, agent_theta
    )
    
    # Calculate midpoints of breakoff lines
    analysis.breakoff_midpoints = calculate_breakoff_line_midpoints(analysis.breakoff_lines)
    
    # Calculate exploration offset points toward unexplored areas
    analysis.exploration_offset_points = calculate_exploration_offset_points(
        analysis.breakoff_midpoints, analysis.breakoff_lines, offset_distance=15.0, spatial_index=spatial_index, agent_x=agent_x, agent_y=agent_y
    )
    
    # Extract nearest nodes information for highlighting
    analysis.exploration_nearest_nodes = []
    if analysis.exploration_offset_points:
        for exploration_point in analysis.exploration_offset_points:
            if len(exploration_point) >= 6 and exploration_point[5] is not None:
                # exploration_point = (offset_x, offset_y, gap_size, category, direction, nearest_node_info)
                node_index, distance = exploration_point[5]
                analysis.exploration_nearest_nodes.append({
                    'node_index': node_index,
                    'distance': distance,
                    'exploration_point': (exploration_point[0], exploration_point[1]),
                    'category': exploration_point[3],
                    'direction': exploration_point[4]
                })
    
    # Convert all environment elements to lines for polygon exploration
    clipped_environment_lines = []
    
    # Convert clipped walls (rectangles) to lines
    if analysis.clipped_walls:
        clipped_environment_lines.extend(convert_rectangles_to_lines(analysis.clipped_walls))
    
    # Convert clipped doors (rectangles) to lines
    if analysis.clipped_doors:
        clipped_environment_lines.extend(convert_rectangles_to_lines(analysis.clipped_doors))
    
    # Calculate polygon exploration paths for breakpoints into unknown areas
    analysis.polygon_exploration_paths = calculate_polygon_exploration_paths(
        analysis.breakoff_lines, agent_x, agent_y, visibility_range, clipped_environment_lines
    )
    
    # Convert visibility rays to wall lines and add to clipped environment
    ray_walls = convert_visibility_rays_to_walls(analysis.visibility_rays, agent_x, agent_y)
    
    # Store the visibility circle as a simple circle (not DDA wall segments)
    analysis.visibility_circle = {
        'center_x': agent_x,
        'center_y': agent_y,
        'radius': visibility_range
    }
    
    # Convert breakoff lines to wall lines
    breakoff_walls = convert_breakoff_lines_to_walls(analysis.breakoff_lines)
    
    # Add ray walls and breakoff walls to the clipped walls so they're part of the environment data
    all_new_walls = []
    if ray_walls:
        all_new_walls.extend(ray_walls)
    if breakoff_walls:
        all_new_walls.extend(breakoff_walls)
    
    if all_new_walls:
        analysis.clipped_walls = list(analysis.clipped_walls) + all_new_walls
        # Update statistics to include new walls
        if analysis.clipping_statistics:
            analysis.clipping_statistics['ray_walls_added'] = len(ray_walls) if ray_walls else 0
            analysis.clipping_statistics['breakoff_walls_added'] = len(breakoff_walls) if breakoff_walls else 0
            analysis.clipping_statistics['total_clipped_objects'] += len(all_new_walls)
    
    # Save lines to file if requested
    if save_lines_to_file:
        save_clipped_environment_to_file(analysis, save_lines_to_file)
    
    return analysis

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
            while i < len(visibility_rays) and not visibility_rays[i][3]:  # not blocked
                i += 1
            sequence_end = i - 1
            
            # Add first and last rays of the sequence
            if sequence_start == sequence_end:
                # Single full-length ray
                rays_to_convert.append(sequence_start)
            else:
                # Multiple consecutive full-length rays - add first and last
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
    import pygame
    import math
    
    circle_walls = []
    
    if visibility_range <= 0:
        return circle_walls
    
    # Calculate angle step for each segment
    angle_step = (2 * math.pi) / num_segments
    
    for i in range(num_segments):
        # Calculate start and end angles for this segment
        start_angle = i * angle_step
        end_angle = (i + 1) * angle_step
        
        # Calculate start and end points on the circle
        start_x = agent_x + visibility_range * math.cos(start_angle)
        start_y = agent_y + visibility_range * math.sin(start_angle)
        end_x = agent_x + visibility_range * math.cos(end_angle)
        end_y = agent_y + visibility_range * math.sin(end_angle)
        
        # Create wall rectangles along the segment using DDA algorithm
        start_x_int, start_y_int = int(start_x), int(start_y)
        end_x_int, end_y_int = int(end_x), int(end_y)
        
        # Calculate the line points using DDA algorithm
        dx = abs(end_x_int - start_x_int)
        dy = abs(end_y_int - start_y_int)
        
        if dx == 0 and dy == 0:
            # Single point
            circle_walls.append(pygame.Rect(start_x_int, start_y_int, wall_thickness, wall_thickness))
            continue
        
        # Determine step direction
        step_x = 1 if start_x_int < end_x_int else -1
        step_y = 1 if start_y_int < end_y_int else -1
        
        # Current position
        x, y = start_x_int, start_y_int
        
        # Create rectangles along the segment
        if dx > dy:
            # X-major line
            error = dx / 2
            while x != end_x_int:
                circle_walls.append(pygame.Rect(x, y, wall_thickness, wall_thickness))
                error -= dy
                if error < 0:
                    y += step_y
                    error += dx
                x += step_x
        else:
            # Y-major line
            error = dy / 2
            while y != end_y_int:
                circle_walls.append(pygame.Rect(x, y, wall_thickness, wall_thickness))
                error -= dx
                if error < 0:
                    x += step_x
                    error += dy
                y += step_y
        
        # Add the final point
        circle_walls.append(pygame.Rect(end_x_int, end_y_int, wall_thickness, wall_thickness))
    
    return circle_walls

def convert_breakoff_lines_to_walls(breakoff_lines, wall_thickness=2):
    """
    Convert breakoff lines to simple line representations.
    
    Args:
        breakoff_lines: List of (start_point, end_point, gap_size, category) tuples
        wall_thickness: Not used anymore, kept for compatibility
        
    Returns:
        List of line dictionaries with 'start' and 'end' points representing breakoff lines
    """
    breakoff_walls = []
    
    if not breakoff_lines:
        return breakoff_walls
    
    for start_point, end_point, gap_size, category in breakoff_lines:
        # Create a simple line representation
        breakoff_walls.append({
            'start': (start_point[0], start_point[1]),
            'end': (end_point[0], end_point[1]),
            'gap_size': gap_size,
            'category': category,
            'type': 'breakoff_line'
        })
    
    return breakoff_walls

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
    
    # Calculate parameter t for the closest point on the line
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    
    # Clamp t to [0, 1] to stay within the line segment
    t = max(0, min(1, t))
    
    # Find the closest point on the line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Return distance from point to closest point on line
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

def calculate_breakoff_line_midpoints(breakoff_lines):
    """
    Calculate midpoints of lines connecting breakoff points.
    
    Args:
        breakoff_lines: List of (start_point, end_point, gap_size, category) tuples
                       from detect_visibility_breakoff_points()
    
    Returns:
        List of (midpoint_x, midpoint_y, gap_size, category) tuples where:
        - midpoint_x, midpoint_y: Coordinates of the line midpoint
        - gap_size: Original gap size from breakoff line
        - category: Category of the breakoff (e.g., 'clockwise_near_far')
    """
    midpoints = []
    
    for start_point, end_point, gap_size, category in breakoff_lines:
        # Calculate midpoint coordinates
        midpoint_x = (start_point[0] + end_point[0]) / 2.0
        midpoint_y = (start_point[1] + end_point[1]) / 2.0
        
        midpoints.append((midpoint_x, midpoint_y, gap_size, category))
    
    return midpoints

def calculate_exploration_offset_points(breakoff_midpoints, breakoff_lines, offset_distance=50.0, spatial_index=None, agent_x=None, agent_y=None):
    """
    Calculate offset points from breakoff midpoints in the direction of unexplored spaces.
    
    Args:
        breakoff_midpoints: List of (midpoint_x, midpoint_y, gap_size, category) tuples
        breakoff_lines: List of (start_point, end_point, gap_size, category) tuples
        offset_distance: Distance to offset from midpoint (default 50 pixels)
        spatial_index: Spatial index for finding nearest map graph nodes (optional)
        agent_x, agent_y: Agent position (required for consistent direction calculation)
    
    Returns:
        List of (offset_x, offset_y, gap_size, category, direction, nearest_node_info) tuples where:
        - offset_x, offset_y: Coordinates of the offset exploration point
        - gap_size: Original gap size from breakoff line
        - category: Category of the breakoff (e.g., 'clockwise_near_far')
        - direction: 'left' or 'right' indicating offset direction relative to line
        - nearest_node_info: (node_index, distance) tuple if spatial_index available, None otherwise
    """
    exploration_points = []
    
    if not breakoff_midpoints or not breakoff_lines:
        return exploration_points
    
    # Match midpoints with their corresponding lines
    for midpoint_x, midpoint_y, gap_size, category in breakoff_midpoints:
        # Find the corresponding breakoff line for this midpoint
        corresponding_line = None
        for start_point, end_point, line_gap_size, line_category in breakoff_lines:
            # Check if this line matches the midpoint (same gap size and category)
            if abs(line_gap_size - gap_size) < 1.0 and line_category == category:
                # Verify this is actually the midpoint of this line
                line_midpoint_x = (start_point[0] + end_point[0]) / 2.0
                line_midpoint_y = (start_point[1] + end_point[1]) / 2.0
                
                # Check if midpoints match (within small tolerance)
                if abs(line_midpoint_x - midpoint_x) < 1.0 and abs(line_midpoint_y - midpoint_y) < 1.0:
                    corresponding_line = (start_point, end_point, line_gap_size, line_category)
                    break
        
        if corresponding_line is None:
            continue
        
        start_point, end_point, _, _ = corresponding_line
        
        # Instead of using line vector direction, use agent position for consistent offset direction
        # Calculate vector from agent to midpoint
        agent_to_midpoint_x = midpoint_x - agent_x if agent_x is not None else 0
        agent_to_midpoint_y = midpoint_y - agent_y if agent_y is not None else 0
        
        # Calculate the line vector
        line_dx = end_point[0] - start_point[0]
        line_dy = end_point[1] - start_point[1]
        
        # Normalize the line vector
        line_length = math.sqrt(line_dx**2 + line_dy**2)
        if line_length == 0:
            continue
        
        line_unit_x = line_dx / line_length
        line_unit_y = line_dy / line_length
        
        # Calculate perpendicular vector (90 degrees counterclockwise from line vector)
        perp_x = -line_unit_y
        perp_y = line_unit_x
        
        # Determine offset direction based on breakoff category
        # Corrected rule:
        # - For 'near_far' transitions: offset to the LEFT (toward the near/open side)
        # - For 'far_near' transitions: offset to the RIGHT (toward the far/blocked side)
        
        if 'near_far' in category:
            # Near to far transition: offset to the left
            direction = 'left'
            offset_multiplier = -1.0  # Negative means opposite to perp vector
        else:  # 'far_near' in category
            # Far to near transition: offset to the right
            direction = 'right'
            offset_multiplier = -1.0  # Positive means in direction of perp vector

        # Calculate the final offset point
        offset_x = midpoint_x + offset_multiplier * perp_x * offset_distance
        offset_y = midpoint_y + offset_multiplier * perp_y * offset_distance
        
        # Find nearest map graph node if spatial index is available
        nearest_node_info = None
        if spatial_index is not None:
            try:
                result = spatial_index.find_nearest_node(offset_x, offset_y)
                if result:
                    nearest_node_info = result  # (node_index, distance) tuple
            except Exception as e:
                # Silently handle spatial index errors
                pass
        
        exploration_points.append((offset_x, offset_y, gap_size, category, direction, nearest_node_info))
    
    return exploration_points

def convert_rectangles_to_lines(rectangles):
    """
    Convert a list of pygame.Rect objects to line segments.
    Each rectangle is converted to 4 line segments (top, right, bottom, left edges).
    
    Args:
        rectangles: List of pygame.Rect objects
        
    Returns:
        List of line segments as ((x1, y1), (x2, y2)) tuples
    """
    lines = []
    
    for rect in rectangles:
        x, y, width, height = rect.x, rect.y, rect.width, rect.height
        
        # Rectangle corners
        top_left = (x, y)
        top_right = (x + width, y)
        bottom_left = (x, y + height) 
        bottom_right = (x + width, y + height)
        
        # Four edges of the rectangle as line segments
        lines.extend([
            (top_left, top_right),      # Top edge
            (top_right, bottom_right),  # Right edge
            (bottom_right, bottom_left), # Bottom edge
            (bottom_left, top_left)     # Left edge
        ])
    
    return lines

def save_clipped_environment_to_file(analysis, filename="clipped_environment_lines.txt"):
    """
    Save all lines from the clipped environment to a text file.
    Converts rectangular walls, doors, and added elements to line segments.
    Circles (visibility boundary) are saved as circle parameters.
    
    Args:
        analysis: EvaderAnalysis object containing clipped environment data
        filename: Output filename (default: "clipped_environment_lines.txt")
    """
    if not analysis or not hasattr(analysis, 'clipped_walls'):
        print("Error: No clipped environment data available")
        return
    
    lines_data = {
        'original_walls': [],
        'original_doors': [], 
        'ray_walls': [],
        'circle_walls': [],
        'breakoff_walls': [],
        'visibility_circle': None
    }
    
    # Get statistics for separating different types of walls
    stats = analysis.clipping_statistics if analysis.clipping_statistics else {}
    num_ray_walls = stats.get('ray_walls_added', 0)
    num_breakoff_walls = stats.get('breakoff_walls_added', 0)
    
    # Total added walls (no longer including circle walls)
    total_added_walls = num_ray_walls + num_breakoff_walls
    
    # Separate different types of walls
    clipped_walls = analysis.clipped_walls
    if total_added_walls > 0 and len(clipped_walls) >= total_added_walls:
        # Original clipped walls (excluding added ones)
        original_walls = clipped_walls[:-total_added_walls]
        
        # Added walls are at the end
        added_walls = clipped_walls[-total_added_walls:]
        
        # Further separate added walls based on order: ray_walls, breakoff_walls
        wall_idx = 0
        if num_ray_walls > 0:
            ray_walls = added_walls[wall_idx:wall_idx + num_ray_walls]
            # Handle ray walls - they are now line dictionaries, not rectangles
            for ray in ray_walls:
                if isinstance(ray, dict) and 'start' in ray and 'end' in ray:
                    # Ray wall is already a line dictionary
                    lines_data['ray_walls'].append((ray['start'], ray['end']))
                else:
                    # Fallback: treat as rectangle and convert to lines
                    lines_data['ray_walls'].extend(convert_rectangles_to_lines([ray]))
            wall_idx += num_ray_walls
            
        if num_breakoff_walls > 0:
            breakoff_walls = added_walls[wall_idx:wall_idx + num_breakoff_walls]
            # Handle breakoff walls - they are now line dictionaries, not rectangles
            for breakoff in breakoff_walls:
                if isinstance(breakoff, dict) and 'start' in breakoff and 'end' in breakoff:
                    # Breakoff wall is already a line dictionary
                    lines_data['breakoff_walls'].append((breakoff['start'], breakoff['end']))
                else:
                    # Fallback: treat as rectangle and convert to lines
                    lines_data['breakoff_walls'].extend(convert_rectangles_to_lines([breakoff]))
    else:
        # No added walls, all are original
        original_walls = clipped_walls
    
    # Convert original walls to lines
    lines_data['original_walls'] = convert_rectangles_to_lines(original_walls)
    
    # Convert clipped doors to lines
    if hasattr(analysis, 'clipped_doors') and analysis.clipped_doors:
        lines_data['original_doors'] = convert_rectangles_to_lines(analysis.clipped_doors)
    
    # Add visibility circle information (keep as circle, not lines)
    if hasattr(analysis, 'visibility_circle') and analysis.visibility_circle:
        lines_data['visibility_circle'] = analysis.visibility_circle
    
    # Write to file
    try:
        with open(filename, 'w') as f:
            f.write("# Clipped Environment Lines\n")
            f.write(f"# Generated from evader analysis at position ({analysis.agent_position[0]:.1f}, {analysis.agent_position[1]:.1f})\n")
            f.write(f"# Visibility range: {analysis.visibility_range:.1f} pixels\n\n")
            
            # Write visibility circle (kept as circle)
            if lines_data['visibility_circle']:
                circle = lines_data['visibility_circle']
                f.write("# VISIBILITY_CIRCLE\n")
                f.write(f"CIRCLE {circle['center_x']:.2f} {circle['center_y']:.2f} {circle['radius']:.2f}\n\n")
            
            # Write original walls as lines
            if lines_data['original_walls']:
                f.write("# ORIGINAL_WALLS\n")
                for line in lines_data['original_walls']:
                    (x1, y1), (x2, y2) = line
                    f.write(f"LINE {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")
                f.write(f"# Total original wall lines: {len(lines_data['original_walls'])}\n\n")
            
            # Write original doors as lines  
            if lines_data['original_doors']:
                f.write("# ORIGINAL_DOORS\n")
                for line in lines_data['original_doors']:
                    (x1, y1), (x2, y2) = line
                    f.write(f"LINE {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")
                f.write(f"# Total original door lines: {len(lines_data['original_doors'])}\n\n")
            
            # Write ray walls as lines
            if lines_data['ray_walls']:
                f.write("# RAY_WALLS (from visibility rays)\n")
                for line in lines_data['ray_walls']:
                    (x1, y1), (x2, y2) = line
                    f.write(f"LINE {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")
                f.write(f"# Total ray wall lines: {len(lines_data['ray_walls'])}\n\n")
            
            # Write breakoff walls as lines
            if lines_data['breakoff_walls']:
                f.write("# BREAKOFF_WALLS (from breakoff lines)\n")
                for line in lines_data['breakoff_walls']:
                    (x1, y1), (x2, y2) = line
                    f.write(f"LINE {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")
                f.write(f"# Total breakoff wall lines: {len(lines_data['breakoff_walls'])}\n\n")
            
            # Write summary
            total_lines = (len(lines_data['original_walls']) + len(lines_data['original_doors']) + 
                          len(lines_data['ray_walls']) + len(lines_data['breakoff_walls']))
            
            f.write("# SUMMARY\n")
            f.write(f"# Total lines: {total_lines}\n")
            f.write(f"# Original walls: {len(lines_data['original_walls'])} lines\n")
            f.write(f"# Original doors: {len(lines_data['original_doors'])} lines\n")
            f.write(f"# Ray walls: {len(lines_data['ray_walls'])} lines\n")
            f.write(f"# Breakoff walls: {len(lines_data['breakoff_walls'])} lines\n")
            f.write(f"# Visibility circle: {'1 circle' if lines_data['visibility_circle'] else '0 circles'}\n")
        
        print(f"Successfully saved {total_lines} lines and {1 if lines_data['visibility_circle'] else 0} circles to {filename}")
        
        # Print breakdown
        print(f"  Original walls: {len(lines_data['original_walls'])} lines")
        print(f"  Original doors: {len(lines_data['original_doors'])} lines")
        print(f"  Ray walls: {len(lines_data['ray_walls'])} lines")
        print(f"  Breakoff walls: {len(lines_data['breakoff_walls'])} lines")
        print(f"  Visibility circle: {'Yes' if lines_data['visibility_circle'] else 'No'}")
        
    except Exception as e:
        print(f"Error saving clipped environment to file: {e}")

def load_lines_from_file(filename="clipped_environment_lines.txt"):
    """
    Load line segments and circles from a text file.
    
    Args:
        filename: Input filename (default: "clipped_environment_lines.txt")
        
    Returns:
        Dictionary with 'lines' (list of line segments) and 'circles' (list of circle parameters)
    """
    lines = []
    circles = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue
                
                # Parse line segments
                if line.startswith('LINE'):
                    parts = line.split()
                    if len(parts) == 5:  # LINE x1 y1 x2 y2
                        x1, y1, x2, y2 = map(float, parts[1:5])
                        lines.append(((x1, y1), (x2, y2)))
                
                # Parse circles
                elif line.startswith('CIRCLE'):
                    parts = line.split()
                    if len(parts) == 4:  # CIRCLE center_x center_y radius
                        center_x, center_y, radius = map(float, parts[1:4])
                        circles.append({
                            'center_x': center_x,
                            'center_y': center_y,
                            'radius': radius
                        })
        
        print(f"Successfully loaded {len(lines)} lines and {len(circles)} circles from {filename}")
        return {'lines': lines, 'circles': circles}
        
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return {'lines': [], 'circles': []}
    except Exception as e:
        print(f"Error loading lines from file: {e}")
        return {'lines': [], 'circles': []}
