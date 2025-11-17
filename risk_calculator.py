#!/usr/bin/env python3
"""
Risk Calculator
Handles reachability calculations, mask transformations, and visibility calculations.
"""

import pickle
import numpy as np
import math
import time
import pygame

# Import environment constants for world boundary checking
try:
    from simulation_config import ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT
except ImportError:
    # Fallback values if simulation_config is not available
    ENVIRONMENT_WIDTH = 1280
    ENVIRONMENT_HEIGHT = 720

# Import polygon exploration functionality
from polygon_exploration_cpp import calculate_polygon_exploration_paths_cpp as calculate_polygon_exploration_paths
# from polygon_exploration import calculate_polygon_exploration_paths  # Fallback if C++ not available

# Import helper modules
from reachability_utils import (
    _get_cached_array,
    load_reachability_mask,
    world_to_grid_coords,
    get_reachability_probabilities_for_fixed_grid,
    get_reachability_at_position
)
from geometry_utils import (
    generate_rectangular_buffer_and_convert,
    point_to_line_distance,
    calculate_breakoff_line_midpoints,
    calculate_exploration_offset_points,
    convert_rectangles_to_lines,
    vectorized_point_in_polygon
)

# Import new utility modules
from visibility_utils import (
    calculate_evader_visibility,
    get_visibility_statistics,
    calculate_visibility_sectors,
    detect_visibility_breakoff_points,
    create_visibility_boundary_polygon,
    convert_visibility_rays_to_walls,
    create_ray_connecting_walls,
    create_visibility_circle_walls
)
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
        self.exploration_graph = None  # Intersection graph from polygon exploration
        self.path_links = None  # Links between overlapping exploration paths
        
        # Reachability overlay processing results (from overlay API)
        self.path_analysis_data = None  # Path analysis data with first edges, orientations, etc.
        self.reachability_overlay_data = None  # Reachability mask and bounds for visualization
        self.sample_points_data = None  # Sample points where reachability values were evaluated
        
        # Environment clipping data for performance optimization
        self.clipped_walls = None  # Walls within visibility range
        self.clipped_doors = None  # Doors within visibility range
        self.visibility_bounding_box = None  # (min_x, min_y, max_x, max_y)
        self.clipping_statistics = None  # Stats about clipping efficiency
        self.visibility_circle = None  # Visibility circle as simple circle parameters
        self.visibility_boundary_polygon = None  # Polygon connecting all visibility ray endpoints
        
        # Future extensibility - easily add new analysis types
        self.rods = None  # For future rod analysis
        self.threat_assessment = None  # For future threat analysis
        self.escape_routes = None  # For future escape route analysis
        self.pursuit_advantage = None  # For future pursuit analysis
        
        # Clipped reachability heatmap within visibility circle
        self.clipped_reachability_grid = None  # Reachability heatmap clipped to visibility circle
        
        # Map graph node highlighting data
        self.highlighted_nodes = None  # Dictionary containing nodes to highlight with categories
        
        # Metadata
        self.agent_position = None
        self.agent_orientation = None
        self.visibility_range = None
        self.analysis_timestamp = None
        self.spatial_index = None  # Spatial index for fast spatial queries


def clip_environment_to_visibility_range(agent_x, agent_y, visibility_range, walls, doors=None):
    """
    Clip walls to only include those within or intersecting the visibility circle.
    This significantly improves performance by reducing the number of collision checks.
    
    Args:
        agent_x, agent_y: Evader position (center of visibility circle)
        visibility_range: Visibility radius
        walls: List of wall rectangles
        doors: Unused parameter, kept for compatibility
        
    Returns:
        Tuple of (clipped_walls, clipped_doors, bounding_box, clipping_stats)
        - clipped_walls: Walls within/intersecting visibility circle
        - clipped_doors: Empty list (doors no longer processed)
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
            # Process wall through shapely buffer before adding
            buffered_wall = generate_rectangular_buffer_and_convert(wall)
            clipped_walls.append(buffered_wall)
    
    # No longer process doors - return empty list for compatibility
    clipped_doors = []
    
    # Calculate clipping statistics
    original_wall_count = len(walls)
    original_door_count = 0  # No doors processed
    clipped_wall_count = len(clipped_walls)
    clipped_door_count = 0   # No doors processed
    
    wall_reduction_percent = ((original_wall_count - clipped_wall_count) / original_wall_count * 100) if original_wall_count > 0 else 0
    door_reduction_percent = 0  # No doors processed
    
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


def detect_and_link_overlapping_paths(polygon_exploration_paths, breakoff_lines):
    """
    Detect when exploration paths overlap and create links between breakpoints.
    If one path contains another breakpoint, they are linked and only the shorter path is kept.
    
    Args:
        polygon_exploration_paths: List of polygon path dictionaries
        breakoff_lines: List of (start_point, end_point, gap_size, category) tuples
        
    Returns:
        Tuple of (filtered_paths, path_links) where:
        - filtered_paths: Original paths with overlapping ones filtered (shorter path kept)
        - path_links: List of link dictionaries showing connections between breakpoints
    """
    if not polygon_exploration_paths or not breakoff_lines:
        return polygon_exploration_paths, []
    
    # Create a mapping from breakoff lines to their exploration paths
    path_to_breakoff = {}
    breakoff_positions = {}
    
    for i, path_data in enumerate(polygon_exploration_paths):
        breakoff_line = path_data.get('breakoff_line')
        if breakoff_line:
            path_to_breakoff[i] = breakoff_line
            # Store breakpoint positions (midpoint of breakoff line)
            start_point, end_point = breakoff_line[0], breakoff_line[1]
            midpoint = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)
            breakoff_positions[i] = midpoint
    
    # Find overlapping paths - both paths must contain each other's breakpoints
    overlapping_groups = []
    path_links = []
    processed_paths = set()
    
    for i, path_data_i in enumerate(polygon_exploration_paths):
        if i in processed_paths:
            continue
            
        path_points_i = path_data_i.get('path_points', [])
        if len(path_points_i) < 3:  # Need at least 3 points for polygon
            continue
            
        overlapping_group = [i]
        
        # Check for mutual containment with other paths
        for j, path_data_j in enumerate(polygon_exploration_paths):
            if i == j or j in processed_paths:
                continue
                
            path_points_j = path_data_j.get('path_points', [])
            if len(path_points_j) < 3:  # Need at least 3 points for polygon
                continue
            
            if i in breakoff_positions and j in breakoff_positions:
                breakpoint_i = breakoff_positions[i]
                breakpoint_j = breakoff_positions[j]
                
                # Check for mutual containment: both breakpoints must be inside the other's path
                i_contains_j = point_in_polygon(breakpoint_j, path_points_i)
                j_contains_i = point_in_polygon(breakpoint_i, path_points_j)
                
                # Both paths must contain each other's breakpoints to be linked
                if i_contains_j and j_contains_i:
                    overlapping_group.append(j)
                    
                    # Create bidirectional link between breakpoints
                    path_links.append({
                        'from_breakpoint': breakpoint_i,
                        'to_breakpoint': breakpoint_j,
                        'from_path_index': i,
                        'to_path_index': j,
                        'link_type': 'mutual_containment'
                    })
        
        # If we found overlapping paths, determine which one to keep (shortest path)
        if len(overlapping_group) > 1:
            overlapping_groups.append(overlapping_group)
            
            # Find the shortest path in the group based on path length
            shortest_path_index = min(overlapping_group, 
                                    key=lambda idx: len(polygon_exploration_paths[idx].get('path_points', [])))
            
            # Mark all but the shortest as processed (will be filtered out)
            for path_idx in overlapping_group:
                if path_idx != shortest_path_index:
                    processed_paths.add(path_idx)
        
        # Mark the current path as processed
        processed_paths.add(i)
    
    # Create filtered list keeping only non-overlapping paths and shortest from each overlapping group
    filtered_paths = []
    for i, path_data in enumerate(polygon_exploration_paths):
        # Keep path if it wasn't filtered out due to overlap
        should_keep = True
        for group in overlapping_groups:
            if i in group:
                # Only keep if this is the shortest path in the group
                shortest_in_group = min(group, 
                                      key=lambda idx: len(polygon_exploration_paths[idx].get('path_points', [])))
                if i != shortest_in_group:
                    should_keep = False
                    break
        
        if should_keep:
            # Add metadata about which paths this one represents
            path_copy = path_data.copy()
            represented_paths = []
            for group in overlapping_groups:
                if i in group:
                    represented_paths = group
                    break
            if represented_paths:
                path_copy['represented_paths'] = represented_paths
                path_copy['is_merged_path'] = True
            else:
                path_copy['represented_paths'] = [i]
                path_copy['is_merged_path'] = False
                
            filtered_paths.append(path_copy)
    
    return filtered_paths, path_links


def point_in_polygon(point, polygon_points):
    """
    Determine if a point is inside a polygon using the ray casting algorithm.
    
    Args:
        point: (x, y) tuple of the point to test
        polygon_points: List of (x, y) tuples forming the polygon vertices
        
    Returns:
        True if point is inside polygon, False otherwise
    """
    if len(polygon_points) < 3:
        return False
    
    x, y = point
    n = len(polygon_points)
    inside = False
    
    p1x, p1y = polygon_points[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_points[i % n]
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def calculate_highlighted_map_nodes(spatial_index, agent_x, agent_y, visibility_range, visibility_bounding_box):
    """
    Calculate which map graph nodes should be highlighted based on visibility circle and bounding box.
    Uses spatial index for efficient O(1) lookups instead of iterating through all nodes.
    
    Args:
        spatial_index: SpatialIndex instance for fast spatial queries
        agent_x, agent_y: Agent position
        visibility_range: Visibility radius
        visibility_bounding_box: (min_x, min_y, max_x, max_y) bounding box
        
    Returns:
        Dictionary with categorized nodes:
        {
            'circle_nodes': [(node_index, node_x, node_y), ...],  # Nodes in visibility circle
            'bbox_nodes': [(node_index, node_x, node_y), ...]    # Nodes only in bounding box
        }
    """
    if spatial_index is None or visibility_bounding_box is None:
        return {'circle_nodes': [], 'bbox_nodes': []}
    
    # Get all nodes within the bounding box using spatial index
    min_x, min_y, max_x, max_y = visibility_bounding_box
    
    # Convert world coordinates to grid indices for efficient querying
    min_i, min_j = spatial_index.coordinates_to_grid_indices(min_x, min_y)
    max_i, max_j = spatial_index.coordinates_to_grid_indices(max_x, max_y)
    
    # Get all nodes in the bounding box region using O(1) grid access
    bbox_node_indices = spatial_index.get_nodes_in_grid_region(min_i, min_j, max_i, max_j)
    
    # Categorize nodes: circle vs bbox-only
    circle_nodes = []
    bbox_only_nodes = []
    visibility_radius_sq = visibility_range ** 2
    
    # Get node positions for the nodes in the bbox
    if hasattr(spatial_index, 'node_positions') and spatial_index.node_positions is not None:
        for node_idx in bbox_node_indices:
            if 0 <= node_idx < len(spatial_index.node_positions):
                node_x, node_y = spatial_index.node_positions[node_idx]
                
                # Check if node is within visibility circle
                dx = node_x - agent_x
                dy = node_y - agent_y
                distance_sq = dx * dx + dy * dy
                
                if distance_sq <= visibility_radius_sq:
                    circle_nodes.append((node_idx, node_x, node_y))
                else:
                    # Node is in bounding box but not in circle
                    bbox_only_nodes.append((node_idx, node_x, node_y))
    
    return {
        'circle_nodes': circle_nodes,
        'bbox_nodes': bbox_only_nodes
    }


def calculate_evader_analysis(agent_x, agent_y, agent_theta, visibility_range, walls, 
                            mask_data=None, num_rays=100, num_sectors=8, min_gap_distance=30, spatial_index=None, save_lines_to_file=None, overlay_api=None):
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
        overlay_api: ReachabilityMaskAPI instance for overlay processing (optional)
        
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
    analysis.overlay_api = overlay_api  # Store overlay API for reachability processing
    
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
        
        # Create clipped version of reachability heatmap within visibility circle (one-time operation)
        analysis.clipped_reachability_grid = create_clipped_reachability_grid(
            agent_x, agent_y, agent_theta, visibility_range, mask_data
        )
    
    # Calculate visibility analysis using clipped environment for better performance
    analysis.visibility_rays = calculate_evader_visibility(
        agent_x, agent_y, visibility_range, analysis.clipped_walls, [], num_rays
    )
    
    # Create visibility boundary polygon from ray endpoints
    analysis.visibility_boundary_polygon = create_visibility_boundary_polygon(
        analysis.visibility_rays, agent_x, agent_y
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
    
    # Convert visibility rays to wall lines and add to clipped environment
    ray_walls = convert_visibility_rays_to_walls(analysis.visibility_rays, agent_x, agent_y)
    
    # Add connecting walls between consecutive ray endpoints (avoiding breakoff line duplicates)
    ray_connecting_walls = create_ray_connecting_walls(analysis.visibility_rays, analysis.breakoff_lines, agent_x, agent_y)
    if ray_connecting_walls:
        ray_walls.extend(ray_connecting_walls)
    
    # Store the visibility circle as a simple circle (not DDA wall segments)
    analysis.visibility_circle = {
        'center_x': agent_x,
        'center_y': agent_y,
        'radius': visibility_range
    }
    
    # Convert breakoff lines to wall lines
    breakoff_walls = convert_breakoff_lines_to_walls(analysis.breakoff_lines)
    
    # Convert all environment elements to lines for polygon exploration
    clipped_environment_lines = []
    
    # Convert clipped walls (rectangles) to lines
    if analysis.clipped_walls:
        clipped_environment_lines.extend(convert_rectangles_to_lines(analysis.clipped_walls))
    
    # Convert clipped doors (rectangles) to lines
    if analysis.clipped_doors:
        clipped_environment_lines.extend(convert_rectangles_to_lines(analysis.clipped_doors))
    
    # Add ray walls to environment lines for polygon exploration
    if ray_walls:
        for ray in ray_walls:
            if isinstance(ray, dict) and 'start' in ray and 'end' in ray:
                clipped_environment_lines.append((ray['start'], ray['end']))
    
    # Add breakoff walls to environment lines for polygon exploration
    if breakoff_walls:
        for breakoff in breakoff_walls:
            if isinstance(breakoff, dict) and 'start' in breakoff and 'end' in breakoff:
                clipped_environment_lines.append((breakoff['start'], breakoff['end']))
    
    # Calculate polygon exploration paths for breakpoints into unknown areas
    # Now includes ray walls and breakoff walls as geometric constraints
    analysis.polygon_exploration_paths, analysis.exploration_graph = calculate_polygon_exploration_paths(
        analysis.breakoff_lines, agent_x, agent_y, visibility_range, clipped_environment_lines
    )
    
    # Detect overlapping paths and create links between breakpoints
    analysis.polygon_exploration_paths, analysis.path_links = detect_and_link_overlapping_paths(
        analysis.polygon_exploration_paths, analysis.breakoff_lines
    )
    
    # Process paths with reachability overlay if API is available
    if overlay_api is not None and analysis.polygon_exploration_paths:
        try:
            # Use the overlay API to process paths with reachability data
            path_analysis_data, reachability_data, sample_points_data = overlay_api.process_paths_with_reachability(
                paths=analysis.polygon_exploration_paths,
                agent_x=agent_x,
                agent_y=agent_y,
                agent_orientation=agent_theta,
                visibility_range=visibility_range,
                visibility_polygon=analysis.visibility_boundary_polygon
            )
            
            # Store the results in the analysis object
            analysis.path_analysis_data = path_analysis_data
            analysis.reachability_overlay_data = reachability_data
            analysis.sample_points_data = sample_points_data
            
        except Exception as e:
            print(f"⚠️ Could not process paths with reachability overlay: {e}")
            # Set fallback values
            analysis.path_analysis_data = []
            analysis.reachability_overlay_data = None
            analysis.sample_points_data = []
    else:
        # No overlay API available
        analysis.path_analysis_data = []
        analysis.reachability_overlay_data = None
        analysis.sample_points_data = []
    
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
    
    # Calculate highlighted map graph nodes using spatial index for efficiency
    if spatial_index is not None and analysis.visibility_bounding_box is not None:
        analysis.highlighted_nodes = calculate_highlighted_map_nodes(
            spatial_index, 
            agent_x, agent_y, 
            visibility_range, 
            analysis.visibility_bounding_box
        )
    
    return analysis

def create_clipped_reachability_grid(agent_x, agent_y, agent_theta, visibility_range, mask_data):
    """
    Create a clipped reachability grid using bounding box clipping for maximum speed.
    Clips the full reachability grid to a square bounding box around the visibility circle.
    
    Args:
        agent_x, agent_y: Agent position (center of visibility area)
        agent_theta: Agent orientation in radians
        visibility_range: Radius of the visibility circle (used for bounding box size)
        mask_data: Full reachability mask data
        
    Returns:
        Dictionary with clipped grid data and metadata, or None if not available
    """
    if mask_data is None:
        return None
    
    # Get the full reachability grid for current agent pose
    full_grid = get_reachability_probabilities_for_fixed_grid(agent_x, agent_y, agent_theta, mask_data)
    
    if full_grid is None:
        return None
    
    grid_size = mask_data.get('grid_size', 0)
    cell_size = mask_data.get('cell_size_px', 1)
    
    if grid_size == 0:
        return None
    
    # Calculate bounding box for clipping (square around the visibility circle)
    center_grid_x = grid_size // 2
    center_grid_y = grid_size // 2
    
    # Convert visibility range to grid cells
    visibility_cells = int(visibility_range / cell_size)
    
    # Calculate bounding box indices (clamp to grid boundaries)
    min_i = max(0, center_grid_x - visibility_cells)
    max_i = min(grid_size, center_grid_x + visibility_cells + 1)
    min_j = max(0, center_grid_y - visibility_cells)
    max_j = min(grid_size, center_grid_y + visibility_cells + 1)
    
    # Fast numpy slicing to extract the bounding box
    clipped_grid = full_grid[min_i:max_i, min_j:max_j].copy()
    
    # Create result structure compatible with drawing functions
    result = {
        'grid_size': grid_size,  # Original grid size for coordinate mapping
        'cell_size_px': cell_size,
        'grid_data': clipped_grid,
        'world_extent_px': mask_data.get('world_extent_px', grid_size * cell_size),
        'visibility_range': visibility_range,
        'clipped_bounds': (min_i, min_j, max_i, max_j),  # Bounds of the clipped region
        'clipped_size': clipped_grid.shape  # Size of the clipped grid
    }
    
    return result

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
            # Blocked ray - always add it
            rays_converted_to_walls.append(i)
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
                rays_converted_to_walls.append(sequence_start)
            else:
                # Multiple consecutive full-length rays - add first and last
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
        next_endpoint = visibility_rays[next_ray_idx][1]
        
        # Round coordinates for comparison
        current_rounded = (round(current_endpoint[0], 2), round(current_endpoint[1], 2))
        next_rounded = (round(next_endpoint[0], 2), round(next_endpoint[1], 2))
        
        # Check if this connection already exists as a breakoff line
        connection_exists = (current_rounded in breakoff_endpoints and 
                           next_rounded in breakoff_endpoints)
        
        # Only add connection if it doesn't already exist and endpoints are different
        if not connection_exists and current_rounded != next_rounded:
            # Calculate distance to avoid very short connections
            distance = math.sqrt(
                (current_endpoint[0] - next_endpoint[0])**2 + 
                (current_endpoint[1] - next_endpoint[1])**2
            )
            
            # Only add if distance is reasonable (avoid noise from very close points)
            if distance > 5.0:  # Minimum distance threshold
                # Check angle between the rays to ensure we don't connect walls on opposite sides
                current_angle = visibility_rays[current_ray_idx][0]  # angle field
                next_angle = visibility_rays[next_ray_idx][0]
                
                # Calculate the clockwise angular difference from current to next ray
                # This measures the angle going clockwise from current_angle to next_angle
                clockwise_angle_diff = next_angle - current_angle
                
                # Normalize to [0, 2π) range for clockwise measurement
                while clockwise_angle_diff < 0:
                    clockwise_angle_diff += 2 * math.pi
                while clockwise_angle_diff >= 2 * math.pi:
                    clockwise_angle_diff -= 2 * math.pi
                
                # Only connect if the clockwise angular difference is less than 180 degrees (π radians)
                # This ensures we're connecting rays that are consecutive in clockwise order
                if clockwise_angle_diff < math.pi:
                    connecting_walls.append({
                        'start': current_endpoint,
                        'end': next_endpoint,
                        'type': 'ray_connecting_line',
                        'distance': distance,
                        'clockwise_angle_diff': clockwise_angle_diff  # Store for debugging/analysis
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

def save_visibility_polygon_to_file(analysis, filename="visibility_polygon.txt"):
    """
    Save the visibility polygon (connecting ray endpoints) along with agent state to a text file.
    
    Args:
        analysis: EvaderAnalysis object containing visibility data
        filename: Output filename (default: "visibility_polygon.txt")
    """
    if not analysis:
        print("Error: No analysis data available")
        return
    
    try:
        with open(filename, 'w') as f:
            f.write("# Visibility Polygon Data\n")
            f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write agent state information
            if hasattr(analysis, 'agent_position') and analysis.agent_position:
                f.write("# AGENT_STATE\n")
                f.write(f"AGENT_POSITION {analysis.agent_position[0]:.6f} {analysis.agent_position[1]:.6f}\n")
                if hasattr(analysis, 'agent_orientation') and analysis.agent_orientation is not None:
                    f.write(f"AGENT_ORIENTATION {analysis.agent_orientation:.6f}\n")
                f.write(f"VISIBILITY_RANGE {analysis.visibility_range:.2f}\n\n")
            
            # Write visibility polygon if available
            if hasattr(analysis, 'visibility_boundary_polygon') and analysis.visibility_boundary_polygon:
                polygon = analysis.visibility_boundary_polygon
                f.write("# VISIBILITY_POLYGON\n")
                f.write(f"# Number of vertices: {len(polygon)}\n")
                f.write("# Format: VERTEX x y\n")
                
                for i, (x, y) in enumerate(polygon):
                    f.write(f"VERTEX {x:.6f} {y:.6f}\n")
                
                f.write(f"\n# POLYGON_SUMMARY\n")
                f.write(f"# Total vertices: {len(polygon)}\n")
                
                # Calculate polygon area using shoelace formula
                if len(polygon) >= 3:
                    area = 0.0
                    n = len(polygon)
                    for i in range(n):
                        j = (i + 1) % n
                        area += polygon[i][0] * polygon[j][1]
                        area -= polygon[j][0] * polygon[i][1]
                    area = abs(area) / 2.0
                    f.write(f"# Polygon area: {area:.2f} square pixels\n")
                
                # Calculate polygon perimeter
                if len(polygon) >= 2:
                    perimeter = 0.0
                    n = len(polygon)
                    for i in range(n):
                        j = (i + 1) % n
                        dx = polygon[j][0] - polygon[i][0]
                        dy = polygon[j][1] - polygon[i][1]
                        perimeter += math.sqrt(dx*dx + dy*dy)
                    f.write(f"# Polygon perimeter: {perimeter:.2f} pixels\n")
            else:
                f.write("# VISIBILITY_POLYGON\n")
                f.write("# No visibility polygon available\n")
            
            # Write visibility rays data if available
            if hasattr(analysis, 'visibility_rays') and analysis.visibility_rays:
                rays = analysis.visibility_rays
                f.write(f"\n# VISIBILITY_RAYS\n")
                f.write(f"# Number of rays: {len(rays)}\n")
                f.write("# Format: RAY angle endpoint_x endpoint_y distance blocked\n")
                
                for angle, endpoint, distance, blocked in rays:
                    blocked_str = "true" if blocked else "false"
                    if endpoint and len(endpoint) >= 2:
                        f.write(f"RAY {angle:.6f} {endpoint[0]:.6f} {endpoint[1]:.6f} {distance:.6f} {blocked_str}\n")
                
                # Calculate ray statistics
                blocked_count = sum(1 for _, _, _, blocked in rays if blocked)
                clear_count = len(rays) - blocked_count
                f.write(f"\n# RAY_SUMMARY\n")
                f.write(f"# Total rays: {len(rays)}\n")
                f.write(f"# Clear rays: {clear_count}\n")
                f.write(f"# Blocked rays: {blocked_count}\n")
                f.write(f"# Visibility ratio: {clear_count/len(rays)*100:.1f}%\n")
        
        # Print success message with breakdown
        polygon_vertices = 0
        ray_count = 0
        if hasattr(analysis, 'visibility_boundary_polygon') and analysis.visibility_boundary_polygon:
            polygon_vertices = len(analysis.visibility_boundary_polygon)
        if hasattr(analysis, 'visibility_rays') and analysis.visibility_rays:
            ray_count = len(analysis.visibility_rays)
        
        print(f"Successfully saved visibility polygon to {filename}")
        print(f"  Agent position: ({analysis.agent_position[0]:.1f}, {analysis.agent_position[1]:.1f})")
        if hasattr(analysis, 'agent_orientation') and analysis.agent_orientation is not None:
            print(f"  Agent orientation: {math.degrees(analysis.agent_orientation):.1f}°")
        print(f"  Visibility range: {analysis.visibility_range:.0f} pixels")
        print(f"  Polygon vertices: {polygon_vertices}")
        print(f"  Visibility rays: {ray_count}")
        
    except Exception as e:
        print(f"Error saving visibility polygon to file: {e}")


def load_visibility_polygon_from_file(filename="visibility_polygon.txt"):
    """
    Load visibility polygon and agent state from a text file.
    
    Args:
        filename: Input filename (default: "visibility_polygon.txt")
        
    Returns:
        Dictionary with 'agent_position', 'agent_orientation', 'visibility_range', 
        'polygon', and 'rays' data
    """
    data = {
        'agent_position': None,
        'agent_orientation': None,
        'visibility_range': None,
        'polygon': [],
        'rays': []
    }
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                if parts[0] == 'AGENT_POSITION' and len(parts) >= 3:
                    data['agent_position'] = (float(parts[1]), float(parts[2]))
                elif parts[0] == 'AGENT_ORIENTATION' and len(parts) >= 2:
                    data['agent_orientation'] = float(parts[1])
                elif parts[0] == 'VISIBILITY_RANGE' and len(parts) >= 2:
                    data['visibility_range'] = float(parts[1])
                elif parts[0] == 'VERTEX' and len(parts) >= 3:
                    data['polygon'].append((float(parts[1]), float(parts[2])))
                elif parts[0] == 'RAY' and len(parts) >= 6:
                    angle = float(parts[1])
                    endpoint = (float(parts[2]), float(parts[3]))
                    distance = float(parts[4])
                    blocked = parts[5].lower() == 'true'
                    data['rays'].append((angle, endpoint, distance, blocked))
        
        print(f"Successfully loaded visibility polygon from {filename}")
        print(f"  Agent position: {data['agent_position']}")
        print(f"  Agent orientation: {data['agent_orientation']}")
        print(f"  Visibility range: {data['visibility_range']}")
        print(f"  Polygon vertices: {len(data['polygon'])}")
        print(f"  Visibility rays: {len(data['rays'])}")
        
        return data
        
    except Exception as e:
        print(f"Error loading visibility polygon from file: {e}")
        return data
