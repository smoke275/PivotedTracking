#!/usr/bin/env python3
"""
Grid Utilities
Functions for grid coordinate transformations, reachability grid creation and manipulation.
"""

import numpy as np
import math
import time


def create_true_reachability_grid(agent_x, agent_y, visibility_range, spatial_index=None, map_width=1280, map_height=720, map_grid_size=120):
    """
    Create a true reachability probability grid using the same grid structure as the map graph.
    Each grid cell represents the actual probability of the evader being reachable at that position.
    
    This simplified version only creates a 1D sparse representation for better performance and memory usage.
    
    Args:
        agent_x, agent_y: Evader position (center of visibility circle)
        visibility_range: Radius of the visibility circle
        spatial_index: Spatial index for checking map grid validity (optional)
        map_width: Width of the map (default 1280, matches ENVIRONMENT_WIDTH)
        map_height: Height of the map (default 720, matches ENVIRONMENT_HEIGHT)
        map_grid_size: Grid size matching the map graph (default 120, matches MAP_GRAPH_GRID_SIZE)
        
    Returns:
        Tuple of (probability_grid, grid_metadata) where:
        - probability_grid: numpy array of shape (cells_in_visibility, ) sparse representation
        - grid_metadata: dictionary with grid parameters, coordinate mappings, and valid_cells list
    """
    # Import environment constants to match map graph
    try:
        from simulation_config import ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT
        map_width, map_height = ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT
    except ImportError:
        pass  # Use defaults
    
    try:
        from multitrack.utils.config import MAP_GRAPH_GRID_SIZE
        map_grid_size = MAP_GRAPH_GRID_SIZE
    except ImportError:
        pass  # Use default
    
    # Calculate map graph cell dimensions (same as in MapGraph class)
    cell_width = map_width / map_grid_size
    cell_height = map_height / map_grid_size
    
    # Calculate visibility circle bounding box
    min_x = agent_x - visibility_range
    max_x = agent_x + visibility_range
    min_y = agent_y - visibility_range  
    max_y = agent_y + visibility_range
    
    # Find which map graph cells fall within the visibility circle
    valid_cells = []
    cell_probabilities = []
    valid_points_count = 0
    circle_center_x, circle_center_y = agent_x, agent_y
    circle_radius_sq = visibility_range ** 2
    
    # Iterate through all map graph cells
    for i in range(map_grid_size):
        for j in range(map_grid_size):
            # Calculate position at center of map graph cell (same logic as MapGraph._sample_nodes)
            world_x = (i + 0.5) * cell_width
            world_y = (j + 0.5) * cell_height
            
            # Check if this cell center is inside the visibility circle
            dx = world_x - circle_center_x
            dy = world_y - circle_center_y
            distance_sq = dx * dx + dy * dy
            
            if distance_sq <= circle_radius_sq:
                # Cell is inside visibility circle
                valid_points_count += 1
                
                # Check if this corresponds to a valid map graph position
                if spatial_index is not None:
                    # Use spatial index to check if there's a valid map node at this position
                    node_idx = spatial_index.get_node_by_coordinates(world_x, world_y)
                    if node_idx is not None:
                        # Valid map graph position - set probability to 1.0 (pure green)
                        probability = 1.0
                    else:
                        # No valid map node at this position - probability is 0.0
                        probability = 0.0
                else:
                    # No spatial index available - set all points in circle to 1.0 (pure green)
                    probability = 1.0
                
                # Store cell information
                valid_cells.append({
                    'map_grid_i': i,
                    'map_grid_j': j,
                    'world_x': world_x,
                    'world_y': world_y,
                    'distance_from_agent': math.sqrt(distance_sq),
                    'probability': probability
                })
                cell_probabilities.append(probability)
    
    # Create numpy array for probabilities (1D sparse representation)
    probability_grid = np.array(cell_probabilities, dtype=np.float32)
    
    # Create metadata for coordinate transformations and rendering
    grid_metadata = {
        'map_grid_size': map_grid_size,
        'map_width': map_width,
        'map_height': map_height,
        'cell_width': cell_width,
        'cell_height': cell_height,
        'visibility_bounding_box': (min_x, min_y, max_x, max_y),
        'agent_position': (agent_x, agent_y),
        'visibility_range': visibility_range,
        'valid_cells': valid_cells,
        'valid_points_count': valid_points_count,
        'total_cells_in_circle': len(valid_cells),
        'cells_with_probability_1': sum(1 for p in cell_probabilities if p > 0.9),
        'grid_coverage_percent': (len(valid_cells) / (map_grid_size * map_grid_size)) * 100,
        'reachable_coverage_percent': (sum(1 for p in cell_probabilities if p > 0.9) / len(valid_cells) * 100) if valid_cells else 0,
        'creation_timestamp': time.time()
    }
    
    return probability_grid, grid_metadata


def world_coords_to_direct_grid_indices(world_x, world_y, map_width=1280, map_height=720, map_grid_size=120):
    """
    Convert world coordinates to direct grid indices for the 2D numpy array.
    This provides O(1) direct access: grid[i][j] where (0,0) is top-left, (119,119) is bottom-right.
    
    Args:
        world_x, world_y: World coordinates (pixels)
        map_width: Width of the map (default 1280)
        map_height: Height of the map (default 720) 
        map_grid_size: Grid size (default 120)
        
    Returns:
        Tuple of (grid_i, grid_j) where:
        - grid_i: Row index in the 2D array (0 to 119)
        - grid_j: Column index in the 2D array (0 to 119)
        - Returns (None, None) if coordinates are outside the map bounds
    """
    try:
        from simulation_config import ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT
        map_width, map_height = ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT
    except ImportError:
        pass
    
    try:
        from multitrack.utils.config import MAP_GRAPH_GRID_SIZE
        map_grid_size = MAP_GRAPH_GRID_SIZE
    except ImportError:
        pass
    
    # Check bounds
    if world_x < 0 or world_x >= map_width or world_y < 0 or world_y >= map_height:
        return (None, None)
    
    # Calculate cell dimensions
    cell_width = map_width / map_grid_size
    cell_height = map_height / map_grid_size
    
    # Direct conversion: world coordinate to grid index
    # The map graph uses (i + 0.5) * cell_width for world_x, so we reverse this:
    # world_x = (i + 0.5) * cell_width  =>  i = (world_x / cell_width) - 0.5
    grid_i = int(world_x / cell_width)
    grid_j = int(world_y / cell_height)
    
    # Clamp to valid range
    grid_i = max(0, min(grid_i, map_grid_size - 1))
    grid_j = max(0, min(grid_j, map_grid_size - 1))
    
    return (grid_i, grid_j)


def direct_grid_indices_to_world_coords(grid_i, grid_j, map_width=1280, map_height=720, map_grid_size=120):
    """
    Convert direct grid indices to world coordinates (center of the cell).
    This is the inverse of world_coords_to_direct_grid_indices.
    
    Args:
        grid_i, grid_j: Grid indices (0 to 119)
        map_width: Width of the map (default 1280)
        map_height: Height of the map (default 720)
        map_grid_size: Grid size (default 120)
        
    Returns:
        Tuple of (world_x, world_y) - center coordinates of the grid cell
    """
    try:
        from simulation_config import ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT
        map_width, map_height = ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT
    except ImportError:
        pass
    
    try:
        from multitrack.utils.config import MAP_GRAPH_GRID_SIZE
        map_grid_size = MAP_GRAPH_GRID_SIZE
    except ImportError:
        pass
    
    # Calculate cell dimensions
    cell_width = map_width / map_grid_size
    cell_height = map_height / map_grid_size
    
    # Convert grid index to world coordinate (center of cell)
    # This matches the map graph logic: world_x = (i + 0.5) * cell_width
    world_x = (grid_i + 0.5) * cell_width
    world_y = (grid_j + 0.5) * cell_height
    
    return (world_x, world_y)


def get_grid_value_at_world_coords(world_x, world_y, probability_grid_2d, map_width=1280, map_height=720, map_grid_size=120):
    """
    Get the probability value directly from the 2D grid using world coordinates.
    This combines coordinate conversion and array access in one function.
    
    Args:
        world_x, world_y: World coordinates
        probability_grid_2d: The 2D numpy array (120x120) - can be either rendering or clean grid
        map_width, map_height, map_grid_size: Grid parameters
        
    Returns:
        float: Probability value at that location, or None if outside bounds
        For rendering grid: -1.0 = outside visibility circle, 0.0 = inside circle but invalid, 1.0 = valid position
        For clean grid: NaN = outside visibility circle, 0.0 = inside circle but invalid, 1.0 = valid position
    """
    grid_i, grid_j = world_coords_to_direct_grid_indices(world_x, world_y, map_width, map_height, map_grid_size)
    
    if grid_i is None or grid_j is None:
        return None
    
    return probability_grid_2d[grid_i, grid_j]


def get_clean_grid_value_at_world_coords(world_x, world_y, evader_analysis, map_width=1280, map_height=720, map_grid_size=120):
    """
    Get the probability value from the sparse grid representation.
    This function is deprecated since 2D grids were removed for simplification.
    
    Args:
        world_x, world_y: World coordinates
        evader_analysis: EvaderAnalysis object containing the grid data
        map_width, map_height, map_grid_size: Grid parameters
        
    Returns:
        float: Probability value (0.0-1.0) or None if not found
    """
    if not hasattr(evader_analysis, 'true_reachability_metadata') or evader_analysis.true_reachability_metadata is None:
        return None
    
    # Search through valid_cells for the matching coordinates
    valid_cells = evader_analysis.true_reachability_metadata.get('valid_cells', [])
    
    # Convert world coordinates to grid indices
    grid_i, grid_j = world_coords_to_direct_grid_indices(world_x, world_y, map_width, map_height, map_grid_size)
    if grid_i is None or grid_j is None:
        return None
    
    # Find matching cell
    for cell in valid_cells:
        if cell['map_grid_i'] == grid_i and cell['map_grid_j'] == grid_j:
            return cell['probability']
    
    return None


def set_grid_value_at_world_coords(world_x, world_y, value, probability_grid_2d, map_width=1280, map_height=720, map_grid_size=120):
    """
    Set the probability value directly in the 2D grid using world coordinates.
    This combines coordinate conversion and array assignment in one function.
    
    Args:
        world_x, world_y: World coordinates
        value: New probability value to set
        probability_grid_2d: The 2D numpy array (120x120) - modified in place
        map_width, map_height, map_grid_size: Grid parameters
        
    Returns:
        bool: True if successful, False if outside bounds
    """
    grid_i, grid_j = world_coords_to_direct_grid_indices(world_x, world_y, map_width, map_height, map_grid_size)
    
    if grid_i is None or grid_j is None:
        return False
    
    probability_grid_2d[grid_i, grid_j] = value
    return True


def set_clean_grid_value_at_world_coords(world_x, world_y, value, evader_analysis, map_width=1280, map_height=720, map_grid_size=120):
    """
    Set the probability value in the sparse grid representation.
    This function is deprecated since 2D grids were removed for simplification.
    
    Args:
        world_x, world_y: World coordinates
        value: New probability value to set (should be 0.0-1.0)
        evader_analysis: EvaderAnalysis object containing the grid data
        map_width, map_height, map_grid_size: Grid parameters
        
    Returns:
        bool: True if successful, False if outside bounds or grids not available
    """
    if not hasattr(evader_analysis, 'true_reachability_metadata') or evader_analysis.true_reachability_metadata is None:
        return False
    
    # Convert world coordinates to grid indices
    grid_i, grid_j = world_coords_to_direct_grid_indices(world_x, world_y, map_width, map_height, map_grid_size)
    if grid_i is None or grid_j is None:
        return False
    
    # Search through valid_cells and update if found
    valid_cells = evader_analysis.true_reachability_metadata.get('valid_cells', [])
    
    # Clamp value to valid probability range
    clamped_value = max(0.0, min(1.0, value))
    
    # Find and update matching cell
    for cell in valid_cells:
        if cell['map_grid_i'] == grid_i and cell['map_grid_j'] == grid_j:
            cell['probability'] = clamped_value
            return True
    
    return False


def world_coords_to_grid_indices(world_x, world_y, grid_metadata):
    """
    Convert world coordinates to map graph cell indices.
    
    Args:
        world_x, world_y: World coordinates
        grid_metadata: Grid metadata from create_true_reachability_grid()
        
    Returns:
        Tuple of (map_grid_i, map_grid_j) or None if outside map bounds or not in valid cells
    """
    map_grid_size = grid_metadata['map_grid_size']
    cell_width = grid_metadata['cell_width']
    cell_height = grid_metadata['cell_height']
    
    # Calculate which map graph cell this world coordinate falls into
    map_grid_i = int(world_x / cell_width)
    map_grid_j = int(world_y / cell_height)
    
    # Check bounds
    if map_grid_i < 0 or map_grid_i >= map_grid_size or map_grid_j < 0 or map_grid_j >= map_grid_size:
        return None
    
    # Check if this cell is in our valid cells list
    for cell in grid_metadata['valid_cells']:
        if cell['map_grid_i'] == map_grid_i and cell['map_grid_j'] == map_grid_j:
            return map_grid_i, map_grid_j
    
    return None


def grid_indices_to_world_coords(map_grid_i, map_grid_j, grid_metadata):
    """
    Convert map graph cell indices to world coordinates (center of cell).
    
    Args:
        map_grid_i, map_grid_j: Map graph cell indices
        grid_metadata: Grid metadata from create_true_reachability_grid()
        
    Returns:
        Tuple of (world_x, world_y)
    """
    cell_width = grid_metadata['cell_width']
    cell_height = grid_metadata['cell_height']
    
    # Calculate world coordinates at center of map graph cell (same as MapGraph logic)
    world_x = (map_grid_i + 0.5) * cell_width
    world_y = (map_grid_j + 0.5) * cell_height
    
    return world_x, world_y


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
    
    # Import reachability utils
    try:
        from reachability_utils import get_reachability_probabilities_for_fixed_grid
    except ImportError:
        print("Warning: reachability_utils not available for clipped reachability grid")
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
