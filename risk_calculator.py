#!/usr/bin/env python3
"""
Risk Calculator
Handles reachability calculations and mask transformations.
"""

import pickle
import numpy as np

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
