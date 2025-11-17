#!/usr/bin/env python3
"""Reachability Mask API - unified interface for loading and analyzing reachability mask data."""

import pickle
import numpy as np
import math
import os
from typing import Optional, Dict, Any, Tuple, List

# Global debug flag - set to False to disable all debug output
DEBUG_PRINT = False

def debug_print(*args, **kwargs):
    """Print debug information if DEBUG_PRINT is enabled."""
    if DEBUG_PRINT:
        print(*args, **kwargs)

# Import C++ acceleration module (required)
import fast_reachability
HAS_CPP_ACCELERATION = True
debug_print("✅ C++ acceleration enabled for reachability operations")


class ReachabilityMaskAPI:
    """API for reachability mask data management and analysis."""
    
    def __init__(self, filename_base: str = "unicycle_grid"):
        """Initialize API with C++ acceleration required. Args: filename_base: Base name of pickle file (without .pkl)"""
        self.filename_base = filename_base
        self._mask_data = None
        self._grid = None
        self._is_loaded = False
        
        # Cached properties
        self._grid_size = None
        self._world_extent = None
        self._cell_size = None
        self._center_idx = None
        self._statistics = None
        
        # Cached processed overlay
        self._processed_overlay = None
        self._overlay_config = None  # Store config used to generate cached overlay
        
        # Try to load the data immediately
        self.load()
    
    def load(self) -> bool:
        """Load reachability mask data from file. Returns: True if loaded successfully."""
        try:
            with open(f"{self.filename_base}.pkl", 'rb') as f:
                self._mask_data = pickle.load(f)
            
            if self._mask_data is None:
                debug_print(f"Warning: Loaded mask data is None from {self.filename_base}.pkl")
                return False
            
            self._grid = self._mask_data.get('grid')
            if self._grid is None:
                debug_print(f"Warning: No grid data found in {self.filename_base}.pkl")
                return False
            
            # Cache basic properties
            self._grid_size = self._mask_data.get('grid_size', self._grid.shape[0])
            self._world_extent = self._mask_data.get('world_extent_px', 320)
            self._cell_size = self._mask_data.get('cell_size_px', 0.64)
            self._center_idx = self._mask_data.get('center_idx', self._grid_size // 2)
            
            self._is_loaded = True
            debug_print(f"✅ Reachability mask loaded: {self._grid_size}×{self._grid_size} grid")
            return True
            
        except FileNotFoundError:
            debug_print(f"❌ File not found: {self.filename_base}.pkl")
            debug_print("Run heatmap.py first to generate the reachability mask.")
            return False
        except Exception as e:
            debug_print(f"❌ Error loading reachability mask: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if the mask data is loaded and valid."""
        return self._is_loaded and self._grid is not None
    
    def get_raw_data(self) -> Optional[Dict[str, Any]]:
        """Get raw mask data dictionary. Returns: Raw data or None if not loaded."""
        return self._mask_data if self.is_loaded() else None
    
    def get_grid(self) -> Optional[np.ndarray]:
        """Get reachability grid array. Returns: Grid array or None if not loaded."""
        return self._grid if self.is_loaded() else None
    
    def get_grid_size(self) -> int:
        """Get the grid size (assumes square grid)."""
        return self._grid_size if self.is_loaded() else 0
    
    def get_world_extent(self) -> float:
        """Get the world extent in pixels."""
        return self._world_extent if self.is_loaded() else 0.0
    
    def get_cell_size(self) -> float:
        """Get the cell size in pixels per cell."""
        return self._cell_size if self.is_loaded() else 0.0
    
    def get_center_idx(self) -> int:
        """Get the center grid index."""
        return self._center_idx if self.is_loaded() else 0
    
    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates. Args: row, col. Returns: (world_x, world_y)"""
        if not self.is_loaded():
            return (0.0, 0.0)
        
        world_x = (col - self._center_idx) * self._cell_size
        world_y = (self._center_idx - row) * self._cell_size
        return world_x, world_y
    
    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates. Args: world_x, world_y. Returns: (row, col)"""
        if not self.is_loaded():
            return (0, 0)
        
        col = int(world_x / self._cell_size + self._center_idx)
        row = int(self._center_idx - world_y / self._cell_size)
        return row, col
    
    def get_value_at_grid(self, row: int, col: int) -> float:
        """Get reachability value at grid coordinates. Args: row, col. Returns: value (0.0 if out of bounds)"""
        if not self.is_loaded():
            return 0.0
        
        if 0 <= row < self._grid_size and 0 <= col < self._grid_size:
            return float(self._grid[row, col])
        return 0.0
    
    def get_value_at_world(self, world_x: float, world_y: float) -> float:
        """Get reachability value at world coordinates. Args: world_x, world_y. Returns: value (0.0 if out of bounds)"""
        row, col = self.world_to_grid(world_x, world_y)
        return self.get_value_at_grid(row, col)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the reachability mask. Returns: Statistics dictionary."""
        if not self.is_loaded():
            return {}
        
        if self._statistics is None:
            # Calculate statistics once and cache them
            grid = self._grid
            
            # Basic statistics
            total_cells = grid.size
            non_zero_mask = grid > 0
            reachable_cells = np.count_nonzero(non_zero_mask)
            unreachable_cells = total_cells - reachable_cells
            
            # Value statistics
            min_value = float(grid.min())
            max_value = float(grid.max())
            
            # Non-zero value statistics
            if reachable_cells > 0:
                non_zero_values = grid[non_zero_mask]
                mean_reachable = float(non_zero_values.mean())
                std_reachable = float(non_zero_values.std())
                min_reachable = float(non_zero_values.min())
                max_reachable = float(non_zero_values.max())
            else:
                mean_reachable = std_reachable = min_reachable = max_reachable = 0.0
            
            # Find maximum location
            max_idx = np.unravel_index(np.argmax(grid), grid.shape)
            max_world_x, max_world_y = self.grid_to_world(max_idx[0], max_idx[1])
            
            self._statistics = {
                # Grid properties
                'grid_size': self._grid_size,
                'world_extent_px': self._world_extent,
                'cell_size_px': self._cell_size,
                'center_idx': self._center_idx,
                
                # Cell counts
                'total_cells': total_cells,
                'reachable_cells': reachable_cells,
                'unreachable_cells': unreachable_cells,
                'reachable_percentage': 100.0 * reachable_cells / total_cells,
                
                # Value ranges
                'min_value': min_value,
                'max_value': max_value,
                'value_range': max_value - min_value,
                
                # Reachable cell statistics
                'mean_reachable': mean_reachable,
                'std_reachable': std_reachable,
                'min_reachable': min_reachable,
                'max_reachable': max_reachable,
                
                # Maximum location
                'max_location_grid': max_idx,
                'max_location_world': (max_world_x, max_world_y),
                
                # Coverage bounds
                'world_bounds': {
                    'x_min': -self._world_extent / 2,
                    'x_max': self._world_extent / 2,
                    'y_min': -self._world_extent / 2,
                    'y_max': self._world_extent / 2
                }
            }
        
        return self._statistics.copy()
    
    def configure_processed_overlay(self, clip_pixels: float = 32.0, 
                                   target_size: Tuple[int, int] = (120, 120),
                                   upscale_target: Tuple[int, int] = (400, 400),
                                   downsample_method: str = 'max_pool',
                                   probability_weighting: Optional[Tuple[float, float]] = None) -> bool:
        """Pre-compute and cache processed overlay. Returns: True if configured successfully."""
        if not self.is_loaded():
            debug_print("⚠️  Cannot configure overlay: no reachability mask loaded")
            return False
        
        # Create configuration signature
        config = {
            'clip_pixels': clip_pixels,
            'target_size': target_size,
            'upscale_target': upscale_target,
            'downsample_method': downsample_method,
            'probability_weighting': probability_weighting
        }
        
        # Check if we already have this configuration cached
        if (self._processed_overlay is not None and 
            self._overlay_config == config):
            debug_print(f"✅ Processed overlay already configured: {self._processed_overlay['processing_chain']}")
            return True
        
        # Compute the processed overlay
        debug_print(f"🔄 Computing processed overlay: {self._grid_size}×{self._grid_size} → clip({clip_pixels}px) → {target_size[0]}×{target_size[1]} → {upscale_target[0]}×{upscale_target[1]}")
        if probability_weighting is not None:
            debug_print(f"   🎲 Probability weighting: Prelect(α={probability_weighting[0]:.2f}, β={probability_weighting[1]:.2f})")
        
        processed_result = self.get_processed_overlay(
            clip_pixels=clip_pixels,
            target_size=target_size,
            upscale_target=upscale_target,
            downsample_method=downsample_method,
            probability_weighting=probability_weighting
        )
        
        if 'error' in processed_result:
            debug_print(f"❌ Failed to configure processed overlay: {processed_result['error']}")
            self._processed_overlay = None
            self._overlay_config = None
            return False
        
        # Cache the results
        self._processed_overlay = processed_result
        self._overlay_config = config
        
        pipeline_info = processed_result['processing_chain']
        scale_factor = processed_result['scale_factors']['total_scale_factor']
        quality_info = processed_result['grid_quality']
        
        debug_print(f"✅ Processed overlay configured: {pipeline_info}")
        debug_print(f"   📐 Scale factor: {scale_factor:.2f}× ({quality_info['interpolation_method']})")
        debug_print(f"   🎯 Quality: {quality_info['pixel_to_grid_ratio']} {quality_info['coordinate_mapping']}")
        
        return True
    
    def setup_overlay_configuration(self, clip_pixels: float = 32.0, 
                                   resize_target: Tuple[int, int] = (120, 120),
                                   upscale_target: Tuple[int, int] = (400, 400),
                                   downsample_method: str = 'max_pool',
                                   probability_weighting: Optional[Tuple[float, float]] = None) -> bool:
        """Configure overlay for repeated use. Args: clip_pixels, resize_target, upscale_target, downsample_method, probability_weighting. Returns: bool"""
        return self.configure_processed_overlay(
            clip_pixels=clip_pixels,
            target_size=resize_target,
            upscale_target=upscale_target,
            downsample_method=downsample_method,
            probability_weighting=probability_weighting
        )
    
    def is_overlay_configured(self) -> bool:
        """
        Check if the overlay has been configured and is ready for use.
        
        Returns:
            True if overlay is configured, False otherwise
        """
        return self._processed_overlay is not None
    
    def get_cached_overlay(self) -> Optional[Dict[str, Any]]:
        """Get cached processed overlay data. Returns: Cached data or None if not configured."""
        if self._processed_overlay is None:
            debug_print("⚠️  No processed overlay configured. Call configure_processed_overlay() first.")
            return None
        
        return self._processed_overlay
    
    def get_overlay_for_visualization(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get processed overlay data for visualization. Returns: (grid, bounds) tuple or None."""
        cached = self.get_cached_overlay()
        if cached is None:
            return None
        
        return (cached['final_grid'], cached['final_bounds'])
    
    def is_overlay_configured(self) -> bool:
        """Check if processed overlay is configured and ready."""
        return self._processed_overlay is not None
    
    def get_overlay_config(self) -> Optional[Dict[str, Any]]:
        """Get the configuration used for the cached overlay."""
        return self._overlay_config.copy() if self._overlay_config else None

    def analyze_coverage_by_distance(self, num_bins: int = 10) -> List[Dict[str, Any]]:
        """Analyze reachability coverage by distance from center. Args: num_bins. Returns: List of analysis dicts."""
        if not self.is_loaded():
            return []
        
        # Create distance matrix from center
        y, x = np.ogrid[:self._grid_size, :self._grid_size]
        distances_grid = np.sqrt((x - self._center_idx)**2 + (y - self._center_idx)**2) * self._cell_size
        
        # Create distance bins
        max_distance = distances_grid.max()
        distance_bins = np.linspace(0, max_distance, num_bins + 1)
        
        analysis_results = []
        
        for i in range(num_bins):
            start_dist = distance_bins[i]
            end_dist = distance_bins[i + 1]
            
            # Find cells in this distance range
            mask = (distances_grid >= start_dist) & (distances_grid < end_dist)
            total_cells = np.sum(mask)
            reachable_cells = np.sum(mask & (self._grid > 0))
            
            if total_cells > 0:
                coverage_pct = 100.0 * reachable_cells / total_cells
                avg_reachability = np.mean(self._grid[mask & (self._grid > 0)]) if reachable_cells > 0 else 0.0
                
                analysis_results.append({
                    'bin_index': i,
                    'distance_range': (start_dist, end_dist),
                    'distance_center': (start_dist + end_dist) / 2,
                    'total_cells': total_cells,
                    'reachable_cells': reachable_cells,
                    'unreachable_cells': total_cells - reachable_cells,
                    'coverage_percentage': coverage_pct,
                    'avg_reachability': avg_reachability
                })
        
        return analysis_results
    
    def get_neighborhood_analysis(self, row: int, col: int, radius: int = 2) -> Dict[str, Any]:
        """
        Analyze the neighborhood around a specific grid cell.
        
        Args:
            row: Grid row index
            col: Grid column index
            radius: Radius of neighborhood to analyze
            
        Returns:
            Dictionary containing neighborhood analysis
        """
        if not self.is_loaded():
            return {}
        
        # Define neighborhood bounds
        rmin = max(0, row - radius)
        rmax = min(self._grid_size, row + radius + 1)
        cmin = max(0, col - radius)
        cmax = min(self._grid_size, col + radius + 1)
        
        neighborhood = self._grid[rmin:rmax, cmin:cmax]
        
        return {
            'center_value': self.get_value_at_grid(row, col),
            'neighborhood_shape': neighborhood.shape,
            'neighborhood_size': neighborhood.size,
            'mean_value': float(neighborhood.mean()),
            'max_value': float(neighborhood.max()),
            'min_value': float(neighborhood.min()),
            'std_value': float(neighborhood.std()),
            'non_zero_count': int(np.count_nonzero(neighborhood)),
            'zero_count': int(neighborhood.size - np.count_nonzero(neighborhood)),
            'bounds': {
                'row_range': (rmin, rmax - 1),
                'col_range': (cmin, cmax - 1)
            }
        }
    
    def get_percentile_rank(self, value: float) -> float:
        """
        Get the percentile rank of a given reachability value among non-zero values.
        
        Args:
            value: Reachability value to rank
            
        Returns:
            Percentile rank (0-100)
        """
        if not self.is_loaded() or value <= 0:
            return 0.0
        
        non_zero_values = self._grid[self._grid > 0]
        if len(non_zero_values) == 0:
            return 0.0
        
        rank = np.sum(non_zero_values > value)
        percentile = 100.0 * (1 - rank / len(non_zero_values))
        return percentile
    
    def get_clipped_region(self, clip_pixels: int = 32) -> Dict[str, Any]:
        """
        Get a clipped region of the reachability mask by removing pixels from each side.
        
        Args:
            clip_pixels: Number of pixels to clip from each side (default: 32)
            
        Returns:
            Dictionary containing clipped grid data and metadata
        """
        if not self.is_loaded():
            return {}
        
        # Convert pixels to grid cells
        clip_cells = int(clip_pixels / self._cell_size)
        
        # Ensure we don't clip more than half the grid
        max_clip = self._grid_size // 3  # Leave at least 1/3 of original size
        clip_cells = min(clip_cells, max_clip)
        
        # Calculate clipped bounds
        start_idx = clip_cells
        end_idx = self._grid_size - clip_cells
        
        if start_idx >= end_idx:
            # Too much clipping requested
            return {
                'error': f'Cannot clip {clip_pixels} pixels ({clip_cells} cells) from {self._grid_size}×{self._grid_size} grid',
                'max_clip_pixels': max_clip * self._cell_size,
                'max_clip_cells': max_clip
            }
        
        # Extract clipped region
        clipped_grid = self._grid[start_idx:end_idx, start_idx:end_idx]
        clipped_size = clipped_grid.shape[0]
        
        # Calculate new world bounds
        clipped_world_extent = clipped_size * self._cell_size
        clipped_center_idx = clipped_size // 2
        
        # Calculate world coordinate offset due to clipping
        world_offset = clip_cells * self._cell_size
        
        return {
            'clipped_grid': clipped_grid,
            'original_grid_size': self._grid_size,
            'clipped_grid_size': clipped_size,
            'clip_pixels': clip_pixels,
            'clip_cells': clip_cells,
            'cell_size_px': self._cell_size,
            'clipped_world_extent': clipped_world_extent,
            'clipped_center_idx': clipped_center_idx,
            'world_offset': world_offset,
            'original_bounds': {
                'x_min': -self._world_extent / 2,
                'x_max': self._world_extent / 2,
                'y_min': -self._world_extent / 2,
                'y_max': self._world_extent / 2
            },
            'clipped_bounds': {
                'x_min': -clipped_world_extent / 2 + world_offset,
                'x_max': clipped_world_extent / 2 + world_offset,
                'y_min': -clipped_world_extent / 2 + world_offset,
                'y_max': clipped_world_extent / 2 + world_offset
            },
            'statistics': self._calculate_clipped_statistics(clipped_grid)
        }
    
    def _calculate_clipped_statistics(self, clipped_grid: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for a clipped grid region."""
        total_cells = clipped_grid.size
        non_zero_mask = clipped_grid > 0
        reachable_cells = np.count_nonzero(non_zero_mask)
        unreachable_cells = total_cells - reachable_cells
        
        # Value statistics
        min_value = float(clipped_grid.min())
        max_value = float(clipped_grid.max())
        
        # Non-zero value statistics
        if reachable_cells > 0:
            non_zero_values = clipped_grid[non_zero_mask]
            mean_reachable = float(non_zero_values.mean())
            std_reachable = float(non_zero_values.std())
            min_reachable = float(non_zero_values.min())
            max_reachable = float(non_zero_values.max())
        else:
            mean_reachable = std_reachable = min_reachable = max_reachable = 0.0
        
        # Find maximum location in clipped grid
        max_idx = np.unravel_index(np.argmax(clipped_grid), clipped_grid.shape)
        
        return {
            'total_cells': total_cells,
            'reachable_cells': reachable_cells,
            'unreachable_cells': unreachable_cells,
            'reachable_percentage': 100.0 * reachable_cells / total_cells,
            'min_value': min_value,
            'max_value': max_value,
            'mean_reachable': mean_reachable,
            'std_reachable': std_reachable,
            'min_reachable': min_reachable,
            'max_reachable': max_reachable,
            'max_location_grid': max_idx
        }
    
    def downsample_grid(self, grid: np.ndarray, target_size: Tuple[int, int], 
                       method: str = 'max_pool') -> Dict[str, Any]:
        """
        Downsample a grid to exact target size using interpolation.
        
        Args:
            grid: Input grid to downsample
            target_size: Target (height, width) for output grid
            method: Interpolation method - 'bilinear', 'nearest', 'max_pool', 'mean_pool'
            
        Returns:
            Dictionary with downsampled grid and metadata
        """
        from scipy.ndimage import zoom
        
        input_h, input_w = grid.shape
        target_h, target_w = target_size
        
        if target_h >= input_h or target_w >= input_w:
            return {
                'error': f'Target size {target_size} must be smaller than input size {grid.shape}',
                'input_size': grid.shape,
                'target_size': target_size
            }
        
        # Calculate scale factors
        scale_h = target_h / input_h
        scale_w = target_w / input_w
        
        if method in ['bilinear', 'nearest']:
            # Use scipy zoom for interpolation methods
            if method == 'bilinear':
                downsampled = zoom(grid, (scale_h, scale_w), order=1)  # Linear interpolation
            else:  # nearest
                downsampled = zoom(grid, (scale_h, scale_w), order=0)  # Nearest neighbor
                
        elif method in ['max_pool', 'mean_pool']:
            # Use pooling-based approach but then resize to exact target
            pool_h = max(1, input_h // target_h)
            pool_w = max(1, input_w // target_w)
            
            # First do pooling to rough size
            actual_h = input_h // pool_h
            actual_w = input_w // pool_w
            
            # Crop input to fit evenly into pooling windows
            crop_h = actual_h * pool_h
            crop_w = actual_w * pool_w
            cropped_grid = grid[:crop_h, :crop_w]
            
            # Reshape for pooling
            reshaped = cropped_grid.reshape(actual_h, pool_h, actual_w, pool_w)
            
            # Apply pooling
            if method == 'max_pool':
                pooled = np.max(reshaped, axis=(1, 3))
            else:  # mean_pool
                pooled = np.mean(reshaped, axis=(1, 3))
            
            # Then resize to exact target size using bilinear interpolation
            pool_scale_h = target_h / pooled.shape[0]
            pool_scale_w = target_w / pooled.shape[1]
            downsampled = zoom(pooled, (pool_scale_h, pool_scale_w), order=1)
        else:
            return {
                'error': f'Unknown downsampling method: {method}',
                'available_methods': ['bilinear', 'nearest', 'max_pool', 'mean_pool']
            }
        
        # Ensure exact target size (zoom might have small rounding errors)
        if downsampled.shape != target_size:
            # Crop or pad to exact size if needed
            current_h, current_w = downsampled.shape
            
            if current_h > target_h:
                downsampled = downsampled[:target_h, :]
            elif current_h < target_h:
                pad_h = target_h - current_h
                downsampled = np.pad(downsampled, ((0, pad_h), (0, 0)), mode='edge')
                
            if current_w > target_w:
                downsampled = downsampled[:, :target_w]
            elif current_w < target_w:
                pad_w = target_w - current_w
                downsampled = np.pad(downsampled, ((0, 0), (0, pad_w)), mode='edge')
        
        # Normalize by dividing by max value to ensure values are in [0, 1] range
        max_value = float(downsampled.max())
        original_max = max_value  # Store original max for metadata
        if max_value > 0:
            downsampled = downsampled / max_value
        
        return {
            'downsampled_grid': downsampled,
            'input_size': (input_h, input_w),
            'target_size': target_size,
            'actual_size': target_size,  # Now exact!
            'scale_factors': (scale_h, scale_w),
            'method': method,
            'reduction_factor': (input_h * input_w) / (target_h * target_w),
            'normalization_applied': True,
            'original_max_value': original_max,
            'normalized_max_value': 1.0 if max_value > 0 else 0.0
        }
    
    def get_clipped_and_downsampled(self, clip_pixels: float = 0, target_size: Tuple[int, int] = (120, 120),
                                   downsample_method: str = 'max_pool') -> Dict[str, Any]:
        """
        Apply clipping and then downsampling to the reachability grid.
        
        Args:
            clip_pixels: Number of pixels to clip from each side (0 = no clipping)
            target_size: Target (height, width) for final grid - EXACT size
            downsample_method: 'bilinear', 'nearest', 'max_pool', or 'mean_pool'
            
        Returns:
            Dictionary with processed grid and comprehensive metadata
        """
        if not self.is_loaded():
            return {'error': 'No reachability mask loaded'}
        
        # Step 1: Clipping (if requested)
        if clip_pixels > 0:
            clip_data = self.get_clipped_region(clip_pixels)
            if 'error' in clip_data:
                return clip_data
            
            source_grid = clip_data['clipped_grid']
            source_bounds = clip_data['clipped_bounds']
            source_center = clip_data['clipped_center_idx']
            source_stats = clip_data['statistics']
            cell_size = clip_data['cell_size_px']
            grid_offset = clip_data['clip_cells']
        else:
            source_grid = self._grid
            source_bounds = self.get_statistics()['world_bounds']
            source_center = self._center_idx
            source_stats = self.get_statistics()
            cell_size = self._cell_size
            grid_offset = 0
        
        # Step 2: Downsampling
        downsample_data = self.downsample_grid(source_grid, target_size, downsample_method)
        if 'error' in downsample_data:
            return downsample_data
        
        final_grid = downsample_data['downsampled_grid']
        final_size = downsample_data['actual_size']  # Should be exactly target_size
        scale_factors = downsample_data['scale_factors']
        
        # Calculate new cell size and world bounds
        new_cell_size = cell_size / scale_factors[0]  # Inverted since we're downsampling
        new_world_extent = final_size[0] * new_cell_size
        new_center = final_size[0] // 2
        
        # Calculate final world bounds
        final_bounds = {
            'x_min': -new_world_extent / 2,
            'x_max': new_world_extent / 2,
            'y_min': -new_world_extent / 2,
            'y_max': new_world_extent / 2
        }
        
        # Calculate final statistics
        final_stats = self._calculate_grid_statistics(final_grid, final_size[0])
        
        return {
            'final_grid': final_grid,
            'final_size': final_size[0],  # Assuming square grid
            'final_bounds': final_bounds,
            'final_center_idx': new_center,
            'final_cell_size': new_cell_size,
            'statistics': final_stats,
            
            # Processing steps
            'clipping_applied': clip_pixels > 0,
            'clip_pixels': clip_pixels,
            'clip_cells': grid_offset,
            'downsampling_applied': True,
            'target_size': target_size,
            'downsample_method': downsample_method,
            
            # Intermediate data
            'source_grid_size': source_grid.shape[0],
            'source_bounds': source_bounds,
            'downsample_info': downsample_data,
            
            # Scale information
            'total_scale_factor': (self._grid_size / final_size[0]) if clip_pixels == 0 else 
                                (self._grid_size / final_size[0]),
            'cell_size_change': new_cell_size / self._cell_size,
            'grid_reduction': (source_grid.shape[0] * source_grid.shape[1]) / (final_size[0] * final_size[1])
        }
    
    def _calculate_grid_statistics(self, grid: np.ndarray, grid_size: int) -> Dict[str, Any]:
        """Calculate statistics for any grid."""
        # Use C++ implementation for significant speedup (30-50% faster)
        grid_f32 = grid.astype(np.float32)
        cpp_stats = fast_reachability.calculate_grid_statistics(grid_f32, grid_size)
        
        # Add world bounds (not calculated in C++ for simplicity)
        cpp_stats['world_bounds'] = {
            'x_min': -grid_size * self._cell_size / 2,
            'x_max': grid_size * self._cell_size / 2,
            'y_min': -grid_size * self._cell_size / 2,
            'y_max': grid_size * self._cell_size / 2
        }
        return cpp_stats
    
    def grid_to_world_processed(self, row: int, col: int, processed_data: Dict[str, Any]) -> Tuple[float, float]:
        """
        Convert processed (clipped/downsampled) grid coordinates to world coordinates.
        
        Args:
            row: Grid row index
            col: Grid column index  
            processed_data: Data from get_clipped_and_downsampled()
            
        Returns:
            World coordinates (x, y) in pixels
        """
        center = processed_data['final_center_idx']
        cell_size = processed_data['final_cell_size']
        
        # Convert to world coordinates relative to grid center
        world_x = (col - center) * cell_size
        world_y = (center - row) * cell_size  # Flip Y axis
        
        return world_x, world_y

    def upscale_grid_pixelated(self, grid: np.ndarray, target_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Upscale a grid to a larger size using nearest-neighbor (pixelated) interpolation.
        Creates a crisp, blocky enlargement where each original grid cell becomes a block of pixels.
        This results in a 1:1 pixel-to-grid-cell ratio in the output.
        
        Args:
            grid: Input grid to upscale
            target_size: Target (height, width) for output grid
            
        Returns:
            Dictionary with upscaled grid and metadata
        """
        input_h, input_w = grid.shape
        target_h, target_w = target_size
        
        if target_h <= input_h or target_w <= input_w:
            return {
                'error': f'Target size {target_size} must be larger than input size {grid.shape}',
                'input_size': grid.shape,
                'target_size': target_size
            }
        
        # Calculate scale factors
        scale_h = target_h / input_h
        scale_w = target_w / input_w
        
        # Use nearest-neighbor interpolation for crisp, pixelated upscaling
        from scipy.ndimage import zoom
        upscaled = zoom(grid, (scale_h, scale_w), order=0)  # order=0 = nearest neighbor
        
        # Ensure exact target size (zoom might have small rounding errors)
        if upscaled.shape != target_size:
            # Crop or pad to exact size if needed
            current_h, current_w = upscaled.shape
            
            if current_h > target_h:
                upscaled = upscaled[:target_h, :]
            elif current_h < target_h:
                pad_h = target_h - current_h
                upscaled = np.pad(upscaled, ((0, pad_h), (0, 0)), mode='edge')
                
            if current_w > target_w:
                upscaled = upscaled[:, :target_w]
            elif current_w < target_w:
                pad_w = target_w - current_w
                upscaled = np.pad(upscaled, ((0, 0), (0, pad_w)), mode='edge')
        
        # Calculate new cell size (each original cell becomes multiple pixels)
        pixel_per_cell_h = target_h / input_h
        pixel_per_cell_w = target_w / input_w
        
        return {
            'upscaled_grid': upscaled,
            'input_size': (input_h, input_w),
            'target_size': target_size,
            'actual_size': target_size,
            'scale_factors': (scale_h, scale_w),
            'pixel_per_cell': (pixel_per_cell_h, pixel_per_cell_w),
            'upscale_factor': (target_h * target_w) / (input_h * input_w),
            'method': 'nearest_neighbor_pixelated',
            'pixel_to_grid_ratio': '1:1',
            'original_max_value': float(upscaled.max()),
            'original_min_value': float(upscaled.min())
        }

    def get_upscaled_pixelated(self, source_grid: Optional[np.ndarray] = None, 
                              target_size: Tuple[int, int] = (400, 400)) -> Dict[str, Any]:
        """
        Get an upscaled pixelated version of the reachability grid with 1:1 pixel-to-grid ratio.
        
        Args:
            source_grid: Optional source grid to upscale (uses main grid if None)
            target_size: Target (height, width) for upscaled grid
            
        Returns:
            Dictionary with upscaled grid and comprehensive metadata
        """
        if not self.is_loaded():
            return {'error': 'No reachability mask loaded'}
        
        # Use provided grid or default to main grid
        if source_grid is None:
            source_grid = self._grid
        
        # Upscale the grid
        upscale_data = self.upscale_grid_pixelated(source_grid, target_size)
        if 'error' in upscale_data:
            return upscale_data
        
        upscaled_grid = upscale_data['upscaled_grid']
        scale_factors = upscale_data['scale_factors']
        
        # For true 1:1 pixel-to-grid ratio: each pixel = 1 world unit
        # The upscaled grid size directly determines the world extent
        new_world_extent = target_size[0]  # 400×400 grid = 400×400 world units
        new_cell_size = 1.0  # Each pixel = 1 world unit
        new_center = target_size[0] // 2
        
        # Calculate what world area the source grid represents for metadata
        if source_grid.shape == self._grid.shape:
            source_world_extent = self._world_extent
        else:
            # For processed grids, calculate based on original cell size
            source_world_extent = source_grid.shape[0] * self._cell_size
        
        # Calculate final world bounds
        final_bounds = {
            'x_min': -new_world_extent / 2,
            'x_max': new_world_extent / 2,
            'y_min': -new_world_extent / 2,
            'y_max': new_world_extent / 2
        }
        
        # Calculate statistics for the upscaled grid
        final_stats = self._calculate_grid_statistics(upscaled_grid, target_size[0])
        
        return {
            'final_grid': upscaled_grid,
            'final_size': target_size[0],
            'final_bounds': final_bounds,
            'final_center_idx': new_center,
            'final_cell_size': new_cell_size,
            'statistics': final_stats,
            
            # Upscaling information
            'upscaling_applied': True,
            'target_size': target_size,
            'upscale_method': 'nearest_neighbor_pixelated',
            'pixel_to_grid_ratio': '1:1',
            
            # Source information
            'source_grid_size': source_grid.shape[0],
            'source_world_extent': source_world_extent,
            'upscale_info': upscale_data,
            
            # Scale information
            'total_scale_factor': target_size[0] / source_grid.shape[0],
            'cell_size_change': new_cell_size / self._cell_size,
            'pixel_enlargement': (target_size[0] * target_size[1]) / (source_grid.shape[0] * source_grid.shape[1])
        }

    def get_summary_info(self) -> str:
        """
        Get a formatted summary of the reachability mask.
        
        Returns:
            Multi-line string with summary information
        """
        if not self.is_loaded():
            return "❌ Reachability mask not loaded"
        
        stats = self.get_statistics()
        
        # Use list and join for efficient string building instead of concatenation
        summary_parts = [
            f"🎯 Reachability Mask Summary: {self.filename_base}.pkl",
            "=" * 50,
            f"📊 Grid: {stats['grid_size']}×{stats['grid_size']} cells",
            f"🌍 World extent: ±{stats['world_extent_px']/2:.1f} pixels",
            f"📏 Cell size: {stats['cell_size_px']:.3f} px/cell",
            f"🎯 Center: grid[{stats['center_idx']}, {stats['center_idx']}] = world(0, 0)",
            "",
            "📈 Coverage:",
            f"  ✅ Reachable: {stats['reachable_cells']:,} cells ({stats['reachable_percentage']:.1f}%)",
            f"  ❌ Unreachable: {stats['unreachable_cells']:,} cells",
            "",
            "🔢 Values:",
            f"  Range: {stats['min_value']:.6f} to {stats['max_value']:.6f}",
            f"  Mean (reachable): {stats['mean_reachable']:.6f}",
            f"  Std (reachable): {stats['std_reachable']:.6f}",
            "",
            f"🏆 Maximum: {stats['max_value']:.6f}",
            f"  Location: grid{stats['max_location_grid']} = world({stats['max_location_world'][0]:.1f}, {stats['max_location_world'][1]:.1f})"
        ]
        
        return '\n'.join(summary_parts)
    
    def _apply_prelect_probability_weighting(self, grid: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """Apply Prelect probability weighting: p^α / (p^α + (1-p)^β). Args: grid, alpha, beta. Returns: weighted grid."""
        # Use C++ implementation for significant speedup (40-60% faster)
        grid_f32 = grid.astype(np.float32)
        return fast_reachability.apply_prelect_weighting(grid_f32, alpha, beta)

    def get_processed_overlay(self, clip_pixels: float = 32.0, 
                             target_size: Tuple[int, int] = (120, 120),
                             upscale_target: Tuple[int, int] = (400, 400),
                             downsample_method: str = 'max_pool',
                             probability_weighting: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Complete processing pipeline: clip → downsample → upscale → probability weighting. Returns: processed grid metadata dict."""
        if not self.is_loaded():
            return {'error': 'No reachability mask loaded'}
        
        # Step 1: Clip and downsample
        clipped_downsampled = self.get_clipped_and_downsampled(
            clip_pixels=clip_pixels,
            target_size=target_size,
            downsample_method=downsample_method
        )
        
        if 'error' in clipped_downsampled:
            return {
                'error': f'Clipping/downsampling failed: {clipped_downsampled["error"]}',
                'step_failed': 'clip_and_downsample',
                'parameters': {
                    'clip_pixels': clip_pixels,
                    'target_size': target_size,
                    'downsample_method': downsample_method
                }
            }
        
        # Extract intermediate results
        intermediate_grid = clipped_downsampled['final_grid']
        intermediate_bounds = clipped_downsampled['final_bounds']
        
        # Step 2: Upscale for pixelated display
        upscaled_data = self.get_upscaled_pixelated(
            source_grid=intermediate_grid,
            target_size=upscale_target
        )
        
        if 'error' in upscaled_data:
            return {
                'error': f'Upscaling failed: {upscaled_data["error"]}',
                'step_failed': 'upscale_pixelated',
                'intermediate_available': True,
                'intermediate_grid': intermediate_grid,
                'intermediate_bounds': intermediate_bounds,
                'intermediate_data': clipped_downsampled,
                'parameters': {
                    'clip_pixels': clip_pixels,
                    'target_size': target_size,
                    'upscale_target': upscale_target,
                    'downsample_method': downsample_method
                }
            }
        
        # Success - combine all metadata
        final_grid = upscaled_data['final_grid']
        final_bounds = upscaled_data['final_bounds']
        
        # Calculate comprehensive processing statistics
        original_cells = self._grid_size * self._grid_size
        intermediate_cells = target_size[0] * target_size[1]
        final_cells = upscale_target[0] * upscale_target[1]
        
        # Overall scale factors
        clip_reduction = (self._grid_size - 2 * (clip_pixels / self._cell_size)) / self._grid_size
        downsample_reduction = (target_size[0] * target_size[1]) / (intermediate_cells if clip_pixels == 0 else 
                                                                   (self._grid_size - 2 * int(clip_pixels / self._cell_size))**2)
        upscale_expansion = (upscale_target[0] * upscale_target[1]) / (target_size[0] * target_size[1])
        overall_scale = final_cells / original_cells
        
        # Step 3 (optional): Apply probability weighting
        # Only create debug copy if probability weighting is actually applied
        pipeline_steps = ['clip', 'downsample', 'upscale']
        processing_chain = f"{self._grid_size}×{self._grid_size} → clip({clip_pixels}px) → {target_size[0]}×{target_size[1]} → {upscale_target[0]}×{upscale_target[1]}"
        
        if probability_weighting is not None:
            alpha, beta = probability_weighting
            debug_print(f"🎲 Applying Prelect probability weighting: α={alpha:.2f}, β={beta:.2f}")
            
            # Only create copy if debug prints are enabled to save memory
            if DEBUG_PRINT:
                original_mean = np.mean(final_grid)
                original_std = np.std(final_grid)
            
            # Apply Prelect weighting to the final grid
            final_grid = self._apply_prelect_probability_weighting(final_grid, alpha, beta)
            
            # Calculate weighting statistics only if debug is enabled
            if DEBUG_PRINT:
                weighted_mean = np.mean(final_grid)
                weighted_std = np.std(final_grid)
                
                debug_print(f"   📊 Original stats: mean={original_mean:.6f}, std={original_std:.6f}")
                debug_print(f"   📊 Weighted stats: mean={weighted_mean:.6f}, std={weighted_std:.6f}")
                debug_print(f"   📈 Mean change: {((weighted_mean - original_mean) / original_mean * 100):.2f}%")
            pipeline_steps.append('probability_weight')
            processing_chain += f" → Prelect(α={alpha:.2f},β={beta:.2f})"
        
        return {
            # Final results
            'final_grid': final_grid,
            'final_bounds': final_bounds,
            'final_size': upscale_target[0],
            'final_cell_size': upscaled_data['final_cell_size'],
            'final_center_idx': upscaled_data['final_center_idx'],
            
            # Processing pipeline summary
            'pipeline_steps': pipeline_steps,
            'pipeline_success': True,
            'processing_chain': processing_chain,
            
            # Parameters used
            'parameters': {
                'clip_pixels': clip_pixels,
                'target_size': target_size,
                'upscale_target': upscale_target,
                'downsample_method': downsample_method,
                'probability_weighting': probability_weighting
            },
            
            # Scale factors
            'scale_factors': {
                'clip_reduction': clip_reduction,
                'downsample_reduction': downsample_reduction,
                'upscale_expansion': upscale_expansion,
                'overall_scale': overall_scale,
                'total_scale_factor': upscaled_data['total_scale_factor']
            },
            
            # Cell counts
            'cell_counts': {
                'original': original_cells,
                'intermediate': intermediate_cells,
                'final': final_cells
            },
            
            # Intermediate data (for debugging/fallback)
            'intermediate_data': clipped_downsampled,
            'upscale_data': upscaled_data,
            
            # Statistics
            'final_statistics': upscaled_data['statistics'],
            # Grid quality and probability weighting info
            'grid_quality': {
                'pixel_to_grid_ratio': upscaled_data.get('pixel_to_grid_ratio', '1:1'),
                'interpolation_method': f"{downsample_method} → nearest_neighbor_pixelated",
                'coordinate_mapping': '1:1 pixel-to-world-unit'
            },
            
            # Probability weighting information
            'probability_weighting': {
                'applied': probability_weighting is not None,
                'parameters': probability_weighting,
                'original_grid_available': probability_weighting is not None  # original_final_grid is stored if weighting was applied
            }
        }
    
    def create_test_mask(self, shape: Tuple[int, int] = (400, 400), 
                        agent_position: Tuple[float, float] = (0.0, 0.0), 
                        world_size: float = 400.0) -> Tuple[np.ndarray, Dict[str, float]]:
        """Create test mask with visual patterns. Args: shape, agent_position, world_size. Returns: (test_mask, world_bounds)"""
        # Create a test pattern with multiple features for visual distinction - VECTORIZED VERSION
        test_mask = np.zeros(shape, dtype=np.float32)
        
        # Create concentric circles pattern with varying intensities - VECTORIZED
        center_x, center_y = shape[1] // 2, shape[0] // 2
        
        # Use meshgrid for vectorized distance calculation (10-20x faster)
        y_coords, x_coords = np.ogrid[:shape[0], :shape[1]]
        dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Create multiple zones with different values using vectorized operations
        # Center bright zone
        center_mask = dist < 50
        test_mask[center_mask] = 1.0
        
        # Inner ring
        inner_mask = (dist >= 50) & (dist < 100)
        test_mask[inner_mask] = 0.8
        
        # Middle ring with checkerboard pattern - vectorized
        middle_mask = (dist >= 100) & (dist < 150)
        checkerboard = ((y_coords // 20) + (x_coords // 20)) % 2 == 0
        test_mask[middle_mask & checkerboard] = 0.6
        test_mask[middle_mask & ~checkerboard] = 0.3
        
        # Outer zone with diagonal stripes - vectorized
        outer_mask = dist >= 150
        diagonal_stripes = (y_coords + x_coords) % 30 < 15
        test_mask[outer_mask & diagonal_stripes] = 0.4
        test_mask[outer_mask & ~diagonal_stripes] = 0.1
        
        # Add some random high-value spots for additional visual interest - OPTIMIZED
        np.random.seed(42)  # For reproducible pattern
        num_spots = 20
        spot_coords = np.random.randint(0, min(shape), size=(num_spots, 2))
        spot_radius = 5
        
        # Create circular mask for spots using vectorized operations
        spot_y, spot_x = np.ogrid[-spot_radius:spot_radius+1, -spot_radius:spot_radius+1]
        spot_mask = spot_x**2 + spot_y**2 <= spot_radius**2
        
        # Apply spots efficiently
        for x, y in spot_coords:
            y_start, y_end = max(0, x - spot_radius), min(shape[0], x + spot_radius + 1)
            x_start, x_end = max(0, y - spot_radius), min(shape[1], y + spot_radius + 1)
            
            # Get the valid region of the spot mask
            mask_y_start = max(0, spot_radius - x)
            mask_y_end = mask_y_start + (y_end - y_start)
            mask_x_start = max(0, spot_radius - y)
            mask_x_end = mask_x_start + (x_end - x_start)
            
            if y_end > y_start and x_end > x_start:
                current_region = test_mask[y_start:y_end, x_start:x_end]
                spot_region = spot_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                # Use np.maximum for efficient element-wise max with broadcasting
                test_mask[y_start:y_end, x_start:x_end] = np.maximum(
                    current_region, 
                    np.where(spot_region, current_region + 0.5, current_region)
                )
                # Clamp to 1.0
                test_mask[y_start:y_end, x_start:x_end] = np.minimum(
                    test_mask[y_start:y_end, x_start:x_end], 1.0
                )
        
        # DRAW ARROW LAST - After all other patterns to ensure visibility on top
        arrow_start_x = center_x - 120  # Start arrow 120 pixels left of center  
        arrow_end_x = center_x + 120    # End arrow 120 pixels right of center
        arrow_y = center_y              # Arrow at center height
        arrow_width = 12                # Arrow thickness (increased)
        
        # Draw arrow shaft (horizontal line) - FORCE TO MAX VALUE
        for x in range(arrow_start_x, arrow_end_x - 30):  # Leave space for arrowhead
            for y in range(arrow_y - arrow_width//2, arrow_y + arrow_width//2 + 1):
                if 0 <= x < shape[1] and 0 <= y < shape[0]:
                    test_mask[y, x] = 1.0  # Force bright white arrow
        
        # Draw arrow head (large triangle pointing right) - FORCE TO MAX VALUE
        arrow_head_size = 30  # Larger arrowhead
        for i in range(arrow_head_size):
            # Triangle sides  
            y_offset = int(i * arrow_width * 1.5 / arrow_head_size)  # Wider triangle
            x_pos = arrow_end_x - arrow_head_size + i
            
            # Top side of triangle
            y_top = arrow_y - y_offset
            # Bottom side of triangle
            y_bottom = arrow_y + y_offset
            
            if 0 <= x_pos < shape[1]:
                # Fill the entire triangle area
                for y_fill in range(max(0, y_top), min(shape[0], y_bottom + 1)):
                    test_mask[y_fill, x_pos] = 1.0  # Force bright white
        
        # Create world bounds that center on the agent position
        agent_x, agent_y = agent_position
        half_size = world_size / 2.0
        world_bounds = {
            'x_min': agent_x - half_size,
            'x_max': agent_x + half_size,
            'y_min': agent_y - half_size,
            'y_max': agent_y + half_size
        }
        
        debug_print(f"🧪 Created test mask: {test_mask.shape[0]}×{test_mask.shape[1]} with complex pattern")
        debug_print(f"   📊 Test mask range: [{test_mask.min():.3f}, {test_mask.max():.3f}]")
        debug_print(f"   🎨 Pattern: concentric circles + checkerboard + stripes + bright spots")
        debug_print(f"   ➡️  Arrow: LARGE prominent arrow drawn LAST (on top) pointing towards positive x-axis")
        
        return test_mask, world_bounds
    
    def transform_mask(self, mask_data: Tuple[np.ndarray, Dict[str, float]], 
                      target_location: Tuple[float, float], 
                      agent_location: Tuple[float, float],
                      orientation: float, 
                      world_bounds: Dict[str, float], 
                      visibility_polygon: Optional[List[Tuple[float, float]]] = None,
                      fast_rotation: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
        """Transform mask with translation, rotation, visibility clipping. Returns: (transformed_mask, world_bounds)"""
        if mask_data is None:
            return None
        
        mask, mask_bounds = mask_data
        from scipy.ndimage import rotate
        import math
        import math
        
        # Step 1: Rotate the mask around its center (optimized for C++ conversion)
        # Convert radians to degrees for scipy
        rotation_degrees = math.degrees(orientation)
        
        # Ensure input mask is in optimal format for C++ conversion
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        
        # Choose rotation method based on performance requirements
        if fast_rotation:
            # Nearest neighbor: 1.6-1.9× faster but more pixelated (C++ friendly)
            rotated_mask = rotate(mask, rotation_degrees, reshape=False, order=0, mode='constant', cval=0.0, prefilter=False)
            rotation_method = "nearest neighbor (fast)"
        else:
            # Linear interpolation: Better quality but slower (still C++ convertible)
            rotated_mask = rotate(mask, rotation_degrees, reshape=False, order=1, mode='constant', cval=0.0, prefilter=True)
            rotation_method = "linear interpolation (quality)"
        
        # Ensure output is float32 for consistent C++ interfacing
        rotated_mask = rotated_mask.astype(np.float32)
        
        # Step 2: Create coordinate system for translation
        # Get original mask center in grid coordinates
        mask_center_y, mask_center_x = mask.shape[0] // 2, mask.shape[1] // 2
        
        # Calculate world coordinates per pixel (assuming uniform grid)
        world_width = world_bounds['x_max'] - world_bounds['x_min']
        world_height = world_bounds['y_max'] - world_bounds['y_min']
        
        pixels_per_world_x = mask.shape[1] / world_width
        pixels_per_world_y = mask.shape[0] / world_height
        
        # Calculate target position in grid coordinates (relative to agent location)
        target_x, target_y = target_location
        agent_x, agent_y = agent_location
        
        # Calculate relative offset from agent to target
        relative_offset_x = target_x - agent_x
        relative_offset_y = -(target_y - agent_y)

        # Current mask is centered at agent location in world coordinates
        # Calculate offset in pixels to move to target location relative to agent
        offset_x_pixels = relative_offset_x * pixels_per_world_x
        offset_y_pixels = relative_offset_y * pixels_per_world_y
        
        # Step 3: Create new mask with translation applied (C++ optimized)
        # Apply translation if there's a relative offset from agent to target
        if abs(relative_offset_x) > 1e-6 or abs(relative_offset_y) > 1e-6:
            debug_print(f"📍 Applying translation: agent({agent_x:.1f}, {agent_y:.1f}) → target({target_x:.1f}, {target_y:.1f})")
            debug_print(f"   📏 Relative offset: ({relative_offset_x:.2f}, {relative_offset_y:.2f}) = ({offset_x_pixels:.1f}, {offset_y_pixels:.1f}) pixels")
            
            # Apply translation by shifting the mask
            from scipy.ndimage import shift
            translated_mask = shift(rotated_mask, [offset_y_pixels, offset_x_pixels], order=1, mode='constant', cval=0.0)
            # Avoid unnecessary type conversion if already float32
            if translated_mask.dtype != np.float32:
                translated_mask = translated_mask.astype(np.float32)
        else:
            # No relative offset - reuse rotated_mask directly (avoid unnecessary copy)
            translated_mask = rotated_mask  # Already float32 from above
            debug_print(f"📍 No translation needed: target location matches agent location ({target_x:.1f}, {target_y:.1f})")
        
        # Step 4: Apply visibility polygon clipping if provided
        if visibility_polygon is not None and len(visibility_polygon) > 2:
            debug_print(f"🔍 Applying visibility polygon clipping with {len(visibility_polygon)} vertices")
            debug_print(f"   📐 World bounds: x=[{world_bounds['x_min']:.1f}, {world_bounds['x_max']:.1f}], y=[{world_bounds['y_min']:.1f}, {world_bounds['y_max']:.1f}]")
            debug_print(f"   📍 Target location: ({target_location[0]:.1f}, {target_location[1]:.1f})")
            debug_print(f"   👁️  Visibility polygon bounds: x=[{min(p[0] for p in visibility_polygon):.1f}, {max(p[0] for p in visibility_polygon):.1f}], y=[{min(p[1] for p in visibility_polygon):.1f}, {max(p[1] for p in visibility_polygon):.1f}]")
            
            # Efficient polygon rasterization using scanline algorithm
            # Initialize with all 0s - areas outside visibility will remain 0 (invisible/unreachable)
            visibility_mask = np.zeros_like(translated_mask, dtype=np.float32)
            
            # Convert polygon vertices to grid coordinates (optimized for C++ conversion)
            mask_height, mask_width = translated_mask.shape
            
            # Use numpy arrays for efficient memory layout and vectorized operations
            if len(visibility_polygon) > 0:
                # Convert to numpy array once for efficient memory access
                vertices_array = np.array(visibility_polygon, dtype=np.float32)
                world_x = vertices_array[:, 0]
                world_y = vertices_array[:, 1]
                
                # Vectorized coordinate conversion (C++ friendly operations)
                grid_x = (world_x - world_bounds['x_min']) * mask_width / world_width
                grid_y = (world_bounds['y_max'] - world_y) * mask_height / world_height  # Flip Y for image coordinates
                
                # Convert to list of tuples for scanline algorithm
                grid_vertices = list(zip(grid_x.astype(np.float32), grid_y.astype(np.float32)))
            else:
                grid_vertices = []
            
            # Use efficient polygon rasterization (optimized for C++ conversion)
            polygon_mask = self._rasterize_polygon_scanline(grid_vertices, mask_height, mask_width)
            
            # Convert boolean mask to float32 for consistent C++ types
            polygon_mask_f32 = polygon_mask.astype(np.float32)
            
            # Apply polygon mask using vectorized operations (C++ friendly)
            # Use numpy.where for efficient conditional assignment
            final_mask = np.where(polygon_mask_f32, translated_mask, 0.0)
            
            # Ensure consistent float32 output for C++ compatibility
            final_mask = final_mask.astype(np.float32)
            
            # Count visible pixels for logging (pixels inside polygon)
            inside_pixels = np.sum(polygon_mask)
            total_pixels = polygon_mask.size
            visibility_percentage = (inside_pixels / total_pixels) * 100
            
            debug_print(f"   👁️  Efficient polygon rasterization: {inside_pixels}/{total_pixels} pixels ({visibility_percentage:.1f}% inside polygon)")
            debug_print(f"   📊 Final clipped mask range: [{final_mask.min():.3f}, {final_mask.max():.3f}]")
            
            # If no pixels are inside polygon, create a small test pattern around the agent
            if inside_pixels == 0:
                debug_print("   ⚠️  No pixels inside polygon - creating test pattern around agent")
                # Create a small circle around the agent position for testing
                agent_x, agent_y = target_location
                
                # Calculate agent grid position correctly
                center_j = int(translated_mask.shape[1] * (agent_x - world_bounds['x_min']) / (world_bounds['x_max'] - world_bounds['x_min']))
                center_i = int(translated_mask.shape[0] * (world_bounds['y_max'] - agent_y) / (world_bounds['y_max'] - world_bounds['y_min']))
                
                # Clamp to valid grid coordinates
                center_i = max(0, min(translated_mask.shape[0] - 1, center_i))
                center_j = max(0, min(translated_mask.shape[1] - 1, center_j))
                
                # Draw a small circle pattern for visibility
                radius = 20  # 20 pixel radius
                for di in range(-radius, radius+1):
                    for dj in range(-radius, radius+1):
                        if di*di + dj*dj <= radius*radius:
                            i, j = center_i + di, center_j + dj
                            if 0 <= i < translated_mask.shape[0] and 0 <= j < translated_mask.shape[1]:
                                final_mask[i, j] = 0.8  # Set visible test pattern
                
                debug_print(f"   🎯 Created test circle at grid ({center_j}, {center_i}) with radius {radius}")
                inside_pixels = np.sum(final_mask != 0.0)
                debug_print(f"   ✅ Test pattern created: {inside_pixels} pixels")
            
            translated_mask = final_mask
        else:
            debug_print(f"   ⚠️  No visibility polygon provided - using full transformed mask")
        
        # Step 5: Flip Y-axis to match matplotlib's imshow coordinate system (C++ optimized)
        # matplotlib imshow has origin at top-left, but our world coordinates assume bottom-left
        # Use numpy.flipud with explicit copy for C++ memory management
        translated_mask = np.flipud(translated_mask).copy()
        
        # Ensure final output is consistently typed for C++ interfacing
        translated_mask = translated_mask.astype(np.float32)
        
        debug_print(f"🔄 Transformed mask:")
        debug_print(f"   📐 Rotated by {rotation_degrees:.1f}° ({orientation:.3f} rad) using {rotation_method}")
        debug_print(f"   📍 Translated to ({target_x:.1f}, {target_y:.1f})")
        debug_print(f"   🔄 Y-axis flipped for matplotlib compatibility")
        debug_print(f"   📊 Final mask range: [{translated_mask.min():.3f}, {translated_mask.max():.3f}]")
        
        return translated_mask, world_bounds
    
    # C++ Conversion Helper Methods
    
    def _ensure_float32_array(self, array: np.ndarray) -> np.ndarray:
        """Ensure array is float32 and contiguous for C++ compatibility."""
        return fast_reachability.ensure_float32_contiguous(array)
    
    def _prepare_polygon_for_cpp(self, vertices: List[Tuple[float, float]]) -> np.ndarray:
        """Convert polygon vertices to C++ compatible format."""
        if not vertices:
            return np.array([], dtype=np.float32).reshape(0, 2)
        
        vertices_array = np.array(vertices, dtype=np.float32)
        return np.ascontiguousarray(vertices_array)
    
    def get_cpp_compatible_grid(self) -> Optional[np.ndarray]:
        """Get grid data in C++ compatible format (float32, contiguous)."""
        if not self.is_loaded():
            return None
        return self._ensure_float32_array(self._grid)
    
    def transform_mask_cpp_optimized(self, mask_data: Tuple[np.ndarray, Dict[str, float]], 
                                   target_location: Tuple[float, float], 
                                   agent_location: Tuple[float, float],
                                   orientation: float, 
                                   world_bounds: Dict[str, float], 
                                   visibility_polygon: Optional[List[Tuple[float, float]]] = None,
                                   fast_rotation: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
        """C++ optimized transform_mask with float32 throughout. Returns: (transformed_mask, world_bounds)"""
        if mask_data is None:
            return None
        
        mask, mask_bounds = mask_data
        
        # Ensure input mask is C++ compatible
        mask = self._ensure_float32_array(mask)
        
        # Convert polygon to C++ compatible format if provided
        if visibility_polygon:
            cpp_vertices = self._prepare_polygon_for_cpp(visibility_polygon)
        else:
            cpp_vertices = None
        
        # Call the regular transform_mask with fast_rotation=True for C++ efficiency
        return self.transform_mask(
            (mask, mask_bounds),
            target_location,
            agent_location,
            orientation,
            world_bounds,
            visibility_polygon,
            fast_rotation=True  # Always use fast mode for C++ conversion
        )
    
    def create_canvas_overlay(self, reachability_data: Optional[Tuple[np.ndarray, Dict[str, Any]]] = None) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Create canvas copy initialized with zeros. Args: reachability_data. Returns: (canvas_mask, world_bounds)"""
        # Use provided data or fall back to cached overlay
        if reachability_data is None:
            reachability_data = self.get_overlay_for_visualization()
        
        if reachability_data is None:
            return None
        
        reachability_mask, world_bounds = reachability_data

        # Create a canvas copy with same shape but filled with 0s
        canvas_mask = np.zeros_like(reachability_mask, dtype=np.float32)

        debug_print(f"🎨 Created canvas overlay: {canvas_mask.shape[0]}×{canvas_mask.shape[1]} filled with 0s")
        debug_print(f"   📐 Canvas dimensions match processed overlay")
        
        return canvas_mask, world_bounds
    
    def apply_mask_to_canvas(self, canvas_data: Optional[Tuple[np.ndarray, Dict[str, Any]]], 
                           mask_data: Optional[Tuple[np.ndarray, Dict[str, Any]]]) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Apply mask to canvas using element-wise maximum. Args: canvas_data, mask_data. Returns: (final_canvas, world_bounds)"""
        if canvas_data is None or mask_data is None:
            return canvas_data  # Return canvas unchanged if no mask data
        
        canvas_mask, canvas_bounds = canvas_data
        mask, mask_bounds = mask_data
        
        # Verify shapes match
        if canvas_mask.shape != mask.shape:
            debug_print(f"⚠️  Shape mismatch: canvas {canvas_mask.shape} vs mask {mask.shape}")
            return canvas_data  # Return canvas unchanged
        
        # Element-wise maximum: canvas = max(canvas, mask) - C++ REQUIRED
        canvas_f32 = canvas_mask.astype(np.float32)
        mask_f32 = mask.astype(np.float32)
        final_canvas = fast_reachability.element_wise_maximum(canvas_f32, mask_f32)
        
        debug_print(f"🎯 Applied mask to canvas: {final_canvas.shape[0]}×{final_canvas.shape[1]}")
        debug_print(f"   📊 Final canvas range: [{final_canvas.min():.3f}, {final_canvas.max():.3f}]")
        
        return final_canvas, canvas_bounds
    
    def _rasterize_polygon_scanline(self, vertices, height, width):
        """Rasterize polygon using scanline algorithm. Args: vertices, height, width. Returns: boolean mask."""
        # Use C++ implementation for significant speedup (60-80% faster)
        vertices_array = np.array(vertices, dtype=np.float32)
        return fast_reachability.rasterize_polygon_scanline(vertices_array, height, width)
    
    def generate_reachability_path_data(self, 
                                      path_analysis_data: List[Dict],
                                      agent_x: float, 
                                      agent_y: float, 
                                      agent_orientation: float,
                                      visibility_polygon: Optional[List[Tuple[float, float]]] = None) -> Tuple[Optional[Tuple[np.ndarray, Dict]], List[Dict]]:
        """Generate reachability data for path analysis. Returns: (reachability_data, sample_points_data)"""
        debug_print("🔄 Setting up reachability overlay...")
        reachability_data = None
        sample_points_data = []
        
        try:
            if not self.is_loaded():
                debug_print(f"⚠️ Failed to load reachability data from {self.filename_base}.pkl")
                return None, []
            
            if not self.is_overlay_configured():
                debug_print(f"⚠️ Overlay not configured. Call setup_overlay_configuration() first.")
                return None, []
            
            # Get the original reachability data
            original_reachability_data = self.get_overlay_for_visualization()
            
            if not original_reachability_data:
                debug_print(f"⚠️ Failed to get original reachability data")
                return None, []
            
            # Create canvas data (copy with all 1s)
            canvas_data = self.create_canvas_overlay(original_reachability_data)
            
            # Create test mask
            test_mask_data = self.create_test_mask(
                shape=(400, 400), 
                agent_position=(agent_x, agent_y),
                world_size=400.0
            )
            
            # Extract components more carefully
            if isinstance(original_reachability_data, tuple) and len(original_reachability_data) >= 2:
                original_mask = original_reachability_data[0]
                original_bounds = original_reachability_data[1]
            else:
                debug_print(f"⚠️ Unexpected original_reachability_data format")
                return None, []
            
            if isinstance(test_mask_data, tuple) and len(test_mask_data) >= 2:
                test_mask = test_mask_data[0]
                test_world_bounds = test_mask_data[1]
            else:
                debug_print(f"⚠️ Unexpected test_mask_data format")
                return None, []
            
            # Only proceed if we have valid data
            if original_mask is None or original_bounds is None or test_mask is None or test_world_bounds is None:
                debug_print(f"⚠️ Missing required data components for reachability overlay")
                return None, []
            
            # Transform test data to agent location and orientation
            debug_print(f"🔄 Calling transform_mask...")
            debug_print(f"🔍 Debug: Using test_world_bounds instead of original_bounds")
            debug_print(f"   Original bounds: {original_bounds}")
            debug_print(f"   Test mask bounds: {test_world_bounds}")
            
            transformed_test_data = self.transform_mask(
                original_reachability_data,  # Pass the full tuple, not just the mask
                target_location=(agent_x, agent_y),
                agent_location=(agent_x, agent_y),
                orientation=agent_orientation,
                world_bounds=test_world_bounds,  # Use bounds from test mask data (centered at agent)
                visibility_polygon=visibility_polygon
            )
            
            debug_print(f"✅ Transform_mask completed successfully")
            debug_print(f"🔍 Debug: transformed_test_data type: {type(transformed_test_data)}")
            if isinstance(transformed_test_data, tuple):
                debug_print(f"🔍 Debug: transformed_test_data tuple length: {len(transformed_test_data)}")
            
            # Initialize reachability_data as canvas_data
            reachability_data = canvas_data
            
            # Create individual masks for each path and apply them sequentially
            debug_print(f"🔄 Creating masks for {len(path_analysis_data)} paths...")
            updated_overlay = canvas_data
            
            for path_idx, path_info in enumerate(path_analysis_data):
                target_point = path_info.get('target_point')
                path_orientation = path_info.get('orientation')
                path_polygon = path_info.get('path_polygon')
                
                if target_point is None or path_orientation is None or path_polygon is None:
                    debug_print(f"  ⚠️ Path {path_idx+1}: Missing data (target_point={target_point is not None}, orientation={path_orientation is not None}, polygon={path_polygon is not None})")
                    continue
                
                # Sample value along first edge direction from transformed_test_data
                target_value = None
                if transformed_test_data is not None and path_info.get('first_edge'):
                    try:
                        # Extract the mask and bounds from transformed_test_data
                        if isinstance(transformed_test_data, tuple) and len(transformed_test_data) >= 2:
                            test_mask = transformed_test_data[0]
                            test_bounds = transformed_test_data[1]
                            
                            # Calculate sampling point along line from edge start towards circle center
                            first_edge = path_info['first_edge']
                            if first_edge and first_edge['start'] and first_edge['end']:
                                edge_start = first_edge['start']
                                edge_end = first_edge['end']
                                
                                # Calculate the length of the first edge
                                edge_vec_x = edge_end[0] - edge_start[0]
                                edge_vec_y = edge_end[1] - edge_start[1]
                                edge_length = math.sqrt(edge_vec_x**2 + edge_vec_y**2)
                                
                                # Calculate direction vector from edge start towards circle center (agent)
                                center_vec_x = agent_x - edge_start[0]
                                center_vec_y = agent_y - edge_start[1]
                                center_distance = math.sqrt(center_vec_x**2 + center_vec_y**2)
                                
                                if center_distance > 0 and edge_length > 0:
                                    # Normalize direction vector towards center
                                    center_dir_x = center_vec_x / center_distance
                                    center_dir_y = center_vec_y / center_distance
                                    
                                    # Move from edge start towards center by edge length + extra distance
                                    extra_distance = 15.0  # Small extra distance for better sampling
                                    total_distance = edge_length + extra_distance
                                    sample_x = edge_start[0] + total_distance * center_dir_x
                                    sample_y = edge_start[1] + total_distance * center_dir_y
                                    
                                    sample_point = (sample_x, sample_y)
                                else:
                                    # Fallback to target point if calculations fail
                                    sample_point = target_point
                            else:
                                # Fallback to target point if first edge data is missing
                                sample_point = target_point
                            
                            # Convert bounds format
                            if isinstance(test_bounds, dict):
                                grid_x_min = test_bounds['x_min']
                                grid_x_max = test_bounds['x_max']
                                grid_y_min = test_bounds['y_min']
                                grid_y_max = test_bounds['y_max']
                            else:
                                grid_x_min, grid_y_min, grid_x_max, grid_y_max = test_bounds
                            
                            # Convert world coordinates to grid indices - OPTIMIZED
                            grid_height, grid_width = test_mask.shape
                            # Pre-calculate scaling factors to avoid repeated division
                            scale_x = grid_width / (grid_x_max - grid_x_min)
                            scale_y = grid_height / (grid_y_max - grid_y_min)
                            
                            # Single calculation with pre-computed scales
                            grid_x = int((sample_point[0] - grid_x_min) * scale_x)
                            grid_y = int((sample_point[1] - grid_y_min) * scale_y)
                            
                            # Clamp to valid grid bounds using numpy clip for efficiency
                            grid_x = np.clip(grid_x, 0, grid_width - 1)
                            grid_y = np.clip(grid_y, 0, grid_height - 1)
                            
                            # Sample value at the calculated point
                            target_value = test_mask[grid_y, grid_x]
                            debug_print(f"  🔍 Value at sample point {sample_point}: {target_value:.6f} (grid coords: {grid_x}, {grid_y})")
                            debug_print(f"      (moved {edge_length:.1f} + 15.0 units from edge start {edge_start} towards circle center)")
                            
                            # Store sample point for visualization
                            sample_points_data.append({
                                'path_idx': path_idx,
                                'sample_point': sample_point,
                                'target_value': target_value,
                                'edge_start': edge_start,
                                'edge_length': edge_length,
                                'extra_distance': 15.0
                            })
                            
                        else:
                            debug_print(f"  🔍 transformed_test_data format unexpected: {type(transformed_test_data)}")
                    except Exception as sample_error:
                        debug_print(f"  ⚠️ Error sampling transformed_test_data along first edge: {sample_error}")
                
                debug_print(f"  🎯 Path {path_idx+1}: Creating mask for target {target_point}, orientation {math.degrees(path_orientation):.1f}°")
                
                try:
                    # Create modified reachability data by multiplying with sampled value
                    modified_reachability_data = original_reachability_data
                    if target_value is not None and target_value != 0:
                        # Raise the value by 30% but cap at 1.0
                        boosted_value = min(1.0, target_value * 1.3)
                        debug_print(f"  🔢 Multiplying original reachability data by boosted value: {target_value:.6f} → {boosted_value:.6f} (+30%, capped at 1.0)")
                        original_mask = original_reachability_data[0]
                        original_bounds = original_reachability_data[1]
                        
                        # Create modified mask by multiplying with boosted value
                        modified_mask = original_mask * boosted_value
                        modified_reachability_data = (modified_mask, original_bounds)
                        debug_print(f"  ✅ Modified reachability data created (values scaled by {boosted_value:.6f})")
                    else:
                        debug_print(f"  ⚠️ Skipping multiplication (target_value={target_value})")
                    
                    # Transform mask for this specific path using modified data
                    path_transformed_data = self.transform_mask(
                        modified_reachability_data,  # Pass the modified tuple instead of original
                        target_location=target_point,  # Use path's target point
                        agent_location=(agent_x, agent_y),  # Keep original agent location
                        orientation=path_orientation,  # Use path's orientation
                        world_bounds=test_world_bounds,  # Use bounds from test mask data
                        visibility_polygon=path_polygon  # Use path's polygon
                    )
                    
                    if path_transformed_data is not None:
                        # Apply this path's mask to the updated overlay
                        temp_overlay = self.apply_mask_to_canvas(updated_overlay, path_transformed_data)
                        if temp_overlay:
                            updated_overlay = temp_overlay
                            debug_print(f"    ✅ Path {path_idx+1}: Mask applied successfully")
                        else:
                            debug_print(f"    ⚠️ Path {path_idx+1}: Failed to apply mask to canvas")
                    else:
                        debug_print(f"    ⚠️ Path {path_idx+1}: Transform_mask returned None")
                        
                except Exception as path_error:
                    debug_print(f"    ❌ Path {path_idx+1}: Error during mask transformation: {path_error}")
                    continue
            
            # Update reachability_data with the final overlay
            if updated_overlay:
                reachability_data = updated_overlay
                debug_print(f"✅ All path masks applied. Final overlay has {len(path_analysis_data)} path contributions.")
            
            # Apply transformed test data to canvas (original behavior)
            if transformed_test_data is not None:
                debug_print(f"🔄 Calling apply_mask_to_canvas...")
                final_overlay = self.apply_mask_to_canvas(updated_overlay, transformed_test_data)
                if final_overlay:
                    reachability_data = final_overlay
                    debug_print(f"✅ Reachability overlay configured and applied with test data")
                    debug_print(f"🔍 Debug: final reachability_data type: {type(reachability_data)}")
                    if isinstance(reachability_data, tuple):
                        debug_print(f"🔍 Debug: final reachability_data tuple length: {len(reachability_data)}")
                else:
                    debug_print(f"⚠️ Failed to apply transformed test data to canvas")
            else:
                debug_print(f"⚠️ Transform_mask returned None for test data")
                
        except Exception as transform_error:
            debug_print(f"⚠️ Error during mask transformation: {transform_error}")
            import traceback
            traceback.print_exc()
            reachability_data = None
            
        return reachability_data, sample_points_data

    def process_paths_with_reachability(self,
                                      paths: List[Dict],
                                      agent_x: float,
                                      agent_y: float,
                                      agent_orientation: float,
                                      visibility_range: float,
                                      visibility_polygon: Optional[List[Tuple[float, float]]] = None) -> Tuple[List[Dict], Optional[Tuple[np.ndarray, Dict]], List[Dict]]:
        """Complete path analysis and reachability processing. Returns: (path_analysis_data, reachability_data, sample_points_data)"""
        debug_print("🔄 Processing path analysis and reachability overlay...")
        
        # Check if overlay is configured
        if not self.is_overlay_configured():
            debug_print("⚠️ Overlay not configured. Call setup_overlay_configuration() first.")
            debug_print("   Proceeding with path analysis only (no reachability overlay)...")
            
        # STEP 1: Compute path analysis data (moved from main function)
        debug_print("🔄 Processing path first edges and polygons...")
        path_analysis_data = []
        
        for i, path_data in enumerate(paths):
            if 'path_points' in path_data and len(path_data['path_points']) > 1:
                points = path_data['path_points']
                path_segments = path_data.get('path_segments', [])
                
                # Extract first edge
                first_edge = None
                if path_segments and len(path_segments) > 0:
                    # Use the actual first segment from path_segments
                    first_segment = path_segments[0]
                    first_edge = {
                        'type': first_segment['type'],
                        'start': first_segment['start'],
                        'end': first_segment['end']
                    }
                    
                    # Add additional data for arc segments
                    if first_segment['type'] == 'arc':
                        edge_data = first_segment.get('edge_data', {})
                        first_edge.update({
                            'center': edge_data.get('center', (agent_x, agent_y)),
                            'radius': edge_data.get('radius', visibility_range),
                            'start_angle': edge_data.get('start_angle', 0),
                            'end_angle': edge_data.get('end_angle', 0)
                        })
                else:
                    # Fallback: create line segment from first two points
                    if len(points) >= 2:
                        first_edge = {
                            'type': 'line',
                            'start': points[0],
                            'end': points[1]
                        }
                
                # Create overall path polygon by linearizing arc segments
                path_polygon = []
                current_position = None
                
                # Build polygon by processing segments to linearize arcs
                if path_segments and len(path_segments) > 0:
                    for seg_idx, segment in enumerate(path_segments):
                        segment_start = segment['start']
                        segment_end = segment['end']
                        
                        # Add start point for first segment only
                        if seg_idx == 0:
                            path_polygon.append(segment_start)
                            current_position = segment_start
                            debug_print(f"           Starting path at: {segment_start}")
                        
                        if segment['type'] == 'line':
                            # For line segments, just add the end point
                            path_polygon.append(segment_end)
                            current_position = segment_end
                            debug_print(f"           Line segment {seg_idx+1}: {segment_start} → {segment_end}")
                        
                        elif segment['type'] == 'arc':
                            # For arc segments, linearize by sampling points along the arc
                            edge_data = segment.get('edge_data', {})
                            center = edge_data.get('center', (agent_x, agent_y))
                            radius = edge_data.get('radius', visibility_range)
                            start_angle = edge_data.get('start_angle', 0)
                            end_angle = edge_data.get('end_angle', 0)
                            
                            # Determine arc direction by checking which direction gets us from start to end
                            # Calculate angles from center to start and end points
                            start_to_center_angle = math.atan2(segment_start[1] - center[1], segment_start[0] - center[0])
                            end_to_center_angle = math.atan2(segment_end[1] - center[1], segment_end[0] - center[0])
                            
                            # Normalize angles to [0, 2π)
                            start_to_center_angle = start_to_center_angle % (2 * math.pi)
                            end_to_center_angle = end_to_center_angle % (2 * math.pi)
                            
                            # Determine shortest arc direction
                            angle_diff = end_to_center_angle - start_to_center_angle
                            if angle_diff > math.pi:
                                angle_diff -= 2 * math.pi
                            elif angle_diff < -math.pi:
                                angle_diff += 2 * math.pi
                            
                            # Use the actual calculated angles instead of edge_data angles if they seem wrong
                            actual_start_angle = start_to_center_angle
                            actual_end_angle = end_to_center_angle
                            
                            # If going in negative direction, adjust end angle
                            if angle_diff < 0:
                                if actual_end_angle > actual_start_angle:
                                    actual_end_angle -= 2 * math.pi
                            else:
                                if actual_end_angle < actual_start_angle:
                                    actual_end_angle += 2 * math.pi
                            
                            # Calculate number of linearization points based on arc length
                            arc_length = abs(actual_end_angle - actual_start_angle)
                            # Use roughly 5 degrees per segment for smooth linearization, minimum 2 points
                            num_points = max(2, int(arc_length / math.radians(5)))
                            
                            # Generate linearized points along the arc (excluding start, including end)
                            for j in range(1, num_points + 1):
                                t = j / num_points
                                angle = actual_start_angle + t * (actual_end_angle - actual_start_angle)
                                arc_x = center[0] + radius * math.cos(angle)
                                arc_y = center[1] + radius * math.sin(angle)
                                path_polygon.append((arc_x, arc_y))
                            
                            current_position = segment_end
                            
                            debug_print(f"           Arc segment {seg_idx+1}: {segment_start} → {segment_end}")
                            debug_print(f"             Center: {center}, Radius: {radius:.1f}")
                            debug_print(f"             Start angle: {math.degrees(actual_start_angle):.1f}°, End angle: {math.degrees(actual_end_angle):.1f}°")
                            debug_print(f"             Arc length: {arc_length:.3f} rad ({math.degrees(arc_length):.1f}°)")
                            debug_print(f"             Linearized into {num_points} points")
                            
                        # Verify connectivity
                        if current_position and seg_idx > 0:
                            expected_start = path_polygon[-2] if len(path_polygon) >= 2 else path_polygon[-1]
                            distance_to_expected = math.sqrt((segment_start[0] - expected_start[0])**2 + (segment_start[1] - expected_start[1])**2)
                            if distance_to_expected > 1e-6:  # Small tolerance for floating point errors
                                debug_print(f"           ⚠️ Gap detected: previous end {expected_start} vs current start {segment_start} (distance: {distance_to_expected:.6f})")
                else:
                    # Fallback: use original points if no segments available
                    path_polygon = points.copy()
                    debug_print(f"           Using original {len(points)} points (no segments available)")
                
                # Calculate orientation: rotate line from second to first point by 45 degrees
                # away from the circle center (agent position)
                orientation = None
                if first_edge and first_edge['start'] and first_edge['end']:
                    start_pt = first_edge['start']
                    end_pt = first_edge['end']
                    
                    # Vector from second point (end) to first point (start)
                    vec_x = start_pt[0] - end_pt[0]
                    vec_y = start_pt[1] - end_pt[1]
                    
                    # Current angle of this vector
                    current_angle = math.atan2(vec_y, vec_x)
                    
                    # Determine which direction to rotate (away from circle center)
                    # Find the perpendicular to the first edge
                    edge_vec_x = end_pt[0] - start_pt[0]
                    edge_vec_y = end_pt[1] - start_pt[1]
                    
                    # Two possible 45-degree rotations
                    angle_option1 = current_angle + math.radians(45)
                    angle_option2 = current_angle - math.radians(45)
                    
                    # Check which direction points away from circle center
                    # Test points at unit distance from the midpoint of the edge
                    mid_x = (start_pt[0] + end_pt[0]) / 2
                    mid_y = (start_pt[1] + end_pt[1]) / 2
                    
                    test_point1_x = mid_x + math.cos(angle_option1)
                    test_point1_y = mid_y + math.sin(angle_option1)
                    
                    test_point2_x = mid_x + math.cos(angle_option2)
                    test_point2_y = mid_y + math.sin(angle_option2)
                    
                    # Distance from test points to circle center (agent)
                    dist1_to_center = math.sqrt((test_point1_x - agent_x)**2 + (test_point1_y - agent_y)**2)
                    dist2_to_center = math.sqrt((test_point2_x - agent_x)**2 + (test_point2_y - agent_y)**2)
                    
                    # Choose the direction that takes us away from the center
                    if dist1_to_center > dist2_to_center:
                        orientation = angle_option1
                    else:
                        orientation = angle_option2
                    
                    # Normalize angle to [0, 2π)
                    while orientation < 0:
                        orientation += 2 * math.pi
                    while orientation >= 2 * math.pi:
                        orientation -= 2 * math.pi
                
                # Store path analysis data
                target_point = None
                if first_edge and first_edge['end']:
                    target_point = first_edge['end']  # Second point of first edge - where orientation arrow starts
                
                path_info = {
                    'path_id': i,
                    'first_edge': first_edge,
                    'path_polygon': path_polygon,
                    'orientation': orientation,
                    'target_point': target_point,  # Second point of first edge (where 45° arrow starts)
                    'completed': path_data.get('completed', False),
                    'breakoff_line': path_data.get('breakoff_line', None),
                    'total_points': len(points),
                    'total_segments': len(path_segments)
                }
                
                path_analysis_data.append(path_info)
                
                # Debug output for first edge and polygon linearization
                if first_edge:
                    orientation_deg = math.degrees(orientation) if orientation is not None else None
                    target_point = path_info['target_point']
                    if first_edge['type'] == 'line':
                        debug_print(f"  Path {i+1}: First edge is LINE from {first_edge['start']} to {first_edge['end']}")
                        debug_print(f"           Target point (45° arrow start): {target_point}")
                        if orientation_deg is not None:
                            debug_print(f"           Orientation: {orientation_deg:.1f}° (away from center)")
                    elif first_edge['type'] == 'arc':
                        start_angle_deg = math.degrees(first_edge.get('start_angle', 0))
                        end_angle_deg = math.degrees(first_edge.get('end_angle', 0))
                        debug_print(f"  Path {i+1}: First edge is ARC from {first_edge['start']} to {first_edge['end']} "
                              f"(angles: {start_angle_deg:.1f}° to {end_angle_deg:.1f}°)")
                        debug_print(f"           Target point (45° arrow start): {target_point}")
                        if orientation_deg is not None:
                            debug_print(f"           Orientation: {orientation_deg:.1f}° (away from center)")
                else:
                    debug_print(f"  Path {i+1}: No first edge found")
                
                # Debug polygon linearization
                original_points = len(points)
                linearized_points = len(path_polygon)
                if linearized_points > original_points:
                    arc_segments = sum(1 for seg in path_segments if seg['type'] == 'arc')
                    line_segments = sum(1 for seg in path_segments if seg['type'] == 'line')
                    debug_print(f"           Polygon linearized: {original_points} → {linearized_points} points")
                    debug_print(f"           Path composition: {line_segments} line segments, {arc_segments} arc segments")
                    debug_print(f"           Arc linearization added {linearized_points - original_points} intermediate points")
                else:
                    debug_print(f"           Polygon points: {linearized_points} (no arcs to linearize)")
        
        debug_print(f"✅ Processed {len(path_analysis_data)} paths with first edge analysis")
        
        # STEP 2: Generate reachability overlay data using the computed path analysis
        if self.is_overlay_configured():
            reachability_data, sample_points_data = self.generate_reachability_path_data(
                path_analysis_data=path_analysis_data,
                agent_x=agent_x,
                agent_y=agent_y,
                agent_orientation=agent_orientation,
                visibility_polygon=visibility_polygon
            )
        else:
            debug_print("⚠️ Skipping reachability overlay generation (overlay not configured)")
            reachability_data = None
            sample_points_data = []
        
        return path_analysis_data, reachability_data, sample_points_data


# No standalone utility functions - use ReachabilityMaskAPI class directly


if __name__ == "__main__":
    # Demo usage
    debug_print("🎯 Reachability Mask API Demo")
    debug_print("=" * 50)
    
    # Create API instance
    api = ReachabilityMaskAPI()
    
    if api.is_loaded():
        # Print summary
        debug_print(api.get_summary_info())
        
        # Demo some API calls
        debug_print(f"\n🔍 Example API calls:")
        debug_print(f"Value at center: {api.get_value_at_world(0, 0):.6f}")
        debug_print(f"Value at (10, 10): {api.get_value_at_world(10, 10):.6f}")
        
        # Distance analysis
        debug_print(f"\n📏 Distance analysis (first 3 bins):")
        distance_analysis = api.analyze_coverage_by_distance(num_bins=10)
        for analysis in distance_analysis[:3]:
            dist_range = analysis['distance_range']
            coverage = analysis['coverage_percentage']
            avg_reach = analysis['avg_reachability']
            debug_print(f"  {dist_range[0]:6.1f}-{dist_range[1]:6.1f}px: {coverage:5.1f}% coverage, avg: {avg_reach:.6f}")
    else:
        debug_print("❌ Could not load reachability mask data")
