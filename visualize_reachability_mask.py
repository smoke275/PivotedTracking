#!/usr/bin/env python3
"""
Reachability Mask Visualizer
Loads and displays the reachability mask data for inspection.
Uses the ReachabilityMaskAPI for data access.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from reachability_mask_api import ReachabilityMaskAPI


def visualize_reachability_mask(filename_base="unicycle_grid"):
    """
    Load and visualize the reachability mask data.
    
    Args:
        filename_base: Base name of the pickle file (without .pkl extension)
    """
    print(f"Loading reachability mask from {filename_base}.pkl...")
    
    # Load the mask data using the API
    api = ReachabilityMaskAPI(filename_base)
    
    if not api.is_loaded():
        print(f"❌ Could not load reachability mask from {filename_base}.pkl")
        print("Make sure the file exists. Run heatmap.py first to generate it.")
        return None
    
    # Get data from API
    grid = api.get_grid()
    stats = api.get_statistics()
    
    print(f"✅ Reachability mask loaded successfully!")
    print(f"📊 Grid size: {stats['grid_size']}x{stats['grid_size']}")
    print(f"🌍 World extent: ±{stats['world_extent_px']/2:.1f} pixels")
    print(f"📏 Cell size: {stats['cell_size_px']:.3f} px/cell")
    print(f"🎯 Center index: {stats['center_idx']}")
    print(f"📈 Grid shape: {grid.shape}")
    print(f"📋 Grid data type: {grid.dtype}")
    print(f"🔢 Grid value range: {stats['min_value']:.6f} to {stats['max_value']:.6f}")
    print(f"🎲 Non-zero values: {stats['reachable_cells']}/{stats['total_cells']} ({stats['reachable_percentage']:.1f}%)")
    print(f"📊 Non-zero statistics:")
    print(f"   Mean: {stats['mean_reachable']:.6f}")
    print(f"   Std:  {stats['std_reachable']:.6f}")
    print(f"   Min:  {stats['min_reachable']:.6f}")
    print(f"   Max:  {stats['max_reachable']:.6f}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # Extract commonly used values
    grid_size = stats['grid_size']
    center_idx = stats['center_idx']
    cell_size = stats['cell_size_px']
    world_extent = stats['world_extent_px']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Raw reachability grid (heatmap)
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(grid, cmap='hot', origin='upper', interpolation='nearest')
    ax1.set_title(f'Raw Reachability Grid\n{grid_size}x{grid_size}, Cell size: {cell_size:.3f}px')
    ax1.set_xlabel('Grid Column (East →)')
    ax1.set_ylabel('Grid Row (North ↑)')
    
    # Add center point marker
    ax1.plot(center_idx, center_idx, 'c+', markersize=15, markeredgewidth=3, label='Center')
    ax1.legend()
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Reachability Probability')
    
    # 2. Log scale visualization (for better contrast)
    ax2 = plt.subplot(2, 3, 2)
    # Use log scale but handle zeros properly
    log_grid = np.log10(grid + 1e-10)  # Add small value to avoid log(0)
    log_grid[grid == 0] = np.nan  # Set zeros to NaN for better visualization
    
    im2 = ax2.imshow(log_grid, cmap='viridis', origin='upper', interpolation='nearest')
    ax2.set_title('Log Scale Reachability\n(Better contrast for small values)')
    ax2.set_xlabel('Grid Column (East →)')
    ax2.set_ylabel('Grid Row (North ↑)')
    
    # Add center point marker
    ax2.plot(center_idx, center_idx, 'w+', markersize=15, markeredgewidth=3, label='Center')
    ax2.legend()
    
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Log₁₀(Reachability Probability)')
    
    # 3. Binary mask (non-zero regions)
    ax3 = plt.subplot(2, 3, 3)
    binary_mask = (grid > 0).astype(int)
    im3 = ax3.imshow(binary_mask, cmap='RdYlBu_r', origin='upper', interpolation='nearest')
    ax3.set_title(f'Binary Reachability Mask\n{np.count_nonzero(binary_mask)} reachable cells')
    ax3.set_xlabel('Grid Column (East →)')
    ax3.set_ylabel('Grid Row (North ↑)')
    
    # Add center point marker
    ax3.plot(center_idx, center_idx, 'k+', markersize=15, markeredgewidth=3, label='Center')
    ax3.legend()
    
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Reachable (1) / Unreachable (0)')
    
    # 4. 3D surface plot
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    
    # Create coordinate grids for 3D plot
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Subsample for performance if grid is large
    stride = max(1, grid_size // 100)
    X_sub = X[::stride, ::stride]
    Y_sub = Y[::stride, ::stride]
    Z_sub = grid[::stride, ::stride]
    
    surf = ax4.plot_surface(X_sub, Y_sub, Z_sub, cmap='hot', alpha=0.8)
    ax4.set_title('3D Reachability Surface')
    ax4.set_xlabel('Grid Column')
    ax4.set_ylabel('Grid Row')
    ax4.set_zlabel('Reachability')
    
    # 5. Cross-section through center
    ax5 = plt.subplot(2, 3, 5)
    
    # Horizontal cross-section through center
    center_row = grid[center_idx, :]
    # Vertical cross-section through center
    center_col = grid[:, center_idx]
    
    x_coords = np.arange(grid_size)
    ax5.plot(x_coords, center_row, 'r-', linewidth=2, label=f'Horizontal (row {center_idx})')
    ax5.plot(x_coords, center_col, 'b-', linewidth=2, label=f'Vertical (col {center_idx})')
    ax5.axvline(x=center_idx, color='k', linestyle='--', alpha=0.5, label='Center')
    
    ax5.set_title('Cross-sections through Center')
    ax5.set_xlabel('Grid Index')
    ax5.set_ylabel('Reachability Probability')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Distribution histogram
    ax6 = plt.subplot(2, 3, 6)
    
    # Plot histogram of non-zero values
    non_zero_values = grid[grid > 0]
    if len(non_zero_values) > 0:
        ax6.hist(non_zero_values, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax6.set_xlabel('Reachability Probability')
        ax6.set_ylabel('Density')
        ax6.set_title(f'Distribution of Non-zero Values\n({len(non_zero_values)} values)')
        ax6.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {stats["mean_reachable"]:.6f}\n'
        stats_text += f'Std: {stats["std_reachable"]:.6f}\n'
        stats_text += f'Min: {stats["min_reachable"]:.6f}\n'
        stats_text += f'Max: {stats["max_reachable"]:.6f}'
        ax6.text(0.65, 0.95, stats_text, transform=ax6.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax6.text(0.5, 0.5, 'No non-zero values found', 
                transform=ax6.transAxes, ha='center', va='center')
        ax6.set_title('No Data to Display')
    
    plt.tight_layout()
    
    # Show coordinate system information
    fig.suptitle(f'Reachability Mask Visualization: {filename_base}.pkl\n'
                f'World coordinates: ±{world_extent/2:.0f}px, Grid: {grid_size}×{grid_size}, Cell: {cell_size:.3f}px',
                fontsize=14, y=0.98)
    
    plt.show()
    
    # Print additional analysis using API
    print(f"\n📋 Additional Analysis:")
    max_loc_world = stats['max_location_world']
    max_loc_grid = stats['max_location_grid']
    print(f"🎯 Center coordinates in world space: (0, 0)")
    print(f"📐 Grid covers: X=[{-world_extent/2:.1f}, {world_extent/2:.1f}], Y=[{-world_extent/2:.1f}, {world_extent/2:.1f}] pixels")
    print(f"🔢 Total grid cells: {stats['total_cells']:,}")
    print(f"✅ Reachable cells: {stats['reachable_cells']:,}")
    print(f"❌ Unreachable cells: {stats['unreachable_cells']:,}")
    print(f"🏆 Maximum reachability: {stats['max_value']:.6f}")
    print(f"   📍 Location: grid[{max_loc_grid[0]}, {max_loc_grid[1]}] = world({max_loc_world[0]:.1f}, {max_loc_world[1]:.1f})")
    
    return api


def analyze_mask_coverage(api):
    """Analyze the coverage patterns in the reachability mask using API."""
    if not api.is_loaded():
        return
    
    print(f"\n🔍 Coverage Pattern Analysis:")
    
    # Use the API's built-in distance analysis
    distance_analysis = api.analyze_coverage_by_distance(num_bins=10)
    
    print(f"📏 Distance analysis (10 equal bins):")
    for analysis in distance_analysis:
        start_dist, end_dist = analysis['distance_range']
        total_cells = analysis['total_cells']
        reachable_cells = analysis['reachable_cells']
        coverage_pct = analysis['coverage_percentage']
        avg_reachability = analysis['avg_reachability']
        
        print(f"   {start_dist:6.1f}-{end_dist:6.1f}px: {reachable_cells:4d}/{total_cells:4d} cells ({coverage_pct:5.1f}%), avg prob: {avg_reachability:.6f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize reachability mask data")
    parser.add_argument("--filename", "-f", default="unicycle_grid", 
                       help="Base filename of the pickle file (default: unicycle_grid)")
    parser.add_argument("--analyze", "-a", action="store_true", 
                       help="Perform detailed coverage analysis")
    
    args = parser.parse_args()
    
    print("🎯 Reachability Mask Visualizer")
    print("=" * 50)
    
    # Load and visualize the mask
    api = visualize_reachability_mask(args.filename)
    
    # Perform additional analysis if requested
    if args.analyze and api is not None:
        analyze_mask_coverage(api)
    
    print("\n✅ Visualization complete!")
