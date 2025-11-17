#!/usr/bin/env python3
"""
Grid Probability Visualizer
Standalone tool to visualize reachability probability grids from CSV files.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
import os
import sys
from pathlib import Path

def load_grid_from_csv(csv_filename):
    """
    Load probability grid from CSV file.
    
    Args:
        csv_filename: Path to the CSV file
        
    Returns:
        tuple: (grid_array, metadata_dict)
    """
    metadata = {}
    
    try:
        # Read the header lines to extract metadata
        with open(csv_filename, 'r') as f:
            lines = f.readlines()
        
        # Parse metadata from header comments
        for line in lines:
            if line.startswith('#'):
                if 'Agent at' in line:
                    # Extract agent position and orientation
                    import re
                    match = re.search(r'Agent at \(([^,]+),([^)]+)\) facing (\d+)', line)
                    if match:
                        metadata['agent_x'] = float(match.group(1))
                        metadata['agent_y'] = float(match.group(2))
                        metadata['agent_orientation'] = float(match.group(3))
                elif 'Grid size:' in line:
                    # Extract grid size
                    import re
                    match = re.search(r'Grid size: (\d+)x(\d+)', line)
                    if match:
                        metadata['grid_width'] = int(match.group(1))
                        metadata['grid_height'] = int(match.group(2))
        
        # Load the actual grid data (skip comment lines)
        grid_data = np.loadtxt(csv_filename, delimiter=',', comments='#')
        
        print(f"✅ Loaded grid from {csv_filename}")
        print(f"   📊 Shape: {grid_data.shape}")
        if metadata:
            if 'agent_x' in metadata:
                print(f"   🎯 Agent: ({metadata['agent_x']:.1f}, {metadata['agent_y']:.1f}) @ {metadata['agent_orientation']:.0f}°")
        
        # Calculate statistics
        non_zero_count = np.count_nonzero(grid_data)
        total_cells = grid_data.size
        max_prob = np.max(grid_data)
        min_prob = np.min(grid_data[grid_data > 0]) if non_zero_count > 0 else 0
        
        print(f"   📈 Values: {non_zero_count:,}/{total_cells:,} non-zero ({non_zero_count/total_cells*100:.1f}%)")
        if non_zero_count > 0:
            print(f"   🔢 Range: {min_prob:.6f} to {max_prob:.6f}")
        
        return grid_data, metadata
        
    except Exception as e:
        print(f"❌ Error loading {csv_filename}: {e}")
        return None, {}

def create_heatmap_visualization(grid_data, metadata, show_colorbar=True, colormap='viridis'):
    """
    Create a heatmap visualization of the probability grid.
    
    Args:
        grid_data: 2D numpy array of probabilities
        metadata: Dictionary with metadata
        show_colorbar: Whether to show the colorbar
        colormap: Matplotlib colormap name
        
    Returns:
        tuple: (figure, axis)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create the heatmap
    # Use log scale for better visualization of small values
    grid_for_display = grid_data.copy()
    
    # Handle zero values for log scale
    min_nonzero = np.min(grid_data[grid_data > 0]) if np.any(grid_data > 0) else 1e-8
    grid_for_display[grid_data == 0] = min_nonzero / 100  # Very small value for zeros
    
    # Create heatmap with log normalization for better visibility
    im = ax.imshow(grid_for_display, 
                   cmap=colormap, 
                   origin='upper',  # Grid coordinates: (0,0) at top-left
                   norm=colors.LogNorm(vmin=min_nonzero/100, vmax=np.max(grid_data)),
                   interpolation='nearest')
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Reachability Probability', rotation=270, labelpad=20)
    
    # Set labels and title
    ax.set_xlabel('Grid Column (X direction)')
    ax.set_ylabel('Grid Row (Y direction)')
    
    # Create title with metadata
    title = 'Reachability Probability Grid'
    if metadata:
        if 'agent_x' in metadata:
            title += f'\nAgent at ({metadata["agent_x"]:.1f}, {metadata["agent_y"]:.1f}), Facing {metadata["agent_orientation"]:.0f}°'
        if 'grid_width' in metadata:
            title += f'\nGrid: {metadata["grid_width"]}×{metadata["grid_height"]} cells'
    
    ax.set_title(title, pad=20)
    
    # Add grid center marker if we have agent position
    if metadata and 'agent_x' in metadata:
        center_row = grid_data.shape[0] // 2
        center_col = grid_data.shape[1] // 2
        ax.plot(center_col, center_row, 'r+', markersize=15, markeredgewidth=3, 
                label=f'Agent Position\n({metadata["agent_x"]:.1f}, {metadata["agent_y"]:.1f})')
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    return fig, ax

def create_3d_visualization(grid_data, metadata, elevation_scale=1000):
    """
    Create a 3D surface plot of the probability grid.
    
    Args:
        grid_data: 2D numpy array of probabilities
        metadata: Dictionary with metadata
        elevation_scale: Scale factor for the Z-axis
        
    Returns:
        tuple: (figure, axis)
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create coordinate meshes
    rows, cols = grid_data.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    Z = grid_data * elevation_scale  # Scale for better visualization
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    # Add colorbar
    plt.colorbar(surf, ax=ax, shrink=0.6, label='Reachability Probability')
    
    # Set labels
    ax.set_xlabel('Grid Column (X direction)')
    ax.set_ylabel('Grid Row (Y direction)')
    ax.set_zlabel(f'Probability × {elevation_scale}')
    
    # Create title
    title = '3D Reachability Probability Surface'
    if metadata and 'agent_x' in metadata:
        title += f'\nAgent at ({metadata["agent_x"]:.1f}, {metadata["agent_y"]:.1f}), Facing {metadata["agent_orientation"]:.0f}°'
    
    ax.set_title(title, pad=20)
    
    # Set viewing angle for better perspective
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    return fig, ax

def create_contour_visualization(grid_data, metadata, num_levels=20):
    """
    Create a contour plot of the probability grid.
    
    Args:
        grid_data: 2D numpy array of probabilities
        metadata: Dictionary with metadata
        num_levels: Number of contour levels
        
    Returns:
        tuple: (figure, axis)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create coordinate meshes
    rows, cols = grid_data.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Create contour plot
    if np.max(grid_data) > 0:
        # Use logarithmic levels for better visibility
        max_val = np.max(grid_data)
        min_val = np.min(grid_data[grid_data > 0]) if np.any(grid_data > 0) else max_val / 1000
        levels = np.logspace(np.log10(min_val), np.log10(max_val), num_levels)
        
        contours = ax.contour(X, Y, grid_data, levels=levels, colors='black', alpha=0.6, linewidths=0.5)
        contourf = ax.contourf(X, Y, grid_data, levels=levels, cmap='viridis', alpha=0.8)
        
        # Add labels to some contours
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2e')
        
        # Add colorbar
        plt.colorbar(contourf, ax=ax, label='Reachability Probability')
    else:
        ax.text(0.5, 0.5, 'No non-zero probabilities to display', 
                transform=ax.transAxes, ha='center', va='center', fontsize=16)
    
    # Set labels and title
    ax.set_xlabel('Grid Column (X direction)')
    ax.set_ylabel('Grid Row (Y direction)')
    
    title = 'Reachability Probability Contours'
    if metadata and 'agent_x' in metadata:
        title += f'\nAgent at ({metadata["agent_x"]:.1f}, {metadata["agent_y"]:.1f}), Facing {metadata["agent_orientation"]:.0f}°'
    
    ax.set_title(title, pad=20)
    
    # Add agent position marker
    if metadata and 'agent_x' in metadata:
        center_row = grid_data.shape[0] // 2
        center_col = grid_data.shape[1] // 2
        ax.plot(center_col, center_row, 'r+', markersize=15, markeredgewidth=3, 
                label='Agent Position')
        ax.legend()
    
    plt.tight_layout()
    return fig, ax

def save_visualization(fig, output_path, dpi=300):
    """Save the visualization to a file."""
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved visualization: {output_path}")
    except Exception as e:
        print(f"❌ Error saving {output_path}: {e}")

def main():
    """Main function for the grid visualizer."""
    parser = argparse.ArgumentParser(description='Visualize reachability probability grids from CSV files')
    parser.add_argument('csv_file', help='Path to the CSV file containing grid data')
    parser.add_argument('--output', '-o', help='Output directory for saving plots (optional)')
    parser.add_argument('--type', '-t', choices=['heatmap', '3d', 'contour', 'all'], 
                       default='all', help='Type of visualization to create')
    parser.add_argument('--colormap', '-c', default='viridis', 
                       help='Matplotlib colormap to use (default: viridis)')
    parser.add_argument('--no-show', action='store_true', 
                       help='Don\'t display plots interactively (useful for batch processing)')
    parser.add_argument('--dpi', type=int, default=300, 
                       help='DPI for saved images (default: 300)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.csv_file):
        print(f"❌ Error: File {args.csv_file} not found")
        return 1
    
    # Load grid data
    print(f"📁 Loading grid data from {args.csv_file}...")
    grid_data, metadata = load_grid_from_csv(args.csv_file)
    
    if grid_data is None:
        return 1
    
    # Create output directory if specified
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
    
    # Get base filename for outputs
    base_name = Path(args.csv_file).stem
    
    # Create visualizations
    figures = []
    
    if args.type in ['heatmap', 'all']:
        print("🎨 Creating heatmap visualization...")
        fig, ax = create_heatmap_visualization(grid_data, metadata, colormap=args.colormap)
        figures.append(('heatmap', fig))
        
        if output_dir:
            save_visualization(fig, output_dir / f"{base_name}_heatmap.png", args.dpi)
    
    if args.type in ['3d', 'all']:
        print("🎨 Creating 3D surface visualization...")
        fig, ax = create_3d_visualization(grid_data, metadata)
        figures.append(('3d', fig))
        
        if output_dir:
            save_visualization(fig, output_dir / f"{base_name}_3d.png", args.dpi)
    
    if args.type in ['contour', 'all']:
        print("🎨 Creating contour visualization...")
        fig, ax = create_contour_visualization(grid_data, metadata)
        figures.append(('contour', fig))
        
        if output_dir:
            save_visualization(fig, output_dir / f"{base_name}_contour.png", args.dpi)
    
    # Show plots interactively unless --no-show is specified
    if not args.no_show:
        print("📊 Displaying visualizations... (Close windows to exit)")
        plt.show()
    else:
        # Close figures to free memory
        for _, fig in figures:
            plt.close(fig)
    
    print("✅ Visualization complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
