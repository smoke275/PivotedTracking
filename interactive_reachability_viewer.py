#!/usr/bin/env python3
"""
Interactive Reachability Mask Viewer
A simple interactive tool to explore the reachability mask with mouse hover and click.
Uses the ReachabilityMaskAPI for data access.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from reachability_mask_api import ReachabilityMaskAPI


class InteractiveReachabilityViewer:
    def __init__(self, filename_base="unicycle_grid", clip_pixels=0, target_size=None, downsample_method='max_pool'):
        """Initialize the interactive viewer."""
        self.filename_base = filename_base
        self.clip_pixels = clip_pixels
        self.target_size = target_size  # None means no downsampling
        self.downsample_method = downsample_method
        self.api = None
        self.grid = None
        self.fig = None
        self.ax = None
        self.im = None
        self.info_text = None
        self.colorbar = None
        
        # Processing data
        self.clip_data = None
        self.processed_data = None
        self.is_clipped = False
        self.is_downsampled = False
        self.grid_offset = 0  # Offset for coordinate conversion in clipped mode
        self.scale_factor = 1.0  # Scale factor for downsampled mode
        
        # Display options
        self.show_log_scale = False
        self.min_threshold = 0.0
        self.max_threshold = 1.0
        
        # Load data
        self.load_data()
        
        if self.api and self.api.is_loaded():
            self.setup_gui()
    
    def load_data(self):
        """Load the reachability mask data using the API."""
        processing_info = []
        if self.clip_pixels > 0:
            processing_info.append(f"{self.clip_pixels}px clipping")
        if self.target_size:
            processing_info.append(f"downsampling to {self.target_size[0]}×{self.target_size[1]} ({self.downsample_method})")
        
        if processing_info:
            print(f"Loading {self.filename_base}.pkl with {' + '.join(processing_info)}...")
        else:
            print(f"Loading {self.filename_base}.pkl...")
            
        self.api = ReachabilityMaskAPI(self.filename_base)
        
        if not self.api.is_loaded():
            print(f"❌ Could not load {self.filename_base}.pkl")
            return
        
        # Apply processing pipeline: clipping + downsampling
        if self.target_size or self.clip_pixels > 0:
            if self.target_size:
                # Both clipping and downsampling requested
                self.processed_data = self.api.get_clipped_and_downsampled(
                    clip_pixels=self.clip_pixels,
                    target_size=self.target_size,
                    downsample_method=self.downsample_method
                )
            else:
                # Only clipping requested, use clipping method
                self.processed_data = self.api.get_clipped_region(self.clip_pixels)
                # Convert to processed data format for consistency
                if 'error' not in self.processed_data:
                    self.processed_data = {
                        'final_grid': self.processed_data['clipped_grid'],
                        'final_size': self.processed_data['clipped_grid_size'],
                        'final_bounds': self.processed_data['clipped_bounds'],
                        'final_center_idx': self.processed_data['clipped_center_idx'],
                        'final_cell_size': self.processed_data['cell_size_px'],
                        'statistics': self.processed_data['statistics'],
                        'clipping_applied': True,
                        'clip_pixels': self.clip_pixels,
                        'clip_cells': self.processed_data['clip_cells'],
                        'downsampling_applied': False,
                        'target_size': None,
                        'downsample_method': None,
                        'source_grid_size': self.processed_data['original_grid_size'],
                        'source_bounds': self.processed_data['original_bounds'],
                        'total_scale_factor': 1.0,
                        'cell_size_change': 1.0,
                        'grid_reduction': 1.0
                    }
            
            if 'error' in self.processed_data:
                print(f"❌ Processing error: {self.processed_data['error']}")
                if 'max_clip_pixels' in self.processed_data:
                    print(f"💡 Maximum clip: {self.processed_data['max_clip_pixels']:.1f} pixels")
                print("🔄 Falling back to original (unprocessed) data...")
                self.clip_pixels = 0
                self.target_size = None
                self.is_clipped = False
                self.is_downsampled = False
                self.grid = self.api.get_grid()
                self.grid_size = self.api.get_grid_size()
                self.center_idx = self.api.get_center_idx()
                self.grid_offset = 0
                self.scale_factor = 1.0
            else:
                print(f"✅ Successfully processed data!")
                self.is_clipped = self.processed_data['clipping_applied']
                self.is_downsampled = self.processed_data['downsampling_applied'] and self.target_size is not None
                self.grid = self.processed_data['final_grid']
                self.grid_size = self.processed_data['final_size']
                self.center_idx = self.processed_data['final_center_idx']
                self.grid_offset = self.processed_data['clip_cells']
                self.scale_factor = self.processed_data.get('total_scale_factor', 1.0)
                
                if self.is_clipped:
                    print(f"� Clipped {self.clip_pixels} pixels from each side")
                if self.is_downsampled:
                    print(f"📉 Downsampled to {self.grid_size}×{self.grid_size} using {self.downsample_method}")
                    print(f"🔍 Scale factor: {self.scale_factor:.2f}x")
                print(f"📊 Final grid: {self.grid_size}×{self.grid_size}")
        else:
            self.is_clipped = False
            self.is_downsampled = False
            self.grid = self.api.get_grid()
            self.grid_size = self.api.get_grid_size()
            self.center_idx = self.api.get_center_idx()
            self.grid_offset = 0
            self.scale_factor = 1.0
        
        self.world_extent = self.api.get_world_extent()
        self.cell_size = self.api.get_cell_size()
        
        # Get statistics from processed or original data
        if self.processed_data:
            stats = self.processed_data['statistics']
        else:
            stats = self.api.get_statistics()
            
        print(f"✅ Loaded {self.grid_size}×{self.grid_size} grid")
        print(f"📏 Cell size: {self.cell_size:.3f}px")
        print(f"🎯 Center: {self.center_idx}")
        print(f"🔢 Value range: {stats['min_value']:.6f} to {stats['max_value']:.6f}")
    
    def setup_gui(self):
        """Set up the interactive GUI."""
        # Create figure with space for controls
        self.fig = plt.figure(figsize=(14, 10))
        
        # Main plot area
        self.ax = plt.subplot2grid((10, 10), (1, 0), rowspan=8, colspan=8)
        
        # Display the grid
        self.update_display()
        
        # Set up mouse event handlers
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        
        # Create controls
        self.setup_controls()
        
        # Info text area
        self.info_text = self.fig.text(0.82, 0.5, "", fontsize=10, verticalalignment='center',
                                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Initial info update
        self.update_info_text("Hover over the grid to explore values\nClick to see detailed information")
        
        plt.tight_layout()
        
        # Instructions
        processing_info = []
        if self.is_clipped:
            processing_info.append(f"clipped: -{self.clip_pixels}px")
        if self.is_downsampled:
            processing_info.append(f"downsampled: {self.grid_size}×{self.grid_size} ({self.downsample_method})")
        
        processing_str = f" ({', '.join(processing_info)})" if processing_info else ""
        
        self.fig.suptitle(f'Interactive Reachability Mask: {self.filename_base}.pkl{processing_str}\n'
                         f'Hover: show values | Click: detailed info | Controls: adjust display',
                         fontsize=12)
    
    def setup_controls(self):
        """Set up interactive controls."""
        # Toggle button for log scale
        ax_log = plt.axes([0.82, 0.85, 0.15, 0.04])
        self.btn_log = Button(ax_log, 'Toggle Log Scale')
        self.btn_log.on_clicked(self.toggle_log_scale)
        
        # Threshold sliders
        ax_min = plt.axes([0.82, 0.75, 0.15, 0.03])
        self.slider_min = Slider(ax_min, 'Min', 0.0, 1.0, valinit=0.0, valfmt='%.4f')
        self.slider_min.on_changed(self.update_thresholds)
        
        ax_max = plt.axes([0.82, 0.70, 0.15, 0.03])
        self.slider_max = Slider(ax_max, 'Max', 0.0, 1.0, valinit=1.0, valfmt='%.4f')
        self.slider_max.on_changed(self.update_thresholds)
        
        # Reset button
        ax_reset = plt.axes([0.82, 0.80, 0.15, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset View')
        self.btn_reset.on_clicked(self.reset_view)
    
    def grid_to_world(self, row, col):
        """Convert grid coordinates to world coordinates using API."""
        if self.api:
            if self.processed_data:
                # Use processed coordinate conversion
                return self.api.grid_to_world_processed(row, col, self.processed_data)
            else:
                # Use original coordinate conversion
                return self.api.grid_to_world(row, col)
        return (0.0, 0.0)
    
    def world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid coordinates using API."""
        if self.api:
            if self.processed_data:
                # For processed data, we need to handle the coordinate conversion manually
                # since we don't have a direct world_to_grid_processed method
                center = self.processed_data['final_center_idx']
                cell_size = self.processed_data['final_cell_size']
                
                # Convert world to grid coordinates
                grid_col = int(world_x / cell_size + center)
                grid_row = int(center - world_y / cell_size)
                
                return (grid_row, grid_col)
            else:
                return self.api.world_to_grid(world_x, world_y)
        return (0, 0)
    
    def update_display(self):
        """Update the main display."""
        # Clear the axis
        self.ax.clear()
        
        # Prepare data for display
        display_grid = self.grid.copy()
        
        # Apply thresholds
        mask = (display_grid >= self.min_threshold) & (display_grid <= self.max_threshold)
        display_grid[~mask] = np.nan
        
        # Apply log scale if enabled
        if self.show_log_scale:
            display_grid = np.log10(display_grid + 1e-10)
            display_grid[self.grid == 0] = np.nan
            colormap = 'viridis'
            label = 'Log₁₀(Reachability)'
        else:
            colormap = 'hot'
            label = 'Reachability Probability'
        
        # Display the grid
        self.im = self.ax.imshow(display_grid, cmap=colormap, origin='upper', 
                               interpolation='nearest', aspect='equal')
        
        # Add center marker
        self.ax.plot(self.center_idx, self.center_idx, 'c+', markersize=15, 
                    markeredgewidth=3, label='Center (0,0)')
        
        # Set labels and title
        self.ax.set_xlabel('Grid Column (East →)')
        self.ax.set_ylabel('Grid Row (North ↑)')
        title = f'{self.grid_size}×{self.grid_size} Grid'
        
        # Add processing information to title
        processing_parts = []
        if self.is_clipped:
            processing_parts.append(f'clipped: -{self.clip_pixels}px')
        if self.is_downsampled:
            processing_parts.append(f'downsampled: {self.downsample_method}')
        
        if processing_parts:
            title += f' ({", ".join(processing_parts)})'
            
        if self.show_log_scale:
            title += ' (Log Scale)'
        if self.min_threshold > 0 or self.max_threshold < 1:
            title += f' [Filtered: {self.min_threshold:.4f}-{self.max_threshold:.4f}]'
        self.ax.set_title(title)
        
        # Update colorbar
        if self.colorbar:
            self.colorbar.remove()
        self.colorbar = plt.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
        self.colorbar.set_label(label)
        
        # Add legend
        self.ax.legend(loc='upper right')
        
        # Refresh
        self.fig.canvas.draw()
    
    def on_mouse_move(self, event):
        """Handle mouse movement over the plot."""
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            col = int(round(event.xdata))
            row = int(round(event.ydata))
            
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                value = self.grid[row, col]
                world_x, world_y = self.grid_to_world(row, col)
                
                # Distance from center
                distance = np.sqrt(world_x**2 + world_y**2)
                
                hover_info = f"Grid: ({row}, {col})\n"
                hover_info += f"World: ({world_x:.1f}, {world_y:.1f})px\n"
                hover_info += f"Distance: {distance:.1f}px\n"
                hover_info += f"Value: {value:.6f}"
                
                if value > 0:
                    hover_info += f"\nLog₁₀: {np.log10(value):.3f}"
                else:
                    hover_info += f"\nLog₁₀: -∞ (zero)"
                
                self.update_info_text(hover_info)
    
    def on_mouse_click(self, event):
        """Handle mouse clicks on the plot."""
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            col = int(round(event.xdata))
            row = int(round(event.ydata))
            
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                value = self.grid[row, col]
                world_x, world_y = self.grid_to_world(row, col)
                distance = np.sqrt(world_x**2 + world_y**2)
                
                # Show detailed information
                click_info = f"🎯 DETAILED INFO\n"
                click_info += f"━━━━━━━━━━━━━━━━\n"
                click_info += f"Grid coords: [{row}, {col}]\n"
                click_info += f"World coords: ({world_x:.2f}, {world_y:.2f})px\n"
                click_info += f"Distance from center: {distance:.2f}px\n"
                click_info += f"Reachability: {value:.8f}\n"
                
                if value > 0:
                    click_info += f"Log₁₀ value: {np.log10(value):.4f}\n"
                    
                    # Find relative rank using API
                    percentile = self.api.get_percentile_rank(value)
                    click_info += f"Percentile: {percentile:.2f}%\n"
                    
                    # Local neighborhood analysis using API
                    neighborhood_analysis = self.api.get_neighborhood_analysis(row, col, radius=2)
                    click_info += f"\n📊 5×5 Neighborhood:\n"
                    click_info += f"Mean: {neighborhood_analysis['mean_value']:.6f}\n"
                    click_info += f"Max: {neighborhood_analysis['max_value']:.6f}\n"
                    click_info += f"Non-zero: {neighborhood_analysis['non_zero_count']}/{neighborhood_analysis['neighborhood_size']}\n"
                else:
                    click_info += f"Status: Unreachable (zero)\n"
                
                self.update_info_text(click_info)
                
                print(f"Clicked on grid[{row}, {col}] = {value:.8f} at world({world_x:.2f}, {world_y:.2f})")
    
    def update_info_text(self, text):
        """Update the info text display."""
        if self.info_text:
            self.info_text.set_text(text)
            self.fig.canvas.draw_idle()
    
    def toggle_log_scale(self, event):
        """Toggle between linear and log scale display."""
        self.show_log_scale = not self.show_log_scale
        print(f"Log scale: {'ON' if self.show_log_scale else 'OFF'}")
        self.update_display()
    
    def update_thresholds(self, val):
        """Update the display thresholds."""
        self.min_threshold = self.slider_min.val
        self.max_threshold = self.slider_max.val
        
        # Ensure min <= max
        if self.min_threshold > self.max_threshold:
            if hasattr(val, 'label') and 'Min' in val.label.get_text():
                self.slider_max.set_val(self.min_threshold)
                self.max_threshold = self.min_threshold
            else:
                self.slider_min.set_val(self.max_threshold)
                self.min_threshold = self.max_threshold
        
        print(f"Threshold range: [{self.min_threshold:.4f}, {self.max_threshold:.4f}]")
        self.update_display()
    
    def reset_view(self, event):
        """Reset all display settings."""
        self.show_log_scale = False
        self.min_threshold = 0.0
        self.max_threshold = 1.0
        self.slider_min.reset()
        self.slider_max.reset()
        print("View reset to defaults")
        self.update_display()
    
    def show(self):
        """Show the interactive viewer."""
        if self.fig:
            plt.show()
        else:
            print("❌ Could not create viewer - no data loaded")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive reachability mask viewer")
    parser.add_argument("--filename", "-f", default="unicycle_grid", 
                       help="Base filename of the pickle file (default: unicycle_grid)")
    parser.add_argument("--clip", "-c", type=float, default=0.0,
                       help="Number of pixels to clip from each side (default: 0 = no clipping)")
    parser.add_argument("--resize", "-r", type=str, default=None,
                       help="Downsample to target size, format: WIDTHxHEIGHT (e.g., 120x120)")
    parser.add_argument("--method", "-m", choices=['bilinear', 'nearest', 'max_pool', 'mean_pool'], default='max_pool',
                       help="Downsampling method (default: max_pool)")
    
    args = parser.parse_args()
    
    # Parse resize argument
    target_size = None
    if args.resize:
        try:
            if 'x' in args.resize.lower():
                w, h = map(int, args.resize.lower().split('x'))
                target_size = (h, w)  # (height, width)
            else:
                # Single number means square
                size = int(args.resize)
                target_size = (size, size)
        except ValueError:
            print(f"❌ Invalid resize format: {args.resize}")
            print("💡 Use format like '120x120' or '120' for square")
            return
    
    print("🎯 Interactive Reachability Mask Viewer")
    print("=" * 50)
    
    processing_steps = []
    if args.clip > 0:
        processing_steps.append(f"🔪 Clipping: {args.clip} pixels from each side")
    if target_size:
        processing_steps.append(f"📉 Downsampling: to EXACTLY {target_size[0]}×{target_size[1]} using {args.method}")
    
    if processing_steps:
        print("Processing Pipeline:")
        for step in processing_steps:
            print(f"  {step}")
    
    print("Controls:")
    print("• Hover: Show grid values and coordinates")
    print("• Click: Detailed information and statistics")
    print("• Toggle Log Scale: Switch between linear and logarithmic display")
    print("• Min/Max sliders: Filter values by threshold")
    print("• Reset View: Restore default settings")
    print("=" * 50)
    
    viewer = InteractiveReachabilityViewer(args.filename, args.clip, target_size, args.method)
    viewer.show()


if __name__ == "__main__":
    main()
