#!/usr/bin/env python3
"""
Interactive Reachability Viewer with Default Settings
Launches the interactive reachability mask viewer with optimized default settings:
- 50px clipping from each side
- Exact 120×120 downsampling 
- max_pool method (preserves peak reachability values)
- Normalized output (values in [0, 1] range)
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from interactive_reachability_viewer import InteractiveReachabilityViewer


def main():
    """Launch interactive reachability viewer with default optimized settings."""
    
    parser = argparse.ArgumentParser(
        description="Interactive Reachability Viewer (Optimized Defaults)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DEFAULT SETTINGS:
  --clip 50        : Remove 50 pixels from each side (focuses on core reachable area)
  --resize 120x120 : Downsample to exactly 120×120 grid for optimal performance
  --method max_pool: Preserve peak reachability values (best for reachability data)

FEATURES:
  • Normalized output: All values in [0, 1] range for easy comparison
  • Exact sizing: Output is guaranteed to be exactly 120×120
  • Interactive controls: Hover, click, zoom, threshold filtering
  • Multiple display modes: Linear and logarithmic scaling
  • Real-time statistics and coordinate conversion

CONTROLS:
  • Hover: Show grid values and world coordinates
  • Click: Detailed cell information and neighborhood analysis
  • Toggle Log Scale: Switch between linear and log display
  • Min/Max sliders: Filter values by threshold range
  • Reset View: Restore default display settings

EXAMPLES:
  python interactive_reachability_default.py                    # Use all defaults
  python interactive_reachability_default.py --file grid2      # Different data file
  python interactive_reachability_default.py --no-clip         # Skip clipping step
  python interactive_reachability_default.py --method bilinear # Use smooth interpolation
        """
    )
    
    parser.add_argument("--file", "-f", default="unicycle_grid", 
                       help="Base filename of pickle file (default: unicycle_grid)")
    
    parser.add_argument("--clip", "-c", type=float, default=50.0,
                       help="Pixels to clip from each side (default: 50.0)")
    
    parser.add_argument("--no-clip", action="store_true",
                       help="Skip clipping step (overrides --clip)")
    
    parser.add_argument("--resize", "-r", type=str, default="120x120",
                       help="Target size WIDTHxHEIGHT (default: 120x120)")
    
    parser.add_argument("--no-resize", action="store_true",
                       help="Skip downsampling step (overrides --resize)")
    
    parser.add_argument("--method", "-m", 
                       choices=['max_pool', 'bilinear', 'nearest', 'mean_pool'], 
                       default='max_pool',
                       help="Downsampling method (default: max_pool)")
    
    args = parser.parse_args()
    
    # Process arguments
    clip_pixels = 0.0 if args.no_clip else args.clip
    
    target_size = None
    if not args.no_resize:
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
            return 1
    
    # Display configuration
    print("🎯 Interactive Reachability Viewer (Optimized Defaults)")
    print("=" * 60)
    print(f"📁 Data file: {args.file}.pkl")
    
    processing_steps = []
    if clip_pixels > 0:
        processing_steps.append(f"🔪 Clipping: {clip_pixels} pixels from each side")
    if target_size:
        processing_steps.append(f"📉 Downsampling: to EXACTLY {target_size[0]}×{target_size[1]} using {args.method}")
    
    if processing_steps:
        print("🔧 Processing Pipeline:")
        for step in processing_steps:
            print(f"  {step}")
    else:
        print("🔧 Processing: None (raw data)")
    
    print("\n✨ Optimizations:")
    print("  • Max-pool preserves peak reachability values")
    print("  • Normalization ensures [0, 1] value range")
    print("  • Exact sizing guarantees precise 120×120 output")
    print("  • Clipping focuses on core reachable region")
    
    print("\n🎮 Interactive Controls:")
    print("  • Hover: Grid values and world coordinates")
    print("  • Click: Detailed cell analysis")
    print("  • Toggle Log Scale: Linear ↔ Logarithmic display")
    print("  • Sliders: Min/Max threshold filtering")
    print("  • Reset: Restore default view settings")
    
    print("=" * 60)
    
    # Create and launch viewer
    try:
        viewer = InteractiveReachabilityViewer(
            filename_base=args.file,
            clip_pixels=clip_pixels,
            target_size=target_size,
            downsample_method=args.method
        )
        
        if not viewer.api or not viewer.api.is_loaded():
            print(f"❌ Failed to load reachability data from {args.file}.pkl")
            print("💡 Make sure the file exists and contains valid reachability mask data")
            return 1
        
        print("🚀 Launching interactive viewer...")
        print("💡 Close the window or press Ctrl+C to exit")
        
        viewer.show()
        
    except KeyboardInterrupt:
        print("\n👋 Viewer closed by user")
        return 0
    except Exception as e:
        print(f"❌ Error launching viewer: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
