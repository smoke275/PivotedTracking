#!/usr/bin/env python3
"""
Run the environment inspection tool.
This script launches a visualization of just the environment without any agents.

🎯 OPTIMIZATION PROJECT ENTRY POINT 🎯
Agent 2 probability generation optimization is currently in progress.

CURRENT STATUS: Phase 1 Complete → Phase 2 Ready (Spatial Optimization)

📋 OPTIMIZATION DOCUMENTATION:
- PRIMARY PLAN: agent2_optimization_plan.md (active, concise <100 lines)
- PHASE 1 RESULTS: PHASE1_COMPLETION_SUMMARY.md (detailed findings)
- HISTORICAL CONTEXT: agent2_optimization_archive.md (full background)
- PERFORMANCE DATA: agent2_performance_log.csv (live measurements)

🎯 TARGET: Lines 2649-2860 in environment_inspection_simulation.py
   Currently: 4.4-4.6s computation per frame → Goal: <16ms (60 FPS)

🔧 TEST PROCEDURE: python inspect_environment.py → Press 'J' → Move with WASD
   Monitor FPS panel for "Agent 2 computation: X ms" timing
"""
import os
import sys
import argparse

# Make sure the project directory is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the environment inspection simulator
from multitrack.simulation.environment_inspection_simulation import run_environment_inspection

def parse_arguments():
    """Parse command line arguments for inspection settings"""
    parser = argparse.ArgumentParser(description="Environment Inspection Tool")
    
    # Add multiprocessing options for map generation and visibility analysis
    parser.add_argument("--multicore", "-m", action="store_true", default=True,
                        help="Use multiple CPU cores for processing (default: True)")
    parser.add_argument("--single-core", "-s", action="store_true",
                        help="Use only a single CPU core (disables parallel processing)")
    parser.add_argument("--num-cores", "-n", type=int, default=None,
                        help="Number of CPU cores to use (default: auto-detect, leaves 1 core free)")
    
    # Visibility analysis options
    parser.add_argument("--visibility-range", "-r", type=int, default=1600,
                        help="Maximum range in pixels for visibility analysis (default: 1600)")
    parser.add_argument("--analyze-on-start", "-a", action="store_true",
                        help="Automatically analyze visibility when starting (default: False)")
    parser.add_argument("--load-visibility", "-l", action="store_true",
                        help="Load cached visibility data on startup (default: False)")
    parser.add_argument("--no-auto-load", action="store_true",
                        help="Disable automatic loading of visibility data (default: False)")
    parser.add_argument("--visibility-cache", "-c", type=str, default=None,
                        help="Custom file path for visibility cache (default: uses config file path)")
    
    # Development options
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable all caching for fresh generation (default: False)")
    
    return parser.parse_args()

if __name__ == "__main__":
    # jarse command line arguments
    args = parse_arguments()
    
    # Handle the single-core flag (overrides multicore)
    if args.single_core:
        args.multicore = False
    
    # Update configuration based on command-line arguments
    from multitrack.utils.config import MAP_GRAPH_VISIBILITY_RANGE
    import multitrack.utils.config as config
    
    # Update visibility range if specified
    if args.visibility_range != 1600:
        config.MAP_GRAPH_VISIBILITY_RANGE = args.visibility_range
        print(f"Using custom visibility range: {args.visibility_range} pixels")
    
    # Handle visibility cache path
    visibility_cache_path = None
    if args.visibility_cache:
        visibility_cache_path = args.visibility_cache
        print("Using custom visibility cache")
        # Make sure the directory exists
        cache_dir = os.path.dirname(visibility_cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
    
    # Handle no-cache option
    if args.no_cache:
        config.MAP_GRAPH_CACHE_ENABLED = False
        print("Cache disabled - all data will be generated fresh")
    
    print("Starting Environment Inspection Mode...")
    print(f"Multicore processing: {'Enabled (' + str(args.num_cores if args.num_cores else 'auto') + ' cores)' if args.multicore else 'Disabled'}")
    print(f"Visibility range: {config.MAP_GRAPH_VISIBILITY_RANGE} pixels")
    
    print("\nControls:")
    print("  Map Graph:")
    print("    G: Toggle map graph display")
    print("    R: Regenerate map graph")
    print("  Visibility Analysis:")
    print("    V: Analyze node visibility and save to cache (may take time for large maps)")
    print("    L: Load visibility data from cache")
    print("  Node Navigation:")
    print("    Mouse: Left-click on nodes to select them")
    print("           Right-click to find path from agent to node")
    print("    N: Go to next node")
    print("    P: Go to previous node")
    print("  Path Visualization:")
    print("    T: Toggle path visibility")
    print("    C: Clear current path")
    print("    Yellow path: Graph-based shortest path")
    print("    Cyan dashed path: Dynamically feasible path respecting agent motion constraints")
    print("  Other:")
    print("    F: Toggle agent-following mode")
    print("    O: Toggle probability overlay (light blue-red color blend, colors nodes based on reachability within time horizon)")
    print("    B: Toggle visibility gaps for first agent (ray casting discontinuities)")
    print("       Blue: Near-to-far transitions, Violet: Far-to-near transitions")
    print("    J: Toggle probability overlay for second agent (pink-green blend, 800px visibility circle)")
    print("       Visibility-based probability: config value for visible nodes, 0 for non-visible")
    print("       Automatically enables rotating rods visualization for agent 2")
    print("    K: Toggle visibility indicators for second agent (camera range: 800px)")
    print("       Cyan: Near-to-far transitions, Green-cyan: Far-to-near transitions")
    print("    H: Toggle extended probability set for agent 1 (gap arcs vs green visibility lines)")
    print("    Y: Toggle rotating rods (needs: visibility data + node + probability ON)")
    print("       Agent 1: Blue/violet arcs with yellow indicators")
    print("       Agent 2: Cyan/green arcs with cyan indicators (also via J key)")
    print("    M: Toggle combined probability mode (multiply Agent 1 & Agent 2 probabilities)")
    print("       Purple-yellow color scheme: low to high combined probability")
    print("       Auto-enables both agent probability overlays when activated")
    print("    Z: Auto-enable agent 1 features (F+O+B+Y+H) - Simplified shortcut")
    print("    +/=: Increase time horizon | -: Decrease time horizon")
    print("    ESC: Quit the application")
    print("")
    print("  Prerequisites for Rotating Rods: 1) Visibility data (V/L) 2) Select node 3) Prob overlay (O) 4) Press Y")
    print("  Quick Start: Press Z to enable agent 1 features (F+O+B+Y+H) in one step")
    
    # Advanced options information
    if args.analyze_on_start or args.load_visibility or args.no_cache or args.no_auto_load:
        print("\nActive Advanced Options:")
        if args.analyze_on_start:
            print("  • Auto-analyze visibility on startup")
        if args.load_visibility:
            print("  • Auto-load visibility data from cache")
        if not args.no_auto_load:
            print("  • Automatically loading visibility cache if available")
        if args.no_auto_load:
            print("  • Automatic visibility cache loading disabled")
        if args.no_cache:
            print("  • Cache disabled for this session")
    
    # Auto-load visibility data by default unless explicitly disabled
    auto_load_visibility = not args.no_auto_load
    
    # Run the environment inspection with additional arguments
    run_environment_inspection(
        multicore=args.multicore, 
        num_cores=args.num_cores,
        auto_analyze=args.analyze_on_start,
        load_visibility=args.load_visibility or auto_load_visibility,  # Always load visibility data unless explicitly disabled
        visibility_cache_file=visibility_cache_path
    )
