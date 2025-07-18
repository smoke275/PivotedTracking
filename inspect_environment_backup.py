#!/usr/bin/env python3
"""
Run the environment inspection tool.
This script launches a visualization of just the environment without any agents.
"""
import os
import sys

# Make sure the project directory is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the environment inspection simulator
from multitrack.simulation.environment_inspection_simulation_backup import run_environment_inspection

if __name__ == "__main__":
    # Use default settings
    multicore = True
    num_cores = None  # Auto-detect
    analyze_on_start = False
    load_visibility = True  # Auto-load visibility data
    visibility_cache_file = None  # Use default cache file
    
    print("Starting Environment Inspection Mode...")
    print("Multicore processing: Enabled (auto cores)")
    print("Visibility range: 1600 pixels")
    
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
    print("    J: Toggle probability overlay for second agent (pink-green blend, dynamic visibility circle)")
    print("       Range: Agent 1 reachability + inter-agent distance (max 800px)")
    print("       Visibility-based probability: config value for visible nodes, 0 for non-visible")
    print("       Automatically enables rotating rods visualization for agent 2")
    print("    K: Toggle visibility indicators for second agent (dynamic camera range)")
    print("       Range: Agent 1 reachability + inter-agent distance (max 800px)")
    print("       Cyan: Near-to-far transitions, Green-cyan: Far-to-near transitions")
    print("    H: Toggle extended probability set for agent 1 (gap arcs vs green visibility lines)")
    print("    Y: Toggle rotating rods (needs: visibility data + node + probability ON)")
    print("       Agent 1: Blue/violet arcs with yellow indicators")
    print("       Agent 2: Cyan/green arcs with cyan indicators (also via J key)")
    print("    M: Toggle combined probability mode (multiply Agent 1 & Agent 2 probabilities)")
    print("       Purple-yellow color scheme: low to high combined probability")
    print("       Auto-enables both agent probability overlays when activated")
    print("       Only shows nodes where both agents have significant probability (>0.1 threshold)")
    print("    I: Toggle probability overlay (requires combined mode to be enabled)")
    print("       Shows live histogram of combined probability distribution on screen")
    print("    U: Toggle threat classification mode (rod-based threat analysis)")
    print("       Classifies threats by which rod sweep created them using different colors")
    print("       Uses same threshold (0.1) as combined mode - all colored nodes become threats")
    print("       Rod colors: Red, Green, Blue, Orange, Magenta, Cyan, Yellow, Purple")
    print("       White nodes: Visibility-only threats, Multiple dots: Overlapping rods")
    print("       Shows mean threat probability for each gap next to gap points")
    print("       Format: 'Rod X: 0.XXX (count)' - mean probability and threat count per rod")
    print("       All nodes shown in M mode will be classified as threats in U mode")
    print("    Z: Auto-enable agent 1 features (F+O+B+Y+H) - Simplified shortcut")
    print("    X: Complete dual-agent system (Z+G+M) - Ultimate visualization mode")
    print("       Combines: Agent 1 features + Map graph + Combined probability mode")
    print("    +/=: Increase time horizon | -: Decrease time horizon")
    print("    ESC: Quit the application")
    print("")
    print("  Prerequisites for Rotating Rods: 1) Visibility data (V/L) 2) Select node 3) Prob overlay (O) 4) Press Y")
    print("  Quick Start: Press Z to enable agent 1 features (F+O+B+Y+H) in one step")
    print("  Ultimate Mode: Press X for complete dual-agent system (Z+G+M) - All features enabled!")
    
    # Run the environment inspection with default arguments
    run_environment_inspection(
        multicore=multicore, 
        num_cores=num_cores,
        auto_analyze=analyze_on_start,
        load_visibility=load_visibility,
        visibility_cache_file=visibility_cache_file
    )
