#!/usr/bin/env python3
"""
Main entry point for the MultiTrack simulation.
This script launches the unicycle simulation with Kalman filter and MPPI controller.
"""
import argparse
import multiprocessing
import time
import os
from multitrack.utils.config import *
from multitrack.simulation import run_simulation, WIDTH, HEIGHT

# Required for Windows multiprocessing - always good practice to include
if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better cross-platform compatibility
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Method already set
        pass

def parse_arguments():
    """Parse command line arguments for simulation settings"""
    parser = argparse.ArgumentParser(description="MultiTrack Simulation")
    
    # Add screen size arguments
    parser.add_argument("--width", type=int, default=WIDTH,
                        help=f"Screen width in pixels (default: {WIDTH})")
    parser.add_argument("--height", type=int, default=HEIGHT,
                        help=f"Screen height in pixels (default: {HEIGHT})")
    
    # Add multiprocessing option for map generation
    parser.add_argument("--multicore", "-m", action="store_true", default=True,
                        help="Use multiple CPU cores for map graph generation (default: True)")
    parser.add_argument("--single-core", "-s", action="store_true",
                        help="Use only a single CPU core for map graph generation")
    parser.add_argument("--num-cores", "-n", type=int, default=None,
                        help="Number of CPU cores to use for map generation (default: all available)")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Apply settings to simulation
    from multitrack.simulation.unicycle_reachability_simulation import (
        ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, SIDEBAR_WIDTH, WIDTH, HEIGHT
    )
    
    # Note: We only need to override ENVIRONMENT_WIDTH and ENVIRONMENT_HEIGHT
    # as WIDTH and HEIGHT will be automatically calculated based on the sidebar width
    from multitrack.simulation.unicycle_reachability_simulation import WIDTH as sim_WIDTH, HEIGHT as sim_HEIGHT
    sim_WIDTH = args.width  # This will become the environment width
    sim_HEIGHT = args.height
    
    # Update the constants in the simulation module
    import multitrack.simulation.unicycle_reachability_simulation as sim
    sim.ENVIRONMENT_WIDTH = args.width
    sim.ENVIRONMENT_HEIGHT = args.height
    sim.WIDTH = args.width + sim.SIDEBAR_WIDTH  # Total width includes sidebar
    
    # Handle the single-core flag (overrides multicore)
    if args.single_core:
        args.multicore = False
    
    print(f"Starting MultiTrack simulation with screen size {args.width}x{args.height}...")
    print(f"Multicore processing: {'Enabled' if args.multicore else 'Disabled'}")
    print("Controls:")
    print("  - Arrow keys: Control the visitor")
    print("  - WASD: Manual control of the escort (when in manual mode)")
    print("  - Q/E: Rapidly rotate the escort's secondary camera (hold for continuous rotation)")
    print("  - A: Toggle camera auto-tracking (camera will automatically follow the visitor)")
    print("  - C: Toggle between manual and auto-tracking modes for the escort")
    print("  - +/-: Adjust following distance")
    print("  - R: Reset escort position")
    print("  - ESC: Quit")
    
    # Run simulation with multiprocessing arguments
    run_simulation(multicore=args.multicore, num_cores=args.num_cores)