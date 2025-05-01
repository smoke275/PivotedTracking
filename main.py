#!/usr/bin/env python3
"""
Main entry point for the MultiTrack simulation.
This script launches the unicycle simulation with Kalman filter and MPPI controller.
"""
import argparse
from multitrack.utils.config import *
from multitrack.simulation import run_simulation, WIDTH, HEIGHT

def parse_arguments():
    """Parse command line arguments for simulation settings"""
    parser = argparse.ArgumentParser(description="MultiTrack Simulation")
    
    # Add screen size arguments
    parser.add_argument("--width", type=int, default=WIDTH,
                        help=f"Screen width in pixels (default: {WIDTH})")
    parser.add_argument("--height", type=int, default=HEIGHT,
                        help=f"Screen height in pixels (default: {HEIGHT})")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Apply settings to simulation
    from multitrack.simulation.unicycle_reachability_simulation import WIDTH as sim_WIDTH, HEIGHT as sim_HEIGHT
    sim_WIDTH = args.width
    sim_HEIGHT = args.height
    
    print(f"Starting MultiTrack simulation with screen size {args.width}x{args.height}...")
    print("Note: The Kalman filter (green circle) updates with control information even when the visitor is out of sight.")
    print("      This is not realistic behavior and should be fixed by modifying visitor_agent.py to prevent control updates")
    print("      when is_visible=False. Currently the escort knows which way the visitor is moving even without seeing it.")
    run_simulation()