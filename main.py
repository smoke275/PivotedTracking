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
    print("Controls:")
    print("  - Arrow keys: Control the visitor")
    print("  - WASD: Manual control of the escort (when in manual mode)")
    print("  - Q/E: Rapidly rotate the escort's secondary camera (hold for continuous rotation)")
    print("  - A: Toggle camera auto-tracking (camera will automatically follow the visitor)")
    print("  - C: Toggle between manual and auto-tracking modes for the escort")
    print("  - +/-: Adjust following distance")
    print("  - R: Reset escort position")
    print("  - ESC: Quit")
    run_simulation()