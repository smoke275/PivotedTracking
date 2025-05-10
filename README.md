# PivotedTracking

A simulation environment for visitor tracking with escort agents using Kalman filters and multiple controller types.

## Project Overview

This project implements a dynamic simulation of a visitor-escort scenario where:

- A **Visitor** (red agent) can be moved manually using arrow keys
- An **Escort** (orange agent) can either:
  - Automatically track and follow the visitor using configurable controllers (MPPI or PID)
  - Be controlled manually using WASD keys

The simulation includes:
- Real-time Kalman filter-based state estimation
- Uncertainty visualization with covariance ellipses
- Dual camera system with primary and secondary vision cones
- Wall and obstacle collision detection
- Dynamic entropy measurement of the visitor's estimated state
- Map graph generation for navigation planning

## Images and Documentation

The `images/` directory contains screenshots and visual documentation of the simulation:
- Implementation diagrams
- System architecture
- Sample tracking scenarios
- UI elements and controls

You can add your own screenshots to this directory by pressing the Print Screen key during simulation or using your preferred screen capture tool.

## Setup Instructions

### Prerequisites

- Python 3.10+ (developed with Python 3.12)
- Pygame
- NumPy
- PyTorch (for MPPI controller)

### Environment Setup

You can set up the environment using the provided `environment.yml` file with Conda:

```bash
# Create and activate a conda environment
conda env create -f environment.yml
conda activate pivoted-tracking
```

Alternatively, you can install the required packages manually:

```bash
pip install pygame numpy torch
```

### Running the Simulation

To run the simulation, execute the main script:

```bash
python main.py
```

## Controls

### Visitor Control
- **Arrow Keys**: Move the visitor (red agent)
  - Up/Down: Move forward/backward
  - Left/Right: Turn left/right

### Escort Control
- **C**: Toggle between auto-tracking and manual control modes
- **WASD** (when in manual mode):
  - W/S: Move forward/backward
  - A/D: Turn left/right
- **P**: Toggle between MPPI and PID controllers
- **+/-**: Adjust following distance
- **R**: Reset escort position

### Camera Control
- **Q/E**: Rotate secondary camera (when manual camera control is active)
- **A**: Toggle camera auto-tracking mode

### Display Options
- **F**: Toggle FPS display
- **K**: Toggle Kalman filter prediction visualization
- **U**: Toggle uncertainty ellipse display
- **M**: Toggle MPPI predictions display
- **T**: Toggle escort agent (enable/disable)
- **D**: Toggle debug view
- **V**: Toggle enhanced visuals
- **Shift+/-** or **Shift++**: Adjust measurement interval

### General
- **ESC**: Quit the simulation

## Project Structure

- `multitrack/models/`: Core agent models
  - `agents/visitor_agent.py`: Visitor agent with Kalman filter tracking
  - `agents/escort_agent.py`: Escort agent with switchable controllers
- `multitrack/controllers/`: Control algorithms
  - `base_controller.py`: Abstract controller interface
  - `mppi_controller.py`: Model Predictive Path Integral controller
  - `pid_controller.py`: PID controller implementation
- `multitrack/filters/`: State estimation
  - `kalman_filter.py`: Kalman filter implementation
- `multitrack/utils/`: Utility functions
  - `vision.py`: Vision and detection utilities
  - `geometry.py`: Geometric calculations
  - `config.py`: Configuration parameters
  - `map_graph.py`: Graph-based navigation map
- `multitrack/visualizations/`: Rendering utilities
  - `enhanced_rendering.py`: Advanced visual effects
  - `information_overlay.py`: Threaded information display
- `multitrack/simulation/`: Simulation environment
  - `unicycle_reachability_simulation.py`: Main simulation loop

## Features

### Visitor Agent
- Unicycle dynamics model
- Kalman filter tracking with visualization
- Adaptive measurement interval
- Collision handling with walls and obstacles
- Entropy calculation for uncertainty measurement
- Smart initialization to avoid starting in walls

### Escort Agent
- Multiple controller options:
  - MPPI-based optimal control for predictive tracking
  - PID controller for simpler, reactive tracking
- Dual vision system:
  - Primary fixed forward-facing camera
  - Secondary rotatable camera (manual or auto-tracking)
- Vision cones with line-of-sight detection
- Manual control mode
- Minimum safety distance enforcement
- Search behavior when visitor is not visible
- Stuck detection and recovery mechanisms
- Intelligent positioning to avoid starting in walls

## Visualization

The simulation provides rich visual feedback:
- Kalman filter state estimates (green)
- Vision cones (green when visitor visible, yellow when not)
- Uncertainty ellipses showing state covariance
- Predicted trajectories for both agents
- Particle effects for movement
- Enhanced human-like rendering

## Technical Notes

1. The Kalman filter implementation uses a unicycle model for state transitions.
2. The MPPI controller optimizes trajectories using GPU acceleration when available.
3. The vision system uses ray casting for line-of-sight detection.
4. Entropy calculation provides a single metric for tracking uncertainty.

## Recent Updates

### May 19, 2025: Controller Architecture Improvements
- Added abstract base controller interface for better extensibility
- Implemented PID controller as an alternative to MPPI
- Fixed controller switching to improve robustness
- Enhanced controller architecture to support seamless switching between controller types
- Improved simulation stability with proper controller error handling

### May 15, 2025: Enhanced Vision System
- Implemented dual camera system (primary fixed and secondary rotatable)
- Added manual camera control with Q/E keys for precise aiming
- Developed camera auto-tracking with PID-based target following
- Improved search patterns when visitor is lost from view
- Enhanced vision cone visualization with camera-specific colors

### May 10, 2025: Threaded Information Overlay System
- Moved information overlay rendering to a separate thread
- Improved performance by decoupling visualization from simulation logic
- Enhanced transparency handling for better visual clarity
- Added thread-safe communication between simulation and visualization systems

### May 7, 2025: Improved Map Generation Responsiveness
- Enhanced map graph generation to maintain UI responsiveness
- Implemented adaptive chunking strategy for better load balancing
- Added frequent UI event processing to prevent freezing
- Optimized both sampling and connection phases with smaller work units
- No reduction in processing power - still uses all available cores for maximum performance

### Performance Options
- **--multicore/-m**: Use multiple CPU cores for map generation (default: enabled)
- **--single-core/-s**: Use only a single core for map generation
- **--num-cores/-n**: Specify exact number of cores to use
- **--limit-cores**: Set maximum number of cores to use (default: 16)
- **--skip-cache**: Force regeneration of the map graph