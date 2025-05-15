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
- Path planning with graph-based and dynamically feasible paths
- Environment inspection tools for map analysis

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

## Additional Tools

### Environment Inspection

The project includes an environment inspection tool that allows you to:
- Visualize and analyze the map graph
- Test visibility between nodes
- Generate and display paths between nodes

To run the environment inspection tool:

```bash
python inspect_environment.py
```

#### Environment Inspection Controls

- **Map Graph:**
  - **G**: Toggle map graph display
  - **R**: Regenerate map graph
- **Visibility Analysis:**
  - **V**: Analyze node visibility and save to cache
  - **L**: Load visibility data from cache
- **Node Navigation:**
  - **Mouse**: Left-click on nodes to select them
  - Right-click to find path from agent to node
  - **N**: Go to next node
  - **P**: Go to previous node
- **Path Visualization:**
  - **T**: Toggle path visibility
  - **C**: Clear current path
  - Yellow path: Graph-based shortest path
  - Cyan dashed path: Dynamically feasible path respecting agent motion constraints

### Reachability Visualization

For advanced analysis, the project also includes a reachability visualization tool:

```bash
python reachability_visualization.py
```

This tool visualizes the reachable set for an agent at its current location with probabilities for each map node.

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

### May 15, 2025: Path Planning and Navigation Enhancements
- Added path planning functionality to the environment inspection tool
- Implemented graph-based shortest path algorithm with A* pathfinding
- Added dynamically feasible path generation that respects agent motion constraints
- Enhanced inspection UI with path visualization controls
- Added path comparison metrics (length, nodes, feasibility)
- Implemented mouse navigation with left-click for node selection and right-click for path finding
- Created new pathfinding utility module for reuse across different simulation modes

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

### May 15, 2025: Environment Inspection UI Improvements
- Simplified visibility cache information display in UI
- Added support for custom visibility cache file locations
- Improved cache handling by showing only relevant information (custom vs. default)
- Enhanced visibility analysis on multi-core systems
- Fixed issue with visibility through doors in map analysis

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

### Environment Inspection Tool

The project includes a dedicated environment inspection tool that allows you to visualize and analyze the simulation environment:

```bash
python inspect_environment.py
```

#### Inspection Tool Options
- **--visibility-cache/-c**: Specify custom file path for visibility cache
- **--visibility-range/-r**: Set maximum visibility range in pixels (default: 1600)
- **--analyze-on-start/-a**: Automatically analyze visibility on startup
- **--load-visibility/-l**: Load visibility data from cache on startup
- **--no-cache**: Disable all caching for fresh generation
- **--multicore/-m**: Use multicore processing (default: enabled)
- **--single-core/-s**: Use single-core processing only
- **--num-cores/-n**: Specify number of CPU cores to use