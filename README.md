# PivotedTracking

A simulation environment for visitor tracking with escort agents using Kalman filters and MPPI control.

## Project Overview

This project implements a dynamic simulation of a visitor-escort scenario where:

- A **Visitor** (red agent) can be moved manually using arrow keys
- An **Escort** (orange agent) can either:
  - Automatically track and follow the visitor using Model Predictive Path Integral (MPPI) control
  - Be controlled manually using WASD keys

The simulation includes:
- Real-time Kalman filter-based state estimation
- Uncertainty visualization with covariance ellipses
- Vision cones for the escort agent (green when visitor is visible, yellow when not visible)
- Wall and obstacle collision detection
- Dynamic entropy measurement of the visitor's estimated state

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
- **+/-**: Adjust following distance
- **R**: Reset escort position

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
  - `agents/escort_agent.py`: Escort agent with MPPI controller
- `multitrack/controllers/`: Control algorithms
  - `mppi_controller.py`: Model Predictive Path Integral controller
- `multitrack/filters/`: State estimation
  - `kalman_filter.py`: Kalman filter implementation
- `multitrack/utils/`: Utility functions
  - `vision.py`: Vision and detection utilities
  - `geometry.py`: Geometric calculations
  - `config.py`: Configuration parameters
- `multitrack/visualizations/`: Rendering utilities
  - `enhanced_rendering.py`: Advanced visual effects
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
- MPPI-based optimal control for tracking
- Vision cone with line-of-sight detection
- Manual control mode
- Minimum safety distance enforcement
- Search behavior when visitor is not visible
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