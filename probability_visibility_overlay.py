#!/usr/bin/env python3
"""
Probability-Based Visibility Overlay System

This module implements a visibility overlay that calculates the probability of being
able to observe different locations based on the agent's current state, heading angle,
dynamics, and time horizon. It combines trajectory prediction with visibility analysis
to create a dynamic probability map.
"""

import os
import sys
import pygame
import numpy as np
import math
import time
import multiprocessing
from math import sin, cos, pi, sqrt, exp
from typing import Dict, List, Tuple, Optional

# Make sure the project directory is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from multitrack.utils.config import *
from multitrack.models.simulation_environment import SimulationEnvironment
from multitrack.utils.map_graph import MapGraph
from multitrack.models.agents.visitor_agent import UnicycleModel
from multitrack.utils.pathfinding import find_shortest_path

# Import environment dimensions
from multitrack.simulation.unicycle_reachability_simulation import (
    ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, WIDTH, HEIGHT, SIDEBAR_WIDTH
)

class ProbabilityVisibilityOverlay:
    """
    Manages probability-based visibility overlay calculations and visualization.
    
    This class combines agent state, motion dynamics, time-based predictions,
    and visibility analysis to create a probability distribution over the 
    environment grid showing where the agent is likely to be visible from.
    """
    
    def __init__(self, map_graph: MapGraph, environment: SimulationEnvironment):
        """
        Initialize the probability visibility overlay system.
        
        Args:
            map_graph: The map graph with visibility data
            environment: The simulation environment with walls and doors
        """
        self.map_graph = map_graph
        self.environment = environment
        self.visibility_map = None
        
        # Grid for probability calculations
        self.grid_resolution = 30  # Increased from 20 to reduce grid size
        self.grid_width = ENVIRONMENT_WIDTH // self.grid_resolution
        self.grid_height = ENVIRONMENT_HEIGHT // self.grid_resolution
        
        # Time horizon parameters
        self.prediction_steps = 20  # Reduced from 30
        self.prediction_dt = 0.15   # Increased from 0.1
        self.time_horizon = self.prediction_steps * self.prediction_dt  # 3 seconds
        
        # Trajectory sampling parameters
        self.num_trajectory_samples = 50  # Reduced from 100
        
        # Visibility parameters
        self.visibility_range = MAP_GRAPH_VISIBILITY_RANGE
        self.detection_radius = 25  # Radius for considering a position "visible"
        
        # Cached probability grid
        self.probability_grid = np.zeros((self.grid_height, self.grid_width))
        self.last_agent_state = None
        self.cache_valid = False
        
    def set_visibility_map(self, visibility_map: Dict[int, List[int]]):
        """Set the visibility map for the overlay system."""
        self.visibility_map = visibility_map
        
    def generate_agent_trajectory_sample(self, agent_state: np.ndarray, 
                                       agent_controls: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Generate a single trajectory sample for the agent using unicycle dynamics.
        
        Args:
            agent_state: Current agent state [x, y, theta, v]
            agent_controls: Current agent controls (linear_vel, angular_vel)
            
        Returns:
            List of (x, y) positions along the predicted trajectory
        """
        x, y, theta, v = agent_state
        linear_vel, angular_vel = agent_controls
        
        trajectory = [(x, y)]
        
        # Add noise to initial state for sampling
        current_x = x + np.random.normal(0, 2)
        current_y = y + np.random.normal(0, 2)
        current_theta = theta + np.random.normal(0, 0.05)
        current_v = max(0, v + np.random.normal(0, 3))
        
        for step in range(self.prediction_steps):
            # Add control variation for realistic uncertainty
            v_noisy = current_v + np.random.normal(0, 5)
            omega_noisy = angular_vel + np.random.normal(0, 0.15)
            
            # Apply unicycle dynamics
            next_x = current_x + v_noisy * cos(current_theta) * self.prediction_dt
            next_y = current_y + v_noisy * sin(current_theta) * self.prediction_dt
            next_theta = current_theta + omega_noisy * self.prediction_dt
            
            # Add process noise
            next_x += np.random.normal(0, 1.5)
            next_y += np.random.normal(0, 1.5)
            next_theta += np.random.normal(0, 0.05)
            
            # Normalize angle
            next_theta = (next_theta + pi) % (2 * pi) - pi
            
            # Apply environment boundaries
            next_x = max(0, min(next_x, ENVIRONMENT_WIDTH))
            next_y = max(0, min(next_y, ENVIRONMENT_HEIGHT))
            
            # Check for collisions with walls
            collision = False
            agent_radius = 16
            
            for wall in self.environment.get_all_walls():
                if hasattr(wall, 'x'):
                    wx, wy, ww, wh = wall.x, wall.y, wall.width, wall.height
                else:
                    wx, wy, ww, wh = wall
                
                # Simple collision check - agent center within expanded wall
                if (wx - agent_radius <= next_x <= wx + ww + agent_radius and
                    wy - agent_radius <= next_y <= wy + wh + agent_radius):
                    
                    # Check if it's a door (passable)
                    is_door = False
                    for door in self.environment.get_doors():
                        if hasattr(door, 'x'):
                            dx, dy, dw, dh = door.x, door.y, door.width, door.height
                        else:
                            dx, dy, dw, dh = door
                        
                        if (dx <= next_x <= dx + dw and dy <= next_y <= dy + dh):
                            is_door = True
                            break
                    
                    if not is_door:
                        collision = True
                        break
            
            if collision:
                # If collision, stop trajectory or bounce back
                break
            
            trajectory.append((next_x, next_y))
            
            # Update state for next iteration
            current_x, current_y, current_theta = next_x, next_y, next_theta
            current_v = v_noisy
            
        return trajectory
    
    def calculate_visibility_probability_from_position(self, position: Tuple[float, float], 
                                                     target_positions: List[Tuple[float, float]]) -> float:
        """
        Calculate the probability of visibility from a given position to target positions.
        
        Args:
            position: Observer position (x, y)
            target_positions: List of target positions to check visibility to
            
        Returns:
            Probability value between 0 and 1
        """
        if not self.visibility_map:
            return 0.0
        
        # Find the closest map graph node to the observer position
        min_distance = float('inf')
        closest_node_index = None
        
        for i, node in enumerate(self.map_graph.nodes):
            distance = math.dist(position, node)
            if distance < min_distance:
                min_distance = distance
                closest_node_index = i
        
        if closest_node_index is None or min_distance > self.detection_radius:
            return 0.0
        
        # Get visible nodes from this position
        if closest_node_index not in self.visibility_map:
            return 0.0
        
        visible_node_indices = self.visibility_map[closest_node_index]
        
        # Count how many target positions are visible
        visible_targets = 0
        total_targets = len(target_positions)
        
        for target_pos in target_positions:
            # Check if target is near any visible node
            for visible_idx in visible_node_indices:
                if visible_idx < len(self.map_graph.nodes):
                    visible_node = self.map_graph.nodes[visible_idx]
                    distance_to_target = math.dist(target_pos, visible_node)
                    
                    if distance_to_target <= self.detection_radius:
                        visible_targets += 1
                        break  # Count each target only once
        
        return visible_targets / total_targets if total_targets > 0 else 0.0
    
    def calculate_temporal_visibility_probability(self, agent) -> np.ndarray:
        """
        Calculate probability grid considering agent dynamics and time horizon.
        
        Args:
            agent: The agent object with state and controls
            
        Returns:
            2D numpy array representing probability values for each grid cell
        """
        agent_state = agent.state
        agent_controls = agent.controls
        
        # Check if we need to recalculate (cache validity)
        if (self.cache_valid and self.last_agent_state is not None and
            np.allclose(self.last_agent_state, agent_state, atol=15.0)):  # Increased tolerance from 5.0 to 15.0
            return self.probability_grid
        
        print(f"Calculating temporal visibility probability for agent at ({agent_state[0]:.1f}, {agent_state[1]:.1f})")
        
        # Generate trajectory samples
        trajectories = []
        for _ in range(self.num_trajectory_samples):
            trajectory = self.generate_agent_trajectory_sample(agent_state, agent_controls)
            trajectories.append(trajectory)
        
        # Initialize probability grid
        probability_grid = np.zeros((self.grid_height, self.grid_width))
        
        # For each grid cell, calculate probability based on trajectory samples
        for grid_y in range(self.grid_height):
            for grid_x in range(self.grid_width):
                # Convert grid coordinates to world coordinates
                world_x = grid_x * self.grid_resolution + self.grid_resolution // 2
                world_y = grid_y * self.grid_resolution + self.grid_resolution // 2
                
                if (world_x >= ENVIRONMENT_WIDTH or world_y >= ENVIRONMENT_HEIGHT):
                    continue
                
                grid_position = (world_x, world_y)
                
                # Calculate probability by sampling over time and trajectories
                total_probability = 0.0
                
                for trajectory in trajectories:
                    # For each point in trajectory (time step), calculate visibility probability
                    for t, agent_pos in enumerate(trajectory):
                        # Time-based weighting (more recent predictions are more certain)
                        time_weight = exp(-t * 0.1)  # Exponential decay
                        
                        # Calculate visibility probability from agent position to grid position
                        visibility_prob = self.calculate_visibility_probability_from_position(
                            agent_pos, [grid_position]
                        )
                        
                        # Add weighted contribution
                        total_probability += visibility_prob * time_weight
                
                # Normalize by number of samples and time steps
                normalized_prob = total_probability / (self.num_trajectory_samples * self.prediction_steps)
                probability_grid[grid_y, grid_x] = min(1.0, normalized_prob)
        
        # Apply spatial smoothing for better visualization
        from scipy.ndimage import gaussian_filter
        probability_grid = gaussian_filter(probability_grid, sigma=1.0)
        
        # Update cache
        self.probability_grid = probability_grid
        self.last_agent_state = agent_state.copy()
        self.cache_valid = True
        
        return probability_grid
    
    def render_probability_overlay(self, screen: pygame.Surface, agent, 
                                 alpha: int = 120, colormap: str = 'hot') -> None:
        """
        Render the probability-based visibility overlay on the screen.
        
        Args:
            screen: Pygame surface to draw on
            agent: The agent object
            alpha: Transparency level (0-255)
            colormap: Color scheme ('hot', 'cool', 'viridis')
        """
        if not self.visibility_map:
            return
        
        # Calculate probability grid
        prob_grid = self.calculate_temporal_visibility_probability(agent)
        
        # Create overlay surface
        overlay = pygame.Surface((ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT), pygame.SRCALPHA)
        
        # Render probability grid
        max_prob = np.max(prob_grid)
        if max_prob > 0:
            for grid_y in range(self.grid_height):
                for grid_x in range(self.grid_width):
                    prob = prob_grid[grid_y, grid_x]
                    
                    if prob > 0.01:  # Only render cells with significant probability
                        # Convert to world coordinates
                        world_x = grid_x * self.grid_resolution
                        world_y = grid_y * self.grid_resolution
                        
                        # Choose color based on probability and colormap
                        intensity = min(255, int(255 * prob / max_prob))
                        
                        if colormap == 'hot':
                            # Red to yellow gradient
                            if prob < max_prob * 0.5:
                                color = (intensity * 2, 0, 0, alpha)
                            else:
                                color = (255, (intensity - 128) * 2, 0, alpha)
                        elif colormap == 'cool':
                            # Blue to cyan gradient
                            color = (0, intensity, 255, alpha)
                        else:  # viridis-like
                            # Purple to yellow gradient
                            color = (intensity, intensity // 2, 255 - intensity, alpha)
                        
                        # Draw the grid cell
                        cell_rect = pygame.Rect(world_x, world_y, 
                                              self.grid_resolution, self.grid_resolution)
                        pygame.draw.rect(overlay, color, cell_rect)
        
        # Blit overlay to screen
        screen.blit(overlay, (0, 0))
        
        # Draw legend
        self.draw_probability_legend(screen, max_prob, colormap)
    
    def draw_probability_legend(self, screen: pygame.Surface, max_prob: float, 
                              colormap: str) -> None:
        """Draw a legend showing the probability color scale."""
        legend_x = ENVIRONMENT_WIDTH + 10
        legend_y = 50
        legend_width = 20
        legend_height = 200
        
        # Draw legend background
        legend_bg = pygame.Rect(legend_x - 5, legend_y - 5, legend_width + 60, legend_height + 10)
        pygame.draw.rect(screen, (0, 0, 0, 180), legend_bg)
        
        # Draw color gradient
        for i in range(legend_height):
            prob_ratio = (legend_height - i) / legend_height
            intensity = int(255 * prob_ratio)
            
            if colormap == 'hot':
                if prob_ratio < 0.5:
                    color = (intensity * 2, 0, 0)
                else:
                    color = (255, (intensity - 128) * 2, 0)
            elif colormap == 'cool':
                color = (0, intensity, 255)
            else:  # viridis-like
                color = (intensity, intensity // 2, 255 - intensity)
            
            pygame.draw.line(screen, color, 
                           (legend_x, legend_y + i), 
                           (legend_x + legend_width, legend_y + i))
        
        # Draw scale labels
        font = pygame.font.SysFont('Arial', 12)
        
        # Top label (maximum)
        max_text = font.render(f"{max_prob:.3f}", True, (255, 255, 255))
        screen.blit(max_text, (legend_x + legend_width + 5, legend_y))
        
        # Middle label
        mid_text = font.render(f"{max_prob/2:.3f}", True, (255, 255, 255))
        screen.blit(mid_text, (legend_x + legend_width + 5, legend_y + legend_height // 2))
        
        # Bottom label (minimum)
        min_text = font.render("0.000", True, (255, 255, 255))
        screen.blit(min_text, (legend_x + legend_width + 5, legend_y + legend_height))
        
        # Title
        title_text = font.render("Visibility", True, (255, 255, 255))
        screen.blit(title_text, (legend_x - 5, legend_y - 20))
        title_text2 = font.render("Probability", True, (255, 255, 255))
        screen.blit(title_text2, (legend_x - 5, legend_y - 5))

def run_probability_visibility_demo(multicore=True, num_cores=None):
    """
    Run a demonstration of the probability-based visibility overlay system.
    """
    print("Starting Probability-Based Visibility Overlay Demo...")
    
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Probability-Based Visibility Overlay")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)
    
    # Initialize environment
    print("Setting up environment...")
    environment = SimulationEnvironment()
    
    # Initialize map graph
    print("Setting up map graph...")
    map_graph = MapGraph(
        ENVIRONMENT_WIDTH,
        ENVIRONMENT_HEIGHT,
        environment.get_all_walls(),
        environment.get_doors(),
        cache_file=MAP_GRAPH_INSPECTION_CACHE_FILE
    )
    
    # Try to load from cache
    cache_loaded = False
    if MAP_GRAPH_CACHE_ENABLED:
        print("Attempting to load map graph from cache...")
        cache_loaded = map_graph.load_from_cache()
        if cache_loaded:
            print(f"Successfully loaded map graph with {len(map_graph.nodes)} nodes.")
        else:
            print("Cache loading failed or no valid cache found.")
    
    # Generate map graph if not loaded from cache
    if not cache_loaded:
        print("Generating new map graph...")
        if multicore:
            cores_to_use = num_cores if num_cores else multiprocessing.cpu_count()
            print(f"Generating map graph using {cores_to_use} CPU cores...")
            map_graph.generate_parallel(None, cores_to_use)
        else:
            print("Generating map graph using single core...")
            map_graph.generate(None)
        
        print(f"Map graph generated with {len(map_graph.nodes)} nodes and {len(map_graph.edges)} edges.")
        
        # Save to cache
        if MAP_GRAPH_CACHE_ENABLED:
            map_graph.save_to_cache()
    
    # Initialize agent
    print("Initializing agent...")
    agent = UnicycleModel()
    agent.state = np.array([100.0, 100.0, 0.0, 0.0])  # x, y, theta, v
    agent.controls = (0.0, 0.0)  # linear_vel, angular_vel
    
    # Initialize probability visibility overlay
    print("Setting up probability visibility overlay...")
    overlay = ProbabilityVisibilityOverlay(map_graph, environment)
    
    # Load visibility data
    print("Loading visibility data...")
    try:
        visibility_map = map_graph.load_visibility_data()
        if visibility_map:
            overlay.set_visibility_map(visibility_map)
            print(f"Loaded visibility data for {len(visibility_map)} nodes")
        else:
            print("No visibility data available. Please run visibility analysis first.")
            return
    except Exception as e:
        print(f"Error loading visibility data: {e}")
        return
    
    # Demo parameters
    show_overlay = True
    colormap = 'hot'
    overlay_alpha = 120
    auto_move = False
    
    print("\nDemo Controls:")
    print("  Arrow keys: Manual agent control")
    print("  Space: Toggle auto-movement")
    print("  O: Toggle overlay display")
    print("  C: Cycle color schemes (hot/cool/viridis)")
    print("  +/-: Adjust overlay transparency")
    print("  ESC: Quit")
    
    # Main loop
    running = True
    last_update = 0
    
    while running:
        dt = clock.tick(60) / 1000.0  # 60 FPS
        current_time = time.time()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_o:
                    show_overlay = not show_overlay
                    print(f"Overlay display: {'On' if show_overlay else 'Off'}")
                elif event.key == pygame.K_c:
                    colormaps = ['hot', 'cool', 'viridis']
                    current_idx = colormaps.index(colormap)
                    colormap = colormaps[(current_idx + 1) % len(colormaps)]
                    print(f"Color scheme: {colormap}")
                    overlay.cache_valid = False  # Force recalculation
                elif event.key == pygame.K_SPACE:
                    auto_move = not auto_move
                    print(f"Auto-movement: {'On' if auto_move else 'Off'}")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    overlay_alpha = min(255, overlay_alpha + 20)
                    print(f"Overlay alpha: {overlay_alpha}")
                elif event.key == pygame.K_MINUS:
                    overlay_alpha = max(20, overlay_alpha - 20)
                    print(f"Overlay alpha: {overlay_alpha}")
        
        # Agent control
        keys = pygame.key.get_pressed()
        linear_vel = 0
        angular_vel = 0
        
        if auto_move:
            # Automatic movement pattern
            linear_vel = 30 + 20 * sin(current_time * 0.5)
            angular_vel = 0.3 * sin(current_time * 0.8)
        else:
            # Manual control
            if keys[pygame.K_UP]:
                linear_vel = 50
            if keys[pygame.K_DOWN]:
                linear_vel = -30
            if keys[pygame.K_LEFT]:
                angular_vel = -1.0
            if keys[pygame.K_RIGHT]:
                angular_vel = 1.0
        
        # Update agent
        agent.set_controls(linear_vel, angular_vel)
        agent.update(dt=dt, walls=environment.get_all_walls(), doors=environment.get_doors())
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw environment
        environment.draw(screen, font)
        
        # Draw map graph (faded)
        for edge in map_graph.edges:
            i, j = edge
            start = map_graph.nodes[i]
            end = map_graph.nodes[j]
            pygame.draw.line(screen, (40, 40, 40), start, end, 1)
        
        for node in map_graph.nodes:
            pygame.draw.circle(screen, (60, 60, 60), node, 2)
        
        # Draw probability overlay
        if show_overlay:
            overlay.render_probability_overlay(screen, agent, overlay_alpha, colormap)
        
        # Draw agent
        x, y = agent.state[0], agent.state[1]
        theta = agent.state[2]
        agent_radius = 16
        
        # Agent body
        pygame.draw.circle(screen, (0, 150, 255), (int(x), int(y)), agent_radius)
        
        # Direction indicator
        end_x = x + agent_radius * cos(theta)
        end_y = y + agent_radius * sin(theta)
        pygame.draw.line(screen, (255, 255, 255), (x, y), (end_x, end_y), 3)
        
        # Draw info
        info_lines = [
            f"Probability-Based Visibility Overlay",
            f"Agent Position: ({x:.1f}, {y:.1f})",
            f"Agent Heading: {math.degrees(theta):.1f}°",
            f"Agent Speed: {agent.state[3]:.1f}",
            f"Color Scheme: {colormap}",
            f"Overlay Alpha: {overlay_alpha}",
            f"Auto-movement: {'On' if auto_move else 'Off'}",
            "",
            "Controls:",
            "Arrow keys: Move agent",
            "Space: Toggle auto-movement",
            "O: Toggle overlay",
            "C: Change colors",
            "+/-: Adjust transparency"
        ]
        
        y_offset = 10
        for line in info_lines:
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (ENVIRONMENT_WIDTH + 10, y_offset))
            y_offset += 20
        
        pygame.display.flip()
    
    pygame.quit()
    print("Demo completed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Probability-Based Visibility Overlay Demo")
    parser.add_argument('--single-core', action='store_true', 
                       help='Use single-core processing instead of multicore')
    parser.add_argument('--cores', type=int, default=None,
                       help='Number of CPU cores to use for processing')
    
    args = parser.parse_args()
    
    run_probability_visibility_demo(
        multicore=not args.single_core,
        num_cores=args.cores
    )
