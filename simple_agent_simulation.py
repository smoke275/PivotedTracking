#!/usr/bin/env python3
"""
Simple Agent Simulation
A standalone simulation that loads environment and agents for basic movement control.
Independent of the inspection tools - creates its own simulation environment.

Key Functionality Integration:

Z Functionality (Complete Agent Tracking & Probability Analysis):
- F: Agent Following - Automatically track and follow specific agents, updating camera/view to center on them
- O: Probability Overlay - Calculate and display probability distributions of where agents might move next
  Based on current position, heading, speed, and reachable areas within a time horizon
- B: Visibility Gaps - Analyze and highlight areas where line-of-sight is obstructed by walls/obstacles
  Shows potential hiding spots and areas of uncertainty for tracking
- Y: Rotating Rods - Dynamic directional analysis showing probable movement vectors and scan patterns
  Rotates based on agent heading and environment constraints
- H: Extended Probability Set - Advanced probability calculations combining distance, direction, gaps, and time decay
  Provides multiple probability layers: base, gap-influenced, directional, time-decay, and combined

M Functionality (Multi-Agent Combined Probability Mode):
- Calculates intersection probabilities when multiple agents are tracked simultaneously
- Multiplies individual agent probability distributions to find areas where agents might converge
- Uses purple-yellow color scheme to visualize combined probability zones
- Helps identify potential meeting points, collision zones, or coordinated movement areas
- Requires at least 2 agents to be active and tracked by the position evaluator

Data Structures Returned:
- Z probabilities: Dict[int, float] mapping node indices to probability values (0.0 to 1.0)
- M probabilities: Dict[int, float] mapping node indices to combined probability values
- Visibility gaps: List[Tuple] of (start_point, end_point, gap_size) for obstruction analysis
- Rotating rods: List[Tuple] of (start_point, end_point, rotation_angle) for directional scanning
"""

import pygame
import sys
import os
import math
import pickle
import time

# Make sure the project directory is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import required modules
from multitrack.models.simulation_environment import SimulationEnvironment
from multitrack.models.agents.visitor_agent import UnicycleModel
from multitrack.utils.map_graph import MapGraph
from multitrack.utils.config import (MAP_GRAPH_INSPECTION_CACHE_FILE, MAP_GRAPH_CACHE_ENABLED,
                                   MAP_GRAPH_VISIBILITY_CACHE_FILE, LEADER_LINEAR_VEL, LEADER_ANGULAR_VEL,
                                   MAP_GRAPH_NODE_COLOR, MAP_GRAPH_EDGE_COLOR)
from position_evaluator import (global_position_evaluator, update_position, get_distance, get_stats, 
                              set_environment_data, find_closest_node, Z)

def main():
    # Initialize Pygame
    pygame.init()
    
    # Screen dimensions (matching the original inspection tool)
    SIDEBAR_WIDTH = 250  # Width of the information sidebar
    ENVIRONMENT_WIDTH = 1280  # Width of the environment area
    ENVIRONMENT_HEIGHT = 720  # Height of the environment area
    WINDOW_WIDTH = ENVIRONMENT_WIDTH + SIDEBAR_WIDTH  # Total window width including sidebar
    WINDOW_HEIGHT = ENVIRONMENT_HEIGHT  # Window height
    
    # Set up the display
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Simple Agent Simulation")
    
    # Initialize font
    font = pygame.font.SysFont('Arial', 16)
    
    # Create the environment
    print("Creating simulation environment...")
    environment = SimulationEnvironment(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT)
    
    # Initialize map graph and try to load from cache
    print("Initializing map graph...")
    map_graph = MapGraph(
        ENVIRONMENT_WIDTH, 
        ENVIRONMENT_HEIGHT, 
        environment.get_all_walls(), 
        environment.get_doors(),
        cache_file=MAP_GRAPH_INSPECTION_CACHE_FILE  # Use inspection-specific cache file
    )
    
    # Try to load map graph from cache
    map_graph_loaded = False
    if MAP_GRAPH_CACHE_ENABLED:
        print("Attempting to load map graph from cache...")
        map_graph_loaded = map_graph.load_from_cache()
        if map_graph_loaded:
            print(f"Successfully loaded map graph from cache with {len(map_graph.nodes)} nodes and {len(map_graph.edges)} edges.")
        else:
            print("No cached map graph found. Map graph will be empty (use inspect_environment.py to generate one).")
    else:
        print("Map graph caching is disabled. Map graph will be empty.")
    
    # Load visibility map if available
    visibility_map = None
    if map_graph_loaded and MAP_GRAPH_CACHE_ENABLED:
        try:
            import pickle
            visibility_cache_file = MAP_GRAPH_VISIBILITY_CACHE_FILE
            if os.path.exists(visibility_cache_file):
                print(f"Loading visibility map from {visibility_cache_file}...")
                with open(visibility_cache_file, 'rb') as f:
                    visibility_map = pickle.load(f)
                print(f"Loaded visibility map with {len(visibility_map)} node sight lines")
            else:
                print(f"No visibility cache found at {visibility_cache_file}")
        except Exception as e:
            print(f"Error loading visibility map: {e}")
    
    # Set up position evaluator with environment data
    print("Configuring position evaluator with environment data...")
    set_environment_data(environment, map_graph if map_graph_loaded else None, visibility_map)
    print("Position evaluator configured with environment, map graph, and visibility data")
    
    # Agent configuration
    AGENT_LINEAR_VEL = LEADER_LINEAR_VEL
    AGENT_ANGULAR_VEL = LEADER_ANGULAR_VEL
    AGENT_COLOR = (255, 0, 255)  # Magenta for agent 1
    AGENT2_COLOR = (0, 255, 255)  # Cyan for agent 2
    AGENT_RADIUS = 16
    
    # Create agents
    print("Creating agents...")
    
    # Agent 1 - controllable with arrow keys
    agent1 = UnicycleModel(
        initial_position=(100, 100),
        walls=environment.get_all_walls(), 
        doors=environment.get_doors()
    )
    
    # Agent 2 - controllable with WASD keys  
    agent2 = UnicycleModel(
        initial_position=(200, 200),
        walls=environment.get_all_walls(), 
        doors=environment.get_doors()
    )
    
    # Try to load agent states if they exist
    AGENT_STATE_FILE = 'agent_state.pkl'
    AGENT2_STATE_FILE = 'agent2_state.pkl'
    
    if os.path.exists(AGENT_STATE_FILE):
        try:
            with open(AGENT_STATE_FILE, 'rb') as f:
                agent_state = pickle.load(f)
                agent1.state = agent_state.copy()
                print("Loaded agent 1 state from file")
        except Exception as e:
            print(f"Could not load agent 1 state: {e}")
    
    if os.path.exists(AGENT2_STATE_FILE):
        try:
            with open(AGENT2_STATE_FILE, 'rb') as f:
                agent2_state = pickle.load(f)
                agent2.state = agent2_state.copy()
                print("Loaded agent 2 state from file")
        except Exception as e:
            print(f"Could not load agent 2 state: {e}")
    
    # Clock for controlling frame rate
    clock = pygame.time.Clock()
    
    # Position evaluator update timing
    position_update_interval = 0.5  # Update position evaluator every 0.5 seconds
    last_position_update = time.time()
    
    # Main simulation loop
    running = True
    show_info = True
    show_map_graph = False  # Start with map graph hidden by default
    show_z_probabilities = False  # Show Z probabilities
    show_m_probabilities = False  # Show M probabilities
    
    print("Starting simulation...")
    print("Controls:")
    print("  Arrow Keys: Control magenta agent (Agent 1)")
    print("  WASD Keys: Control cyan agent (Agent 2)")
    print("  G: Toggle map graph display")
    print("  I: Toggle info display")
    print("  S: Save agent states")
    print("  ESC: Quit")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_i:
                    show_info = not show_info
                    print(f"Info display: {'ON' if show_info else 'OFF'}")
                elif event.key == pygame.K_g:
                    show_map_graph = not show_map_graph
                    print(f"Map graph display: {'ON' if show_map_graph else 'OFF'}")
                elif event.key == pygame.K_s:
                    # Save agent states
                    try:
                        with open(AGENT_STATE_FILE, 'wb') as f:
                            pickle.dump(agent1.state, f)
                        with open(AGENT2_STATE_FILE, 'wb') as f:
                            pickle.dump(agent2.state, f)
                        print("Agent states saved")
                    except Exception as e:
                        print(f"Error saving agent states: {e}")
        
        # Get current key states for continuous movement
        keys = pygame.key.get_pressed()
        
        # Agent 1 control (arrow keys)
        linear_vel1 = 0
        angular_vel1 = 0
        if keys[pygame.K_UP]:
            linear_vel1 = AGENT_LINEAR_VEL
        if keys[pygame.K_DOWN]:
            linear_vel1 = -AGENT_LINEAR_VEL
        if keys[pygame.K_LEFT]:
            angular_vel1 = -AGENT_ANGULAR_VEL
        if keys[pygame.K_RIGHT]:
            angular_vel1 = AGENT_ANGULAR_VEL
        
        # Agent 2 control (WASD keys)
        linear_vel2 = 0
        angular_vel2 = 0
        if keys[pygame.K_w]:
            linear_vel2 = AGENT_LINEAR_VEL
        if keys[pygame.K_s]:
            linear_vel2 = -AGENT_LINEAR_VEL
        if keys[pygame.K_a]:
            angular_vel2 = -AGENT_ANGULAR_VEL
        if keys[pygame.K_d]:
            angular_vel2 = AGENT_ANGULAR_VEL
        
        # Update agents
        agent1.set_controls(linear_vel1, angular_vel1)
        agent1.update(dt=0.1, walls=environment.get_all_walls(), doors=environment.get_doors())
        
        agent2.set_controls(linear_vel2, angular_vel2)
        agent2.update(dt=0.1, walls=environment.get_all_walls(), doors=environment.get_doors())
        
        # Update position evaluator periodically
        current_time = time.time()
        probabilities = {}  # Store probabilities for drawing
        if current_time - last_position_update >= position_update_interval:
            # Update agent positions in the evaluator
            update_position("agent1", agent1.state[0], agent1.state[1], agent1.state[2], (linear_vel1, angular_vel1))
            update_position("agent2", agent2.state[0], agent2.state[1], agent2.state[2], (linear_vel2, angular_vel2))
            
            # Get distance between agents
            distance = get_distance("agent1", "agent2")
            if distance is not None:
                print(f"Position Evaluator - Distance between agents: {distance:.2f} pixels")
            
            # Get evaluator statistics
            stats = get_stats()
            print(f"Position Evaluator - Tracking {stats.get('agent_count', 0)} agents")
            print(f"Position Evaluator - Environment data: {'Available' if stats.get('has_environment', False) else 'Not available'}")
            
            # Get Z probabilities
            probabilities = Z()
            print(f"Position Evaluator - Z probabilities: {len(probabilities)} nodes with probabilities")
            if probabilities:
                max_prob = max(probabilities.values())
                print(f"Position Evaluator - Max probability: {max_prob:.3f}")
            
            # Test closest node functionality if map graph is available
            if map_graph_loaded:
                closest_node_agent1 = find_closest_node("agent1")
                closest_node_agent2 = find_closest_node("agent2")
                if closest_node_agent1 is not None:
                    print(f"Position Evaluator - Agent 1 closest to node {closest_node_agent1}")
                if closest_node_agent2 is not None:
                    print(f"Position Evaluator - Agent 2 closest to node {closest_node_agent2}")
            
            last_position_update = current_time
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw environment
        environment.draw(screen, font)
        
        # Draw map graph if enabled and loaded
        if show_map_graph and map_graph_loaded:
            # Draw the graph edges first
            for edge in map_graph.edges:
                i, j = edge
                start = map_graph.nodes[i]
                end = map_graph.nodes[j]
                pygame.draw.line(screen, MAP_GRAPH_EDGE_COLOR, start, end, 1)
            
            # Draw the graph nodes
            for i, node in enumerate(map_graph.nodes):
                pygame.draw.circle(screen, MAP_GRAPH_NODE_COLOR, node, 4)
        
        # Draw Z probabilities if available
        if probabilities and map_graph_loaded:
            for node_idx, probability in probabilities.items():
                if node_idx < len(map_graph.nodes):
                    node_pos = map_graph.nodes[node_idx]
                    
                    # Color based on probability value - blue (low) to red (high)
                    r = int(255 * probability)
                    g = 0
                    b = int(255 * (1 - probability))
                    color = (r, g, b)
                    
                    # Draw probability circle - size based on probability
                    radius = max(6, int(12 * probability))
                    pygame.draw.circle(screen, color, node_pos, radius)
        
        # Draw agents
        def draw_agent(agent, color, label):
            x, y, theta = agent.state[0], agent.state[1], agent.state[2]
            
            # Draw agent body (circle)
            pygame.draw.circle(screen, color, (int(x), int(y)), AGENT_RADIUS)
            
            # Draw direction indicator (line showing heading)
            end_x = x + AGENT_RADIUS * 1.5 * math.cos(theta)
            end_y = y + AGENT_RADIUS * 1.5 * math.sin(theta)
            pygame.draw.line(screen, (255, 255, 255), (int(x), int(y)), (int(end_x), int(end_y)), 3)
            
            # Draw agent label
            label_surface = font.render(label, True, (255, 255, 255))
            screen.blit(label_surface, (int(x) - 20, int(y) - AGENT_RADIUS - 25))
        
        # Draw both agents
        draw_agent(agent1, AGENT_COLOR, "Agent 1")
        draw_agent(agent2, AGENT2_COLOR, "Agent 2")
        
        # Draw info panel if enabled
        if show_info:
            info_lines = [
                "Simple Agent Simulation",
                "",
                "Controls:",
                "Arrow Keys: Control Agent 1 (Magenta)",
                "WASD Keys: Control Agent 2 (Cyan)",
                "G: Toggle map graph display",
                "I: Toggle this info display",
                "S: Save agent states to file",
                "ESC: Quit simulation",
                "",
                f"Agent 1 Position: ({agent1.state[0]:.1f}, {agent1.state[1]:.1f})",
                f"Agent 1 Heading: {math.degrees(agent1.state[2]):.1f}°",
                f"Agent 2 Position: ({agent2.state[0]:.1f}, {agent2.state[1]:.1f})",
                f"Agent 2 Heading: {math.degrees(agent2.state[2]):.1f}°",
                "",
                f"Distance between agents: {math.sqrt((agent1.state[0] - agent2.state[0])**2 + (agent1.state[1] - agent2.state[1])**2):.1f}",
                f"FPS: {int(clock.get_fps())}",
                "",
                "Position Evaluator:",
                f"Evaluator Distance: {get_distance('agent1', 'agent2'):.1f}" if get_distance('agent1', 'agent2') else "Evaluator Distance: N/A",
                f"Tracked Agents: {get_stats().get('agent_count', 0)}",
                f"Environment Data: {'Yes' if get_stats().get('has_environment', False) else 'No'}",
                f"Update Interval: {position_update_interval}s",
                "",
                f"Map Graph: {'ON' if show_map_graph else 'OFF'}",
            ]
            
            # Add map graph info if loaded
            if map_graph_loaded:
                stats = get_stats()
                info_lines.extend([
                    f"Nodes: {stats.get('map_nodes', 0)}",
                    f"Edges: {stats.get('map_edges', 0)}"
                ])
                
                # Add closest node info
                closest_node_agent1 = find_closest_node("agent1")
                closest_node_agent2 = find_closest_node("agent2")
                if closest_node_agent1 is not None:
                    info_lines.append(f"Agent 1 → Node {closest_node_agent1}")
                if closest_node_agent2 is not None:
                    info_lines.append(f"Agent 2 → Node {closest_node_agent2}")
            else:
                info_lines.append("Map Graph: Not loaded")
                info_lines.append("(Run inspect_environment.py first)")
            
            # Create info background
            info_bg = pygame.Surface((300, len(info_lines) * 20 + 20))
            info_bg.fill((0, 0, 0))
            info_bg.set_alpha(180)
            
            # Position info panel on the right side (in the sidebar area)
            info_x = ENVIRONMENT_WIDTH + 10  # Position in sidebar
            info_y = 20
            screen.blit(info_bg, (info_x, info_y))
            
            # Draw info text
            for i, line in enumerate(info_lines):
                if line:  # Skip empty lines
                    text_surface = font.render(line, True, (255, 255, 255))
                    screen.blit(text_surface, (info_x + 10, info_y + 10 + i * 20))
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        clock.tick(60)  # 60 FPS
    
    # Save agent states on exit
    try:
        with open(AGENT_STATE_FILE, 'wb') as f:
            pickle.dump(agent1.state, f)
        with open(AGENT2_STATE_FILE, 'wb') as f:
            pickle.dump(agent2.state, f)
        print("Agent states saved on exit")
    except Exception as e:
        print(f"Error saving agent states on exit: {e}")
    
    # Cleanup
    pygame.quit()
    print("Simulation ended")

if __name__ == "__main__":
    main()
