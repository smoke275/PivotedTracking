#!/usr/bin/env python3
"""
Minimal Agent Simulation
A simple simulation where you can create an environment and move an agent around.
Starting from scratch with minimal dependencies.

Controls:
- Arrow Keys: Move the agent (Up/Down = forward/backward, Left/Right = rotate)
- ESC: Quit
"""

import os
import sys
import math
import pygame

# Make sure the project directory is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import only what we need
from multitrack.models.simulation_environment import SimulationEnvironment
from multitrack.models.agents.visitor_agent import UnicycleModel
from multitrack.utils.config import LEADER_LINEAR_VEL, LEADER_ANGULAR_VEL


def main():
    """Main simulation loop"""
    # Initialize Pygame
    pygame.init()
    
    # Screen dimensions
    ENVIRONMENT_WIDTH = 1280
    ENVIRONMENT_HEIGHT = 720
    
    # Set up the display
    screen = pygame.display.set_mode((ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT))
    pygame.display.set_caption("Minimal Agent Simulation")
    
    # Initialize font for basic info
    font = pygame.font.SysFont('Arial', 14)
    
    # Create the environment with walls and doors
    print("Creating simulation environment...")
    environment = SimulationEnvironment(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT)
    print("Environment created!")
    
    # Agent configuration
    AGENT_LINEAR_VEL = LEADER_LINEAR_VEL  # 50.0 pixels/second
    AGENT_ANGULAR_VEL = LEADER_ANGULAR_VEL  # 1.0 radians/second
    AGENT_COLOR = (255, 0, 255)  # Magenta
    AGENT_RADIUS = 16
    
    # Create the agent at a starting position
    print("Creating agent...")
    agent = UnicycleModel(
        initial_position=(200, 200),  # Start position
        walls=environment.get_all_walls(),
        doors=environment.get_doors()
    )
    print("Agent created at position (200, 200)")
    
    # Clock for controlling frame rate
    clock = pygame.time.Clock()
    
    # Main simulation loop
    running = True
    show_info = True
    
    print("\nSimulation started!")
    print("Controls:")
    print("  Arrow Keys: Control agent (Up/Down = move, Left/Right = rotate)")
    print("  I: Toggle info display")
    print("  ESC: Quit")
    print("")
    
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
        
        # Get current key states for continuous movement
        keys = pygame.key.get_pressed()
        
        # Agent control (arrow keys)
        linear_vel = 0
        angular_vel = 0
        if keys[pygame.K_UP]:
            linear_vel = AGENT_LINEAR_VEL
        if keys[pygame.K_DOWN]:
            linear_vel = -AGENT_LINEAR_VEL
        if keys[pygame.K_LEFT]:
            angular_vel = AGENT_ANGULAR_VEL
        if keys[pygame.K_RIGHT]:
            angular_vel = -AGENT_ANGULAR_VEL
        
        # Update agent
        agent.set_controls(linear_vel, angular_vel)
        agent.update(
            dt=0.1,
            walls=environment.get_all_walls(),
            doors=environment.get_doors()
        )
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw environment (walls, doors, etc.)
        environment.draw(screen, font)
        
        # Draw agent
        agent_x = int(agent.state[0])
        agent_y = int(agent.state[1])
        agent_theta = agent.state[2]
        
        # Draw agent as a circle
        pygame.draw.circle(screen, AGENT_COLOR, (agent_x, agent_y), AGENT_RADIUS)
        
        # Draw direction indicator (a line showing which way the agent is facing)
        direction_length = AGENT_RADIUS + 10
        end_x = agent_x + direction_length * math.cos(agent_theta)
        end_y = agent_y + direction_length * math.sin(agent_theta)
        pygame.draw.line(screen, (255, 255, 255), (agent_x, agent_y), (int(end_x), int(end_y)), 3)
        
        # Draw info overlay if enabled
        if show_info:
            info_lines = [
                "Minimal Agent Simulation",
                "",
                f"Position: ({agent.state[0]:.1f}, {agent.state[1]:.1f})",
                f"Heading: {math.degrees(agent.state[2]):.1f}°",
                f"FPS: {int(clock.get_fps())}",
                "",
                "Controls:",
                "  Up/Down: Move forward/backward",
                "  Left/Right: Rotate",
                "  I: Toggle this info",
                "  ESC: Quit",
            ]
            
            # Draw semi-transparent background for info
            info_height = len(info_lines) * 20 + 10
            info_surface = pygame.Surface((300, info_height), pygame.SRCALPHA)
            info_surface.fill((0, 0, 0, 180))
            screen.blit(info_surface, (10, 10))
            
            # Draw info text
            y_offset = 15
            for line in info_lines:
                text_surface = font.render(line, True, (255, 255, 255))
                screen.blit(text_surface, (15, y_offset))
                y_offset += 20
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate (30 FPS)
        clock.tick(30)
    
    # Clean up
    pygame.quit()
    print("\nSimulation ended. Goodbye!")


if __name__ == "__main__":
    main()
