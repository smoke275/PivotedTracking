"""
Visitor Agent (UnicycleModel) implementation

This file contains the UnicycleModel class which represents the "visitor" agent
in the simulation. It includes the dynamics model and visualization functionality.
"""
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

import numpy as np
import pygame
from math import sin, cos, pi
from multitrack.utils.config import *

class UnicycleModel:
    def __init__(self, initial_position=None, walls=None, doors=None):
        # State: [x, y, theta, v]
        if initial_position is not None:
            self.state = np.array([initial_position[0], initial_position[1], 0.0, 0.0])
        else:
            # Default to center if no position is provided
            self.state = np.array([WIDTH//2, HEIGHT//2, 0.0, 0.0])
            
            # If walls are provided, find a valid starting position
            if walls is not None:
                self.state = self._find_valid_position(walls, doors)
        
        # Control inputs: [v, omega] (linear and angular velocity)
        self.controls = np.array([0.0, 0.0])
        
        # Add measurement noise - for realistic monitoring
        self.measurement_noise_pos = KF_MEASUREMENT_NOISE_POS
        self.measurement_noise_angle = KF_MEASUREMENT_NOISE_ANGLE
        
        # For collision handling
        self.prev_state = self.state.copy()
        
        # Store the latest noisy position for visibility detection
        self.noisy_position = np.array([self.state[0], self.state[1], self.state[2]])
    
    def _find_valid_position(self, walls, doors=None):
        """Find a valid position that doesn't collide with walls"""
        if doors is None:
            doors = []
        
        # Define safe areas (room centers) to try first
        # These positions are roughly the centers of different rooms
        safe_positions = [
            (WIDTH * 0.15, HEIGHT * 0.15),    # Bedroom
            (WIDTH * 0.6, HEIGHT * 0.15),     # Kitchen
            (WIDTH * 0.25, HEIGHT * 0.40),    # Living Room
            (WIDTH * 0.12, HEIGHT * 0.45),    # Study
            (WIDTH * 0.52, HEIGHT * 0.60),    # Bathroom
            (WIDTH * 0.80, HEIGHT * 0.15),    # Dining Room
            (WIDTH * 0.80, HEIGHT * 0.40),    # Center of house
            (WIDTH * 0.80, HEIGHT * 0.80),    # Lower area
            (WIDTH * 0.20, HEIGHT * 0.80),    # Lower left
        ]
        
        # Try safe positions first
        for pos in safe_positions:
            agent_rect = pygame.Rect(
                int(pos[0]) - 10,  # x
                int(pos[1]) - 10,  # y
                20, 20  # width, height (agent size)
            )
            
            # Check if position is valid
            if self._is_valid_position(agent_rect, walls, doors):
                return np.array([pos[0], pos[1], 0.0, 0.0])
        
        # Fall back to random positions if safe positions fail
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate random position within screen bounds with padding
            x = np.random.uniform(30, WIDTH - 30)
            y = np.random.uniform(30, HEIGHT - 30)
            
            agent_rect = pygame.Rect(
                int(x) - 10,  # x
                int(y) - 10,  # y
                20, 20  # width, height (agent size)
            )
            
            # Check if position is valid
            if self._is_valid_position(agent_rect, walls, doors):
                return np.array([x, y, 0.0, 0.0])
        
        # If all attempts fail, default to center (though it might be a wall)
        return np.array([WIDTH//2, HEIGHT//2, 0.0, 0.0])
    
    def _is_valid_position(self, agent_rect, walls, doors):
        """Check if a position is valid (not colliding with walls)"""
        for wall in walls:
            if agent_rect.colliderect(wall):
                # Check if we're in a door
                in_door = False
                for door in doors:
                    if agent_rect.colliderect(door):
                        in_door = True
                        break
                
                if not in_door:
                    return False
        return True

    def update(self, dt=0.1, elapsed_time=0, walls=None, doors=None, is_visible=False):
        # Store previous state for collision detection
        self.prev_state = self.state.copy()
        
        v, omega = self.controls
        theta = self.state[2]
        
        # Unicycle model dynamics
        self.state[0] += v * cos(theta) * dt
        self.state[1] += v * sin(theta) * dt
        self.state[2] += omega * dt
        self.state[3] = v  # Update velocity state
        
        # Normalize angle to [-pi, pi]
        self.state[2] = (self.state[2] + pi) % (2 * pi) - pi
        
        # Handle collisions if walls are provided
        if walls is not None:
            self.handle_collision(walls, doors)
        
        # Get actual screen dimensions from pygame (to fix boundary issues)
        screen_width, screen_height = pygame.display.get_surface().get_size()
        
        # Boundary conditions using actual screen dimensions
        self.state[0] = np.clip(self.state[0], 0, screen_width)
        self.state[1] = np.clip(self.state[1], 0, screen_height)
        
        # Generate noisy measurement at every time step for visibility detection
        self.noisy_position = np.array([
            self.state[0] + np.random.normal(0, self.measurement_noise_pos),
            self.state[1] + np.random.normal(0, self.measurement_noise_pos),
            self.state[2] + np.random.normal(0, self.measurement_noise_angle)
        ])
    
    def set_controls(self, linear_vel, angular_vel):
        self.controls = np.array([linear_vel, angular_vel])
    
    def draw(self, screen):
        # Draw unicycle agent (red circle - true position)
        x, y, theta, _ = self.state
        radius = 10
        pygame.draw.circle(screen, RED, (int(x), int(y)), radius)
        end_x = x + radius * cos(theta)
        end_y = y + radius * sin(theta)
        pygame.draw.line(screen, WHITE, (x, y), (end_x, end_y), 2)
        
        # Draw noisy measurement position (magenta circle)
        # This is what the escort agent can actually detect
        noisy_x, noisy_y = self.noisy_position[0], self.noisy_position[1]
        noisy_theta = self.noisy_position[2]
        
        # Draw noisy measurement
        pygame.draw.circle(screen, (255, 0, 255), (int(noisy_x), int(noisy_y)), 8, 3)  # Thicker outline
        end_x = noisy_x + 15 * cos(noisy_theta)
        end_y = noisy_y + 15 * sin(noisy_theta)
        pygame.draw.line(screen, (255, 0, 255), (noisy_x, noisy_y), (end_x, end_y), 3)  # Thicker line
        
        # Draw text label for the measurement
        font = pygame.font.SysFont('Arial', 12)
        text = font.render("Measurement", True, (255, 0, 255))
        screen.blit(text, (int(noisy_x) - text.get_width() // 2, int(noisy_y) - 20))
    
    def handle_collision(self, walls, doors):
        """
        Check if agent collides with walls and handle collision
        
        Args:
            walls: List of wall rectangles
            doors: List of door rectangles or None
        """
        if doors is None:
            doors = []
            
        # Create a rect for collision detection
        agent_rect = pygame.Rect(
            int(self.state[0]) - 10,  # x
            int(self.state[1]) - 10,  # y
            20, 20  # width, height (agent size)
        )
        
        collision = False
        for wall in walls:
            if agent_rect.colliderect(wall):
                # Check if we're in a door
                in_door = False
                for door in doors:
                    if agent_rect.colliderect(door):
                        in_door = True
                        break
                
                if not in_door:
                    collision = True
                    break
        
        # Reset position if collision occurred
        if collision:
            self.state[0] = self.prev_state[0]
            self.state[1] = self.prev_state[1]
            self.state[3] *= 0.5  # Reduce velocity on collision