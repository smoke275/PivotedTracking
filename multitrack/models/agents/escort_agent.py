"""
Escort Agent (FollowerAgent) implementation

This file contains the FollowerAgent class which represents the "escort" agent
in the simulation. It uses MPPI control to follow the main visitor agent.
"""
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

import numpy as np
import pygame
from math import sin, cos, pi
from multitrack.controllers.mppi_controller import MPPIController
from multitrack.utils.vision import is_agent_in_vision_cone, get_vision_cone_points
from multitrack.utils.config import *

class FollowerAgent:
    def __init__(self, initial_state=None, target_distance=FOLLOWER_TARGET_DISTANCE):
        """
        Follower agent that uses an MPPI controller to follow a target
        
        Parameters:
        - initial_state: [x, y, theta, v] or None for random initialization
        - target_distance: Desired following distance
        """
        # Initialize state
        if initial_state is None:
            # Random initialization away from the center
            x = np.random.uniform(WIDTH * 0.2, WIDTH * 0.8)
            y = np.random.uniform(HEIGHT * 0.2, HEIGHT * 0.8)
            theta = np.random.uniform(-pi, pi)
            self.state = np.array([x, y, theta, 0.0])
        else:
            self.state = initial_state.copy()
        
        # Control inputs
        self.controls = np.array([0.0, 0.0])
        
        # Initialize MPPI controller
        self.mppi = MPPIController(horizon=MPPI_HORIZON, samples=MPPI_SAMPLES, dt=0.1)
        
        # Last predicted trajectory from MPPI
        self.predicted_trajectory = None
        
        # Target distance to maintain from the leader
        self.target_distance = target_distance
        
        # History of states for visualization
        self.history = []
        self.max_history = 20
        
        # For collision detection
        self.prev_state = self.state.copy()
        
        # Vision system
        self.vision_range = DEFAULT_VISION_RANGE
        self.vision_angle = VISION_ANGLE
        self.target_visible = False
        self.last_seen_position = None
        self.search_mode = False
        self.search_timer = 0
        self.search_duration = 100  # Frames to continue searching
        self.vision_cone_points = []  # Store vision cone for visualization
        
        # Manual control mode
        self.manual_mode = False
    
    def set_manual_controls(self, linear_vel, angular_vel):
        """
        Set manual controls for the follower when not in automatic tracking mode
        
        Parameters:
        - linear_vel: Forward/backward velocity
        - angular_vel: Turning velocity
        """
        self.manual_mode = True
        self.controls = np.array([linear_vel, angular_vel])
    
    def set_auto_mode(self, enabled=True):
        """Toggle between automatic and manual control modes"""
        self.manual_mode = not enabled
        if self.manual_mode:
            # Reset controls when switching to manual
            self.controls = np.array([0.0, 0.0])
    
    def update(self, dt, leader_state, obstacles=None, walls=None, doors=None):
        """
        Update follower agent state based on leader position
        
        Parameters:
        - dt: Time step
        - leader_state: State of the leader agent [x, y, theta, v]
        - obstacles: List of obstacle positions [(x, y, radius), ...] or None
        - walls: List of wall rectangles for collision detection
        - doors: List of door rectangles for collision detection
        """
        # Store previous state for collision detection
        self.prev_state = self.state.copy()
        
        # Update vision cone for visualization
        if walls is not None and doors is not None:
            self.vision_cone_points = get_vision_cone_points(
                self, self.vision_range, self.vision_angle, walls, doors
            )
        
        # Check if leader is visible
        leader = type('Leader', (), {'state': leader_state})  # Create simple object with state attribute
        target_visible = is_agent_in_vision_cone(
            self, leader, self.vision_range, self.vision_angle, walls, doors
        )
        
        # Always update the visibility status regardless of control mode
        self.target_visible = target_visible
        
        # Update last seen position if target is visible
        if target_visible:
            self.last_seen_position = leader_state[:2].copy()  # Store position only
            self.search_mode = False
            self.search_timer = 0
        elif not self.search_mode and self.last_seen_position is not None:
            # Just lost sight of leader - start search mode
            self.search_mode = True
            self.search_timer = 0
        
        # Skip automatic control calculations if in manual mode
        if not self.manual_mode:
            # Vision-based targeting logic
            if target_visible:
                # Generate target trajectory (follow leader at a distance)
                target_trajectory = self._generate_target_trajectory(leader_state)
                
                # Compute optimal control using MPPI
                optimal_control, predicted_trajectory = self.mppi.compute_control(
                    self.state, target_trajectory, obstacles)
                
                # Store predicted trajectory for visualization
                self.predicted_trajectory = predicted_trajectory
                
                # Apply control
                self.controls = optimal_control
            else:
                # Leader not visible but we have a last seen position
                if self.last_seen_position is not None:
                    # In search mode, head toward last seen position
                    if self.search_timer < self.search_duration:
                        # Create a target state at the last seen position
                        target_x, target_y = self.last_seen_position
                        
                        # Calculate vector to last seen position
                        dx = target_x - self.state[0]
                        dy = target_y - self.state[1]
                        target_theta = np.arctan2(dy, dx)
                        
                        # Create a target state with the agent's position and the target orientation
                        # This will make the agent navigate to the last seen position
                        target_state = np.array([target_x, target_y, target_theta, 0.0])
                        target_trajectory = np.tile(target_state, (self.mppi.horizon + 1, 1))
                        
                        # Compute control to move to last seen position
                        optimal_control, predicted_trajectory = self.mppi.compute_control(
                            self.state, target_trajectory, obstacles)
                        
                        # Store predicted trajectory for visualization
                        self.predicted_trajectory = predicted_trajectory
                        
                        # Apply control
                        self.controls = optimal_control
                        
                        # Increment search timer
                        self.search_timer += 1
                    else:
                        # Search duration exceeded - stop moving
                        self.controls = np.array([0.0, 0.0])
                else:
                    # Never seen leader - don't move
                    self.controls = np.array([0.0, 0.0])
        
        # Update state using unicycle dynamics
        x, y, theta, _ = self.state
        v, omega = self.controls
        
        # Unicycle model dynamics
        x += v * cos(theta) * dt
        y += v * sin(theta) * dt
        theta += omega * dt
        
        # Normalize angle to [-pi, pi]
        theta = (theta + pi) % (2 * pi) - pi
        
        # Update state
        self.state = np.array([x, y, theta, v])
        
        # Handle collisions if walls are provided
        if walls is not None:
            self.handle_collision(walls, doors)
            
        # Get actual screen dimensions from pygame
        screen_width, screen_height = pygame.display.get_surface().get_size()
        
        # Boundary conditions using actual screen dimensions
        self.state[0] = np.clip(self.state[0], 0, screen_width)
        self.state[1] = np.clip(self.state[1], 0, screen_height)
        
        # Add current state to history
        self.history.append(self.state.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def handle_collision(self, walls, doors=None):
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
    
    def _generate_target_trajectory(self, leader_state):
        """
        Generate a target trajectory to follow leader at a specified distance
        
        Parameters:
        - leader_state: State of the leader agent [x, y, theta, v]
        
        Returns:
        - target_trajectory: Array of target states
        """
        lx, ly, ltheta, lv = leader_state
        
        # Calculate current distance to leader
        dx = lx - self.state[0]
        dy = ly - self.state[1]
        current_distance = (dx**2 + dy**2)**0.5
        
        # Target position is behind the leader at the specified distance
        tx = lx - self.target_distance * cos(ltheta)
        ty = ly - self.target_distance * sin(ltheta)
        
        # If we're too close to the leader, adjust target to enforce minimum distance
        if current_distance < FOLLOWER_SAFETY_DISTANCE:
            # Direction vector from leader to follower
            direction_x = self.state[0] - lx
            direction_y = self.state[1] - ly
            
            # Normalize direction vector
            direction_length = (direction_x**2 + direction_y**2)**0.5
            if direction_length > 0:  # Avoid division by zero
                direction_x /= direction_length
                direction_y /= direction_length
            
                # Set target position at least minimum distance away
                tx = lx + direction_x * max(self.target_distance, FOLLOWER_SAFETY_DISTANCE + 20)
                ty = ly + direction_y * max(self.target_distance, FOLLOWER_SAFETY_DISTANCE + 20)
        
        # Calculate target heading - now points toward leader for better following
        # This makes the escort face the visitor rather than matching orientation
        target_dx = lx - tx
        target_dy = ly - ty
        ttheta = np.arctan2(target_dy, target_dx)
        
        # Target velocity should adjust based on distance to leader
        # Slow down when close to target position
        target_pos_dx = tx - self.state[0]
        target_pos_dy = ty - self.state[1]
        target_pos_distance = (target_pos_dx**2 + target_pos_dy**2)**0.5
        
        # Adaptive velocity: faster when far, slower when close
        tv = min(lv, target_pos_distance * 0.2)
        
        # For simplicity, repeat the target state for the entire horizon
        target_state = np.array([tx, ty, ttheta, tv])
        target_trajectory = np.tile(target_state, (self.mppi.horizon + 1, 1))
        
        return target_trajectory
    
    def draw(self, screen):
        """Draw the follower agent and its predicted trajectory"""
        # Draw trajectory history as a fading trail
        if len(self.history) > 1:
            for i in range(1, len(self.history)):
                alpha = int(255 * i / len(self.history))
                color = (min(255, ORANGE[0]), 
                        min(255, ORANGE[1]), 
                        min(255, ORANGE[2]))
                
                prev_pos = self.history[i-1][:2]
                curr_pos = self.history[i][:2]
                
                pygame.draw.line(screen, color, 
                            (int(prev_pos[0]), int(prev_pos[1])), 
                            (int(curr_pos[0]), int(curr_pos[1])), 2)
        
        # Draw MPPI predicted trajectory
        if SHOW_MPPI_PREDICTIONS and self.predicted_trajectory is not None:
            for i in range(1, len(self.predicted_trajectory)):
                pred_prev = self.predicted_trajectory[i-1]
                pred_curr = self.predicted_trajectory[i]
                
                # Draw line from previous prediction to current
                pygame.draw.line(screen, MPPI_PREDICTION_COLOR, 
                                (int(pred_prev[0]), int(pred_prev[1])), 
                                (int(pred_curr[0]), int(pred_curr[1])), 1)
                
                # Draw small circle at each prediction point (less frequently to avoid clutter)
                if i % 3 == 0:
                    pygame.draw.circle(screen, MPPI_PREDICTION_COLOR, 
                                    (int(pred_curr[0]), int(pred_curr[1])), 2)
        
        # Draw follower agent
        x, y, theta, _ = self.state
        radius = 10
        pygame.draw.circle(screen, ORANGE, (int(x), int(y)), radius)
        end_x = x + radius * cos(theta)
        end_y = y + radius * sin(theta)
        pygame.draw.line(screen, WHITE, (x, y), (end_x, end_y), 2)
        
        # Draw vision cone if we have points
        if self.vision_cone_points and len(self.vision_cone_points) > 2:
            # Convert vision cone points to pygame format
            vision_points = [(int(p[0]), int(p[1])) for p in self.vision_cone_points]
            
            # Create a semi-transparent surface for the vision cone
            vision_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            
            # Set color based on whether target is visible
            if self.target_visible:
                # Green when target is visible
                vision_color = (0, 255, 0, VISION_TRANSPARENCY)
            else:
                # Yellow when searching
                vision_color = (255, 255, 0, VISION_TRANSPARENCY)
            
            # Draw polygon for vision cone
            pygame.draw.polygon(vision_surface, vision_color, vision_points)
            
            # Draw lines around the edge of the vision cone
            for i in range(1, len(vision_points)):
                pygame.draw.line(vision_surface, (*vision_color[:3], 150), 
                            vision_points[0], vision_points[i], 1)
            
            # Draw connection between consecutive points along the arc
            for i in range(1, len(vision_points)-1):
                pygame.draw.line(vision_surface, (*vision_color[:3], 150), 
                            vision_points[i], vision_points[i+1], 1)
            
            # Blit the vision surface onto the screen
            screen.blit(vision_surface, (0, 0))
            
        # Draw circle at last seen position if in search mode
        if self.search_mode and self.last_seen_position is not None:
            x, y = self.last_seen_position
            pygame.draw.circle(screen, (255, 100, 100, 150), (int(x), int(y)), 15, 2)
            
            # Draw line from agent to last seen position
            pygame.draw.line(screen, (255, 100, 100, 150), 
                          (int(self.state[0]), int(self.state[1])), 
                          (int(x), int(y)), 1)
        
        # Indicate manual mode if active
        if self.manual_mode:
            pos = (int(self.state[0]), int(self.state[1] - 25))
            font = pygame.font.SysFont('Arial', 12)
            text = font.render("MANUAL", True, (255, 255, 255))
            screen.blit(text, (pos[0] - text.get_width() // 2, pos[1]))