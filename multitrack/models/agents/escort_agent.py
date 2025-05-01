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
from multitrack.filters.kalman_filter import UnicycleKalmanFilter

class FollowerAgent:
    def __init__(self, initial_state=None, target_distance=FOLLOWER_TARGET_DISTANCE, 
                 search_duration=FOLLOWER_SEARCH_DURATION, walls=None, doors=None):
        """
        Follower agent that uses an MPPI controller to follow a target
        
        Parameters:
        - initial_state: [x, y, theta, v] or None for random initialization
        - target_distance: Desired following distance
        - search_duration: Frames to continue searching after losing sight of target
        - walls: List of wall rectangles for valid position checking
        - doors: List of door rectangles for valid position checking
        """
        # Initialize state
        if initial_state is None:
            # Position initialization with obstacle awareness
            if walls is not None:
                # Find a valid position that doesn't collide with walls
                self.state = self._find_valid_position(walls, doors)
            else:
                # Fallback to random initialization away from the center
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
        
        # Primary Vision system (fixed forward)
        self.vision_range = DEFAULT_VISION_RANGE
        self.vision_angle = VISION_ANGLE
        self.target_visible = False
        self.vision_cone_points = []  # Store vision cone for visualization
        
        # Secondary Vision system (rotatable camera)
        self.secondary_vision_range = DEFAULT_VISION_RANGE
        self.secondary_vision_angle = VISION_ANGLE
        self.secondary_vision_orientation = self.state[2]  # Initial orientation matches escort
        self.secondary_camera_offset = 0.0  # Offset angle from escort's orientation
        self.secondary_target_visible = False
        self.secondary_vision_cone_points = []
        
        # Enhanced camera controls with angular velocity
        self.secondary_camera_angular_vel = 0.0  # Current angular velocity of the camera
        self.secondary_camera_max_angular_vel = SECONDARY_CAMERA_MAX_ANGULAR_VEL  # Maximum angular velocity
        self.secondary_camera_angular_accel = SECONDARY_CAMERA_ANGULAR_ACCEL  # Angular acceleration per frame
        self.secondary_camera_angular_decel = SECONDARY_CAMERA_ANGULAR_DECEL  # Angular deceleration when not rotating
        
        # Camera auto-tracking system
        self.camera_auto_track = CAMERA_AUTO_TRACK_ENABLED  # Auto-tracking toggle
        self.camera_pid_p = CAMERA_PID_P  # Proportional gain
        self.camera_pid_i = CAMERA_PID_I  # Integral gain
        self.camera_pid_d = CAMERA_PID_D  # Derivative gain
        self.camera_error_integral = 0.0  # For the integral term
        self.camera_prev_error = 0.0  # For the derivative term
        self.camera_search_mode = True  # Start in search mode (active from startup)
        self.camera_track_timer = 0  # Timer for tracking timeout
        self.camera_last_seen_angle = 0.0  # Last angle visitor was seen at
        self.camera_search_direction = 1  # Direction to search in
        self.measurement_update_timer = 0  # Count frames without measurement updates
    
        # Target tracking
        self.last_seen_position = None
        self.search_mode = False
        self.search_timer = 0
        self.search_duration = search_duration
        
        # Manual control mode
        self.manual_mode = False
        
        # Kalman filter for tracking the visitor
        self.kalman_filter = None
        self.kalman_filter_active = False
        self.kalman_predictions = []
        self.last_measurement_time = 0
        self.measurement_interval = DEFAULT_MEASUREMENT_INTERVAL
        self.prediction_horizon = PREDICTION_STEPS
        
        # For entropy tracking
        self.entropy_history = []
        self.entropy_times = []
        self.max_history_points = 100  # Maximum number of points to display
        self.current_entropy = 0.0
    
    def _find_valid_position(self, walls, doors=None):
        """Find a valid position that doesn't collide with walls"""
        if doors is None:
            doors = []
        
        # Define safe areas (room centers) to try first
        # These positions are roughly the centers of different rooms
        # Use different positions than visitor to avoid starting together
        safe_positions = [
            (WIDTH * 0.60, HEIGHT * 0.60),    # Lower right of original house
            (WIDTH * 0.25, HEIGHT * 0.15),    # Bedroom
            (WIDTH * 0.80, HEIGHT * 0.15),    # Dining Room
            (WIDTH * 0.15, HEIGHT * 0.40),    # Living Room
            (WIDTH * 0.85, HEIGHT * 0.60),    # Library area
            (WIDTH * 0.30, HEIGHT * 0.80),    # Lower left area
            (WIDTH * 0.85, HEIGHT * 0.30),    # Game room
            (WIDTH * 0.85, HEIGHT * 0.80),    # Storage area
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
        
        # If all attempts fail, default to a reasonable position (might be a wall)
        return np.array([WIDTH - 100, HEIGHT - 100, 0.0, 0.0])
    
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
        # Create simple object with state attribute and add noisy_position attribute
        leader = type('Leader', (), {'state': leader_state, 'noisy_position': leader_state})
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
        
        # Draw Kalman filter predictions if available
        if SHOW_PREDICTIONS and self.kalman_filter_active and self.kalman_predictions:
            # Draw prediction path
            for i in range(1, len(self.kalman_predictions)):
                pred_prev = self.kalman_predictions[i-1]
                pred_curr = self.kalman_predictions[i]
                
                # Draw line from previous prediction to current
                pygame.draw.line(screen, PREDICTION_COLOR, 
                                (pred_prev[0], pred_prev[1]), 
                                (pred_curr[0], pred_curr[1]), 2)
                
                # Draw small circle at each prediction point
                if i % 5 == 0:  # Draw every 5th point to avoid clutter
                    pygame.draw.circle(screen, PREDICTION_COLOR, 
                                      (int(pred_curr[0]), int(pred_curr[1])), 3)
            
            # Draw final predicted position
            if len(self.kalman_predictions) > 1:
                final_pred = self.kalman_predictions[-1]
                pygame.draw.circle(screen, YELLOW, 
                                  (int(final_pred[0]), int(final_pred[1])), 5)
        
        # Draw uncertainty ellipse if Kalman filter is active
        if SHOW_UNCERTAINTY and self.kalman_filter_active and self.kalman_filter is not None:
            # Get uncertainty ellipse points from Kalman filter
            ellipse_points = self.kalman_filter.get_prediction_ellipse(0.95)
            
            # Convert points to integers for pygame
            ellipse_points_int = [(int(p[0]), int(p[1])) for p in ellipse_points]
            
            # Draw ellipse as polygon
            if len(ellipse_points_int) > 2:
                pygame.draw.polygon(screen, UNCERTAINTY_COLOR, ellipse_points_int, 1)
        
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
        
        # Draw Kalman filter estimated position (green circle) - if active
        if self.kalman_filter_active and self.kalman_filter is not None:
            kf_x, kf_y = self.kalman_filter.state[0], self.kalman_filter.state[1]
            kf_theta = self.kalman_filter.state[2]
            
            # Determine if Kalman filter is in a high uncertainty state
            is_high_uncertainty = self.current_entropy > 8.0  # Threshold for high uncertainty
            
            # Draw estimated position as circle with direction indicator
            # If high uncertainty, use a different style to indicate "unreliable estimate"
            if is_high_uncertainty:
                # Draw with dashed line for high uncertainty (estimate is unreliable)
                pygame.draw.circle(screen, GREEN, (int(kf_x), int(kf_y)), 8, 1)
                # Draw a second circle to make it look "faded"
                pygame.draw.circle(screen, GREEN, (int(kf_x), int(kf_y)), 12, 1)
                
                # Add "RESET" label to indicate Kalman filter has been reset
                font = pygame.font.SysFont('Arial', 12)
                reset_text = font.render("RESET", True, GREEN)
                screen.blit(reset_text, (int(kf_x) - reset_text.get_width() // 2, int(kf_y) - 25))
            else:
                # Regular solid circle for normal uncertainty
                pygame.draw.circle(screen, GREEN, (int(kf_x), int(kf_y)), 8, 2)
                
            # Draw direction indicator
            end_x = kf_x + 15 * cos(kf_theta)
            end_y = kf_y + 15 * sin(kf_theta)
            # Also make the direction line dashed/faded when uncertainty is high
            if is_high_uncertainty:
                # Draw a thinner, more transparent line for high uncertainty
                line_color = (0, 200, 0, 128)  # Transparent green
                line_surface = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
                pygame.draw.line(line_surface, line_color, (kf_x, kf_y), (end_x, end_y), 1)
                screen.blit(line_surface, (0, 0))
            else:
                pygame.draw.line(screen, GREEN, (kf_x, kf_y), (end_x, end_y), 2)
        
        # Draw primary vision cone (fixed forward)
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
        
        # Draw secondary vision cone (rotatable camera)
        if self.secondary_vision_cone_points and len(self.secondary_vision_cone_points) > 2:
            # Convert vision cone points to pygame format
            vision_points = [(int(p[0]), int(p[1])) for p in self.secondary_vision_cone_points]
            
            # Create a semi-transparent surface for the vision cone
            vision_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            
            # Set color based on whether target is visible with secondary camera
            # Use a different color scheme for the secondary camera
            if self.secondary_target_visible:
                # Cyan when target is visible with secondary camera
                vision_color = (0, 200, 255, VISION_TRANSPARENCY)
            else:
                # Light purple when searching with secondary camera
                vision_color = (200, 100, 255, VISION_TRANSPARENCY)
            
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
            
            # Draw a small indicator showing the camera orientation
            cam_x, cam_y = self.state[0], self.state[1]
            cam_dir_x = cam_x + 18 * cos(self.secondary_vision_orientation)
            cam_dir_y = cam_y + 18 * sin(self.secondary_vision_orientation)
            pygame.draw.line(vision_surface, (255, 255, 255, 200), 
                         (int(cam_x), int(cam_y)), 
                         (int(cam_dir_x), int(cam_dir_y)), 3)
            
            # Draw a small camera icon at the position
            pygame.draw.circle(vision_surface, (255, 255, 255, 200), 
                          (int(cam_x), int(cam_y)), 5, 2)
            
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
        
        # Draw entropy plot if Kalman filter is active
        if self.kalman_filter_active and len(self.entropy_history) > 1:
            # Get actual screen dimensions
            screen_width, screen_height = screen.get_size()
            
            # Entropy plot dimensions and position - moved to extreme right top corner
            plot_width, plot_height = 150, 80
            plot_x, plot_y = screen_width - plot_width - 5, 5
            
            # Draw plot background with transparency
            plot_bg = pygame.Surface((plot_width, plot_height))
            plot_bg.fill((50, 50, 50))
            plot_bg.set_alpha(150)  # More transparent background
            screen.blit(plot_bg, (plot_x, plot_y))
            pygame.draw.rect(screen, WHITE, (plot_x, plot_y, plot_width, plot_height), 1)
            
            # Normalize entropy values to plot height
            if len(self.entropy_history) > 0:
                min_entropy = min(self.entropy_history)
                max_entropy = max(max(self.entropy_history), min_entropy + 0.1)  # Avoid division by zero
                
                # Draw entropy line graph
                points = []
                for i, entropy in enumerate(self.entropy_history):
                    x = plot_x + i * (plot_width / self.max_history_points)
                    # Invert y-axis so higher entropy is higher on the plot
                    normalized_entropy = 1.0 - (entropy - min_entropy) / (max_entropy - min_entropy)
                    y = plot_y + normalized_entropy * plot_height
                    points.append((x, y))
                
                # Draw the entropy line
                if len(points) > 1:
                    pygame.draw.lines(screen, (255, 100, 100), False, points, 2)
            
            # Draw plot labels
            font = pygame.font.SysFont('Arial', 12)
            title = font.render("Entropy (Position)", True, WHITE)
            current = font.render(f"Current: {self.current_entropy:.2f}", True, WHITE)
            screen.blit(title, (plot_x + 5, plot_y + 5))
            screen.blit(current, (plot_x + 5, plot_y + plot_height - 15))    
    
    def reset_kalman_filter(self):
        """Reset the Kalman filter to a high uncertainty state and deactivate it"""
        # Reset the Kalman filter with the current state but high uncertainty
        # Initialize with a default state if no prediction available
        if self.last_seen_position is not None:
            initial_state = np.array([
                self.last_seen_position[0],
                self.last_seen_position[1],
                0.0,  # Default theta
                0.0   # Default velocity
            ])
        else:
            # If no last seen position, just use the escort's position with some offset
            initial_state = np.array([
                self.state[0] + 100,  # Arbitrary offset to prevent starting at escort's position
                self.state[1] + 100,
                0.0,
                0.0
            ])
            
        self.kalman_filter = UnicycleKalmanFilter(initial_state, dt=0.1)
        # Increase initial uncertainty to reflect complete lack of knowledge
        self.kalman_filter.P = np.diag([50.0, 50.0, 1.0, 10.0, 1.0])  # High initial uncertainty
        self.kalman_predictions = []
        
        # Calculate new entropy
        self.current_entropy = self.kalman_filter.calculate_entropy(position_only=True)
        
        # Add to history
        self.entropy_history.append(self.current_entropy)
        self.entropy_times.append(self.last_measurement_time)
        
        # Deactivate Kalman filter - will be reactivated only when visitor is seen again
        self.kalman_filter_active = False
        print("Kalman filter deactivated - MPPI can no longer use its values until visitor is seen again")
    
    def update_kalman_filter(self, noisy_measurement, elapsed_time, is_visible):
        """
        Update the Kalman filter with a new noisy measurement from the visitor
        
        Parameters:
        - noisy_measurement: Noisy position and orientation of the visitor [x, y, theta]
        - elapsed_time: Current simulation time for interval tracking
        - is_visible: Whether the visitor is visible to the escort
        """
        # Track frames without measurement updates
        if is_visible:
            self.measurement_update_timer = 0
        else:
            self.measurement_update_timer += 1
            
        # If we haven't received measurements for a while, ensure camera is in search mode
        # This works regardless of whether the camera is in auto-tracking mode
        if self.measurement_update_timer > 10 and not self.camera_search_mode:
            self.camera_search_mode = True
            self.camera_track_timer = 0
        
        # Initialize Kalman filter if it doesn't exist yet
        if self.kalman_filter is None:
            if is_visible and noisy_measurement is not None:
                # Initialize with the first measurement
                initial_state = np.array([
                    noisy_measurement[0],
                    noisy_measurement[1],
                    noisy_measurement[2],
                    0.0  # Initial velocity
                ])
                self.kalman_filter = UnicycleKalmanFilter(initial_state, dt=0.1)
                self.kalman_filter_active = True
                self.last_measurement_time = elapsed_time
                print("Kalman filter initialized with first measurement")
            else:
                # Can't initialize without a measurement, so just return
                return
        
        # Reactivate Kalman filter if visitor is visible again
        if is_visible and not self.kalman_filter_active:
            self.kalman_filter_active = True
            # Reinitialize filter with current noisy measurement as starting point
            self.kalman_filter = UnicycleKalmanFilter(
                np.array([
                    noisy_measurement[0],
                    noisy_measurement[1], 
                    noisy_measurement[2],
                    0.0  # Reset velocity
                ]), dt=0.1)
            print("Visitor visible again - Kalman filter reactivated with new measurement")
        
        # Update Kalman filter monitoring system only if it's active or visitor is visible
        if self.kalman_filter_active or is_visible:
            if elapsed_time - self.last_measurement_time >= self.measurement_interval:
                # Only update with measurement if the visitor is visible to the escort
                if is_visible:
                    # When visible, provide only measurement (no control inputs)
                    # We don't know the visitor's control inputs, just their noisy position
                    self.kalman_filter.update(noisy_measurement, None)  # Pass None for controls
                else:
                    # When not visible, just perform the prediction step with no controls or measurements
                    # This will properly increase uncertainty over time
                    self.kalman_filter.predict_step(None)
                
                self.last_measurement_time = elapsed_time
                
                # Generate predictions for visualization (always with no controls)
                # This simulates trying to predict where visitor might go without knowing controls
                self.kalman_predictions = self.kalman_filter.predict(None, self.prediction_horizon)
            else:
                # For regular updates between measurement intervals
                if is_visible:
                    # Even for intermediate updates, never pass control inputs
                    self.kalman_filter.update(None, None)
                else:
                    # When not visible, just run prediction with no controls to increase uncertainty
                    self.kalman_filter.predict_step(None)
        
        # Calculate entropy after every update (only if filter is active)
        if self.kalman_filter_active:
            self.current_entropy = self.kalman_filter.calculate_entropy(position_only=True)
            
            # Track entropy history
            self.entropy_history.append(self.current_entropy)
            self.entropy_times.append(elapsed_time)
            
            # Keep history limited to max_history_points
            if len(self.entropy_history) > self.max_history_points:
                self.entropy_history.pop(0)
                self.entropy_times.pop(0)
    
    def adjust_measurement_interval(self, delta):
        """Adjust the measurement interval by the specified delta amount"""
        self.measurement_interval = max(MIN_MEASUREMENT_INTERVAL, 
                                       min(MAX_MEASUREMENT_INTERVAL, 
                                           self.measurement_interval + delta))
        return self.measurement_interval
    
    def rotate_secondary_camera(self, direction):
        """
        Rotate the secondary camera in the specified direction
        
        Parameters:
        - direction: 1 for clockwise, -1 for counter-clockwise
        """
        # Update the offset angle of the secondary camera
        self.secondary_camera_offset += direction * self.secondary_vision_rotation_speed
        
        # Normalize angle to [-pi, pi]
        self.secondary_camera_offset = (self.secondary_camera_offset + pi) % (2 * pi) - pi
        
        # Calculate absolute orientation by adding offset to escort's orientation
        self.secondary_vision_orientation = (self.state[2] + self.secondary_camera_offset + pi) % (2 * pi) - pi
    
    def update_secondary_camera(self, rotation_input, dt):
        """
        Update the secondary camera's rotation based on angular velocity and user input
        
        Parameters:
        - rotation_input: -1 for counterclockwise (Q), 1 for clockwise (E), 0 for no rotation
        - dt: Time step in seconds
        """
        # Update angular velocity based on input
        if rotation_input != 0:
            # Accelerate in the direction of rotation input
            target_vel = rotation_input * self.secondary_camera_max_angular_vel
            # Gradually approach the target velocity
            if self.secondary_camera_angular_vel < target_vel:
                self.secondary_camera_angular_vel = min(target_vel, 
                    self.secondary_camera_angular_vel + self.secondary_camera_angular_accel)
            elif self.secondary_camera_angular_vel > target_vel:
                self.secondary_camera_angular_vel = max(target_vel, 
                    self.secondary_camera_angular_vel - self.secondary_camera_angular_accel)
        else:
            # Decelerate when no input
            if abs(self.secondary_camera_angular_vel) < self.secondary_camera_angular_decel:
                self.secondary_camera_angular_vel = 0  # Stop completely if nearly stopped
            elif self.secondary_camera_angular_vel > 0:
                self.secondary_camera_angular_vel -= self.secondary_camera_angular_decel
            else:
                self.secondary_camera_angular_vel += self.secondary_camera_angular_decel
        
        # Update the camera offset based on angular velocity
        if self.secondary_camera_angular_vel != 0:
            self.secondary_camera_offset += self.secondary_camera_angular_vel * dt
            
            # Normalize angle to [-pi, pi]
            self.secondary_camera_offset = (self.secondary_camera_offset + pi) % (2 * pi) - pi
            
            # Calculate absolute orientation by adding offset to escort's orientation
            self.secondary_vision_orientation = (self.state[2] + self.secondary_camera_offset + pi) % (2 * pi) - pi
    
    def update_secondary_vision(self, leader, walls, doors):
        """
        Update the secondary vision cone and check if the target is visible
        
        Parameters:
        - leader: Leader agent with state and noisy_position
        - walls: List of wall rectangles
        - doors: List of door rectangles
        
        Returns:
        - True if leader is visible with the secondary camera, False otherwise
        """
        # Update the secondary camera orientation based on escort's current orientation
        self.secondary_vision_orientation = (self.state[2] + self.secondary_camera_offset + pi) % (2 * pi) - pi
        
        # Create a temporary agent for vision cone calculation
        # This represents the secondary camera with its own orientation
        secondary_camera = type('Camera', (), {
            'state': np.array([
                self.state[0],  # x - same as escort
                self.state[1],  # y - same as escort
                self.secondary_vision_orientation,  # theta - updated orientation based on offset
                0.0  # v - not needed for vision
            ])
        })
        
        # Calculate vision cone points for the secondary camera
        self.secondary_vision_cone_points = get_vision_cone_points(
            secondary_camera, 
            self.secondary_vision_range, 
            self.secondary_vision_angle, 
            walls, 
            doors
        )
        
        # Check if the leader is visible with the secondary camera
        self.secondary_target_visible = is_agent_in_vision_cone(
            secondary_camera,
            leader,
            self.secondary_vision_range,
            self.secondary_vision_angle,
            walls,
            doors
        )
        
        return self.secondary_target_visible
    
    def toggle_camera_auto_track(self):
        """Toggle camera auto-tracking mode on/off"""
        self.camera_auto_track = not self.camera_auto_track
        
        # Reset PID controller when enabling auto-tracking
        if self.camera_auto_track:
            self.camera_error_integral = 0.0
            self.camera_prev_error = 0.0
            self.camera_search_mode = False
            self.camera_track_timer = 0
            print("Camera auto-tracking activated")
        else:
            print("Camera auto-tracking deactivated")
        
        return self.camera_auto_track
    
    def update_camera_auto_tracking(self, leader_state, dt):
        """
        Update camera position using PID controller to track the visitor or search for them when lost
        
        Parameters:
        - leader_state: Noisy position of the leader [x, y, theta]
        - dt: Time step in seconds
        
        Returns:
        - rotation_input: Camera rotation input (-1, 0, or 1)
        """
        # If target is visible and auto-tracking is enabled, directly track it with PID controller
        if self.secondary_target_visible and self.camera_auto_track:
            # Reset search mode and store the last seen position when visitor is visible
            self.camera_search_mode = False
            self.camera_track_timer = 0
            
            # Calculate angle to target
            target_x, target_y = leader_state[0], leader_state[1]
            escort_x, escort_y = self.state[0], self.state[1]
            target_angle = np.arctan2(target_y - escort_y, target_x - escort_x)
            
            # Store the angle where we last saw the target (for all modes)
            self.camera_last_seen_angle = target_angle
            
            # Calculate error (difference between current camera angle and desired angle)
            error = target_angle - self.secondary_vision_orientation
            error = (error + pi) % (2 * pi) - pi  # Normalize to [-pi, pi]
            
            # Calculate integral term (with anti-windup)
            self.camera_error_integral += error * dt
            self.camera_error_integral = max(-CAMERA_MAX_ERROR_INTEGRAL, 
                                         min(CAMERA_MAX_ERROR_INTEGRAL, 
                                             self.camera_error_integral))
            
            # Calculate derivative term
            error_derivative = (error - self.camera_prev_error) / dt if dt > 0 else 0
            self.camera_prev_error = error
            
            # PID control equation
            control = (self.camera_pid_p * error + 
                      self.camera_pid_i * self.camera_error_integral + 
                      self.camera_pid_d * error_derivative)
            
            # Convert control output to angular velocity
            self.secondary_camera_angular_vel = max(-self.secondary_camera_max_angular_vel,
                                                min(self.secondary_camera_max_angular_vel, 
                                                    control))
        # If target is visible but auto-tracking is off, just store the angle (no movement)
        elif self.secondary_target_visible and not self.camera_auto_track:
            # Even when auto-tracking is off, still keep track of where we saw the visitor
            target_x, target_y = leader_state[0], leader_state[1]
            escort_x, escort_y = self.state[0], self.state[1]
            self.camera_last_seen_angle = np.arctan2(target_y - escort_y, target_x - escort_x)
            self.camera_search_mode = False
            self.camera_track_timer = 0
            # Don't adjust angular velocity - let manual control handle it
            return 0
        # Target is not visible - search mode
        else:
            # If we're not already in search mode, initialize it
            if not self.camera_search_mode:
                self.camera_search_mode = True
                self.camera_track_timer = 0
                
                # Always set search direction to clockwise (positive value)
                self.camera_search_direction = 1
            
            # Always increment tracking timeout when in search mode
            self.camera_track_timer += 1
            
            # First try to return to the last known position (if we have one)
            if self.camera_track_timer < CAMERA_TRACK_TIMEOUT and self.camera_last_seen_angle != 0.0:
                # Calculate error (difference between current camera angle and last seen angle)
                error = self.camera_last_seen_angle - self.secondary_vision_orientation
                error = (error + pi) % (2 * pi) - pi  # Normalize to [-pi, pi]
                
                # If we're close to the last seen angle and still don't see the target,
                # start the search pattern
                if abs(error) < 0.1:  # About 5.7 degrees
                    # Set a constant angular velocity for clockwise sweeping search
                    self.secondary_camera_angular_vel = CAMERA_SEARCH_SPEED
                else:
                    # Move toward the last seen angle faster than normal
                    self.secondary_camera_angular_vel = max(-self.secondary_camera_max_angular_vel,
                                                        min(self.secondary_camera_max_angular_vel, 
                                                            5.0 * error))  # Faster convergence
            else:
                # Full 360° search mode - constant angular velocity in clockwise direction
                self.secondary_camera_angular_vel = CAMERA_SEARCH_SPEED
                
                # No longer reverse direction - always rotate clockwise
        
        # Always update the camera position based on angular velocity
        self.secondary_camera_offset += self.secondary_camera_angular_vel * dt
        
        # Normalize angle to [-pi, pi]
        self.secondary_camera_offset = (self.secondary_camera_offset + pi) % (2 * pi) - pi
        
        # Calculate absolute orientation by adding offset to escort's orientation
        self.secondary_vision_orientation = (self.state[2] + self.secondary_camera_offset + pi) % (2 * pi) - pi
        
        return 0