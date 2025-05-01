"""
Visitor Agent (UnicycleModel) implementation

This file contains the UnicycleModel class which represents the "visitor" agent
in the simulation. It includes the dynamics model, Kalman filter tracking, and
visualization functionality.
"""
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

import numpy as np
import pygame
from math import sin, cos, pi
from multitrack.filters.kalman_filter import UnicycleKalmanFilter
from multitrack.utils.config import *

# Ensure we're importing the latest value of the flag
import sys
# Get the module where SHOW_UNCERTAINTY is defined
config_module = sys.modules.get('multitrack.utils.config')

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
        # Process noise parameters
        self.noise_pos = KF_PROCESS_NOISE_POS
        self.noise_angle = KF_PROCESS_NOISE_ANGLE
        
        # Initialize Kalman filter for monitoring
        self.kalman_filter = UnicycleKalmanFilter(self.state, dt=0.1)
        self.kalman_predictions = []
        self.last_measurement_time = 0
        self.measurement_interval = DEFAULT_MEASUREMENT_INTERVAL  # How often to take measurements (seconds)
        self.prediction_horizon = PREDICTION_STEPS
        
        # Add measurement noise - for realistic monitoring
        self.measurement_noise_pos = KF_MEASUREMENT_NOISE_POS
        self.measurement_noise_angle = KF_MEASUREMENT_NOISE_ANGLE
        
        # For entropy tracking
        self.entropy_history = []
        self.entropy_times = []
        self.max_history_points = 100  # Maximum number of points to display
        self.current_entropy = 0.0
        
        # For collision handling
        self.prev_state = self.state.copy()
        
        # Store the latest noisy position for visibility detection
        self.noisy_position = np.array([self.state[0], self.state[1], self.state[2]])
        
        # Flag to indicate if Kalman filter is active (deactivated when search duration expires)
        self.kalman_filter_active = True
    
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
        
        # Reactivate Kalman filter if visitor is visible again
        if is_visible and not self.kalman_filter_active:
            self.kalman_filter_active = True
            # Reinitialize filter with current noisy measurement as starting point
            self.kalman_filter = UnicycleKalmanFilter(
                np.array([self.noisy_position[0], self.noisy_position[1], 
                          self.noisy_position[2], self.state[3]]), dt=0.1)
            print("Visitor visible again - Kalman filter reactivated with new measurement")
        
        # Update Kalman filter monitoring system only if it's active or visitor is visible
        if self.kalman_filter_active or is_visible:
            if elapsed_time - self.last_measurement_time >= self.measurement_interval:
                # Only update with measurement if the visitor is visible to the escort
                if is_visible:
                    # When visible, provide only measurement (no control inputs)
                    # The escort can see where the visitor is, but not know its control inputs
                    self.kalman_filter.update(self.noisy_position, None)  # Pass None for controls
                else:
                    # When not visible, just perform the prediction step with no controls or measurements
                    # This will properly increase uncertainty over time
                    self.kalman_filter.predict_step(None)
                
                self.last_measurement_time = elapsed_time
                
                # Generate predictions for visualization (always with no controls)
                # This simulates the escort trying to predict where visitor might go without knowing controls
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
    
    def set_controls(self, linear_vel, angular_vel):
        self.controls = np.array([linear_vel, angular_vel])
    
    def adjust_measurement_interval(self, delta):
        """Adjust the measurement interval by the specified delta amount"""
        self.measurement_interval = max(MIN_MEASUREMENT_INTERVAL, 
                                       min(MAX_MEASUREMENT_INTERVAL, 
                                           self.measurement_interval + delta))
        return self.measurement_interval
    
    def draw(self, screen):
        # Draw Kalman filter monitoring visualizations
        if SHOW_PREDICTIONS and self.kalman_predictions:
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
        
        # Draw uncertainty ellipse - get current value from the simulation module
        # Get the current value from the module where it's being toggled
        from multitrack.simulation.unicycle_reachability_simulation import SHOW_UNCERTAINTY
        if SHOW_UNCERTAINTY:
            # Get uncertainty ellipse points from Kalman filter
            ellipse_points = self.kalman_filter.get_prediction_ellipse(0.95)
            
            # Convert points to integers for pygame
            ellipse_points_int = [(int(p[0]), int(p[1])) for p in ellipse_points]
            
            # Draw ellipse as polygon
            if len(ellipse_points_int) > 2:
                pygame.draw.polygon(screen, UNCERTAINTY_COLOR, ellipse_points_int, 1)
        
        # Draw unicycle agent (red circle - true position)
        x, y, theta, _ = self.state
        radius = 10
        pygame.draw.circle(screen, RED, (int(x), int(y)), radius)
        end_x = x + radius * cos(theta)
        end_y = y + radius * sin(theta)
        pygame.draw.line(screen, WHITE, (x, y), (end_x, end_y), 2)
        
        # Draw Kalman filter estimated position (green circle)
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
        
        # Draw noisy measurement position last so it's on top (magenta circle)
        # This is what the escort agent can actually detect
        noisy_x, noisy_y = self.noisy_position[0], self.noisy_position[1]
        noisy_theta = self.noisy_position[2]
        
        # Draw noisy measurement more prominently
        pygame.draw.circle(screen, (255, 0, 255), (int(noisy_x), int(noisy_y)), 8, 3)  # Thicker outline
        end_x = noisy_x + 15 * cos(noisy_theta)
        end_y = noisy_y + 15 * sin(noisy_theta)
        pygame.draw.line(screen, (255, 0, 255), (noisy_x, noisy_y), (end_x, end_y), 3)  # Thicker line
        
        # Draw text label for the measurement
        font = pygame.font.SysFont('Arial', 12)
        text = font.render("Measurement", True, (255, 0, 255))
        screen.blit(text, (int(noisy_x) - text.get_width() // 2, int(noisy_y) - 20))
        
        # Draw entropy plot
        if len(self.entropy_history) > 1:
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
    
    def reset_kalman_filter(self):
        """Reset the Kalman filter to a high uncertainty state and deactivate it"""
        # Reset the Kalman filter with the current state but high uncertainty
        self.kalman_filter = UnicycleKalmanFilter(self.state, dt=0.1)
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