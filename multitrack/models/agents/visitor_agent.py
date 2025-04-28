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

class UnicycleModel:
    def __init__(self):
        # State: [x, y, theta, v]
        self.state = np.array([WIDTH//2, HEIGHT//2, 0.0, 0.0])
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

    def update(self, dt=0.1, elapsed_time=0, walls=None, doors=None):
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
        
        # Update Kalman filter monitoring system
        if elapsed_time - self.last_measurement_time >= self.measurement_interval:
            # Create a noisy measurement
            noisy_measurement = np.array([
                self.state[0] + np.random.normal(0, self.measurement_noise_pos),
                self.state[1] + np.random.normal(0, self.measurement_noise_pos),
                self.state[2] + np.random.normal(0, self.measurement_noise_angle)
            ])
            
            # Update Kalman filter with measurement and control input
            self.kalman_filter.update(noisy_measurement, self.controls)
            self.last_measurement_time = elapsed_time
            
            # Generate predictions for visualization
            self.kalman_predictions = self.kalman_filter.predict(self.controls, self.prediction_horizon)
        else:
            # Just update the filter prediction without measurement
            self.kalman_filter.update(None, self.controls)
        
        # Calculate entropy after every update
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
        
        # Draw uncertainty ellipse
        if SHOW_UNCERTAINTY:
            # Get uncertainty ellipse points from Kalman filter
            ellipse_points = self.kalman_filter.get_prediction_ellipse(0.95)
            
            # Convert points to integers for pygame
            ellipse_points_int = [(int(p[0]), int(p[1])) for p in ellipse_points]
            
            # Draw ellipse as polygon
            if len(ellipse_points_int) > 2:
                pygame.draw.polygon(screen, UNCERTAINTY_COLOR, ellipse_points_int, 1)
        
        # Draw Kalman filter estimated position
        kf_x, kf_y = self.kalman_filter.state[0], self.kalman_filter.state[1]
        kf_theta = self.kalman_filter.state[2]
        
        # Draw estimated position as circle with direction indicator
        pygame.draw.circle(screen, GREEN, (int(kf_x), int(kf_y)), 8, 2)
        end_x = kf_x + 15 * cos(kf_theta)
        end_y = kf_y + 15 * sin(kf_theta)
        pygame.draw.line(screen, GREEN, (kf_x, kf_y), (end_x, end_y), 2)
        
        # Draw unicycle agent
        x, y, theta, _ = self.state
        radius = 10
        pygame.draw.circle(screen, RED, (int(x), int(y)), radius)
        end_x = x + radius * cos(theta)
        end_y = y + radius * sin(theta)
        pygame.draw.line(screen, WHITE, (x, y), (end_x, end_y), 2)
        
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