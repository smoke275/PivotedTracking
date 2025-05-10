"""
PID Controller for unicycle model.

This file implements a PID controller for the unicycle model used in the PivotedTracking project.
It provides a simpler alternative to the MPPI controller with lower computational requirements.
"""
import os
import sys
import time
import numpy as np
from math import sin, cos, pi, sqrt, atan2
from collections import deque

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from multitrack.controllers.base_controller import BaseController
from multitrack.utils.config import *

# Device info for monitoring
DEVICE_INFO = "CPU (PID Controller)"

class PIDController(BaseController):
    """
    PID Controller implementation for a unicycle model.
    
    This controller uses a proportional-integral-derivative control law to track
    a target trajectory. It's simpler than MPPI but still effective for many scenarios.
    """
    
    def __init__(self, horizon=20, dt=0.1, control_limits=None):
        """
        Initialize PID controller for unicycle model
        
        Parameters:
        - horizon: Planning horizon (number of time steps) for prediction
        - dt: Time step duration
        - control_limits: Dictionary with max/min limits for controls
        """
        self.horizon = horizon
        self.dt = dt
        
        # Default control limits if not provided
        if control_limits is None:
            control_limits = {
                'v_min': FOLLOWER_LINEAR_VEL_MIN,
                'v_max': FOLLOWER_LINEAR_VEL_MAX,
                'omega_min': FOLLOWER_ANGULAR_VEL_MIN,
                'omega_max': FOLLOWER_ANGULAR_VEL_MAX
            }
        self.control_limits = control_limits
        
        # PID gains for position control
        self.position_kp = 1.0  # Proportional gain
        self.position_ki = 0.01  # Integral gain
        self.position_kd = 0.5  # Derivative gain
        
        # PID gains for orientation control
        self.orientation_kp = 2.0  # Proportional gain
        self.orientation_ki = 0.01  # Integral gain
        self.orientation_kd = 0.5  # Derivative gain
        
        # Error accumulators for integral terms
        self.position_error_sum = 0.0
        self.orientation_error_sum = 0.0
        
        # Previous errors for derivative terms
        self.prev_position_error = 0.0
        self.prev_orientation_error = 0.0
        
        # Maximum error sum to prevent integral windup
        self.max_error_sum = 100.0
        
        # Minimum safety distance
        self.safety_distance = FOLLOWER_SAFETY_DISTANCE
        
        # Performance monitoring
        self.computation_times = deque(maxlen=10)
        self.last_compute_time = 0
        
        # Last computed trajectory for visualization
        self.last_trajectory = None
    
    def compute_control(self, current_state, target_trajectory, obstacles=None):
        """
        Compute optimal control using PID control laws
        
        Parameters:
        - current_state: Current state [x, y, theta, v]
        - target_trajectory: Sequence of target states to follow
        - obstacles: List of obstacle positions [(x, y, radius), ...]
        
        Returns:
        - optimal_control: Optimal control [v, omega]
        - predicted_trajectory: Predicted trajectory using the computed control
        """
        # Start timing
        start_time = time.time()
        
        # Extract current state
        x, y, theta, _ = current_state
        
        # Get target state (first point in the trajectory)
        target_state = target_trajectory[0]
        tx, ty, target_theta, _ = target_state
        
        # Calculate errors
        dx = tx - x
        dy = ty - y
        
        # Position error (distance to target)
        position_error = sqrt(dx**2 + dy**2)
        
        # Desired heading to target
        desired_heading = atan2(dy, dx)
        
        # Heading error (normalized to [-pi, pi])
        heading_error = (desired_heading - theta + pi) % (2 * pi) - pi
        
        # Update integral terms with anti-windup
        self.position_error_sum += position_error * self.dt
        self.position_error_sum = max(-self.max_error_sum, min(self.max_error_sum, self.position_error_sum))
        
        self.orientation_error_sum += heading_error * self.dt
        self.orientation_error_sum = max(-self.max_error_sum, min(self.max_error_sum, self.orientation_error_sum))
        
        # Calculate derivative terms
        position_error_derivative = (position_error - self.prev_position_error) / self.dt
        orientation_error_derivative = (heading_error - self.prev_orientation_error) / self.dt
        
        # Update previous errors
        self.prev_position_error = position_error
        self.prev_orientation_error = heading_error
        
        # PID control for linear velocity
        desired_v = (self.position_kp * position_error + 
                     self.position_ki * self.position_error_sum + 
                     self.position_kd * position_error_derivative)
        
        # If we need to go backwards (heading error > 90 degrees), reverse velocity
        if abs(heading_error) > pi/2:
            desired_v = -desired_v
            # Adjust heading error for reverse motion
            heading_error = (heading_error + pi) % (2 * pi) - pi
        
        # PID control for angular velocity
        desired_omega = (self.orientation_kp * heading_error + 
                         self.orientation_ki * self.orientation_error_sum + 
                         self.orientation_kd * orientation_error_derivative)
        
        # Apply control limits
        desired_v = max(self.control_limits['v_min'], min(self.control_limits['v_max'], desired_v))
        desired_omega = max(self.control_limits['omega_min'], min(self.control_limits['omega_max'], desired_omega))
        
        # Stop if very close to target
        if position_error < 10.0:
            desired_v *= position_error / 10.0  # Scale down velocity as we get closer
            
        # Obstacle avoidance
        if obstacles:
            # Simple reactive obstacle avoidance
            for ox, oy, radius in obstacles:
                # Calculate distance to obstacle
                obstacle_dist = sqrt((x - ox)**2 + (y - oy)**2) - radius
                
                if obstacle_dist < self.safety_distance:
                    # Calculate vector from obstacle to agent
                    avoid_dx = x - ox
                    avoid_dy = y - oy
                    avoid_dist = sqrt(avoid_dx**2 + avoid_dy**2)
                    
                    # Normalize
                    if avoid_dist > 0:
                        avoid_dx /= avoid_dist
                        avoid_dy /= avoid_dist
                    
                    # Calculate avoidance heading
                    avoid_heading = atan2(avoid_dy, avoid_dx)
                    
                    # Blend avoidance heading with desired heading
                    # Weight by how close we are to the obstacle
                    blend_factor = 1.0 - min(1.0, obstacle_dist / self.safety_distance)
                    adjusted_heading = (1 - blend_factor) * desired_heading + blend_factor * avoid_heading
                    
                    # Recalculate heading error with adjusted heading
                    adjusted_heading_error = (adjusted_heading - theta + pi) % (2 * pi) - pi
                    
                    # Apply to angular velocity
                    obstacle_omega = self.orientation_kp * adjusted_heading_error * 2.0
                    desired_omega = (1 - blend_factor) * desired_omega + blend_factor * obstacle_omega
                    
                    # Reduce speed near obstacles
                    speed_factor = max(0.2, obstacle_dist / self.safety_distance)
                    desired_v *= speed_factor
        
        # Create optimal control vector
        optimal_control = np.array([desired_v, desired_omega])
        
        # Compute predicted trajectory
        predicted_trajectory = self._compute_predicted_trajectory(current_state, optimal_control)
        self.last_trajectory = predicted_trajectory
        
        # Record computation time
        self.last_compute_time = time.time() - start_time
        self.computation_times.append(self.last_compute_time)
        
        return optimal_control, predicted_trajectory
    
    def reset(self):
        """Reset the controller's internal state"""
        # Reset error accumulators
        self.position_error_sum = 0.0
        self.orientation_error_sum = 0.0
        self.prev_position_error = 0.0
        self.prev_orientation_error = 0.0
        self.last_trajectory = None
    
    def get_computation_stats(self):
        """Return performance statistics for monitoring"""
        if not self.computation_times:
            return {"avg_time": 0, "last_time": 0, "device": DEVICE_INFO}
            
        avg_time = sum(self.computation_times) / len(self.computation_times)
        return {
            "avg_time": avg_time * 1000,  # Convert to ms
            "last_time": self.last_compute_time * 1000,  # Convert to ms
            "device": DEVICE_INFO
        }
    
    def _compute_predicted_trajectory(self, initial_state, control):
        """
        Compute a predicted trajectory using the current control values
        
        Parameters:
        - initial_state: Initial state [x, y, theta, v]
        - control: Control values [v, omega] to apply
        
        Returns:
        - states: Predicted states over the horizon [horizon+1, 4]
        """
        states = np.zeros((self.horizon + 1, 4))
        states[0] = initial_state.copy()
        
        # Extract control
        v, omega = control
        
        # Simple forward simulation with constant control
        for t in range(self.horizon):
            x, y, theta, _ = states[t]
            
            # Apply unicycle dynamics
            new_x = x + v * cos(theta) * self.dt
            new_y = y + v * sin(theta) * self.dt
            new_theta = (theta + omega * self.dt) % (2 * pi)
            
            # Update state
            states[t+1] = [new_x, new_y, new_theta, v]
            
        return states