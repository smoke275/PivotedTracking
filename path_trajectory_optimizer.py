#!/usr/bin/env python3
"""
Path Trajectory Optimizer Integration
Integrates the RobotTrajectoryOptimizer with the agent simulation system.
Generates optimized trajectories for clicked RRT paths.
Includes threaded and multiprocessing trajectory calculators for high performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.optimize import minimize
import math
from typing import List, Tuple, Optional, Dict, Any
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Check for dubins library availability
try:
    import dubins
    DUBINS_AVAILABLE = True
    print("✅ pydubins is available")
except ImportError:
    DUBINS_AVAILABLE = False
    print("❌ pydubins not found. Install with: pip install dubins")

try:
    from matplotlib.widgets import Slider, Button
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("❌ matplotlib widgets not available")


class RobotTrajectoryOptimizer:
    def __init__(self, max_velocity=2.0, max_acceleration=1.5, max_turning_rate=1.0):
        """
        Initialize the trajectory optimizer
        
        Args:
            max_velocity: Maximum linear velocity (m/s)
            max_acceleration: Maximum acceleration (m/s²)
            max_turning_rate: Maximum angular velocity/turning rate (rad/s)
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_turning_rate = max_turning_rate
    
    def calculate_turn_radius(self, velocity, angular_velocity=None):
        """
        Calculate turn radius from velocity and angular velocity
        
        Args:
            velocity: Linear velocity (m/s)
            angular_velocity: Angular velocity (rad/s). If None, uses max_turning_rate
            
        Returns:
            turn_radius: Turn radius (m)
        """
        if angular_velocity is None:
            angular_velocity = self.max_turning_rate
        
        # Avoid division by zero
        if abs(angular_velocity) < 1e-6:
            return float('inf')  # Straight line
        
        return abs(velocity / angular_velocity)
    
    def get_dynamic_turn_radius(self, velocity_for_dubins=None):
        """
        Get dynamic turn radius based on robot constraints
        
        Args:
            velocity_for_dubins: Velocity to use for Dubins calculation. 
                                If None, uses a fraction of max_velocity
        
        Returns:
            turn_radius: Computed turn radius (m)
        """
        if velocity_for_dubins is None:
            # Use a conservative velocity for Dubins planning (e.g., 15% of max)
            velocity_for_dubins = self.max_velocity * 0.15

        return self.calculate_turn_radius(velocity_for_dubins, self.max_turning_rate)
        
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi
        return angle
    
    def angle_difference(self, angle1, angle2):
        """Calculate the shortest angular difference between two angles"""
        diff = angle2 - angle1
        return self.normalize_angle(diff)
    
    def get_dubins_path(self, start_pos, start_angle, end_pos, end_angle, velocity_for_dubins=None, step_size=0.1):
        """
        Generate Dubins path using pydubins library with dynamic turn radius
        
        Args:
            start_pos: (x, y) starting position
            start_angle: starting orientation in radians
            end_pos: (x, y) ending position  
            end_angle: ending orientation in radians
            velocity_for_dubins: velocity to use for turn radius calculation
            step_size: distance between path points
            
        Returns:
            path_points: array of (x, y) coordinates
            path_length: total path length
            path_type: string describing the path type
            turn_radius_used: the turn radius used for this path
        """
        if not DUBINS_AVAILABLE:
            # Fallback to straight line
            path_points = np.array([start_pos, end_pos])
            path_length = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
            return path_points, path_length, "straight (no pydubins)", float('inf')
        
        # Calculate dynamic turn radius
        turn_radius = self.get_dynamic_turn_radius(velocity_for_dubins)
        
        # Create start and end configurations: (x, y, theta)
        start_config = (start_pos[0], start_pos[1], start_angle)
        end_config = (end_pos[0], end_pos[1], end_angle)
        
        try:
            # Generate Dubins path
            path = dubins.shortest_path(start_config, end_config, turn_radius)
            
            # Get path length
            path_length = path.path_length()
            
            # Get path type
            path_type = path.path_type()
            
            # Sample points along the path
            configurations, _ = path.sample_many(step_size)
            
            # Extract x, y coordinates
            path_points = np.array([[config[0], config[1]] for config in configurations])
            
            return path_points, path_length, path_type, turn_radius
            
        except Exception as e:
            print(f"Error generating Dubins path: {e}")
            # Fallback to straight line
            path_points = np.array([start_pos, end_pos])
            path_length = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
            return path_points, path_length, "error_fallback", turn_radius
    
    def smooth_path_with_orientation(self, waypoints, orientations=None, num_points=100, smoothing_factor=0):
        """
        Create a smooth path through waypoints considering orientations
        
        Args:
            waypoints: List of (x, y) tuples
            orientations: List of orientations at waypoints (rad). If None, calculated from path
            num_points: Number of interpolated points
            smoothing_factor: Smoothing parameter (0 = interpolation, >0 = approximation)
        
        Returns:
            Smoothed path as numpy array of shape (num_points, 2)
            Smoothed orientations as numpy array of shape (num_points,)
        """
        waypoints = np.array(waypoints)
        
        # If orientations not provided, calculate them from waypoint directions
        if orientations is None or any(o is None for o in orientations):
            calculated_orientations = np.zeros(len(waypoints))
            for i in range(len(waypoints)):
                if i == 0 and len(waypoints) > 1:
                    # First waypoint: look towards second waypoint
                    dx = waypoints[1, 0] - waypoints[0, 0]
                    dy = waypoints[1, 1] - waypoints[0, 1]
                    calculated_orientations[i] = np.arctan2(dy, dx)
                elif i == len(waypoints) - 1:
                    # Last waypoint: maintain direction from previous
                    dx = waypoints[i, 0] - waypoints[i-1, 0]
                    dy = waypoints[i, 1] - waypoints[i-1, 1]
                    calculated_orientations[i] = np.arctan2(dy, dx)
                else:
                    # Middle waypoints: average direction
                    dx1 = waypoints[i, 0] - waypoints[i-1, 0]
                    dy1 = waypoints[i, 1] - waypoints[i-1, 1]
                    dx2 = waypoints[i+1, 0] - waypoints[i, 0]
                    dy2 = waypoints[i+1, 1] - waypoints[i, 1]
                    angle1 = np.arctan2(dy1, dx1)
                    angle2 = np.arctan2(dy2, dx2)
                    # Take average of angles, handling wraparound
                    calculated_orientations[i] = angle1 + self.normalize_angle(angle2 - angle1) / 2
            
            # If orientations was partially None, fill in the None values
            if orientations is not None:
                orientations = np.array(orientations, dtype=object)
                for i in range(len(orientations)):
                    if orientations[i] is None:
                        orientations[i] = calculated_orientations[i]
                orientations = orientations.astype(float)
            else:
                orientations = calculated_orientations
        
        # Use parametric spline interpolation for position
        tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], 
                        s=smoothing_factor, k=min(3, len(waypoints)-1))
        
        # Generate smooth path
        u_new = np.linspace(0, 1, num_points)
        smooth_path = np.array(splev(u_new, tck)).T
        
        # Interpolate orientations, handling angle wraparound
        orientations = np.array(orientations, dtype=float)
        
        # Unwrap angles to avoid interpolation issues
        unwrapped_orientations = np.unwrap(orientations)
        
        # Interpolate unwrapped angles
        u_waypoints = np.linspace(0, 1, len(waypoints))
        smooth_orientations = np.interp(u_new, u_waypoints, unwrapped_orientations)
        
        # Wrap back to [-pi, pi]
        smooth_orientations = np.array([self.normalize_angle(angle) for angle in smooth_orientations])
        
        return smooth_path, smooth_orientations
    
    def calculate_path_derivatives(self, path):
        """
        Calculate first and second derivatives of the path
        
        Args:
            path: Array of path points (n, 2)
            
        Returns:
            dx, dy, ddx, ddy: First and second derivatives
        """
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        return dx, dy, ddx, ddy
    
    def calculate_curvature(self, path):
        """
        Calculate curvature at each point along the path
        
        Args:
            path: Array of path points (n, 2)
            
        Returns:
            Array of curvature values
        """
        dx, dy, ddx, ddy = self.calculate_path_derivatives(path)
        
        # Curvature formula: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
        numerator = dx * ddy - dy * ddx
        denominator = (dx**2 + dy**2)**(3/2)
        
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-6, denominator)
        curvature = numerator / denominator
        
        return np.abs(curvature)
    
    def calculate_orientation_constraints(self, path_orientations, target_orientations):
        """
        Calculate angular velocity constraints based on orientation differences
        
        Args:
            path_orientations: Current path orientations
            target_orientations: Target orientations at waypoints
            
        Returns:
            orientation_errors: Angular errors at each point
        """
        orientation_errors = np.zeros(len(path_orientations))
        
        for i in range(len(path_orientations)):
            # Find closest target orientation (this is simplified)
            orientation_errors[i] = abs(self.angle_difference(path_orientations[i], target_orientations[i]))
        
        return orientation_errors
    
    def calculate_turning_rate_limited_velocity(self, path, orientations):
        """
        Calculate maximum velocity based on turning rate constraints
        
        Args:
            path: Smoothed path array
            orientations: Path orientations
            
        Returns:
            velocity_limits: Maximum velocity at each point based on turning rate
        """
        curvature = self.calculate_curvature(path)
        
        # Calculate angular velocity requirements from orientation changes
        angular_velocities = np.zeros(len(orientations))
        for i in range(1, len(orientations)):
            # Simple finite difference for angular velocity
            dtheta = self.angle_difference(orientations[i-1], orientations[i])
            # Use path distance as approximation for time difference
            ds = np.linalg.norm(path[i] - path[i-1]) if i > 0 else 1e-6
            if ds > 1e-6:
                # Estimate time based on a reference velocity
                dt = ds / max(self.max_velocity * 0.5, 0.1)  # Conservative estimate
                angular_velocities[i] = abs(dtheta / dt)
        
        # For a robot following a curved path: ω = v * κ
        # Therefore: v_max = ω_max / κ
        velocity_limits = np.full(len(path), self.max_velocity)
        
        # Apply turning rate constraint from curvature
        valid_curvature = curvature > 1e-6
        velocity_limits[valid_curvature] = np.minimum(
            velocity_limits[valid_curvature],
            self.max_turning_rate / curvature[valid_curvature]
        )
        
        # Apply turning rate constraint from orientation requirements
        valid_angular_vel = angular_velocities > 1e-6
        for i in range(len(velocity_limits)):
            if valid_angular_vel[i]:
                # Limit velocity to ensure we don't exceed turning rate
                max_vel_for_orientation = self.max_turning_rate / max(angular_velocities[i], 1e-6)
                velocity_limits[i] = min(velocity_limits[i], max_vel_for_orientation)
        
        return velocity_limits
    
    def optimize_velocity_profile(self, path, orientations):
        """
        Generate velocity profile considering kinematic constraints and orientations
        
        Args:
            path: Smoothed path array
            orientations: Path orientations
            
        Returns:
            velocity_profile: Velocity at each path point
            time_stamps: Time stamps for each point
        """
        n_points = len(path)
        
        # Calculate path segments
        distances = np.zeros(n_points)
        for i in range(1, n_points):
            distances[i] = distances[i-1] + np.linalg.norm(path[i] - path[i-1])
        
        # Calculate turning rate limited velocities considering orientations
        turning_rate_velocities = self.calculate_turning_rate_limited_velocity(path, orientations)
        
        # Initialize velocity profile with turning rate limits
        velocities = np.minimum(turning_rate_velocities, self.max_velocity)
        
        # Consider initial orientation alignment
        # If robot needs to turn significantly at start, reduce initial velocity
        if len(orientations) > 1:
            initial_turn = abs(self.angle_difference(orientations[0], orientations[1]))
            if initial_turn > 0.1:  # Significant initial turn
                velocities[0] = min(velocities[0], self.max_turning_rate / (initial_turn * 2))
            else:
                velocities[0] = 0  # Start from rest
        else:
            velocities[0] = 0
        
        velocities[-1] = 0  # End at rest
        
        # Forward pass: acceleration constraints
        for i in range(1, n_points-1):
            ds = distances[i] - distances[i-1]
            if ds > 0:
                v_max_accel = np.sqrt(velocities[i-1]**2 + 2 * self.max_acceleration * ds)
                velocities[i] = min(velocities[i], v_max_accel)
        
        # Backward pass: deceleration constraints
        for i in range(n_points-2, 0, -1):
            ds = distances[i+1] - distances[i]
            if ds > 0:
                v_max_decel = np.sqrt(velocities[i+1]**2 + 2 * self.max_acceleration * ds)
                velocities[i] = min(velocities[i], v_max_decel)
        
        # Calculate time stamps
        time_stamps = np.zeros(n_points)
        for i in range(1, n_points):
            ds = distances[i] - distances[i-1]
            if ds > 0:
                if velocities[i] > 1e-6 or velocities[i-1] > 1e-6:
                    avg_velocity = max((velocities[i] + velocities[i-1]) / 2, 1e-3)
                    time_stamps[i] = time_stamps[i-1] + ds / avg_velocity
                else:
                    time_stamps[i] = time_stamps[i-1] + 0.1  # Small time step
        
        return velocities, time_stamps
    
    def calculate_waypoint_arrival_times(self, waypoints, path, time_stamps):
        """
        Calculate the time when each waypoint is reached during trajectory execution
        
        Args:
            waypoints: Original waypoint coordinates
            path: Generated trajectory path points
            time_stamps: Time stamps for each path point
            
        Returns:
            waypoint_arrival_times: List of arrival times for each waypoint
            waypoint_arrival_indices: List of path indices closest to each waypoint
        """
        waypoint_arrival_times = []
        waypoint_arrival_indices = []
        
        for i, waypoint in enumerate(waypoints):
            waypoint_pos = np.array(waypoint)
            
            # Find the closest point on the trajectory path to this waypoint
            distances_to_waypoint = np.linalg.norm(path - waypoint_pos, axis=1)
            closest_idx = np.argmin(distances_to_waypoint)
            
            # Get the arrival time at this waypoint
            arrival_time = time_stamps[closest_idx]
            
            waypoint_arrival_times.append(arrival_time)
            waypoint_arrival_indices.append(closest_idx)
            
        return waypoint_arrival_times, waypoint_arrival_indices
    
    def generate_trajectory(self, waypoints, orientations=None, num_points=100):
        """
        Generate complete optimized trajectory with Dubins path from first to second waypoint
        
        Args:
            waypoints: List of waypoint coordinates (first waypoint is current robot position)
            orientations: List of orientations at waypoints (optional)
            num_points: Number of trajectory points
            
        Returns:
            Dictionary containing trajectory data including Dubins connection and time comparison
        """
        original_waypoints = list(waypoints)  # Keep original for plotting
        dubins_path_info = None
        
        # First, create pure spline trajectory through all waypoints for comparison
        pure_spline_path, pure_spline_orientations = self.smooth_path_with_orientation(
            waypoints, orientations, num_points)
        pure_spline_velocities, pure_spline_times = self.optimize_velocity_profile(
            pure_spline_path, pure_spline_orientations)
        pure_spline_total_time = pure_spline_times[-1]
        
        # Calculate pure spline distance
        pure_spline_distance = 0
        for i in range(1, len(pure_spline_path)):
            pure_spline_distance += np.linalg.norm(pure_spline_path[i] - pure_spline_path[i-1])
        
        # Calculate time for WP1→WP2 segment in pure spline (for proper comparison)
        wp1_to_wp2_spline_time = 0
        if len(waypoints) >= 2:
            # Find the approximate time it takes to go from WP1 to WP2 in the pure spline
            # We need to estimate which portion of the pure spline corresponds to WP1→WP2
            wp1_pos = np.array(waypoints[0])
            wp2_pos = np.array(waypoints[1])
            
            # Find closest points in pure spline path to WP1 and WP2
            distances_to_wp1 = np.linalg.norm(pure_spline_path - wp1_pos, axis=1)
            distances_to_wp2 = np.linalg.norm(pure_spline_path - wp2_pos, axis=1)
            wp1_idx = np.argmin(distances_to_wp1)
            wp2_idx = np.argmin(distances_to_wp2)
            
            # Time for WP1→WP2 segment in pure spline
            if wp2_idx > wp1_idx:
                wp1_to_wp2_spline_time = pure_spline_times[wp2_idx] - pure_spline_times[wp1_idx]
            else:
                # Fallback: estimate based on waypoint ratio
                wp1_to_wp2_spline_time = pure_spline_total_time / (len(waypoints) - 1)
        
        # Generate Dubins path from first to second waypoint if we have at least 2 waypoints
        hybrid_total_time = pure_spline_total_time  # Default to pure spline time
        dubins_time = 0
        remaining_spline_time = 0
        
        if len(waypoints) >= 2 and orientations is not None and len(orientations) >= 2:
            # Use first waypoint orientation from slider
            first_wp_orientation = orientations[0]
            
            # Get the actual tangent direction of the spline at the second waypoint
            # Create spline for position only to get tangent
            waypoints_array = np.array(waypoints)
            tck, u = splprep([waypoints_array[:, 0], waypoints_array[:, 1]], 
                            s=0, k=min(3, len(waypoints)-1))
            
            # Find parameter value at second waypoint
            # Since waypoints are at parameter values 0, 1/(n-1), 2/(n-1), ..., 1
            u_at_wp2 = 1.0 / (len(waypoints) - 1) if len(waypoints) > 1 else 0.0
            
            # Get first derivative (tangent) at second waypoint
            dx_du, dy_du = splev(u_at_wp2, tck, der=1)
            
            # Calculate tangent angle at second waypoint
            second_wp_orientation = np.arctan2(dy_du, dx_du)
            
            # Generate Dubins path from WP1 to WP2
            dubins_path_points, dubins_length, dubins_type, turn_radius_used = self.get_dubins_path(
                start_pos=waypoints[0],
                start_angle=first_wp_orientation,
                end_pos=waypoints[1],
                end_angle=second_wp_orientation,
                velocity_for_dubins=None,
                step_size=0.1
            )
            
            # Calculate time for Dubins segment
            # Use conservative velocity for Dubins path
            dubins_velocity = self.max_velocity * 0.7
            dubins_time = dubins_length / dubins_velocity if dubins_length > 0 else 0
            
            # Calculate remaining spline path from WP2 to end
            if len(waypoints) > 2:
                remaining_waypoints = waypoints[1:]  # Start from WP2
                remaining_orientations = orientations[1:] if orientations else None
                
                # Adjust first orientation to match Dubins end orientation
                if remaining_orientations:
                    remaining_orientations = list(remaining_orientations)
                    remaining_orientations[0] = second_wp_orientation
                
                remaining_spline_path, remaining_spline_orientations = self.smooth_path_with_orientation(
                    remaining_waypoints, remaining_orientations, num_points)
                remaining_spline_velocities, remaining_spline_times = self.optimize_velocity_profile(
                    remaining_spline_path, remaining_spline_orientations)
                remaining_spline_time = remaining_spline_times[-1]
            else:
                remaining_spline_time = 0
            
            # CORRECT hybrid time calculation: 
            # Pure spline time - WP1→WP2 spline time + Dubins time + remaining spline time
            # This accounts for replacing the WP1→WP2 spline segment with Dubins path
            hybrid_total_time = pure_spline_total_time - wp1_to_wp2_spline_time + dubins_time
            
            dubins_path_info = {
                'path_points': dubins_path_points,
                'path_length': dubins_length,
                'path_type': dubins_type,
                'turn_radius_used': turn_radius_used,
                'start_pose': (waypoints[0][0], waypoints[0][1], first_wp_orientation),
                'end_pose': (waypoints[1][0], waypoints[1][1], second_wp_orientation),
                'dubins_time': dubins_time,
                'wp1_to_wp2_spline_time': wp1_to_wp2_spline_time,
                'remaining_spline_time': remaining_spline_time,
                'hybrid_total_time': hybrid_total_time,
                'pure_spline_time': pure_spline_total_time,
                'time_difference': hybrid_total_time - pure_spline_total_time,
                'time_savings_percent': ((pure_spline_total_time - hybrid_total_time) / pure_spline_total_time * 100) if pure_spline_total_time > 0 else 0
            }
        
        # Use hybrid path as the main trajectory if Dubins connection exists
        if dubins_path_info is not None:
            # Combine Dubins path with remaining spline path
            dubins_points = dubins_path_info['path_points']
            
            if len(waypoints) > 2:
                # Get remaining spline path from WP2 to end
                remaining_waypoints = waypoints[1:]  # Start from WP2
                remaining_orientations = orientations[1:] if orientations else None
                
                # Adjust first orientation to match Dubins end orientation
                if remaining_orientations:
                    remaining_orientations = list(remaining_orientations)
                    remaining_orientations[0] = dubins_path_info['end_pose'][2]  # Use Dubins end orientation
                
                remaining_spline_path, remaining_spline_orientations = self.smooth_path_with_orientation(
                    remaining_waypoints, remaining_orientations, num_points)
                
                # Combine paths: Dubins + remaining spline
                smooth_path = np.vstack([dubins_points, remaining_spline_path[1:]])  # Skip first point to avoid duplication
                
                # Combine orientations: Dubins orientations + remaining spline orientations
                dubins_orientations = []
                for i in range(len(dubins_points)-1):
                    dx = dubins_points[i+1][0] - dubins_points[i][0]
                    dy = dubins_points[i+1][1] - dubins_points[i][1]
                    if np.sqrt(dx**2 + dy**2) > 1e-6:
                        dubins_orientations.append(np.arctan2(dy, dx))
                    else:
                        dubins_orientations.append(dubins_orientations[-1] if dubins_orientations else 0)
                # Add final Dubins orientation
                dubins_orientations.append(dubins_path_info['end_pose'][2])
                
                smooth_orientations = np.concatenate([dubins_orientations, remaining_spline_orientations[1:]])
            else:
                # Only two waypoints, use just Dubins path
                smooth_path = dubins_points
                
                # Calculate orientations along Dubins path
                smooth_orientations = []
                for i in range(len(dubins_points)-1):
                    dx = dubins_points[i+1][0] - dubins_points[i][0]
                    dy = dubins_points[i+1][1] - dubins_points[i][1]
                    if np.sqrt(dx**2 + dy**2) > 1e-6:
                        smooth_orientations.append(np.arctan2(dy, dx))
                    else:
                        smooth_orientations.append(smooth_orientations[-1] if smooth_orientations else 0)
                # Add final orientation
                smooth_orientations.append(dubins_path_info['end_pose'][2])
                smooth_orientations = np.array(smooth_orientations)
        else:
            # No Dubins connection, use pure spline path
            smooth_path, smooth_orientations = pure_spline_path, pure_spline_orientations
        
        # Optimize velocity profile for the hybrid path
        velocities, time_stamps = self.optimize_velocity_profile(smooth_path, smooth_orientations)
        
        # Update num_points to match actual path length
        actual_num_points = len(smooth_path)
        
        # Calculate angular velocities (turning rates) from orientation profile
        angular_velocities = np.zeros(actual_num_points)
        for i in range(1, actual_num_points):
            dt = time_stamps[i] - time_stamps[i-1]
            if dt > 1e-6:
                dtheta = self.angle_difference(smooth_orientations[i-1], smooth_orientations[i])
                angular_velocities[i] = dtheta / dt
        
        # Calculate actual curvature and turning rates for analysis
        curvature = self.calculate_curvature(smooth_path)
        actual_turning_rates = velocities * curvature
        
        # Calculate orientation errors (difference between path direction and desired orientation)
        path_headings = np.zeros(actual_num_points)
        for i in range(1, actual_num_points):
            dx = smooth_path[i, 0] - smooth_path[i-1, 0]
            dy = smooth_path[i, 1] - smooth_path[i-1, 1]
            if np.sqrt(dx**2 + dy**2) > 1e-6:
                path_headings[i] = np.arctan2(dy, dx)
            else:
                path_headings[i] = path_headings[i-1]
        path_headings[0] = path_headings[1] if len(path_headings) > 1 else 0
        
        orientation_errors = np.array([abs(self.angle_difference(path_headings[i], smooth_orientations[i])) 
                                     for i in range(actual_num_points)])
        
        # Calculate total distance of the actual hybrid path
        path_diffs = np.diff(smooth_path, axis=0)
        total_distance = np.sum(np.linalg.norm(path_diffs, axis=1))
        
        # Calculate waypoint arrival times and indices
        waypoint_arrival_times, waypoint_arrival_indices = self.calculate_waypoint_arrival_times(
            original_waypoints, smooth_path, time_stamps)
        
        # Calculate time intervals between waypoints
        waypoint_time_intervals = []
        for i in range(1, len(waypoint_arrival_times)):
            interval = waypoint_arrival_times[i] - waypoint_arrival_times[i-1]
            waypoint_time_intervals.append(interval)
        
        trajectory_data = {
            'path': smooth_path,
            'velocities': velocities,
            'time_stamps': time_stamps,
            'orientations': smooth_orientations,
            'path_headings': path_headings,
            'angular_velocities': angular_velocities,
            'curvature': curvature,
            'actual_turning_rates': actual_turning_rates,
            'orientation_errors': orientation_errors,
            'total_time': time_stamps[-1],
            'total_distance': total_distance,
            'original_waypoints': original_waypoints,
            'dubins_connection': dubins_path_info,
            'pure_spline_distance': pure_spline_distance,
            'is_hybrid_path': dubins_path_info is not None,
            'pure_spline_path': pure_spline_path,
            'pure_spline_orientations': pure_spline_orientations,
            # Waypoint arrival information
            'waypoint_arrival_times': waypoint_arrival_times,
            'waypoint_arrival_indices': waypoint_arrival_indices,
            'waypoint_time_intervals': waypoint_time_intervals
        }
        
        return trajectory_data
    
    def plot_trajectory(self, waypoints, trajectory=None, orientations=None, show_constraints=True):
        """
        Plot the trajectory and waypoints with orientation visualization
        """
        if trajectory is None:
            trajectory = self.generate_trajectory(waypoints, orientations)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Path plot with orientations
        ax1 = axes[0, 0]
        waypoints_array = np.array(trajectory.get('original_waypoints', waypoints))
        
        # Plot the main trajectory path
        ax1.plot(trajectory['path'][:, 0], trajectory['path'][:, 1], 'b-', 
                label='Robot Trajectory' + (' (Hybrid)' if trajectory.get('is_hybrid_path', False) else ' (Spline)'), 
                linewidth=3)
        
        # Plot Dubins segment if available
        dubins_info = trajectory.get('dubins_connection')
        if dubins_info is not None:
            dubins_path = dubins_info['path_points']
            # Highlight the Dubins segment within the main path
            ax1.plot(dubins_path[:, 0], dubins_path[:, 1], 'c-', 
                    label=f'Dubins Segment ({dubins_info["path_type"]})', 
                    linewidth=4, alpha=0.8)
            
            # Plot the COMPLETE pure spline path (created at the beginning before Dubins)
            original_waypoints = trajectory.get('original_waypoints', waypoints)
            pure_spline_path = trajectory.get('pure_spline_path')
            if pure_spline_path is not None:
                # Only plot the WP1→WP2 segment for detailed comparison
                if len(original_waypoints) >= 2:
                    # Find the portion of the full spline that corresponds to WP1→WP2
                    wp1_pos = np.array(original_waypoints[0])
                    wp2_pos = np.array(original_waypoints[1])
                    
                    # Find closest points in the full spline path to WP1 and WP2
                    distances_to_wp1 = np.linalg.norm(pure_spline_path - wp1_pos, axis=1)
                    distances_to_wp2 = np.linalg.norm(pure_spline_path - wp2_pos, axis=1)
                    wp1_idx = np.argmin(distances_to_wp1)
                    wp2_idx = np.argmin(distances_to_wp2)
                    
                    # Extract the WP1→WP2 segment from the full spline
                    if wp2_idx > wp1_idx:
                        pure_wp1_wp2_path = pure_spline_path[wp1_idx:wp2_idx+1]
                    else:
                        # Fallback: create linear path
                        pure_wp1_wp2_path = np.array([original_waypoints[0], original_waypoints[1]])
                    
                    ax1.plot(pure_wp1_wp2_path[:, 0], pure_wp1_wp2_path[:, 1], 'r--', 
                            label='Pure Spline (WP1→WP2)', linewidth=4, alpha=1.0)            # Add text with Dubins info
            ax1.text(0.02, 0.98, f'Hybrid Path\nDubins: {dubins_info["path_type"]}\nLength: {dubins_info["path_length"]:.2f}m\nTurn Radius: {dubins_info["turn_radius_used"]:.2f}m\n(WP1 → WP2)', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax1.plot(waypoints_array[:, 0], waypoints_array[:, 1], 'ro-', label='Waypoints', markersize=10, linewidth=2)
        
        # Mark first waypoint (current position) differently
        ax1.plot(waypoints_array[0, 0], waypoints_array[0, 1], 'gs', markersize=12, label='Current Position')
        
        # Add waypoint arrival time annotations
        waypoint_arrivals = trajectory.get('waypoint_arrival_times', [])
        if waypoint_arrivals and len(waypoint_arrivals) > 0:
            for i, (waypoint, arrival_time) in enumerate(zip(waypoints_array, waypoint_arrivals)):
                # Offset text position slightly to avoid overlapping with waypoint marker
                offset_x = 0.2
                offset_y = 0.2
                ax1.annotate(f'WP{i+1}\n{arrival_time:.1f}s', 
                           xy=(waypoint[0], waypoint[1]),
                           xytext=(waypoint[0] + offset_x, waypoint[1] + offset_y),
                           fontsize=9, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', lw=1))
        
        # Add velocity color coding to path
        scatter = ax1.scatter(trajectory['path'][:, 0], trajectory['path'][:, 1], 
                             c=trajectory['velocities'], cmap='viridis', s=20, alpha=0.7)
        plt.colorbar(scatter, ax=ax1, label='Velocity (m/s)')
        
        # Draw orientation arrows along path
        skip = max(1, len(trajectory['path']) // 20)  # Show ~20 arrows
        for i in range(0, len(trajectory['path']), skip):
            x, y = trajectory['path'][i]
            theta = trajectory['orientations'][i]
            arrow_length = 0.3
            dx = arrow_length * np.cos(theta)
            dy = arrow_length * np.sin(theta)
            ax1.arrow(x, y, dx, dy, head_width=0.1, head_length=0.05, 
                     fc='red', ec='red', alpha=0.6)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Robot Trajectory' + (' (Hybrid: Dubins + Spline)' if trajectory.get('is_hybrid_path', False) else ' (Pure Spline)'))
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # Velocity profile
        ax2 = axes[0, 1]
        ax2.plot(trajectory['time_stamps'], trajectory['velocities'], 'g-', linewidth=3, label='Actual Velocity')
        if show_constraints:
            ax2.axhline(y=self.max_velocity, color='r', linestyle='--', linewidth=2, 
                       label=f'Max Velocity ({self.max_velocity} m/s)')
        
        # Add waypoint arrival markers on velocity plot
        waypoint_arrivals = trajectory.get('waypoint_arrival_times', [])
        waypoint_indices = trajectory.get('waypoint_arrival_indices', [])
        if waypoint_arrivals and waypoint_indices:
            for i, (arrival_time, arrival_idx) in enumerate(zip(waypoint_arrivals, waypoint_indices)):
                if arrival_idx < len(trajectory['velocities']):
                    velocity_at_wp = trajectory['velocities'][arrival_idx]
                    ax2.axvline(x=arrival_time, color='red', linestyle='--', alpha=0.5)
                    ax2.plot(arrival_time, velocity_at_wp, 'ro', markersize=6)
                    ax2.annotate(f'WP{i+1}', xy=(arrival_time, velocity_at_wp), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Velocity Profile & Waypoint Arrivals')
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim(0, self.max_velocity * 1.1)
        
        # Orientation profile
        ax3 = axes[0, 2]
        ax3.plot(trajectory['time_stamps'], np.degrees(trajectory['orientations']), 'purple', 
                linewidth=2, label='Desired Orientation')
        ax3.plot(trajectory['time_stamps'], np.degrees(trajectory['path_headings']), 'orange', 
                linewidth=2, label='Path Direction', alpha=0.7)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Orientation (degrees)')
        ax3.set_title('Orientation Profile')
        ax3.legend()
        ax3.grid(True)
        
        # Turning rate comparison
        ax4 = axes[1, 0]
        ax4.plot(trajectory['time_stamps'], np.abs(trajectory['angular_velocities']), 'b-', 
                linewidth=2, label='Angular Velocity')
        ax4.plot(trajectory['time_stamps'], trajectory['actual_turning_rates'], 'm-', 
                linewidth=2, label='v×κ (Curvature Rate)')
        if show_constraints:
            ax4.axhline(y=self.max_turning_rate, color='r', linestyle='--', linewidth=2, 
                       label=f'Max Turning Rate ({self.max_turning_rate} rad/s)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Turning Rate (rad/s)')
        ax4.set_title('Turning Rate Analysis')
        ax4.legend()
        ax4.grid(True)
        ax4.set_ylim(0, max(self.max_turning_rate * 1.2, 
                           np.max(np.abs(trajectory['angular_velocities'])) * 1.1))
        
        # Orientation error
        ax5 = axes[1, 1]
        ax5.plot(trajectory['time_stamps'], np.degrees(trajectory['orientation_errors']), 'red', 
                linewidth=2, label='Orientation Error')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Orientation Error (degrees)')
        ax5.set_title('Orientation Tracking Error')
        ax5.legend()
        ax5.grid(True)
        
        # Curvature with waypoint arrivals
        ax6 = axes[1, 2]
        ax6.plot(trajectory['time_stamps'], trajectory['curvature'], 'c-', linewidth=2, label='Path Curvature')
        
        # Add waypoint arrival markers on the curvature plot
        waypoint_arrivals = trajectory.get('waypoint_arrival_times', [])
        waypoint_indices = trajectory.get('waypoint_arrival_indices', [])
        if waypoint_arrivals and waypoint_indices:
            for i, (arrival_time, arrival_idx) in enumerate(zip(waypoint_arrivals, waypoint_indices)):
                if arrival_idx < len(trajectory['curvature']):
                    curvature_at_wp = trajectory['curvature'][arrival_idx]
                    ax6.axvline(x=arrival_time, color='red', linestyle='--', alpha=0.7)
                    ax6.plot(arrival_time, curvature_at_wp, 'ro', markersize=8, 
                           label=f'WP{i+1}' if i < 3 else None)  # Only label first few to avoid clutter
                    ax6.annotate(f'WP{i+1}', xy=(arrival_time, curvature_at_wp), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Curvature (1/m)')
        ax6.set_title('Path Curvature & Waypoint Arrivals')
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_waypoint_timeline(self, trajectory):
        """
        Create a timeline visualization showing waypoint arrivals and segment times
        
        Args:
            trajectory: Trajectory data dictionary
            
        Returns:
            matplotlib figure
        """
        waypoint_arrivals = trajectory.get('waypoint_arrival_times', [])
        waypoint_intervals = trajectory.get('waypoint_time_intervals', [])
        original_waypoints = trajectory.get('original_waypoints', [])
        
        if not waypoint_arrivals:
            print("No waypoint arrival data available for timeline plot.")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Timeline plot
        ax1.set_xlim(0, max(waypoint_arrivals) * 1.1)
        ax1.set_ylim(-0.5, 0.5)
        
        # Draw timeline
        ax1.axhline(y=0, color='black', linewidth=2)
        
        # Plot waypoint markers
        for i, arrival_time in enumerate(waypoint_arrivals):
            waypoint_coord = original_waypoints[i] if i < len(original_waypoints) else "Unknown"
            
            # Waypoint marker
            ax1.plot(arrival_time, 0, 'ro', markersize=12)
            ax1.annotate(f'WP{i+1}\n({waypoint_coord[0]:.1f}, {waypoint_coord[1]:.1f})\n{arrival_time:.1f}s', 
                        xy=(arrival_time, 0), xytext=(arrival_time, 0.3 if i % 2 == 0 else -0.3),
                        ha='center', va='bottom' if i % 2 == 0 else 'top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', lw=1))
            
            # Segment time annotations
            if i > 0 and i-1 < len(waypoint_intervals):
                segment_time = waypoint_intervals[i-1]
                mid_time = (waypoint_arrivals[i-1] + waypoint_arrivals[i]) / 2
                ax1.annotate(f'{segment_time:.1f}s', xy=(mid_time, 0), xytext=(mid_time, 0.1),
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_title('Waypoint Arrival Timeline')
        ax1.set_yticks([])
        ax1.grid(True, alpha=0.3)
        
        # Bar chart of segment times
        if waypoint_intervals:
            segment_labels = [f'WP{i+1}→WP{i+2}' for i in range(len(waypoint_intervals))]
            bars = ax2.bar(range(len(waypoint_intervals)), waypoint_intervals, 
                          color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral'][:len(waypoint_intervals)])
            
            # Add value labels on bars
            for i, (bar, interval) in enumerate(zip(bars, waypoint_intervals)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{interval:.1f}s', ha='center', va='bottom', fontweight='bold')
            
            ax2.set_xlabel('Trajectory Segments')
            ax2.set_ylabel('Travel Time (seconds)')
            ax2.set_title('Segment Travel Times')
            ax2.set_xticks(range(len(segment_labels)))
            ax2.set_xticklabels(segment_labels)
            ax2.grid(True, alpha=0.3)
            
            # Highlight longest and shortest segments
            if len(waypoint_intervals) > 1:
                max_idx = np.argmax(waypoint_intervals)
                min_idx = np.argmin(waypoint_intervals)
                bars[max_idx].set_color('red')
                bars[min_idx].set_color('green')
                
                ax2.text(0.98, 0.95, f'Longest: {segment_labels[max_idx]} ({waypoint_intervals[max_idx]:.1f}s)\nShortest: {segment_labels[min_idx]} ({waypoint_intervals[min_idx]:.1f}s)',
                        transform=ax2.transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def print_trajectory_stats(self, trajectory):
        """Print detailed trajectory statistics"""
        print("="*70)
        print("TRAJECTORY OPTIMIZATION RESULTS")
        print("="*70)
        print(f"Robot Constraints:")
        print(f"  Max Velocity:     {self.max_velocity:.2f} m/s")
        print(f"  Max Acceleration: {self.max_acceleration:.2f} m/s²")
        print(f"  Max Turning Rate: {self.max_turning_rate:.2f} rad/s ({np.degrees(self.max_turning_rate):.1f} deg/s)")
        
        # Show calculated turn radius
        calculated_turn_radius = self.get_dynamic_turn_radius()
        print(f"  Calculated Turn Radius: {calculated_turn_radius:.2f} m (v={self.max_velocity*0.7:.1f}m/s ÷ ω={self.max_turning_rate:.2f}rad/s)")
        print()
        
        # Dubins connection info
        dubins_info = trajectory.get('dubins_connection')
        if dubins_info is not None:
            print(f"Dubins Connection (WP1 → WP2):")
            print(f"  Path Type:        {dubins_info['path_type']}")
            print(f"  Path Length:      {dubins_info['path_length']:.2f} m")
            print(f"  Turn Radius Used: {dubins_info['turn_radius_used']:.2f} m")
            print(f"  Start (WP1):      ({dubins_info['start_pose'][0]:.2f}, {dubins_info['start_pose'][1]:.2f}, {np.degrees(dubins_info['start_pose'][2]):.1f}°)")
            print(f"  End (WP2):        ({dubins_info['end_pose'][0]:.2f}, {dubins_info['end_pose'][1]:.2f}, {np.degrees(dubins_info['end_pose'][2]):.1f}°)")
            print()
            
            print(f"⏱️  TIME COMPARISON:")
            print(f"  Pure Spline Time:         {dubins_info['pure_spline_time']:.2f} seconds")
            print(f"  Hybrid Path Time:")
            print(f"    - Pure spline (WP1→WP2): {dubins_info['wp1_to_wp2_spline_time']:.2f} seconds")
            print(f"    - Dubins (WP1→WP2):      {dubins_info['dubins_time']:.2f} seconds")
            print(f"    - Net change (WP1→WP2):  {dubins_info['dubins_time'] - dubins_info['wp1_to_wp2_spline_time']:.2f} seconds")
            print(f"    - Total Hybrid:          {dubins_info['hybrid_total_time']:.2f} seconds")
            print(f"  Time Difference:          {dubins_info['time_difference']:.2f} seconds")
            if dubins_info['time_difference'] < 0:
                print(f"  ✅ Hybrid is FASTER by {abs(dubins_info['time_difference']):.2f}s ({abs(dubins_info['time_savings_percent']):.1f}%)")
            elif dubins_info['time_difference'] > 0:
                print(f"  ❌ Hybrid is SLOWER by {dubins_info['time_difference']:.2f}s ({dubins_info['time_savings_percent']:.1f}%)")
            else:
                print(f"  ⚖️ Both paths take the same time")
            print()
        
        print(f"Trajectory Statistics:")
        if trajectory.get('is_hybrid_path', False):
            print(f"  Path Type:        Hybrid (Dubins + Spline)")
        else:
            print(f"  Path Type:        Pure Spline")
        print(f"  Total Time:       {trajectory['total_time']:.2f} seconds")
        print(f"  Total Distance:   {trajectory['total_distance']:.2f} meters")
        print(f"  Average Speed:    {trajectory['total_distance']/trajectory['total_time']:.2f} m/s")
        
        print(f"  Peak Velocity:    {np.max(trajectory['velocities']):.2f} m/s")
        print(f"  Peak Turning Rate:{np.max(np.abs(trajectory['angular_velocities'])):.2f} rad/s ({np.degrees(np.max(np.abs(trajectory['angular_velocities']))):.1f} deg/s)")
        print(f"  Peak v×κ Rate:    {np.max(trajectory['actual_turning_rates']):.2f} rad/s")
        print(f"  Max Curvature:    {np.max(trajectory['curvature']):.4f} 1/m")
        
        # Orientation statistics
        max_orientation_error = np.max(trajectory['orientation_errors'])
        avg_orientation_error = np.mean(trajectory['orientation_errors'])
        print(f"  Max Orient. Error:{np.degrees(max_orientation_error):.2f} degrees")
        print(f"  Avg Orient. Error:{np.degrees(avg_orientation_error):.2f} degrees")
        
        # Check constraint violations
        vel_violations = np.sum(trajectory['velocities'] > self.max_velocity + 1e-6)
        turn_violations = np.sum(np.abs(trajectory['angular_velocities']) > self.max_turning_rate + 1e-6)
        actual_turn_violations = np.sum(trajectory['actual_turning_rates'] > self.max_turning_rate + 1e-6)
        
        print()
        print(f"Constraint Violations:")
        print(f"  Velocity:         {vel_violations} points")
        print(f"  Angular Velocity: {turn_violations} points")
        print(f"  v×κ Turning Rate: {actual_turn_violations} points")
        
        # Waypoint arrival information
        waypoint_arrivals = trajectory.get('waypoint_arrival_times', [])
        waypoint_intervals = trajectory.get('waypoint_time_intervals', [])
        original_waypoints = trajectory.get('original_waypoints', [])
        
        if waypoint_arrivals and len(waypoint_arrivals) > 0:
            print()
            print(f"WAYPOINT ARRIVAL SCHEDULE:")
            for i, arrival_time in enumerate(waypoint_arrivals):
                waypoint_coord = original_waypoints[i] if i < len(original_waypoints) else "Unknown"
                if i == 0:
                    print(f"  WP{i+1} at ({waypoint_coord[0]:.2f}, {waypoint_coord[1]:.2f}): {arrival_time:.2f}s (START)")
                else:
                    interval = waypoint_intervals[i-1] if i-1 < len(waypoint_intervals) else 0
                    print(f"  WP{i+1} at ({waypoint_coord[0]:.2f}, {waypoint_coord[1]:.2f}): {arrival_time:.2f}s (+{interval:.2f}s from WP{i})")
            
            # Summary of segment times
            if len(waypoint_intervals) > 0:
                print(f"\n  Segment Travel Times:")
                total_segment_time = 0
                for i, interval in enumerate(waypoint_intervals):
                    print(f"    WP{i+1} → WP{i+2}: {interval:.2f}s")
                    total_segment_time += interval
                print(f"    Total: {total_segment_time:.2f}s")
                
                # Show which segment takes longest/shortest
                if len(waypoint_intervals) > 1:
                    max_interval_idx = np.argmax(waypoint_intervals)
                    min_interval_idx = np.argmin(waypoint_intervals)
                    print(f"    Longest segment: WP{max_interval_idx+1} → WP{max_interval_idx+2} ({waypoint_intervals[max_interval_idx]:.2f}s)")
                    print(f"    Shortest segment: WP{min_interval_idx+1} → WP{min_interval_idx+2} ({waypoint_intervals[min_interval_idx]:.2f}s)")
        
        print("="*70)

    def plot_interactive_trajectory(self, waypoints, initial_orientations=None):
        """
        Create an interactive plot with sliders to adjust the first and second waypoint orientations
        """
        if not WIDGETS_AVAILABLE:
            print("❌ Interactive widgets not available. Using static plot instead.")
            return self.plot_trajectory(waypoints, orientations=initial_orientations)
        
        # Initialize orientations if not provided
        if initial_orientations is None:
            initial_orientations = [0.0] * len(waypoints)
        else:
            initial_orientations = list(initial_orientations)
        
        # Ensure we have the right number of orientations
        while len(initial_orientations) < len(waypoints):
            initial_orientations.append(0.0)
        
        # Calculate default second waypoint orientation from spline tangent (with pseudo waypoint)
        default_second_orientation = 0.0
        if len(waypoints) >= 2:
            # Temporarily add pseudo waypoint for better orientation calculation
            waypoints_with_pseudo = list(waypoints)
            wp1 = np.array(waypoints[0])
            wp2 = np.array(waypoints[1])
            direction = wp2 - wp1
            direction_normalized = direction / np.linalg.norm(direction)
            pseudo_distance = 0.5
            pseudo_waypoint = wp2 + direction_normalized * pseudo_distance
            waypoints_with_pseudo.insert(2, pseudo_waypoint.tolist())
            
            waypoints_array = np.array(waypoints_with_pseudo)
            tck, u = splprep([waypoints_array[:, 0], waypoints_array[:, 1]], 
                            s=0, k=min(3, len(waypoints_with_pseudo)-1))
            # Find u parameter corresponding to original WP2
            wp2_pos = np.array(waypoints[1])
            u_samples = np.linspace(0, 1, 100)
            spline_points = np.array(splev(u_samples, tck)).T
            distances = np.linalg.norm(spline_points - wp2_pos, axis=1)
            u_at_wp2 = u_samples[np.argmin(distances)]
            
            dx_du, dy_du = splev(u_at_wp2, tck, der=1)
            default_second_orientation = np.arctan2(dy_du, dx_du)
        
        # Create the figure and subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Main trajectory plot (larger)
        ax_main = plt.subplot2grid((2, 4), (0, 0), colspan=3, rowspan=2)
        
        # Smaller plots
        ax_vel = plt.subplot2grid((2, 4), (0, 3))
        ax_orient = plt.subplot2grid((2, 4), (1, 3))
        
        # Create sliders for first and second waypoint orientations
        ax_slider1 = plt.axes([0.15, 0.08, 0.4, 0.03])
        slider1 = Slider(ax_slider1, 'WP1 Orientation', -180, 180, 
                        valinit=np.degrees(initial_orientations[0]), 
                        valfmt='%0.0f°')
        
        ax_slider2 = plt.axes([0.15, 0.02, 0.4, 0.03])
        slider2 = Slider(ax_slider2, 'WP2 Orientation', -180, 180, 
                        valinit=np.degrees(default_second_orientation), 
                        valfmt='%0.0f°')
        
        # Function to update the plot
        def update_plot(val=None):
            # Clear axes
            ax_main.clear()
            ax_vel.clear()
            ax_orient.clear()
            
            # Get current orientations for first and second waypoints
            current_orientations = list(initial_orientations)
            current_orientations[0] = np.radians(slider1.val)  # First waypoint orientation
            
            # Second waypoint orientation (only update if we have at least 2 waypoints)
            if len(waypoints) >= 2:
                current_orientations[1] = np.radians(slider2.val)  # Second waypoint orientation
            
            # Generate trajectory with current orientations
            trajectory = self.generate_trajectory(waypoints, current_orientations)
            
            # Plot main trajectory
            waypoints_array = np.array(waypoints)
            
            # Plot main trajectory (hybrid if available)
            ax_main.plot(trajectory['path'][:, 0], trajectory['path'][:, 1], 'b-', 
                        label='Robot Trajectory' + (' (Hybrid)' if trajectory.get('is_hybrid_path', False) else ' (Spline)'), 
                        linewidth=3)
            
            # Plot Dubins segment if available
            dubins_info = trajectory.get('dubins_connection')
            if dubins_info is not None:
                dubins_path = dubins_info['path_points']
                ax_main.plot(dubins_path[:, 0], dubins_path[:, 1], 'c-', 
                            label=f'Dubins Segment ({dubins_info["path_type"]})', linewidth=4, alpha=0.8)
                
                # Plot the WP1→WP2 segment for comparison
                if len(waypoints) >= 2:
                    # Use the STORED pure spline path from trajectory generation
                    pure_spline_path = trajectory.get('pure_spline_path')
                    if pure_spline_path is not None:
                        # Find the portion of the full spline that corresponds to WP1→WP2
                        wp1_pos = np.array(waypoints[0])
                        wp2_pos = np.array(waypoints[1])
                        
                        # Find closest points in the full spline path to WP1 and WP2
                        distances_to_wp1 = np.linalg.norm(pure_spline_path - wp1_pos, axis=1)
                        distances_to_wp2 = np.linalg.norm(pure_spline_path - wp2_pos, axis=1)
                        wp1_idx = np.argmin(distances_to_wp1)
                        wp2_idx = np.argmin(distances_to_wp2)
                        
                        # Extract the WP1→WP2 segment from the full spline
                        if wp2_idx > wp1_idx:
                            pure_wp1_wp2_path = pure_spline_path[wp1_idx:wp2_idx+1]
                        else:
                            # Fallback: create linear path
                            pure_wp1_wp2_path = np.array([waypoints[0], waypoints[1]])
                        
                        ax_main.plot(pure_wp1_wp2_path[:, 0], pure_wp1_wp2_path[:, 1], 'r--', 
                                    label='Pure Spline (WP1→WP2)', linewidth=4, alpha=1.0)
                
                # Draw orientation arrows along Dubins path
                skip_dubins = max(1, len(dubins_path) // 8)
                for i in range(0, len(dubins_path), skip_dubins):
                    if i < len(dubins_path) - 1:
                        x, y = dubins_path[i]
                        # Calculate orientation from path direction
                        dx_path = dubins_path[i+1][0] - dubins_path[i][0]
                        dy_path = dubins_path[i+1][1] - dubins_path[i][1]
                        if np.sqrt(dx_path**2 + dy_path**2) > 1e-6:
                            theta = np.arctan2(dy_path, dx_path)
                            arrow_length = 0.25
                            dx = arrow_length * np.cos(theta)
                            dy = arrow_length * np.sin(theta)
                            ax_main.arrow(x, y, dx, dy, head_width=0.08, head_length=0.04, 
                                         fc='cyan', ec='cyan', alpha=0.8)
            
            # Mark first waypoint (current position) differently
            ax_main.plot(waypoints_array[0, 0], waypoints_array[0, 1], 'gs', markersize=15, label='Current Position')
            
            # Draw current orientation arrow at first waypoint
            arrow_length = 0.8
            dx = arrow_length * np.cos(current_orientations[0])
            dy = arrow_length * np.sin(current_orientations[0])
            ax_main.arrow(waypoints_array[0, 0], waypoints_array[0, 1], dx, dy, 
                         head_width=0.2, head_length=0.15, fc='green', ec='green', linewidth=3)
            
            # Draw orientation arrow at second waypoint if we have at least 2 waypoints
            if len(waypoints) >= 2:
                ax_main.plot(waypoints_array[1, 0], waypoints_array[1, 1], 'bs', markersize=12, label='WP2 (Dubins End)')
                arrow_length2 = 0.6
                dx2 = arrow_length2 * np.cos(current_orientations[1])
                dy2 = arrow_length2 * np.sin(current_orientations[1])
                ax_main.arrow(waypoints_array[1, 0], waypoints_array[1, 1], dx2, dy2, 
                             head_width=0.15, head_length=0.1, fc='blue', ec='blue', linewidth=2)
            
            ax_main.plot(waypoints_array[:, 0], waypoints_array[:, 1], 'ro-', 
                        label='Waypoints', markersize=8, linewidth=2)
            
            # Add velocity color coding
            scatter = ax_main.scatter(trajectory['path'][:, 0], trajectory['path'][:, 1], 
                                     c=trajectory['velocities'], cmap='viridis', s=15, alpha=0.7)
            
            # Draw orientation arrows along path
            skip = max(1, len(trajectory['path']) // 15)
            for i in range(0, len(trajectory['path']), skip):
                x, y = trajectory['path'][i]
                theta = trajectory['orientations'][i]
                arrow_length = 0.3
                dx = arrow_length * np.cos(theta)
                dy = arrow_length * np.sin(theta)
                ax_main.arrow(x, y, dx, dy, head_width=0.1, head_length=0.05, 
                             fc='red', ec='red', alpha=0.6)
            
            # Add current orientation text with time comparison
            dubins_text = ""
            dubins_info = trajectory.get('dubins_connection')
            if dubins_info is not None:
                time_diff = dubins_info['time_difference']
                if time_diff < 0:
                    time_status = f"Hybrid FASTER by {abs(time_diff):.1f}s"
                elif time_diff > 0:
                    time_status = f"Hybrid SLOWER by {time_diff:.1f}s"
                else:
                    time_status = "Same time"
                
                dubins_text = f'\nDubins: {dubins_info["path_type"]}, R={dubins_info["turn_radius_used"]:.1f}m\nPure Spline: {dubins_info["pure_spline_time"]:.1f}s vs Hybrid: {dubins_info["hybrid_total_time"]:.1f}s\n{time_status}'
            
            # Display orientations for both waypoints
            orientation_text = f'WP1 Orientation: {slider1.val:.0f}°'
            if len(waypoints) >= 2:
                orientation_text += f'\nWP2 Orientation: {slider2.val:.0f}°'
                # Show default tangent orientation for comparison
                default_tangent_deg = np.degrees(default_second_orientation)
                orientation_text += f' (tangent: {default_tangent_deg:.0f}°)'
            
            # Add waypoint arrival summary
            waypoint_arrivals = trajectory.get('waypoint_arrival_times', [])
            if waypoint_arrivals:
                orientation_text += f'\n\nWaypoint Arrivals:'
                for i, arrival_time in enumerate(waypoint_arrivals[:4]):  # Show first 4 to avoid clutter
                    orientation_text += f'\n  WP{i+1}: {arrival_time:.1f}s'
                if len(waypoint_arrivals) > 4:
                    orientation_text += f'\n  ... +{len(waypoint_arrivals)-4} more'
            
            ax_main.text(0.02, 0.98, f'{orientation_text}{dubins_text}', 
                        transform=ax_main.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            ax_main.set_xlabel('X (m)')
            ax_main.set_ylabel('Y (m)')
            ax_main.set_title('Interactive Robot Trajectory' + (' (Hybrid: Dubins + Spline)' if trajectory.get('is_hybrid_path', False) else ' (Pure Spline)'))
            ax_main.legend(loc='upper left')
            ax_main.grid(True)
            ax_main.axis('equal')
            
            # Velocity profile
            ax_vel.plot(trajectory['time_stamps'], trajectory['velocities'], 'g-', linewidth=2)
            ax_vel.axhline(y=self.max_velocity, color='r', linestyle='--', alpha=0.7)
            ax_vel.set_ylabel('Velocity (m/s)')
            ax_vel.set_title('Velocity')
            ax_vel.grid(True)
            ax_vel.set_ylim(0, self.max_velocity * 1.1)
            
            # Orientation profile
            ax_orient.plot(trajectory['time_stamps'], np.degrees(trajectory['orientations']), 'purple', linewidth=2)
            ax_orient.set_ylabel('Orientation (°)')
            ax_orient.set_xlabel('Time (s)')
            ax_orient.set_title('Orientation')
            ax_orient.grid(True)
            
            plt.tight_layout()
            fig.canvas.draw()
        
        # Connect sliders to update function
        slider1.on_changed(update_plot)
        slider2.on_changed(update_plot)
        
        # Add reset buttons
        ax_reset1 = plt.axes([0.57, 0.08, 0.06, 0.03])
        button_reset1 = Button(ax_reset1, 'Reset WP1')
        
        ax_reset2 = plt.axes([0.57, 0.02, 0.06, 0.03])
        button_reset2 = Button(ax_reset2, 'Reset WP2')
        
        def reset_wp1_orientation(event):
            slider1.reset()
            
        def reset_wp2_orientation(event):
            # Reset to calculated tangent orientation
            slider2.set_val(np.degrees(default_second_orientation))
        
        button_reset1.on_clicked(reset_wp1_orientation)
        button_reset2.on_clicked(reset_wp2_orientation)
        
        # Initial plot
        update_plot()
        
        # Add instruction text
        fig.text(0.02, 0.14, 'Instructions:\n• WP1 Orientation: Control current robot orientation\n• WP2 Orientation: Control second waypoint orientation (affects Dubins curve)\n• "Reset WP1": Restore initial orientation • "Reset WP2": Restore tangent-based orientation', 
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.show()
        return fig


# Simplified function interface for easy use
def generate_robot_trajectory(waypoints, initial_orientation=0.0, orientations=None, 
                            max_velocity=3.0, max_acceleration=2.0, max_turning_rate=1.0, 
                            num_points=200, return_detailed=False):
    """
    Simplified function to generate robot trajectory with time and distance analysis
    
    Args:
        waypoints: List of (x, y) waypoint coordinates. First waypoint is current robot position.
        initial_orientation: Initial robot orientation in degrees (default: 0°)
        orientations: Optional list of orientations at each waypoint in degrees. 
                     If None, calculated from path direction.
        max_velocity: Maximum robot velocity in m/s (default: 3.0)
        max_acceleration: Maximum robot acceleration in m/s² (default: 2.0)
        max_turning_rate: Maximum robot turning rate in rad/s (default: 1.0)
        num_points: Number of trajectory points (default: 200)
        return_detailed: If True, returns full trajectory data. If False, returns summary.
    
    Returns:
        If return_detailed=False (default):
            Dictionary with keys: 'total_time', 'path_length', 'dubins_length', 
                                'hybrid_time', 'pure_spline_time', 'time_savings'
        If return_detailed=True:
            Full trajectory dictionary with all data
    
    Example:
        # Simple usage
        waypoints = [(-1, -0.5), (3, 2), (5, 5), (8, 4)]
        result = generate_robot_trajectory(waypoints, initial_orientation=30)
        
        # Advanced usage with custom orientations
        orientations = [30, 45, 90, 180]  # degrees at each waypoint
        result = generate_robot_trajectory(waypoints, orientations=orientations, 
                                         max_velocity=2.5, return_detailed=True)
    """
    # Convert orientations from degrees to radians
    if orientations is not None:
        orientations_rad = [np.radians(angle) for angle in orientations]
    else:
        # Use initial orientation for first waypoint, calculate others from path
        orientations_rad = [np.radians(initial_orientation)] + [None] * (len(waypoints) - 1)
    
    # Create optimizer with specified parameters
    optimizer = RobotTrajectoryOptimizer(
        max_velocity=max_velocity,
        max_acceleration=max_acceleration, 
        max_turning_rate=max_turning_rate
    )
    
    # Generate trajectory
    trajectory = optimizer.generate_trajectory(waypoints, orientations_rad, num_points)
    
    if return_detailed:
        return trajectory
    else:
        # Return simplified summary
        dubins_info = trajectory.get('dubins_connection')
        
        summary = {
            'total_time': trajectory['total_time'],
            'total_distance': trajectory['total_distance'],
            'path_length': trajectory['total_distance'],  # Main path length (hybrid if available)
            'is_hybrid_path': trajectory.get('is_hybrid_path', False),
            'dubins_length': dubins_info['path_length'] if dubins_info else 0,
            'hybrid_time': dubins_info['hybrid_total_time'] if dubins_info else trajectory['total_time'],
            'pure_spline_time': dubins_info['pure_spline_time'] if dubins_info else trajectory['total_time'],
            'time_savings': abs(dubins_info['time_difference']) if dubins_info else 0,
            'is_hybrid_faster': dubins_info['time_difference'] < 0 if dubins_info else False,
            'dubins_path_type': dubins_info['path_type'] if dubins_info else 'none',
            'turn_radius_used': dubins_info['turn_radius_used'] if dubins_info else 0,
            # Waypoint arrival information
            'waypoint_arrival_times': trajectory.get('waypoint_arrival_times', []),
            'waypoint_time_intervals': trajectory.get('waypoint_time_intervals', []),
            'num_waypoints': len(trajectory.get('original_waypoints', []))
        }
        return summary

class PathTrajectoryIntegrator:
    """
    Integrates trajectory optimization with the agent simulation system.
    Takes RRT paths and generates smooth, kinematically feasible trajectories.
    """
    
    def __init__(self, max_velocity=2.0, max_acceleration=1.5, max_turning_rate=1.0):
        """
        Initialize the path trajectory integrator.
        
        Args:
            max_velocity: Maximum linear velocity (m/s or pixels/s)
            max_acceleration: Maximum acceleration (m/s² or pixels/s²)
            max_turning_rate: Maximum angular velocity/turning rate (rad/s)
        """
        self.optimizer = RobotTrajectoryOptimizer(
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            max_turning_rate=max_turning_rate
        )
        
        # Store the last generated trajectory for access
        self.last_trajectory = None
        self.last_agent_id = None
        self.trajectory_timestamp = None
        
    def rrt_nodes_to_waypoints(self, rrt_path: List) -> List[Tuple[float, float]]:
        """
        Convert RRT nodes to waypoint coordinates.
        
        Args:
            rrt_path: List of RRT nodes from position_evaluator
            
        Returns:
            List of (x, y) waypoint tuples
        """
        waypoints = []
        for node in rrt_path:
            waypoints.append((float(node.x), float(node.y)))
        return waypoints
    
    def generate_optimized_trajectory(self, rrt_path: List, agent_id: str, num_points: int = 100) -> Optional[Dict[str, Any]]:
        """
        Generate an optimized trajectory from an RRT path.
        
        Args:
            rrt_path: List of RRT nodes
            agent_id: ID of the agent for this trajectory
            num_points: Number of points in the optimized trajectory
            
        Returns:
            Dictionary containing optimized trajectory data or None if failed
        """
        if not rrt_path or len(rrt_path) < 2:
            print("Error: RRT path must have at least 2 nodes")
            return None
        
        try:
            # Convert RRT nodes to waypoints
            waypoints = self.rrt_nodes_to_waypoints(rrt_path)
            print(f"Converting {len(waypoints)} waypoints for {agent_id}")
            
            # Generate optimized trajectory using the simplified interface
            trajectory = generate_robot_trajectory(
                waypoints=waypoints,
                initial_orientation=0.0,  # Default initial orientation
                orientations=None,  # Let the system calculate orientations from path
                max_velocity=self.optimizer.max_velocity,
                max_acceleration=self.optimizer.max_acceleration,
                max_turning_rate=self.optimizer.max_turning_rate,
                num_points=num_points,
                return_detailed=True  # Return full trajectory data
            )
            
            # Store for visualization
            self.last_trajectory = trajectory
            self.last_agent_id = agent_id
            self.trajectory_timestamp = time.time()
            
            # Add metadata
            trajectory['agent_id'] = agent_id
            trajectory['original_waypoints'] = waypoints
            trajectory['rrt_path'] = rrt_path
            trajectory['generation_time'] = self.trajectory_timestamp
            
            print(f"Generated optimized trajectory for {agent_id}:")
            print(f"  Total time: {trajectory['total_time']:.2f} seconds")
            print(f"  Total distance: {trajectory['total_distance']:.2f} units")
            print(f"  Peak velocity: {np.max(trajectory['velocities']):.2f} units/s")
            print(f"  Path type: {'Hybrid (Dubins+Spline)' if trajectory.get('is_hybrid_path', False) else 'Pure Spline'}")
            
            return trajectory
            
        except Exception as e:
            print(f"Error generating trajectory for {agent_id}: {e}")
            return None
    
    def _calculate_path_distance(self, path: np.ndarray) -> float:
        """Calculate total distance along a path."""
        if len(path) < 2:
            return 0.0
        
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return np.sum(distances)
    def get_trajectory_info(self) -> Dict[str, Any]:
        """
        Get information about the current trajectory for display.
        
        Returns:
            Dictionary with trajectory information
        """
        if not self.last_trajectory:
            return {"status": "No trajectory generated"}
        
        traj = self.last_trajectory
        path_distance = self._calculate_path_distance(traj['path'])
        avg_velocity = path_distance / traj['total_time'] if traj['total_time'] > 0 else 0.0
        
        return {
            "status": "Active",
            "agent_id": self.last_agent_id,
            "total_time": f"{traj['total_time']:.2f}s",
            "total_distance": f"{path_distance:.1f} px",
            "peak_velocity": f"{np.max(traj['velocities']):.1f} px/s",
            "avg_velocity": f"{avg_velocity:.1f} px/s",
        }
    
    def clear_trajectory(self) -> None:
        """Clear the current trajectory."""
        self.last_trajectory = None
        self.last_agent_id = None
        self.trajectory_timestamp = None
        print("Trajectory cleared")

# Global trajectory integrator instance
trajectory_integrator = None

def initialize_trajectory_integrator(max_velocity=150.0, max_acceleration=100.0, max_turning_rate=1.5):
    """
    Initialize the global trajectory integrator.
    
    Args:
        max_velocity: Maximum velocity in pixels/second
        max_acceleration: Maximum acceleration in pixels/second²
        max_turning_rate: Maximum turning rate in rad/s
    """
    global trajectory_integrator
    trajectory_integrator = PathTrajectoryIntegrator(
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        max_turning_rate=max_turning_rate
    )
    print(f"Trajectory integrator initialized with max_vel={max_velocity}, max_accel={max_acceleration}, max_turn_rate={max_turning_rate}")

def generate_trajectory_for_path(rrt_path: List, agent_id: str, num_points: int = 100) -> Optional[Dict[str, Any]]:
    """
    Generate optimized trajectory for an RRT path.
    
    Args:
        rrt_path: List of RRT nodes
        agent_id: Agent identifier
        num_points: Number of trajectory points
        
    Returns:
        Trajectory data dictionary or None
    """
    if trajectory_integrator is None:
        print("Error: Trajectory integrator not initialized. Call initialize_trajectory_integrator() first.")
        return None
    
    return trajectory_integrator.generate_optimized_trajectory(rrt_path, agent_id, num_points)

def get_current_trajectory() -> Optional[Dict[str, Any]]:
    """Get the current trajectory data for drawing."""
    if trajectory_integrator and trajectory_integrator.last_trajectory:
        return trajectory_integrator.last_trajectory
    return None

def get_trajectory_info() -> Dict[str, Any]:
    """Get information about the current trajectory."""
    if trajectory_integrator:
        return trajectory_integrator.get_trajectory_info()
    return {"status": "Integrator not initialized"}

def clear_trajectory() -> None:
    """Clear the current trajectory."""
    if trajectory_integrator:
        trajectory_integrator.clear_trajectory()


class ThreadedTrajectoryCalculator:
    """
    Handles trajectory calculations in background threads.
    Uses ThreadPoolExecutor for concurrent trajectory computations.
    """
    
    def __init__(self, max_workers=4):
        """
        Initialize the threaded trajectory calculator.
        
        Args:
            max_workers: Maximum number of background threads for calculations
        """
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self.travel_time_cache = {}  # Cache for travel time results
        self.travel_time_futures = {}  # Active futures for travel time calculations
        self.trajectory_futures = {}  # Active futures for trajectory calculations
        
        # Persistent cache for individual node travel times across calculations
        self.node_travel_times_cache = {}  # {agent_id: {node_id: travel_time}}
        
        self.lock = threading.Lock()
        
        print(f"ThreadedTrajectoryCalculator initialized with {max_workers} threads")
        
    def get_travel_times_async(self, agent_id):
        """
        Get travel times for an agent, using cached results or starting async calculation.
        Uses threading for trajectory calculations.
        Returns cached results immediately or None if still calculating.
        """
        with self.lock:
            # Return cached result if available
            if agent_id in self.travel_time_cache:
                return self.travel_time_cache[agent_id]
            
            # Start calculation if not already running
            if agent_id not in self.travel_time_futures:
                print(f"Starting background travel time calculation for {agent_id} with threading...")
                future = self.thread_executor.submit(self._calculate_travel_times_with_threading, agent_id)
                self.travel_time_futures[agent_id] = future
                return None  # Still calculating
                
            # Check if calculation is complete
            future = self.travel_time_futures[agent_id]
            if future.done():
                try:
                    result = future.result()
                    self.travel_time_cache[agent_id] = result
                    del self.travel_time_futures[agent_id]
                    print(f"Travel time calculation completed for {agent_id}: {len(result)} paths analyzed")
                    return result
                except Exception as e:
                    print(f"Error calculating travel times for {agent_id}: {e}")
                    del self.travel_time_futures[agent_id]
                    return []
            
            return None  # Still calculating
    
    def is_calculating(self, agent_id):
        """
        Check if travel time calculation is currently in progress for an agent.
        """
        with self.lock:
            return agent_id in self.travel_time_futures
    
    def generate_trajectory_async(self, path, agent_id, num_points=150, callback=None):
        """
        Generate trajectory asynchronously and call callback when done.
        """
        def trajectory_worker():
            try:
                result = generate_trajectory_for_path(path, agent_id, num_points)
                if callback:
                    callback(result, agent_id)
                return result
            except Exception as e:
                print(f"Error generating trajectory: {e}")
                return None
        
        future = self.thread_executor.submit(trajectory_worker)
        return future
    
    def _calculate_travel_times_with_threading(self, agent_id):
        """
        Calculate travel times using optimized DFS approach with intermediate caching.
        For each trajectory calculation, we store travel times to ALL intermediate nodes.
        Uses persistent caching across multiple calculations.
        This runs in a background thread and coordinates trajectory calculations.
        """
        # Import here to avoid circular imports in the main thread
        from position_evaluator import get_agent_rrt_tree, get_path_to_node
        
        tree = get_agent_rrt_tree(agent_id)
        if not tree:
            return []
        
        # Get trajectory optimizer parameters from global instance
        if trajectory_integrator is None:
            print(f"Warning: Trajectory integrator not initialized for {agent_id}")
            return []
        
        # Load existing node travel times from persistent cache
        with self.lock:
            if agent_id not in self.node_travel_times_cache:
                self.node_travel_times_cache[agent_id] = {}
            node_travel_times = self.node_travel_times_cache[agent_id].copy()
        
        # Create node index mapping for fast lookup
        node_to_index = {id(node): i for i, node in enumerate(tree)}
        
        non_root_nodes = [n for n in tree if n.parent is not None]
        print(f"Calculating travel times for {len(non_root_nodes)} nodes using optimized DFS with persistent caching...")
        print(f"  - Starting with {len(node_travel_times)} cached node travel times")
        
        calculated_trajectories = 0
        reused_intermediate_times = 0
        cached_nodes_used = 0
        
        # Process nodes in DFS order (deepest nodes first to maximize reuse)
        nodes_by_depth = sorted(non_root_nodes, key=lambda n: self._get_node_depth(n), reverse=True)
        
        for node in nodes_by_depth:
            node_idx = node_to_index[id(node)]
            
            # Skip if we already have travel time for this node
            if node_idx in node_travel_times:
                cached_nodes_used += 1
                continue
                
            # Get path to this node
            path = get_path_to_node(agent_id, node)
            if not path or len(path) < 2:
                continue
                
            try:
                # Convert path to waypoints
                waypoints = [(float(n.x), float(n.y)) for n in path]
                
                # Generate optimized trajectory using the simplified interface
                trajectory = generate_robot_trajectory(
                    waypoints=waypoints,
                    initial_orientation=0.0,
                    orientations=None,
                    max_velocity=trajectory_integrator.optimizer.max_velocity,
                    max_acceleration=trajectory_integrator.optimizer.max_acceleration,
                    max_turning_rate=trajectory_integrator.optimizer.max_turning_rate,
                    num_points=50,  # Fewer points for speed
                    return_detailed=True
                )
                
                if trajectory and 'total_time' in trajectory and 'waypoint_arrival_times' in trajectory:
                    calculated_trajectories += 1
                    
                    # Extract waypoint arrival times (cumulative times at each waypoint)
                    waypoint_arrival_times = trajectory.get('waypoint_arrival_times', [])
                    total_time = trajectory['total_time']
                    
                    # Store travel times for all nodes on this path
                    for i, path_node in enumerate(path):
                        path_node_idx = node_to_index[id(path_node)]
                        
                        # Get the arrival time for this waypoint
                        if i < len(waypoint_arrival_times):
                            arrival_time = waypoint_arrival_times[i]
                        elif i == len(path) - 1:
                            # Last node gets the total time
                            arrival_time = total_time
                        else:
                            # Estimate proportional time if waypoint data is incomplete
                            arrival_time = (i / (len(path) - 1)) * total_time
                        
                        # Only update if we don't have a time yet, or if this is a shorter path
                        if path_node_idx not in node_travel_times or arrival_time < node_travel_times[path_node_idx]:
                            if path_node_idx in node_travel_times:
                                reused_intermediate_times += 1
                            node_travel_times[path_node_idx] = arrival_time
                
            except Exception as e:
                print(f"Error calculating trajectory for node at ({node.x}, {node.y}): {e}")
                continue
        
        # Update persistent cache
        with self.lock:
            self.node_travel_times_cache[agent_id] = node_travel_times
        
        # Convert to the expected format: list of (node, travel_time, path_length)
        travel_times = []
        for node in non_root_nodes:
            node_idx = node_to_index[id(node)]
            if node_idx in node_travel_times:
                path = get_path_to_node(agent_id, node)
                path_length = len(path) if path else 0
                travel_times.append((node, node_travel_times[node_idx], path_length))
        
        # Sort by travel time (longest first)
        travel_times.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Optimized travel time calculation completed:")
        print(f"  - Used {cached_nodes_used} nodes from persistent cache")
        print(f"  - Calculated {calculated_trajectories} new full trajectories")
        print(f"  - Reused {reused_intermediate_times} intermediate travel times within calculations")
        print(f"  - Total nodes with travel times: {len(travel_times)}")
        total_efficiency = (cached_nodes_used + reused_intermediate_times)
        print(f"  - Overall efficiency: {total_efficiency}/{len(travel_times)} ({100*total_efficiency/max(len(travel_times),1):.1f}% cache reuse)")
        
        return travel_times
    
    def _get_node_depth(self, node):
        """Helper method to calculate the depth of a node in the tree."""
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
    
    def invalidate_cache(self, agent_id=None):
        """
        Clear cached results when RRT trees are regenerated.
        """
        with self.lock:
            if agent_id:
                self.travel_time_cache.pop(agent_id, None)
                self.node_travel_times_cache.pop(agent_id, None)  # Clear node-level cache too
                # Cancel any running calculation for this agent
                if agent_id in self.travel_time_futures:
                    self.travel_time_futures[agent_id].cancel()
                    del self.travel_time_futures[agent_id]
                print(f"Cleared travel time cache for {agent_id}")
            else:
                # Clear all caches
                self.travel_time_cache.clear()
                self.node_travel_times_cache.clear()  # Clear all node-level caches
                for future in self.travel_time_futures.values():
                    future.cancel()
                self.travel_time_futures.clear()
                print("Cleared all travel time caches")
    
    def shutdown(self):
        """
        Cleanup executors on shutdown.
        """
        print("Shutting down trajectory calculation threads...")
        self.thread_executor.shutdown(wait=False)


# Global trajectory calculator instance
global_trajectory_calculator = None

def initialize_trajectory_calculator(max_workers=4):
    """
    Initialize the global trajectory calculator.
    
    Args:
        max_workers: Maximum number of background threads
    """
    global global_trajectory_calculator
    global_trajectory_calculator = ThreadedTrajectoryCalculator(
        max_workers=max_workers
    )
    return global_trajectory_calculator

def get_trajectory_calculator():
    """Get the global trajectory calculator instance."""
    return global_trajectory_calculator

def calculate_all_travel_times(agent_id):
    """
    Legacy function for compatibility - now uses threaded calculator.
    """
    if global_trajectory_calculator:
        return global_trajectory_calculator.get_travel_times_async(agent_id) or []
    return []
