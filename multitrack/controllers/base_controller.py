"""
Base controller interface for the PivotedTracking project.

This module defines the interface that all controllers should implement.
"""
import os
import sys
import numpy as np
from abc import ABC, abstractmethod

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

class BaseController(ABC):
    """
    Abstract base class for all controllers.
    
    Any controller used in the PivotedTracking project should inherit from this class
    and implement its methods.
    """
    
    @abstractmethod
    def compute_control(self, current_state, target_trajectory, obstacles=None):
        """
        Compute optimal control input.
        
        Parameters:
        - current_state: Current state [x, y, theta, v]
        - target_trajectory: Sequence of target states to follow
        - obstacles: List of obstacle positions [(x, y, radius), ...] or None
        
        Returns:
        - optimal_control: Optimal control [v, omega] for the current time step
        - predicted_trajectory: Predicted states over the horizon
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the controller's internal state."""
        pass
    
    @abstractmethod
    def get_computation_stats(self):
        """
        Return performance statistics for monitoring.
        
        Returns:
        - dict: Dictionary with computation statistics
        """
        pass