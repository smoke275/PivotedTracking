#!/usr/bin/env python3
"""
Fast Visibility Calculator
C++ optimized version of visibility calculations with fallback to Python implementation.
"""

import math
import sys
import warnings

# Try to import the C++ extension, fall back to Python implementation
try:
    import fast_visibility
    CPP_AVAILABLE = True
    print("✅ Fast C++ visibility extension loaded successfully")
except ImportError as e:
    CPP_AVAILABLE = False
    print(f"⚠️  C++ extension not available: {e}")
    print("💡 Run 'python setup.py build_ext --inplace' to build the fast C++ extension")
    print("🐍 Falling back to Python implementation")

def convert_rectangles_to_tuples(rectangles):
    """Convert pygame.Rect objects to (x, y, width, height) tuples for C++."""
    return [(rect.x, rect.y, rect.width, rect.height) for rect in rectangles]

def calculate_visibility_optimized(agent_x, agent_y, visibility_range, walls, doors, num_rays=100, force_python=False):
    """
    Calculate 360-degree visibility with automatic C++/Python fallback.
    
    Args:
        agent_x, agent_y: Agent position
        visibility_range: Maximum visibility distance
        walls: List of wall rectangles (pygame.Rect objects)
        doors: List of door rectangles (pygame.Rect objects)
        num_rays: Number of rays to cast
        force_python: Force use of Python implementation for testing
        
    Returns:
        List of (angle, endpoint_x, endpoint_y, distance, blocked) tuples
    """
    
    # Use C++ implementation if available and not forced to use Python
    if CPP_AVAILABLE and not force_python:
        try:
            wall_tuples = convert_rectangles_to_tuples(walls)
            door_tuples = convert_rectangles_to_tuples(doors)
            
            # Call C++ function
            results = fast_visibility.calculate_fast_visibility(
                agent_x, agent_y, visibility_range, wall_tuples, door_tuples, num_rays
            )
            
            # Convert to expected format: (angle, (endpoint_x, endpoint_y), distance, blocked)
            visibility_data = []
            for angle, endpoint_x, endpoint_y, distance, blocked in results:
                visibility_data.append((angle, (endpoint_x, endpoint_y), distance, blocked))
            
            return visibility_data
            
        except Exception as e:
            warnings.warn(f"C++ implementation failed: {e}. Falling back to Python.")
    
    # Fallback to original Python implementation
    from multitrack.utils.vision import cast_vision_ray
    
    visibility_data = []
    angle_step = (2 * math.pi) / num_rays
    
    for i in range(num_rays):
        angle = i * angle_step
        
        # Cast ray in this direction
        endpoint = cast_vision_ray(
            agent_x, agent_y, angle, visibility_range, walls, doors
        )
        
        # Calculate distance to endpoint
        distance = math.sqrt((endpoint[0] - agent_x)**2 + (endpoint[1] - agent_y)**2)
        
        # Check if ray was blocked (distance less than max range)
        blocked = distance < visibility_range - 1  # Small tolerance
        
        visibility_data.append((angle, endpoint, distance, blocked))
    
    return visibility_data

class FastVisibilityCalculator:
    """
    Reusable visibility calculator that maintains wall/door data for multiple calculations.
    Optimized for scenarios where the environment doesn't change but agent position does.
    """
    
    def __init__(self, walls=None, doors=None):
        self.walls = walls or []
        self.doors = doors or []
        self._cpp_calculator = None
        
        if CPP_AVAILABLE:
            try:
                self._cpp_calculator = fast_visibility.FastVisibilityCalculator()
                self._update_cpp_data()
            except Exception as e:
                warnings.warn(f"Failed to initialize C++ calculator: {e}")
                self._cpp_calculator = None
    
    def _update_cpp_data(self):
        """Update C++ calculator with current wall/door data."""
        if self._cpp_calculator:
            try:
                wall_tuples = convert_rectangles_to_tuples(self.walls)
                door_tuples = convert_rectangles_to_tuples(self.doors)
                self._cpp_calculator.set_walls(wall_tuples)
                self._cpp_calculator.set_doors(door_tuples)
            except Exception as e:
                warnings.warn(f"Failed to update C++ data: {e}")
                self._cpp_calculator = None
    
    def set_environment(self, walls, doors):
        """Update the environment (walls and doors)."""
        self.walls = walls
        self.doors = doors
        if self._cpp_calculator:
            self._update_cpp_data()
    
    def calculate_visibility(self, agent_x, agent_y, visibility_range, num_rays=100, force_python=False):
        """
        Calculate visibility from agent position.
        
        Args:
            agent_x, agent_y: Agent position
            visibility_range: Maximum visibility distance
            num_rays: Number of rays to cast
            force_python: Force use of Python implementation
            
        Returns:
            List of (angle, (endpoint_x, endpoint_y), distance, blocked) tuples
        """
        
        # Try C++ implementation first
        if self._cpp_calculator and not force_python:
            try:
                results = self._cpp_calculator.calculate_visibility(
                    agent_x, agent_y, visibility_range, num_rays
                )
                
                # Convert to expected format
                visibility_data = []
                for angle, endpoint_x, endpoint_y, distance, blocked in results:
                    visibility_data.append((angle, (endpoint_x, endpoint_y), distance, blocked))
                
                return visibility_data
                
            except Exception as e:
                warnings.warn(f"C++ calculation failed: {e}. Using Python fallback.")
        
        # Fallback to Python implementation
        return calculate_visibility_optimized(
            agent_x, agent_y, visibility_range, self.walls, self.doors, num_rays, force_python=True
        )

def benchmark_implementations(agent_x=100, agent_y=100, visibility_range=200, num_rays=100, num_iterations=100):
    """
    Benchmark C++ vs Python implementations.
    """
    import time
    import pygame
    
    # Create some test walls and doors
    walls = [
        pygame.Rect(50, 50, 100, 10),
        pygame.Rect(200, 80, 80, 120),
        pygame.Rect(300, 150, 60, 80),
    ]
    doors = [pygame.Rect(250, 100, 20, 40)]
    
    print(f"🔬 Benchmarking visibility calculations:")
    print(f"   Agent: ({agent_x}, {agent_y})")
    print(f"   Range: {visibility_range}, Rays: {num_rays}")
    print(f"   Iterations: {num_iterations}")
    print(f"   Walls: {len(walls)}, Doors: {len(doors)}")
    
    # Benchmark Python implementation
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        result_python = calculate_visibility_optimized(
            agent_x, agent_y, visibility_range, walls, doors, num_rays, force_python=True
        )
    python_time = time.perf_counter() - start_time
    
    print(f"🐍 Python implementation: {python_time:.4f}s ({python_time/num_iterations*1000:.2f}ms per call)")
    
    # Benchmark C++ implementation if available
    if CPP_AVAILABLE:
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            result_cpp = calculate_visibility_optimized(
                agent_x, agent_y, visibility_range, walls, doors, num_rays, force_python=False
            )
        cpp_time = time.perf_counter() - start_time
        
        print(f"⚡ C++ implementation: {cpp_time:.4f}s ({cpp_time/num_iterations*1000:.2f}ms per call)")
        print(f"🚀 Speedup: {python_time/cpp_time:.2f}x faster")
        
        # Verify results are similar
        if len(result_python) == len(result_cpp):
            max_diff = 0
            for (a1, e1, d1, b1), (a2, e2, d2, b2) in zip(result_python, result_cpp):
                diff = abs(d1 - d2)
                max_diff = max(max_diff, diff)
            print(f"✅ Results match (max distance difference: {max_diff:.6f})")
        else:
            print("❌ Results length mismatch!")
    else:
        print("⚠️  C++ implementation not available for comparison")

if __name__ == "__main__":
    # Run benchmark if called directly
    benchmark_implementations()
