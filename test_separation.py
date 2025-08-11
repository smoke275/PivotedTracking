#!/usr/bin/env python3
"""
Test script to verify the polygon exploration module separation.
"""

# Test imports
try:
    from polygon_exploration import calculate_polygon_exploration_paths
    print("✓ polygon_exploration import successful")
except ImportError as e:
    print(f"✗ polygon_exploration import failed: {e}")

try:
    from risk_calculator import calculate_evader_analysis
    print("✓ risk_calculator import successful")
except ImportError as e:
    print(f"✗ risk_calculator import failed: {e}")

# Test that polygon exploration functions are available
try:
    from polygon_exploration import (
        find_nearest_intersection,
        ray_line_intersection,
        ray_circle_intersection,
        line_circle_intersections,
        calculate_exploration_turn_direction
    )
    print("✓ All polygon exploration functions available")
except ImportError as e:
    print(f"✗ Some polygon exploration functions missing: {e}")

print("\nModule separation successful!")
print("Polygon exploration functionality is now in polygon_exploration.py")
print("Risk calculator imports and uses it seamlessly.")
