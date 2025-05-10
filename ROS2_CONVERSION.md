# Converting MultiTrack to ROS 2

## Summary

This document outlines a plan for converting the PivotedTracking simulation to ROS 2. The current Python-based simulation with Pygame visualization can be transformed into a distributed ROS 2 system with multiple nodes communicating via topics and services.

## Architecture Overview

### ROS 2 Nodes

1. **Visitor Agent Node**
   - Handles visitor dynamics
   - Publishes visitor state
   - Subscribes to control inputs

2. **Escort Agent Node**
   - Handles escort dynamics and tracking behavior
   - Publishes escort state
   - Implements Kalman filter for tracking
   - Contains camera vision system
   - Subscribes to visitor state

3. **Environment Node**
   - Maintains walls, doors, and other environment elements
   - Publishes environment data
   - Handles collision detection

4. **Visualization Node**
   - Subscribes to all state topics
   - Provides visualization through RViz or custom interface
   - Could maintain Pygame visualization as a transition step

5. **Controller Node**
   - Implements MPPI and PID controllers
   - Publishes control commands
   - Could be part of the Escort Agent or separate

### ROS 2 Topics

- `/visitor/state` - Visitor position, orientation, and velocity
- `/visitor/controls` - Control inputs for the visitor
- `/escort/state` - Escort position, orientation, and velocity 
- `/escort/controls` - Control inputs for the escort
- `/escort/target_distance` - Following distance parameter
- `/escort/camera/orientation` - Camera orientation information
- `/escort/visibility` - Visibility status information
- `/environment/walls` - Wall information
- `/environment/doors` - Door information
- `/kalman/predictions` - Kalman filter predictions
- `/kalman/uncertainty` - Covariance matrix for visualization

### ROS 2 Services

- `/escort/set_controller_type` - Switch between MPPI and PID controllers
- `/escort/toggle_camera_auto_track` - Toggle camera auto-tracking
- `/escort/reset_position` - Reset escort position
- `/simulation/toggle_pause` - Pause/resume simulation

### ROS 2 Parameters

- Screen dimensions
- Vision ranges and angles
- Controller parameters
- Follower parameters

## Implementation Steps

1. **Create ROS 2 workspace and package structure**:
   ```
   multitrack_ros/
   ├── multitrack_msgs/       # Custom messages
   ├── multitrack_core/       # Core algorithms
   ├── multitrack_agents/     # Agent implementations 
   ├── multitrack_simulation/ # Simulation environment
   ├── multitrack_viz/        # Visualization
   └── multitrack_controllers/ # Controllers
   ```

2. **Define custom message types** for agent states, control inputs, vision cones, etc.

3. **Convert each module to a ROS 2 node**:
   - Start with the agent models (visitor and escort)
   - Implement the controllers as separate nodes or libraries
   - Create a visualization node using RViz or a custom visualization

4. **Use ROS 2 launch files** to orchestrate the system startup

5. **Implement keyboard input through ROS 2** teleop mechanisms

## Benefits of Converting to ROS 2

1. **Distributed system**: Run components on different machines
2. **Hardware integration**: Easily connect to real robots, sensors, and actuators
3. **Tools ecosystem**: Access to ROS 2 tools like rosbag, RViz, tf2, etc.
4. **Standardized communication**: Well-defined interfaces through message types
5. **Real-time capabilities**: ROS 2 has better real-time performance than ROS 1

## Challenges

1. **Performance overhead**: ROS 2 adds communication overhead
2. **Learning curve**: If not familiar with ROS 2
3. **Visualization adaptation**: Switching from Pygame to RViz or custom ROS 2 visualizers
4. **Message definition**: Creating appropriate message types for all data structures

## Phase 1: Core Structure

Initial phase focuses on setting up the ROS 2 package structure and implementing basic communication between nodes:

1. Create basic ROS 2 package structure
2. Define core message types
3. Implement basic visitor and escort nodes
4. Create simple visualization

## Phase 2: Feature Migration

1. Implement Kalman filter as ROS 2 node
2. Convert controllers to ROS 2
3. Add environment with collision detection
4. Implement vision system
5. Add parameter handling

## Phase 3: Advanced Features

1. Improve visualization with RViz markers
2. Add rosbag recording for simulation playback
3. Create web interface using ROS 2 bridge
4. Implement hardware integration
5. Add testing framework

## Resources

- [ROS 2 Documentation](https://docs.ros.org/en/humble/index.html)
- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [ROS 2 Migration Guide](https://docs.ros.org/en/humble/Contributing/Migration-Guide.html)
- [ROS 2 Humble Packages](https://github.com/ros2/ros2/wiki/Humble-Overview)