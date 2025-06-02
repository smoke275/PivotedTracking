# Animated Rotating Rods Implementation for Agent 2

## Overview
This implementation adds smooth back-and-forth animation to the gap rods for agent 2, with a very narrow bar that highlights only the cells/nodes immediately below the gaps during the rotation animation.

## Features Implemented

### 1. Back-and-Forth Animation
- **Animation Period**: 4 seconds (4000ms) for a complete back-and-forth cycle
- **Motion Type**: Smooth sinusoidal motion using `math.sin()` for natural acceleration/deceleration
- **Animation Range**: From gap line position (0°) to maximum angle (45°) and back
- **Direction**: Cyan gaps rotate counterclockwise, green-cyan gaps rotate clockwise
- **Starting Point**: Rod starts at the gap line position and rotates outward

### 2. Narrow Highlighting Bar
- **Bar Width**: 5 pixels (ultra-narrow for precision)
- **Highlight Color**: Bright yellow (255, 255, 0) with 180 alpha transparency
- **Target**: Map graph nodes that fall within the narrow bar width under the rotating rod
- **Visual Style**: 
  - Highlighted circles (radius 5) for nodes under the rod
  - White border (1px width) for enhanced visibility

### 3. Enhanced Visual Design
- **Rod Appearance**: Thicker animated rod (4px width) in gap color (cyan/green-cyan)
- **Background Swept Area**: More transparent (40 alpha) to reduce visual clutter
- **Rotation Indicators**: Smaller, dimmer arrows to minimize distraction
- **Pivot Points**: Maintained cyan circles to show rotation centers

## Technical Implementation

### Animation Timing
```python
current_time = pygame.time.get_ticks()
animation_period = 4000  # 4 seconds
animation_speed = 2 * math.pi / animation_period
animation_phase = (current_time * animation_speed) % (2 * math.pi)
animation_progress = (math.sin(animation_phase) + 1) / 2  # 0 to 1 range
```

### Rod Position Calculation
```python
current_rotation_offset = max_rotation * rotation_direction * animation_progress
current_rod_angle = initial_rod_angle + current_rotation_offset
current_rod_end = (
    rod_base[0] + rod_length * math.cos(current_rod_angle),
    rod_base[1] + rod_length * math.sin(current_rod_angle)
)
```

### Node Highlighting Algorithm
For each map graph node, the system:
1. Calculates the point-to-line distance from the node to the current rod position
2. Projects the node onto the rod line to find the closest point
3. Measures the perpendicular distance
4. Highlights nodes within the 15-pixel bar width

### Distance Calculation
```python
# Vector from rod start to rod end
rod_dx = rod_x2 - rod_x1
rod_dy = rod_y2 - rod_y1
rod_length_sq = rod_dx * rod_dx + rod_dy * rod_dy

# Project node onto rod line
t = max(0, min(1, (node_dx * rod_dx + node_dy * rod_dy) / rod_length_sq))

# Closest point on rod line to the node
closest_x = rod_x1 + t * rod_dx
closest_y = rod_y1 + t * rod_dy

# Distance from node to rod line
distance_to_rod = math.sqrt((node_x - closest_x)**2 + (node_y - closest_y)**2)
```

## Controls

### Activation
- **Method 1**: Press `J` to enable agent 2 probability overlay (automatically enables rotating rods)
- **Method 2**: Press `K` to enable agent 2 visibility gaps, then `Y` for rotating rods
- **Global**: Press `Y` when visibility data is loaded (affects both agents)

### Prerequisites
- Visibility data must be loaded (automatic in environment inspection)
- Map graph must be available (automatic in environment inspection)
- Agent 2 must be visible on screen

## Visual Behavior

### Animation Cycle
1. **Phase 1** (0-1s): Rod starts at gap line and rotates to maximum angle
2. **Phase 2** (1-2s): Rod rotates back to gap line position
3. **Phase 3** (2-3s): Rod continues from gap line to maximum angle again
4. **Phase 4** (3-4s): Rod returns to gap line, cycle repeats

The rod always starts from the detected gap line (where the discontinuity occurs) and sweeps outward to its maximum rotation angle, then returns.

### Highlighting Behavior
- Yellow highlighted nodes follow the rod position in real-time
- Only nodes immediately under the current rod position are highlighted (5-pixel precision)
- Highlighting updates smoothly with the animation
- Nodes outside the ultra-narrow bar remain unaffected

## Color Coding
- **Cyan Rods** (0, 200, 255): Near-to-far gaps, rotate counterclockwise
- **Green-Cyan Rods** (0, 240, 180): Far-to-near gaps, rotate clockwise
- **Yellow Highlights** (255, 255, 0): Nodes under the rotating rod
- **Cyan Pivot Points** (0, 255, 255): Rod rotation centers

## Performance Considerations
- Animation uses efficient trigonometric calculations
- Node highlighting uses optimized point-to-line distance algorithm with 5-pixel precision
- Transparency surfaces are created per frame for smooth blending
- Only significant gaps (>50 pixel difference) are processed

## Integration
The animated rotating rods are fully integrated with the existing environment inspection system and work alongside:
- Agent 2 probability overlay
- Agent 2 visibility gaps visualization  
- Map graph display
- Path visualization
- All existing controls and features

The animation runs continuously when activated and does not interfere with other visualization elements or user interactions.
