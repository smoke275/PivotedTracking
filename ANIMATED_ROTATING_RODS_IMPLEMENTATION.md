# Instantaneous Sweep-Based Probability System for Agent 2

## Overview
This implementation replaces the animated rotating rods with an instantaneous sweep-based probability system. The system processes all angles in the 45-degree sweep range at once, assigns probabilities to nodes under the sweep area, and integrates these with Agent 2's existing visibility-based probability system.

## Features Implemented

### 1. Instantaneous Sweep Processing
- **Processing Method**: All angles processed at once (no animation)
- **Sweep Range**: Full 45-degree range from gap line position to maximum angle
- **Angle Resolution**: 20 discrete angles across the sweep range for high precision
- **Probability Assignment**: 0.8-1.0 range, increasing linearly with sweep angle
- **Direction**: Cyan gaps sweep counterclockwise, green-cyan gaps sweep clockwise

### 2. Gap Probability Calculation
- **Base Probability**: 0.8 for nodes at the starting angle
- **Maximum Probability**: 1.0 for nodes at the maximum sweep angle
- **Progression**: Linear increase: `0.8 + (angle_progress * 0.2)`
- **Node Detection**: 15-pixel width sweep bar for node detection
- **Conflict Resolution**: Maximum probability used when nodes hit by multiple angles

### 3. Integrated Probability System
- **Data Structure**: `agent2_gap_probabilities` dictionary stores gap-based probabilities
- **Integration Method**: Visibility-based probabilities override gap-based probabilities
- **Fallback Support**: Works with or without visibility data
- **Color Coding**: 
  - Pink to Green gradient based on probability (0.0 to 1.0)
  - Gap-based high probabilities show as bright green
  - Visual glow effect for probabilities > 0.7

## Technical Implementation

### Instantaneous Sweep Processing
```python
# Number of discrete angles to sweep through
num_sweep_angles = 20

for angle_step in range(num_sweep_angles + 1):
    angle_progress = angle_step / num_sweep_angles
    current_angle = sweep_start_angle + angle_progress * (sweep_end_angle - sweep_start_angle)
    
    # Calculate gap probability: 0.8 to 1.0 range
    gap_probability = 0.8 + (angle_progress * 0.2)
```

### Probability Integration
```python
# Visibility-based probabilities override gap-based ones
for node_index, gap_prob in agent2_gap_probabilities.items():
    if node_index not in agent2_node_probabilities:
        # Only gap probability exists (no visibility): use gap probability
        agent2_node_probabilities[node_index] = gap_prob
    # If visibility probability exists, it overrides gap probability (no action needed)
```
```python
current_time = pygame.time.get_ticks()
animation_period = 4000  # 4 seconds
animation_speed = 2 * math.pi / animation_period
### Node Detection Algorithm
For each map graph node and each sweep angle, the system:
1. Calculates the current rod position for the given angle
2. Computes the point-to-line distance from the node to the rod
3. Projects the node onto the rod line to find the closest point
4. Assigns probability to nodes within the 15-pixel sweep bar width

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

# If node is within sweep bar width, assign probability
if distance_to_rod <= bar_width:
    agent2_gap_probabilities[node_index] = gap_probability
```

# Closest point on rod line to the node
## Controls

### Activation
- **Method 1**: Press `J` to enable agent 2 probability overlay (automatically enables sweep system)
- **Method 2**: Press `K` to enable agent 2 visibility gaps, then `Y` for sweep processing
- **Global**: Press `Y` when visibility data is loaded (affects both agents)

### Prerequisites
- Visibility data must be loaded (automatic in environment inspection)
- Map graph must be available (automatic in environment inspection)
- Agent 2 must be visible on screen

## Visual Behavior

### Instantaneous Processing
1. **Gap Detection**: System identifies visibility gaps automatically
2. **Sweep Processing**: All 20 angles in 45-degree range processed at once
3. **Probability Assignment**: Nodes receive probabilities (0.8-1.0) based on sweep coverage
4. **Integration**: Gap probabilities merged with visibility-based probabilities
5. **Visualization**: Final probabilities displayed as colored nodes (pink to green)

### Static Swept Area Display
- Static swept area visualization shows the full 45-degree coverage zone
- Rod position displayed at initial gap line position (no animation)
- Transparency used to avoid visual clutter with other elements
- Swept areas clearly indicate coverage zones for gap-based probabilities

## Color Coding
- **Pink Nodes**: Low probability (0.1-0.3) - mostly visibility-based
- **Orange Nodes**: Medium probability (0.4-0.6) - mixed visibility/gap
- **Green Nodes**: High probability (0.7-1.0) - strong gap-based probabilities
- **Glow Effect**: Applied to nodes with probability > 0.7 for emphasis
- **Cyan Vision Circle**: Agent 2's 800px vision range indicator

## Performance Advantages
- **No Animation**: Eliminates per-frame calculations and smooth rendering overhead
- **Instantaneous Processing**: All probabilities calculated in single pass
- **Efficient Integration**: Direct dictionary merging with existing probability system
- **Optimized Storage**: Only stores probabilities > 0 for memory efficiency
- **Scalable Resolution**: 20-angle discretization provides good coverage without excess computation

## Integration
The instantaneous sweep system is fully integrated with the existing environment inspection system:
- **Agent 2 Probability System**: Seamlessly merges with visibility-based probabilities
- **Visual Overlay**: Works with existing node visualization and color coding
- **Map Graph Integration**: Operates on same node indices as visibility system
- **Control Compatibility**: Activated by same keys as previous animated system
- **Configuration Respect**: Uses `AGENT2_BASE_PROBABILITY` from config for baseline values

The system provides immediate probability updates without animation delays and maintains compatibility with all existing features.
