"""
Threat Calculator Module

This module handles dual-agent tracking, environment-aware visibility calculations, 
and threat assessment functionality. Provides a modular interface for tracking 
agent movement, updating visibility calculations, and performing threat analysis.

KEY FEATURES:
- Dual-agent position tracking and movement detection
- Environment-aware visibility calculations using walls and doors
- Map graph integration for node-based threat assessment
- Modular architecture for extensible threat calculation methods
- Real-time agent position updates for continuous threat monitoring
"""

import math
import numpy as np
import time
from multitrack.utils.pathfinding import find_closest_node


# OPTIMIZATION: Agent2ComputationOptimizer for selective computation
class Agent2ComputationOptimizer:
    """
    Selective computation system to avoid unnecessary recomputations
    when agent state hasn't changed significantly.
    """
    def __init__(self):
        self.last_position = None
        self.last_angle = None
        self.last_result = None
        self.last_gap_probabilities = None
        
        # Configurable thresholds for detecting significant changes
        self.position_threshold = 5.0  # pixels - recompute if agent moves >5px
        self.angle_threshold = 0.1     # radians (~5.7 degrees) - recompute if angle changes significantly
        
        # Performance tracking
        self.computation_skipped = 0
        self.computation_performed = 0
        
        # Cache invalidation tracking
        self.frames_since_last_computation = 0
        self.max_frames_without_computation = 30  # Force recomputation every 30 frames max
    
    def needs_recomputation(self, current_pos, current_angle):
        """
        Determine if Agent2 computation needs to be performed based on state changes.
        
        Args:
            current_pos: Current agent position (x, y)
            current_angle: Current agent angle in radians
            
        Returns:
            bool: True if recomputation is needed, False if cached result can be used
        """
        # Always compute on first run
        if self.last_position is None:
            return True
            
        # Force recomputation after max frames to prevent stale data
        if self.frames_since_last_computation >= self.max_frames_without_computation:
            return True
        
        # Check position change
        pos_delta = math.sqrt((current_pos[0] - self.last_position[0])**2 + 
                             (current_pos[1] - self.last_position[1])**2)
        
        # Check angle change
        angle_delta = abs(current_angle - self.last_angle)
        # Handle angle wraparound (e.g., from 359° to 1°)
        angle_delta = min(angle_delta, 2*math.pi - angle_delta)
        
        return (pos_delta > self.position_threshold or 
                angle_delta > self.angle_threshold)
    
    def update_state(self, position, angle, gap_probabilities):
        """Update cached state after computation"""
        self.last_position = position
        self.last_angle = angle
        self.last_gap_probabilities = gap_probabilities.copy() if gap_probabilities else {}
        self.computation_performed += 1
        self.frames_since_last_computation = 0
    
    def skip_computation(self):
        """Record that computation was skipped and return cached result"""
        self.computation_skipped += 1
        self.frames_since_last_computation += 1
        return self.last_gap_probabilities if self.last_gap_probabilities else {}
    
    def get_performance_stats(self):
        """Get performance statistics for monitoring"""
        total_calls = self.computation_performed + self.computation_skipped
        skip_rate = self.computation_skipped / total_calls if total_calls > 0 else 0
        return {
            'total_calls': total_calls,
            'computations_performed': self.computation_performed,
            'computations_skipped': self.computation_skipped,
            'skip_rate_percent': skip_rate * 100,
            'frames_since_last': self.frames_since_last_computation
        }


# OPTIMIZATION: Mathematical shortcuts and fast approximations
class FastMathOptimizations:
    """
    Fast mathematical operations for improved performance.
    Provides approximations and optimized calculations that maintain visual quality.
    """
    
    @staticmethod
    def fast_distance_squared(p1, p2):
        """Fast squared distance calculation (avoids sqrt when only comparison needed)"""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy
    
    @staticmethod
    def fast_distance_check(p1, p2, threshold):
        """Fast distance check using squared distance to avoid sqrt"""
        return FastMathOptimizations.fast_distance_squared(p1, p2) < (threshold * threshold)
    
    @staticmethod
    def fast_normalize_angle(angle):
        """Fast angle normalization to [0, 2π] range"""
        while angle < 0:
            angle += 2 * math.pi
        while angle >= 2 * math.pi:
            angle -= 2 * math.pi
        return angle
    
    @staticmethod
    def fast_angle_difference(angle1, angle2):
        """Fast calculation of minimum angle difference (handles wraparound)"""
        diff = abs(angle1 - angle2)
        return min(diff, 2 * math.pi - diff)


class TrigLookupTable:
    """
    Pre-computed trigonometric lookup table for common angles.
    Trades memory for computation speed.
    """
    
    def __init__(self, precision=1000):
        self.precision = precision
        self.angle_step = 2 * math.pi / precision
        
        # Pre-compute sin/cos for common angles
        self.angles = np.linspace(0, 2 * math.pi, precision, endpoint=False)
        self.cos_table = np.cos(self.angles)
        self.sin_table = np.sin(self.angles)
    
    def fast_cos(self, angle):
        """Fast cosine lookup with linear interpolation"""
        # Normalize angle to [0, 2π]
        angle = angle % (2 * math.pi)
        
        # Find nearest table index
        index = int(angle / self.angle_step)
        index = min(index, self.precision - 1)
        
        return self.cos_table[index]
    
    def fast_sin(self, angle):
        """Fast sine lookup with linear interpolation"""
        # Normalize angle to [0, 2π]
        angle = angle % (2 * math.pi)
        
        # Find nearest table index
        index = int(angle / self.angle_step)
        index = min(index, self.precision - 1)
        
        return self.sin_table[index]


def cast_rays_and_find_gaps(position_x, position_y, visibility_range, environment):
    """
    Consolidated ray casting and gap finding logic.
    Moved from environment_inspection_simulation_backup.py for modularity.
    
    Args:
        position_x: X coordinate of the ray casting origin
        position_y: Y coordinate of the ray casting origin  
        visibility_range: Maximum range for ray casting
        environment: SimulationEnvironment instance for wall/door collision detection
    
    Returns:
        tuple: (ray_endpoints, gap_lines) where gap_lines contains (start_point, end_point, gap_size)
    """
    from multitrack.utils.vision import cast_vision_ray
    
    # Cast rays in all directions to find visibility discontinuities
    num_rays = 360  # Every 1 degree for finer resolution
    angle_step = (2 * math.pi) / num_rays
    ray_endpoints = []
    
    # Cast rays in all directions
    for i in range(num_rays):
        angle = i * angle_step
        endpoint = cast_vision_ray(
            position_x, 
            position_y, 
            angle, 
            visibility_range,
            environment.get_all_walls(),
            environment.get_doors()  # Doors allow vision through
        )
        ray_endpoints.append(endpoint)
    
    # Find discontinuities in ray distances and connect successive rays with abrupt changes
    min_gap_distance = 30  # Minimum distance difference to consider a gap
    gap_lines = []
    
    for i in range(num_rays):
        current_endpoint = ray_endpoints[i]
        next_endpoint = ray_endpoints[(i + 1) % num_rays]  # Wrap around
        
        # Calculate distances from position
        current_dist = math.dist((position_x, position_y), current_endpoint)
        next_dist = math.dist((position_x, position_y), next_endpoint)
        
        # Check for significant distance change (gap) between successive rays
        distance_diff = abs(current_dist - next_dist)
        if distance_diff > min_gap_distance:
            # Record this gap line connecting the two successive ray endpoints
            gap_lines.append((current_endpoint, next_endpoint, distance_diff))
    
    return ray_endpoints, gap_lines


class ThreatCalculator:
    """
    Handles dual-agent movement tracking and threat assessment calculations.
    
    This class encapsulates agent-following functionality and environment-aware
    threat calculations, making it modular and reusable for complex scenarios.
    """
    
    def __init__(self, map_graph, environment=None, movement_threshold=20.0):
        """
        Initialize the threat calculator with dual-agent tracking and environment awareness.
        
        Args:
            map_graph: The MapGraph instance containing nodes and visibility data
            environment: SimulationEnvironment instance for walls, doors, and visibility calculations
            movement_threshold: Minimum distance to consider agent has moved (pixels)
        """
        self.map_graph = map_graph
        self.environment = environment
        self.movement_threshold = movement_threshold
        
        # Agent 1 (primary) tracking state
        self.agent1_last_position = None
        self.agent1_following_node_index = None
        
        # Agent 2 (secondary) tracking state  
        self.agent2_last_position = None
        self.agent2_following_node_index = None
        
        # Current agent positions for calculations
        self.agent1_current_position = None
        self.agent2_current_position = None
        
        # OPTIMIZATION: Initialize optimizer instances
        self.agent2_optimizer = Agent2ComputationOptimizer()
        self.fast_math = FastMathOptimizations()
        self.trig_lookup = TrigLookupTable(precision=360000)  # 0.001 degree precision
        
        # Initialize if map graph has nodes
        if self.map_graph and self.map_graph.nodes:
            print("ThreatCalculator: Initialized with dual-agent tracking and environment awareness")
            print("ThreatCalculator: Optimizations enabled (Agent2Optimizer, FastMath, TrigLookup)")
            if self.environment:
                print("ThreatCalculator: Environment data available for visibility calculations")
    
    def update_agent_positions(self, agent1, agent2):
        """
        Update current positions of both agents before threat calculations.
        
        This should be called at every step in the main loop before other calculations.
        
        Args:
            agent1: Primary agent object with state [x, y, theta, ...]
            agent2: Secondary agent object with state [x, y, theta, ...]
        """
        self.agent1_current_position = (agent1.state[0], agent1.state[1])
        self.agent2_current_position = (agent2.state[0], agent2.state[1])
    
    def initialize_agent_following(self, agent1, agent2=None, visibility_map=None):
        """
        Initialize dual-agent following functionality.
        
        Args:
            agent1: Primary agent object with state [x, y, theta, ...]
            agent2: Optional secondary agent object with state [x, y, theta, ...]
            visibility_map: Optional visibility map for enhanced initialization
            
        Returns:
            tuple: (selected_node_index, message) - the initial node index and status message
        """
        if not self.map_graph or not self.map_graph.nodes:
            return None, "No map nodes available for agent following"
        
        # Initialize agent 1 position tracking
        self.agent1_last_position = (agent1.state[0], agent1.state[1])
        self.agent1_current_position = self.agent1_last_position
        
        # Initialize agent 2 position tracking if provided
        if agent2:
            self.agent2_last_position = (agent2.state[0], agent2.state[1])
            self.agent2_current_position = self.agent2_last_position
        
        # Find closest nodes to current agent positions
        agent1_pos = (agent1.state[0], agent1.state[1])
        self.agent1_following_node_index = find_closest_node(self.map_graph.nodes, agent1_pos)
        
        if agent2:
            agent2_pos = (agent2.state[0], agent2.state[1])
            self.agent2_following_node_index = find_closest_node(self.map_graph.nodes, agent2_pos)
        
        if self.agent1_following_node_index is not None and visibility_map:
            message = f"Following agent1 at node {self.agent1_following_node_index}"
            if agent2 and self.agent2_following_node_index is not None:
                message += f", agent2 at node {self.agent2_following_node_index}"
            return self.agent1_following_node_index, message
        elif self.agent1_following_node_index is not None:
            message = f"Following agent1 at node {self.agent1_following_node_index} (visibility data pending)"
            if agent2 and self.agent2_following_node_index is not None:
                message += f", agent2 at node {self.agent2_following_node_index}"
            return self.agent1_following_node_index, message
        else:
            return None, "Agent position not near any map nodes"
    
    def update_agent_following(self, agent1, agent2=None, visibility_map=None):
        """
        Update agent-following functionality in the main loop.
        
        Args:
            agent1: Primary agent object with state [x, y, theta, ...]
            agent2: Optional secondary agent object with state [x, y, theta, ...]
            visibility_map: Optional visibility map for checking node validity
            
        Returns:
            tuple: (selected_node_index, agent_moved, debug_message) - updated node index, 
                   whether agent moved significantly, and optional debug message
        """
        if not visibility_map:
            return self.agent1_following_node_index, False, None
        
        # Update current positions (this should also be called separately via update_agent_positions)
        current_agent1_position = (agent1.state[0], agent1.state[1])
        self.agent1_current_position = current_agent1_position
        
        if agent2:
            current_agent2_position = (agent2.state[0], agent2.state[1])
            self.agent2_current_position = current_agent2_position
        
        # Check if agent1 has moved significantly
        agent1_moved = False
        if self.agent1_last_position is None:
            agent1_moved = True
        else:
            distance_moved = math.sqrt(
                (current_agent1_position[0] - self.agent1_last_position[0]) ** 2 + 
                (current_agent1_position[1] - self.agent1_last_position[1]) ** 2
            )
            if distance_moved >= self.movement_threshold:
                agent1_moved = True
        
        # Update visibility display if agent1 moved significantly
        if agent1_moved:
            # Find the closest node to agent1's current position
            new_closest_node = find_closest_node(self.map_graph.nodes, current_agent1_position)
            
            # Only update if we found a node and it's different from current
            if (new_closest_node is not None and 
                new_closest_node != self.agent1_following_node_index and
                new_closest_node in visibility_map):
                
                old_node = self.agent1_following_node_index
                self.agent1_following_node_index = new_closest_node
                self.agent1_last_position = current_agent1_position
                
                # Generate debug message
                visible_count = len(visibility_map[self.agent1_following_node_index])
                debug_message = f"Agent1 moved - now following node {self.agent1_following_node_index} with {visible_count} visible nodes"
                
                return self.agent1_following_node_index, True, debug_message
        
        # Update agent2 tracking if provided (for future threat calculations)
        if agent2:
            agent2_moved = False
            if self.agent2_last_position is None:
                agent2_moved = True
            else:
                distance_moved = math.sqrt(
                    (current_agent2_position[0] - self.agent2_last_position[0]) ** 2 + 
                    (current_agent2_position[1] - self.agent2_last_position[1]) ** 2
                )
                if distance_moved >= self.movement_threshold:
                    agent2_moved = True
                    self.agent2_last_position = current_agent2_position
                    
                    # Update agent2 following node
                    new_agent2_node = find_closest_node(self.map_graph.nodes, current_agent2_position)
                    if new_agent2_node is not None:
                        self.agent2_following_node_index = new_agent2_node
        
        return self.agent1_following_node_index, agent1_moved, None
    
    def calculate_inter_agent_distance(self):
        """
        Calculate distance between the two agents.
        
        Returns:
            float or None: Distance between agents in pixels, or None if positions not available
        """
        if self.agent1_current_position and self.agent2_current_position:
            return math.sqrt(
                (self.agent1_current_position[0] - self.agent2_current_position[0]) ** 2 + 
                (self.agent1_current_position[1] - self.agent2_current_position[1]) ** 2
            )
        return None
    
    def get_agent_threat_context(self):
        """
        Get comprehensive context for threat calculations.
        
        Returns:
            dict: Dictionary containing all relevant agent and environment information
        """
        return {
            'agent1_position': self.agent1_current_position,
            'agent2_position': self.agent2_current_position,
            'agent1_node': self.agent1_following_node_index,
            'agent2_node': self.agent2_following_node_index,
            'inter_agent_distance': self.calculate_inter_agent_distance(),
            'has_environment': self.environment is not None,
            'has_map_graph': self.map_graph is not None,
            'movement_threshold': self.movement_threshold
        }
    
    def can_perform_environment_visibility_check(self, position1, position2):
        """
        Check if a direct visibility check between two positions is possible using environment data.
        
        Args:
            position1: (x, y) tuple for first position
            position2: (x, y) tuple for second position
            
        Returns:
            bool: True if environment visibility check is possible
        """
        if not self.environment:
            return False
        
        # Additional checks could be added here for position validation
        return True
    
    def future_threat_assessment_placeholder(self, threat_parameters=None):
        """
        Placeholder method for future threat assessment implementations.
        
        Args:
            threat_parameters: Dictionary of parameters for threat assessment
            
        Returns:
            dict: Placeholder return structure for threat assessment results
        """
        return {
            'threat_level': 0.0,
            'assessment_type': 'placeholder',
            'agent_context': self.get_agent_threat_context(),
            'parameters': threat_parameters or {},
            'message': 'Future threat assessment functionality will be implemented here'
        }
    
    def get_current_node_index(self):
        """
        Get the current node index that agent1 is following.
        
        Returns:
            int or None: Current node index being followed by agent1
        """
        return self.agent1_following_node_index
    
    def get_agent1_node_index(self):
        """
        Get the current node index that agent1 is following.
        
        Returns:
            int or None: Current node index being followed by agent1
        """
        return self.agent1_following_node_index
    
    def get_agent2_node_index(self):
        """
        Get the current node index that agent2 is following.
        
        Returns:
            int or None: Current node index being followed by agent2
        """
        return self.agent2_following_node_index
    
    def get_agent_positions(self):
        """
        Get current positions of both agents.
        
        Returns:
            tuple: (agent1_position, agent2_position) where each is (x, y) or None if not set
        """
        return self.agent1_current_position, self.agent2_current_position
    
    def get_agent_last_position(self):
        """
        Get the last recorded agent1 position (for backward compatibility).
        
        Returns:
            tuple or None: (x, y) position of agent1's last recorded location
        """
        return self.agent1_last_position
    
    def get_agent1_last_position(self):
        """
        Get the last recorded agent1 position.
        
        Returns:
            tuple or None: (x, y) position of agent1's last recorded location
        """
        return self.agent1_last_position
    
    def get_agent2_last_position(self):
        """
        Get the last recorded agent2 position.
        
        Returns:
            tuple or None: (x, y) position of agent2's last recorded location
        """
        return self.agent2_last_position
    
    def get_environment(self):
        """
        Get the environment instance for visibility calculations.
        
        Returns:
            SimulationEnvironment or None: Environment instance if available
        """
        return self.environment
    
    def can_calculate_visibility(self):
        """
        Check if environment-based visibility calculations are possible.
        
        Returns:
            bool: True if environment data is available for visibility calculations
        """
        return self.environment is not None
    
    def reset_tracking(self):
        """
        Reset all agent tracking state.
        Useful when restarting or changing scenarios.
        """
        self.agent1_last_position = None
        self.agent1_following_node_index = None
        self.agent1_current_position = None
        
        self.agent2_last_position = None
        self.agent2_following_node_index = None
        self.agent2_current_position = None
        
        print("ThreatCalculator: All agent tracking state reset")
    
    def set_movement_threshold(self, threshold):
        """
        Update the movement threshold for detecting significant agent movement.
        
        Args:
            threshold: New movement threshold in pixels
        """
        self.movement_threshold = threshold
        print(f"ThreatCalculator: Movement threshold updated to {threshold} pixels")
    
    def calculate_agent_probabilities(self, agent, visibility_map, selected_node_index, time_horizon, agent_speed):
        """
        Calculate node probabilities for Agent 1 (O key functionality).
        
        This method encapsulates the probability overlay calculation that was previously
        done in the main simulation loop, providing a modular interface for probability
        calculations based on agent position, heading, visibility, and time horizon.
        
        Args:
            agent: Agent object with state [x, y, theta, ...]
            visibility_map: Dictionary mapping node indices to lists of visible node indices
            selected_node_index: Index of the currently selected node
            time_horizon: Look-ahead time in seconds
            agent_speed: Agent movement speed (typically LEADER_LINEAR_VEL from config)
            
        Returns:
            dict: Dictionary mapping node indices to probability values (0.0 to 1.0)
        """
        node_probabilities = {}
        
        # Validate inputs
        if not visibility_map or selected_node_index not in visibility_map:
            return node_probabilities
        
        if not self.map_graph or not self.map_graph.nodes:
            return node_probabilities
        
        # Extract agent state
        agent_x, agent_y = agent.state[0], agent.state[1]
        agent_theta = agent.state[2]  # Agent's heading angle
        
        # Calculate maximum reachable distance based on time horizon and agent speed
        max_reachable_distance = time_horizon * agent_speed
        
        # Use the reachable distance directly for probability calculation
        max_distance = max_reachable_distance
        
        # Get the list of nodes visible from the selected node
        visible_node_indices = set(visibility_map[selected_node_index])
        
        # Check if time horizon is too restrictive or too permissive
        reachable_count = 0
        total_visible_count = len(visible_node_indices)
        
        # First pass: count reachable nodes
        for i, node in enumerate(self.map_graph.nodes):
            if i in visible_node_indices:
                node_x, node_y = node
                distance = math.sqrt((node_x - agent_x)**2 + (node_y - agent_y)**2)
                if distance <= max_distance:
                    reachable_count += 1
        
        # Handle edge cases
        edge_case_handled = False
        if reachable_count == 0 and total_visible_count > 0:
            # Time horizon too low - no nodes are reachable
            # Set very small probabilities for closest visible nodes
            closest_distances = []
            for i, node in enumerate(self.map_graph.nodes):
                if i in visible_node_indices:
                    node_x, node_y = node
                    distance = math.sqrt((node_x - agent_x)**2 + (node_y - agent_y)**2)
                    closest_distances.append((distance, i))
            
            # Sort by distance and assign small probabilities to closest 3 nodes
            closest_distances.sort()
            for idx, (dist, node_idx) in enumerate(closest_distances[:3]):
                node_probabilities[node_idx] = 0.1 * (1.0 - idx * 0.3)  # 0.1, 0.07, 0.04
            edge_case_handled = True
        
        elif reachable_count == total_visible_count and total_visible_count > 1:
            # Time horizon too high - all visible nodes are reachable
            # Use the current max_distance but ensure probabilities are well distributed
            # Don't change max_distance here - let it use the time horizon-based value
            edge_case_handled = False  # Let normal calculation proceed with time horizon distance
        
        # Normal probability calculation (or edge case for "all reachable")
        if not edge_case_handled:
            for i, node in enumerate(self.map_graph.nodes):
                # Only calculate probability for nodes that are visible from the selected node
                if i in visible_node_indices:
                    node_x, node_y = node
                    distance = math.sqrt((node_x - agent_x)**2 + (node_y - agent_y)**2)
                    
                    if distance <= max_distance:
                        # Calculate distance-based probability (1.0 at agent position, 0.0 at max_distance)
                        distance_prob = max(0, 1.0 - (distance / max_distance))
                        
                        # Calculate heading angle bias
                        if distance > 1.0:  # Avoid division by zero for nodes very close to agent
                            # Calculate angle from agent to node
                            node_angle = math.atan2(node_y - agent_y, node_x - agent_x)
                            
                            # Calculate angular difference between agent heading and direction to node
                            angle_diff = abs(agent_theta - node_angle)
                            # Normalize to [0, π] (shortest angular distance)
                            if angle_diff > math.pi:
                                angle_diff = 2 * math.pi - angle_diff
                            
                            # Convert angle difference to heading bias factor
                            heading_bias = max(0, 1.0 - (angle_diff / math.pi))
                            
                            # Weight the heading bias (0.3 = 30% distance, 70% heading)
                            probability = distance_prob * (0.3 + 0.7 * heading_bias)
                        else:
                            # For very close nodes, use distance-only probability
                            probability = distance_prob
                        
                        if probability > 0.05:  # Only store significant probabilities
                            node_probabilities[i] = probability
        
        return node_probabilities

    def calculate_visibility_gaps(self, agent, visibility_range, selected_node_index=None):
        """
        Calculate visibility gaps for Agent 1 (B key functionality).
        
        This method encapsulates the visibility gaps calculation that was previously
        done in the main simulation loop, providing a modular interface for detecting
        visibility discontinuities and gap visualization.
        
        Args:
            agent: Agent object with state [x, y, theta, ...]
            visibility_range: Maximum range for ray casting (typically MAP_GRAPH_VISIBILITY_RANGE)
            selected_node_index: Optional node index to use as the origin (if None, uses agent position)
            
        Returns:
            tuple: (ray_endpoints, gap_lines) where gap_lines contains (start_point, end_point, gap_size)
                   gap_lines can be used for visualization with blue/violet color coding
        """
        if not self.environment:
            return [], []
        
        # Determine ray casting origin
        if selected_node_index is not None and self.map_graph and self.map_graph.nodes:
            if selected_node_index < len(self.map_graph.nodes):
                origin_x, origin_y = self.map_graph.nodes[selected_node_index]
            else:
                # Fallback to agent position if invalid node index
                origin_x, origin_y = agent.state[0], agent.state[1]
        else:
            # Use agent position as origin
            origin_x, origin_y = agent.state[0], agent.state[1]
        
        # Use the consolidated ray casting function
        ray_endpoints, gap_lines = cast_rays_and_find_gaps(
            origin_x, 
            origin_y, 
            visibility_range, 
            self.environment
        )
        
        return ray_endpoints, gap_lines

    def calculate_gap_visualization_data(self, gap_lines, origin_position):
        """
        Process gap lines for visualization with blue/violet color coding.
        
        This method processes the raw gap data from calculate_visibility_gaps()
        and prepares it for rendering with appropriate colors and line widths.
        
        Args:
            gap_lines: List of (start_point, end_point, gap_size) tuples from calculate_visibility_gaps()
            origin_position: (x, y) tuple of the ray casting origin
            
        Returns:
            list: List of dictionaries with visualization data:
                 [{'start_point': (x,y), 'end_point': (x,y), 'color': (r,g,b), 'line_width': int, 'circle_size': int}]
        """
        visualization_data = []
        
        for start_point, end_point, gap_size in gap_lines:
            # Determine gap orientation relative to clockwise ray casting
            start_dist = math.dist(origin_position, start_point)
            end_dist = math.dist(origin_position, end_point)
            
            # Classify gap type based on distance progression
            is_near_to_far = start_dist < end_dist  # Near point first, far point second
            is_far_to_near = start_dist > end_dist  # Far point first, near point second
            
            # Choose base color based on gap orientation
            if is_near_to_far:
                # Blue for near-to-far transitions (expanding gaps)
                base_color = (0, 100, 255) if gap_size > 150 else (50, 150, 255) if gap_size > 80 else (100, 200, 255)
            elif is_far_to_near:
                # Violet for far-to-near transitions (contracting gaps)
                base_color = (150, 0, 255) if gap_size > 150 else (180, 50, 255) if gap_size > 80 else (200, 100, 255)
            else:
                # Fallback color for equal distances (rare case)
                base_color = (100, 100, 100)
            
            # Determine line width based on gap size
            if gap_size > 150:
                line_width = 3
            elif gap_size > 80:
                line_width = 2
            else:
                line_width = 1
            
            # Calculate circle size for gap endpoints
            circle_size = max(2, min(5, int(gap_size / 30)))
            
            visualization_data.append({
                'start_point': start_point,
                'end_point': end_point,
                'color': base_color,
                'line_width': line_width,
                'circle_size': circle_size,
                'gap_size': gap_size,
                'is_expanding': is_near_to_far
            })
        
        return visualization_data

    def calculate_extended_probabilities_from_gaps(self, agent, gap_lines, node_probabilities, time_horizon, agent_speed):
        """
        Calculate extended probabilities from visibility gaps (part of B functionality).
        
        This method processes visibility gaps to compute extended probability propagation
        that was previously done inline in the main simulation loop.
        
        Args:
            agent: Agent object with state [x, y, theta, ...]
            gap_lines: List of (start_point, end_point, gap_size) tuples from calculate_visibility_gaps()
            node_probabilities: Dictionary of existing base node probabilities
            time_horizon: Look-ahead time in seconds
            agent_speed: Agent movement speed for reachability calculations
            
        Returns:
            dict: Dictionary mapping node indices to extended probability values
        """
        propagated_probabilities = {}
        
        if not gap_lines or not node_probabilities or not self.map_graph:
            return propagated_probabilities
        
        # Get agent position and max reachable distance
        agent_x, agent_y = agent.state[0], agent.state[1]
        max_reachable_distance = time_horizon * agent_speed
        
        # Get selected node position (ray casting origin)
        selected_node_index = self.get_current_node_index()
        if selected_node_index is None or selected_node_index >= len(self.map_graph.nodes):
            return propagated_probabilities
        
        selected_node = self.map_graph.nodes[selected_node_index]
        
        # Process gaps to compute extended probabilities
        for start_point, end_point, gap_size in gap_lines:
            # Only process significant gaps
            if gap_size < 50:
                continue
            
            # Determine gap orientation and near/far points
            start_dist = math.dist(selected_node, start_point)
            end_dist = math.dist(selected_node, end_point)
            
            # Determine near point (pivot point) and far point
            if start_dist < end_dist:
                near_point = start_point
                far_point = end_point
                is_blue_gap = True  # Near-to-far (expanding)
            else:
                near_point = end_point
                far_point = start_point
                is_blue_gap = False  # Far-to-near (contracting)
            
            # Calculate initial rod angle (along the gap line from near to far point)
            initial_rod_angle = math.atan2(far_point[1] - near_point[1], far_point[0] - near_point[0])
            
            # Calculate rod length: should extend from near point to the edge of reachability circle
            distance_to_rod_base = math.sqrt((near_point[0] - agent_x)**2 + (near_point[1] - agent_y)**2)
            
            # Calculate maximum rod length that stays within reachability circle
            if distance_to_rod_base >= max_reachable_distance:
                # Rod base is outside reachability circle, use minimal rod
                rod_length = 10
            else:
                # Rod length = remaining distance from base to circle edge
                remaining_distance_to_circle = max_reachable_distance - distance_to_rod_base
                
                # Also consider the original gap size as a constraint
                original_gap_rod_length = math.dist(near_point, far_point)
                
                # Use the smaller of the two: gap size or remaining circle distance
                rod_length = min(remaining_distance_to_circle, original_gap_rod_length)
                
                # Ensure minimum rod length for visibility
                rod_length = max(20, rod_length)
            
            max_rotation = math.pi / 4  # Maximum 45 degrees rotation
            
            # Determine rotation direction based on gap color
            if is_blue_gap:
                rotation_direction = -1  # Anticlockwise (counterclockwise)
            else:
                rotation_direction = 1   # Clockwise
            
            # SINGLE DIRECTION ROTATION: Rod pivots at near point, rotates in one direction only
            rod_base = near_point
            
            # Calculate the swept arc range
            sweep_start_angle = initial_rod_angle
            sweep_end_angle = initial_rod_angle + max_rotation * rotation_direction
            
            # Find probability at near point (rod base)
            near_point_prob = 0.0
            min_distance_near = float('inf')
            for node_idx, prob in node_probabilities.items():
                node_pos = self.map_graph.nodes[node_idx]
                dist_to_near = math.dist(node_pos, near_point)
                if dist_to_near < min_distance_near and dist_to_near < 50:  # Within 50px
                    min_distance_near = dist_to_near
                    near_point_prob = prob
            
            # Find probability at far point (gap end)
            far_point_prob = 0.0
            min_distance_far = float('inf')
            far_point_actual = (
                rod_base[0] + rod_length * math.cos(initial_rod_angle),
                rod_base[1] + rod_length * math.sin(initial_rod_angle)
            )
            for node_idx, prob in node_probabilities.items():
                node_pos = self.map_graph.nodes[node_idx]
                dist_to_far = math.dist(node_pos, far_point_actual)
                if dist_to_far < min_distance_far and dist_to_far < 50:  # Within 50px
                    min_distance_far = dist_to_far
                    far_point_prob = prob
            
            # If no probabilities found nearby, use default values
            if near_point_prob == 0.0 and far_point_prob == 0.0:
                near_point_prob = 0.3  # Default probability at near point
                far_point_prob = 0.1   # Lower probability at far point
            
            # OPTIMIZED GRID PROCESSING - reduced grid density for better performance
            angle_steps = 15  # Reduced from 40
            radius_steps = 8  # Reduced from 20
            
            # PRE-FILTER: Only consider nodes that are NOT already probabilized and in general area
            candidate_nodes = []
            arc_center_x = rod_base[0] + (rod_length / 2) * math.cos(initial_rod_angle)
            arc_center_y = rod_base[1] + (rod_length / 2) * math.sin(initial_rod_angle)
            filter_radius = rod_length + 50  # General area around the arc
            
            for j, node in enumerate(self.map_graph.nodes):
                if j not in node_probabilities:  # Skip nodes with existing probabilities
                    # Quick distance check to arc center
                    dx = node[0] - arc_center_x
                    dy = node[1] - arc_center_y
                    if dx*dx + dy*dy <= filter_radius*filter_radius:  # Avoid sqrt
                        candidate_nodes.append((j, node))
            
            # Process grid points with optimized node search
            search_radius = 25
            search_radius_sq = search_radius * search_radius  # Avoid sqrt in distance calc
            
            for a in range(angle_steps + 1):
                for r in range(1, radius_steps + 1):
                    angle_progress = a / angle_steps
                    radius_progress = r / radius_steps
                    current_angle = sweep_start_angle + angle_progress * (sweep_end_angle - sweep_start_angle)
                    current_radius = radius_progress * rod_length
                    
                    sweep_x = rod_base[0] + current_radius * math.cos(current_angle)
                    sweep_y = rod_base[1] + current_radius * math.sin(current_angle)
                    
                    # Boundary check (avoid sqrt)
                    dx_agent = sweep_x - agent_x
                    dy_agent = sweep_y - agent_y
                    dist_sq_agent = dx_agent*dx_agent + dy_agent*dy_agent
                    
                    if dist_sq_agent <= max_reachable_distance*max_reachable_distance:
                        # Calculate probability with stronger distance-based decay
                        base_probability = (1 - radius_progress) * near_point_prob + radius_progress * far_point_prob
                        
                        # Stronger angular decay - harder to reach at wider angles
                        angular_decay = max(0.1, 1.0 - (angle_progress * 0.8))
                        
                        # Distance-based decay - exponential decay with distance from rod base
                        distance_decay = max(0.2, 1.0 - (radius_progress ** 1.5) * 0.7)
                        
                        # Overall propagation decay
                        propagation_decay = 0.7
                        
                        final_probability = base_probability * angular_decay * distance_decay * propagation_decay
                        
                        if final_probability > 0.03:  # Only store significant probabilities
                            # FAST node search - only through pre-filtered candidates
                            closest_node_idx = None
                            closest_distance_sq = search_radius_sq
                            
                            for node_idx, node in candidate_nodes:
                                dx = node[0] - sweep_x
                                dy = node[1] - sweep_y
                                dist_sq = dx*dx + dy*dy  # Avoid sqrt
                                
                                if dist_sq < closest_distance_sq:
                                    closest_distance_sq = dist_sq
                                    closest_node_idx = node_idx
                            
                            # Assign probability
                            if closest_node_idx is not None:
                                if closest_node_idx in propagated_probabilities:
                                    propagated_probabilities[closest_node_idx] = max(
                                        propagated_probabilities[closest_node_idx], final_probability)
                                else:
                                    propagated_probabilities[closest_node_idx] = final_probability
        
        return propagated_probabilities

    def get_tracking_stats(self):
        """
        Get current tracking statistics and status for both agents.
        
        Returns:
            dict: Dictionary containing comprehensive tracking information
        """
        return {
            'movement_threshold': self.movement_threshold,
            'agent1_current_node_index': self.agent1_following_node_index,
            'agent1_last_position': self.agent1_last_position,
            'agent1_current_position': self.agent1_current_position,
            'agent2_current_node_index': self.agent2_following_node_index,
            'agent2_last_position': self.agent2_last_position,
            'agent2_current_position': self.agent2_current_position,
            'has_map_graph': self.map_graph is not None,
            'has_environment': self.environment is not None,
            'node_count': len(self.map_graph.nodes) if self.map_graph else 0,
            'can_calculate_visibility': self.can_calculate_visibility()
        }

    # OPTIMIZATION: Optimizer access methods
    def get_agent2_optimizer(self):
        """
        Get the Agent2ComputationOptimizer instance.
        
        Returns:
            Agent2ComputationOptimizer: The optimizer instance for selective computation
        """
        return self.agent2_optimizer
    
    def get_fast_math(self):
        """
        Get the FastMathOptimizations instance.
        
        Returns:
            FastMathOptimizations: The fast math optimizer instance
        """
        return self.fast_math
    
    def get_trig_lookup(self):
        """
        Get the TrigLookupTable instance.
        
        Returns:
            TrigLookupTable: The trigonometric lookup table instance
        """
        return self.trig_lookup
    
    def get_optimizer_stats(self):
        """
        Get performance statistics from all optimizers.
        
        Returns:
            dict: Dictionary containing optimizer performance statistics
        """
        return {
            'agent2_optimizer': self.agent2_optimizer.get_performance_stats(),
            'trig_lookup_precision': self.trig_lookup.precision,
            'fast_math_available': True
        }

    def calculate_agent2_sweep_probabilities(self, agent2, gap_lines, time_horizon, agent_speed, 
                                           show_threat_classification=False, rod_gap_points=None, 
                                           rod_threat_stats=None, DEFAULT_VISION_RANGE=800):
        """
        Calculate Agent 2 instantaneous sweep-based probability assignment (modularized computation).
        
        This method encapsulates the complex Agent 2 computation phase that was previously
        done inline in the main simulation loop. It handles dynamic range calculation,
        selective computation optimization, vectorized gap processing, and rod-based threat tracking.
        
        Args:
            agent2: Agent 2 object with state [x, y, theta, ...]
            gap_lines: List of (start_point, end_point, gap_size) tuples from visibility gaps
            time_horizon: Look-ahead time in seconds for dynamic range calculation
            agent_speed: Agent movement speed (typically LEADER_LINEAR_VEL)
            show_threat_classification: Whether to enable threat classification mode
            rod_gap_points: Dictionary to store rod-to-gap-point mapping (for threat classification)
            rod_threat_stats: Dictionary to store rod threat statistics (for threat classification)
            DEFAULT_VISION_RANGE: Maximum vision range cap (default 800px)
            
        Returns:
            dict: {
                'agent2_gap_probabilities': dict mapping node indices to probabilities,
                'agent2_rod_node_mapping': dict mapping node indices to list of (rod_id, probability),
                'computation_stats': dict with performance statistics,
                'computation_skipped': bool indicating if computation was optimized away
            }
        """
        if not gap_lines or not self.map_graph or not self.agent1_current_position:
            return {
                'agent2_gap_probabilities': {},
                'agent2_rod_node_mapping': {},
                'computation_stats': {'message': 'Insufficient data for computation'},
                'computation_skipped': False
            }
        
        # Extract agent positions and calculate dynamic range
        agent1_x, agent1_y = self.agent1_current_position
        agent2_x, agent2_y = agent2.state[0], agent2.state[1]
        agent2_theta = agent2.state[2]
        
        # DYNAMIC RANGE CALCULATION: Agent 1 reachability + inter-agent distance (max 800px)
        agent1_reachability = time_horizon * agent_speed  # Agent 1's reachable distance
        inter_agent_distance = math.dist((agent1_x, agent1_y), (agent2_x, agent2_y))
        
        # Calculate dynamic vision range: reachability + distance between agents, capped at 800
        agent2_dynamic_vision_range = min(agent1_reachability + inter_agent_distance, DEFAULT_VISION_RANGE)
        
        # OPTIMIZATION: Selective computation - check if recomputation is needed
        current_agent2_pos = (agent2_x, agent2_y)
        current_agent2_angle = math.atan2(math.sin(agent2_theta), math.cos(agent2_theta))  # Normalize angle
        
        # Initialize computation variables and data storage
        computation_stats = {
            'total_gaps_processed': 0,
            'total_nodes_checked': 0,
            'total_probabilities_assigned': 0,
            'computation_time_ms': 0,
            'optimization_skipped': False
        }
        
        # Check if we can skip computation based on agent state changes
        if not self.agent2_optimizer.needs_recomputation(current_agent2_pos, current_agent2_angle):
            # Skip computation, use cached result
            cached_probabilities = self.agent2_optimizer.skip_computation()
            
            computation_stats.update({
                'total_probabilities_assigned': len(cached_probabilities),
                'optimization_skipped': True,
                'message': 'Used cached computation result'
            })
            
            return {
                'agent2_gap_probabilities': cached_probabilities,
                'agent2_rod_node_mapping': {},  # Rod mapping not cached currently
                'computation_stats': computation_stats,
                'computation_skipped': True
            }
        
        # Perform full computation
        computation_start = time.perf_counter()
        
        # Initialize gap probability dictionary and rod tracking
        agent2_gap_probabilities = {}
        agent2_rod_node_mapping = {}   # Track which rod created which node probabilities
        
        # INSTANTANEOUS SWEEP: Process all angles at once to assign gap probabilities
        rod_id = 0  # Track rod ID for threat classification
        
        for start_point, end_point, gap_size in gap_lines:
            # Only process significant gaps
            if gap_size < 50:
                continue
            
            computation_stats['total_gaps_processed'] += 1
            
            # Determine gap orientation relative to clockwise ray casting
            start_dist = math.dist((agent2_x, agent2_y), start_point)
            end_dist = math.dist((agent2_x, agent2_y), end_point)
            
            # Determine near point (pivot point) and far point
            if start_dist < end_dist:
                near_point = start_point
                far_point = end_point
                is_cyan_gap = True  # Near-to-far (expanding)
            else:
                near_point = end_point
                far_point = start_point
                is_cyan_gap = False  # Far-to-near (contracting)
            
            # Store rod-to-gap-point mapping for threat classification
            if show_threat_classification and rod_gap_points is not None and rod_threat_stats is not None:
                rod_gap_points[rod_id] = near_point
                rod_threat_stats[rod_id] = {'probabilities': [], 'mean_prob': 0.0}
            
            # Calculate initial rod angle (along the gap line from near to far point)
            initial_rod_angle = math.atan2(far_point[1] - near_point[1], far_point[0] - near_point[0])
            
            # Calculate rod length based on the gap size
            original_gap_rod_length = math.dist(near_point, far_point)
            rod_length = max(20, original_gap_rod_length)
            
            max_rotation = math.pi / 4  # Maximum 45 degrees rotation
            
            # Determine rotation direction based on gap color
            if is_cyan_gap:
                rotation_direction = -1  # Anticlockwise (counterclockwise)
                gap_color = (0, 200, 255)  # Cyan tone
            else:
                rotation_direction = 1   # Clockwise
                gap_color = (0, 240, 180)  # Green-cyan tone
            
            rod_base = near_point
            
            # INSTANTANEOUS SWEEP: Calculate probabilities for all angles at once
            sweep_start_angle = initial_rod_angle
            sweep_end_angle = initial_rod_angle + max_rotation * rotation_direction
            total_sweep_angle = abs(sweep_end_angle - sweep_start_angle)
            
            # VECTORIZED OPTIMIZATION: Replace nested loops with NumPy operations
            # This reduces computation from O(angles×nodes) to O(1) vectorized operations
            
            # Pre-filter nodes within vision range once for this gap
            if self.map_graph and self.map_graph.nodes:
                # Convert all nodes to NumPy arrays for vectorized operations
                all_nodes = np.array(self.map_graph.nodes)
                
                # OPTIMIZATION: Enhanced spatial filtering with multiple criteria
                agent2_pos = np.array([agent2_x, agent2_y])
                
                # Primary filter: Distance-based (use fast squared distance)
                distances_squared = np.sum((all_nodes - agent2_pos)**2, axis=1)
                vision_range_squared = agent2_dynamic_vision_range * agent2_dynamic_vision_range
                within_range_mask = distances_squared <= vision_range_squared
                
                # Secondary filter: Gap-relevance filter (only consider nodes near gap direction)
                if len(all_nodes) > 100:  # Only apply for large node sets
                    # Calculate gap center direction
                    gap_center = np.array([(near_point[0] + far_point[0]) / 2, 
                                          (near_point[1] + far_point[1]) / 2])
                    gap_direction = gap_center - agent2_pos
                    gap_direction_norm = np.linalg.norm(gap_direction)
                    
                    if gap_direction_norm > 0:
                        gap_direction = gap_direction / gap_direction_norm
                        
                        # Calculate angle between agent->node and agent->gap vectors
                        node_directions = all_nodes - agent2_pos
                        node_distances = np.linalg.norm(node_directions, axis=1)
                        
                        # Avoid division by zero
                        valid_nodes_mask = node_distances > 1e-6
                        
                        if np.any(valid_nodes_mask):
                            node_directions_norm = node_directions[valid_nodes_mask] / node_distances[valid_nodes_mask, np.newaxis]
                            
                            # Dot product for angle calculation
                            dot_products = np.dot(node_directions_norm, gap_direction)
                            
                            # Keep nodes within ±60 degrees of gap direction
                            angle_threshold = math.cos(math.pi / 3)  # 60 degrees
                            gap_relevant_mask = np.zeros(len(all_nodes), dtype=bool)
                            gap_relevant_mask[valid_nodes_mask] = dot_products >= angle_threshold
                            
                            # Combine filters
                            within_range_mask = within_range_mask & gap_relevant_mask
                
                within_range_indices = np.where(within_range_mask)[0]
                within_range_nodes = all_nodes[within_range_mask]
                
                # Track statistics
                computation_stats['total_nodes_checked'] += len(within_range_nodes)
                
                if len(within_range_nodes) > 0:
                    # Generate all sweep angles at once
                    num_sweep_angles = 20
                    
                    # Vectorized angle generation
                    angle_steps = np.linspace(0, 1, num_sweep_angles + 1)
                    sweep_angles = sweep_start_angle + angle_steps * (sweep_end_angle - sweep_start_angle)
                    
                    # Vectorized gap probabilities (0.8 to 1.0 range)
                    base_gap_probability = 0.8
                    gap_probabilities = base_gap_probability + (angle_steps * 0.2)
                    
                    # Vectorized rod end positions for all angles
                    rod_ends_x = rod_base[0] + rod_length * np.cos(sweep_angles)
                    rod_ends_y = rod_base[1] + rod_length * np.sin(sweep_angles)
                    
                    # For each angle, calculate distances from all nodes to the rod line
                    bar_width = 15.0  # Wider sweep for probability assignment
                    
                    # Initialize result arrays
                    node_max_probabilities = np.zeros(len(within_range_nodes))
                    
                    # Process all angles and nodes simultaneously
                    for angle_idx, (angle, gap_prob, rod_end_x, rod_end_y) in enumerate(
                        zip(sweep_angles, gap_probabilities, rod_ends_x, rod_ends_y)
                    ):
                        # Vectorized rod line calculations
                        rod_base_arr = np.array(rod_base)
                        rod_end_arr = np.array([rod_end_x, rod_end_y])
                        
                        # Rod vector
                        rod_vec = rod_end_arr - rod_base_arr
                        rod_length_sq = np.dot(rod_vec, rod_vec)
                        
                        if rod_length_sq > 0:
                            # Vectors from rod start to all nodes
                            node_vecs = within_range_nodes - rod_base_arr
                            
                            # Vectorized projection onto rod line
                            projections = np.dot(node_vecs, rod_vec) / rod_length_sq
                            projections = np.clip(projections, 0, 1)
                            
                            # Vectorized closest points on rod line
                            closest_points = rod_base_arr + projections[:, np.newaxis] * rod_vec
                            
                            # Vectorized distances from nodes to rod line
                            distance_vectors = within_range_nodes - closest_points
                            distances_to_rod = np.linalg.norm(distance_vectors, axis=1)
                            
                            # Boolean mask for nodes within bar width
                            within_bar_mask = distances_to_rod <= bar_width
                            
                            # Update maximum probabilities for affected nodes
                            node_max_probabilities[within_bar_mask] = np.maximum(
                                node_max_probabilities[within_bar_mask], gap_prob
                            )
                    
                    # Assign final probabilities to nodes
                    significant_prob_mask = node_max_probabilities > 0
                    significant_indices = within_range_indices[significant_prob_mask]
                    significant_probs = node_max_probabilities[significant_prob_mask]
                    
                    for idx, prob in zip(significant_indices, significant_probs):
                        if idx in agent2_gap_probabilities:
                            agent2_gap_probabilities[idx] = max(agent2_gap_probabilities[idx], prob)
                        else:
                            agent2_gap_probabilities[idx] = prob
                        
                        # Track rod source for threat classification
                        if idx not in agent2_rod_node_mapping:
                            agent2_rod_node_mapping[idx] = []
                        agent2_rod_node_mapping[idx].append((rod_id, prob))
                        
                        computation_stats['total_probabilities_assigned'] += 1
            
            # Increment rod ID for next gap
            rod_id += 1
        
        # Record computation time
        computation_end = time.perf_counter()
        computation_stats['computation_time_ms'] = (computation_end - computation_start) * 1000
        
        # Update optimizer state after computation
        self.agent2_optimizer.update_state(current_agent2_pos, current_agent2_angle, agent2_gap_probabilities)
        
        return {
            'agent2_gap_probabilities': agent2_gap_probabilities,
            'agent2_rod_node_mapping': agent2_rod_node_mapping,
            'computation_stats': computation_stats,
            'computation_skipped': False
        }

    def calculate_agent2_probability_overlay_computation(self, agent2, visibility_map, time_horizon, agent_speed, 
                                                       agent2_gap_probabilities=None, DEFAULT_VISION_RANGE=800):
        """
        Calculate Agent 2 probability overlay computation phase (J key functionality - computation part).
        
        This method encapsulates the computation phase from the "elif show_agent2_probability_overlay 
        and not show_combined_probability_mode:" block, separating computation from visualization 
        for better modularity. Handles dynamic range calculation, visibility-based probabilities,
        and gap probability integration.
        
        Args:
            agent2: Agent 2 object with state [x, y, theta, ...]
            visibility_map: Dictionary mapping node indices to lists of visible node indices
            time_horizon: Look-ahead time in seconds for dynamic range calculation
            agent_speed: Agent movement speed (typically LEADER_LINEAR_VEL)
            agent2_gap_probabilities: Optional dictionary of gap-based probabilities to integrate
            DEFAULT_VISION_RANGE: Maximum vision range cap (default 800px)
            
        Returns:
            dict: {
                'agent2_probability_node_data': list of probability node visualization data,
                'agent2_range_circle_data': dict with range circle visualization data,
                'agent2_node_probabilities': dict mapping node indices to final probabilities,
                'computation_stats': dict with performance statistics
            }
        """
        # Initialize computation result structure
        result = {
            'agent2_probability_node_data': [],
            'agent2_range_circle_data': {},
            'agent2_node_probabilities': {},
            'computation_stats': {
                'visibility_nodes_processed': 0,
                'gap_probabilities_integrated': 0,
                'final_probabilities_assigned': 0,
                'computation_time_ms': 0,
                'dynamic_range': 0
            }
        }
        
        if not self.map_graph or not self.agent1_current_position:
            result['computation_stats']['message'] = 'Insufficient data for computation'
            return result
        
        computation_start = time.perf_counter()
        
        # Calculate agent position and basic parameters
        agent2_x, agent2_y = agent2.state[0], agent2.state[1]
        
        # DYNAMIC RANGE CALCULATION: Agent 1 reachability + inter-agent distance (max 800px)
        agent1_x, agent1_y = self.agent1_current_position  # Agent 1 position
        agent1_reachability = time_horizon * agent_speed  # Agent 1's reachable distance
        inter_agent_distance = math.dist((agent1_x, agent1_y), (agent2_x, agent2_y))
        
        # Calculate dynamic vision range: reachability + distance between agents, capped at 800
        dynamic_vision_range = min(agent1_reachability + inter_agent_distance, DEFAULT_VISION_RANGE)
        agent2_vision_range = dynamic_vision_range
        
        result['computation_stats']['dynamic_range'] = agent2_vision_range
        
        # Store range circle data for later visualization
        result['agent2_range_circle_data'] = {
            'position': (int(agent2_x), int(agent2_y)),
            'radius': agent2_vision_range,
            'color': (0, 200, 200),
            'line_width': 2
        }
        
        # Initialize agent 2 probability list/dictionary (similar to agent 1)
        agent2_node_probabilities = {}
        
        # Get visibility data for agent 2 if available
        if visibility_map and self.map_graph:
            # Start timing visibility-based probability calculation
            visibility_calculation_start = time.perf_counter()
            
            # Find closest node to agent 2's position
            agent2_pos = (agent2_x, agent2_y)
            agent2_node_index = find_closest_node(self.map_graph.nodes, agent2_pos)
            
            if agent2_node_index is not None:
                # Get visible nodes from the visibility map
                visible_node_indices = set(visibility_map[agent2_node_index])
                
                # Calculate and store probabilities for all map graph nodes within range
                for i, node in enumerate(self.map_graph.nodes):
                    node_x, node_y = node
                    
                    # Calculate distance from agent 2 to this node
                    dist_to_node = math.dist((agent2_x, agent2_y), (node_x, node_y))
                    
                    # Only process nodes within the dynamic range
                    if dist_to_node <= agent2_vision_range:
                        # Check if this node is actually visible to agent 2
                        if i in visible_node_indices:
                            # Node is visible: use fixed base probability (only store if > 0)
                            from multitrack.utils.config import AGENT2_BASE_PROBABILITY
                            if AGENT2_BASE_PROBABILITY > 0:
                                agent2_node_probabilities[i] = AGENT2_BASE_PROBABILITY
                                result['computation_stats']['visibility_nodes_processed'] += 1
                        # Note: nodes not visible or with 0 probability are not stored (optimization)
            
            # End timing visibility calculation
            visibility_calculation_end = time.perf_counter()
            visibility_calculation_time = visibility_calculation_end - visibility_calculation_start
            
        else:
            # Fallback: if no visibility data available, show uniform probability (original behavior)
            fallback_start = time.perf_counter()
            
            for i, node in enumerate(self.map_graph.nodes):
                node_x, node_y = node
                
                # Calculate distance from agent 2 to this node
                dist_to_node = math.dist((agent2_x, agent2_y), (node_x, node_y))
                
                # Simple uniform probability within range (fallback behavior)
                if dist_to_node <= agent2_vision_range:
                    # Use configured base probability for fallback (only store if > 0)
                    from multitrack.utils.config import AGENT2_BASE_PROBABILITY
                    if AGENT2_BASE_PROBABILITY > 0:
                        agent2_node_probabilities[i] = AGENT2_BASE_PROBABILITY
                        result['computation_stats']['visibility_nodes_processed'] += 1
            
            # End timing fallback calculation
            fallback_end = time.perf_counter()
            visibility_calculation_time = fallback_end - fallback_start
        
        # INTEGRATE GAP PROBABILITIES: Visibility-based probabilities override gap-based ones
        # This happens after visibility calculations but before drawing
        if agent2_gap_probabilities:
            for node_index, gap_prob in agent2_gap_probabilities.items():
                if node_index not in agent2_node_probabilities:
                    # Only gap probability exists (no visibility): use gap probability
                    agent2_node_probabilities[node_index] = gap_prob
                    result['computation_stats']['gap_probabilities_integrated'] += 1
                # If visibility probability exists, it overrides gap probability (no action needed)
        
        # Store final probabilities for return
        result['agent2_node_probabilities'] = agent2_node_probabilities
        
        # Process all probability nodes and calculate visualization data
        for i, probability in agent2_node_probabilities.items():
            # All stored probabilities are > 0 by design (optimization)
            node_x, node_y = self.map_graph.nodes[i]
            
            # Color blending scheme: probability determines mix between pink and green
            # Low probability (e.g., 0.1): mostly pink (0.9) + little green (0.1)
            # High probability (e.g., 0.9): mostly green (0.9) + little pink (0.1)
            
            # Define base colors
            pink_color = (255, 105, 180)  # Hot pink
            green_color = (0, 255, 100)   # Bright green
            
            # Use probability directly for color blending (0.0 to 1.0)
            green_weight = probability  # How much green
            pink_weight = 1.0 - probability  # How much pink
            
            # Blend the colors
            red = int(pink_color[0] * pink_weight + green_color[0] * green_weight)
            green = int(pink_color[1] * pink_weight + green_color[1] * green_weight)
            blue = int(pink_color[2] * pink_weight + green_color[2] * green_weight)
            
            color = (red, green, blue)
            
            # Make circles smaller (reduced from 3-8 to 2-4)
            min_size, max_size = 2, 4
            node_size = int(min_size + probability * (max_size - min_size))
            
            # Store node visualization data
            node_data = {
                'position': (node_x, node_y),
                'color': color,
                'size': node_size,
                'probability': probability,
                'glow_effect': probability > 0.7
            }
            
            # Add glow data if needed
            if probability > 0.7:
                node_data['glow_color'] = (0, 255, 150)  # Green-cyan glow
                node_data['glow_size'] = node_size + 1  # Smaller glow
            
            result['agent2_probability_node_data'].append(node_data)
            result['computation_stats']['final_probabilities_assigned'] += 1
        
        # Record computation time
        computation_end = time.perf_counter()
        result['computation_stats']['computation_time_ms'] = (computation_end - computation_start) * 1000
        
        return result

    def calculate_combined_probability_computation(self, agent1, agent2, node_probabilities, visibility_map, 
                                                 time_horizon, agent_speed, threat_threshold=0.1, 
                                                 show_threat_classification=False, propagated_probabilities=None, 
                                                 agent2_gap_probabilities=None, DEFAULT_VISION_RANGE=800):
        """
        Calculate Combined Probability Mode computation phase (M key functionality - computation part).
        
        This method encapsulates the computation phase that was previously done inline in the 
        "COMBINED PROBABILITY MODE: Multiply Agent 1 and Agent 2 probabilities" section,
        separating computation from visualization for better modularity.
        
        Args:
            agent1: Agent 1 object with state [x, y, theta, ...]
            agent2: Agent 2 object with state [x, y, theta, ...]
            node_probabilities: Dictionary of Agent 1 base node probabilities
            visibility_map: Dictionary mapping node indices to lists of visible node indices
            time_horizon: Look-ahead time in seconds for dynamic range calculation
            agent_speed: Agent movement speed (typically LEADER_LINEAR_VEL)
            threat_threshold: Probability threshold for threat classification (default 0.1)
            show_threat_classification: Whether to enable threat classification mode
            propagated_probabilities: Optional dict of Agent 1 extended probabilities from gaps
            agent2_gap_probabilities: Optional dict of Agent 2 gap-based probabilities
            DEFAULT_VISION_RANGE: Maximum vision range cap (default 800px)
            
        Returns:
            dict: {
                'combined_threat_nodes_data': list of threat node visualization data (when classification enabled),
                'combined_probability_nodes_data': list of combined probability node data (when classification disabled),
                'combined_visibility_ranges_data': list of visibility range circle data,
                'combined_info_display_data': dict with info panel display data,
                'agent1_probabilities': dict of final Agent 1 probabilities,
                'agent2_probabilities': dict of final Agent 2 probabilities,
                'combined_probabilities': dict of final combined probabilities,
                'computation_stats': dict with performance statistics
            }
        """
        # Initialize computation result structure
        result = {
            'combined_threat_nodes_data': [],
            'combined_probability_nodes_data': [],
            'combined_visibility_ranges_data': [],
            'combined_info_display_data': {},
            'agent1_probabilities': {},
            'agent2_probabilities': {},
            'combined_probabilities': {},
            'computation_stats': {
                'agent1_nodes_processed': 0,
                'agent2_nodes_processed': 0,
                'combined_nodes_processed': 0,
                'threat_nodes_classified': 0,
                'computation_time_ms': 0
            }
        }
        
        if not self.map_graph or not self.agent1_current_position or not self.agent2_current_position:
            result['computation_stats']['message'] = 'Insufficient data for computation'
            return result
        
        computation_start = time.perf_counter()
        
        # Extract agent positions
        agent1_x, agent1_y = agent1.state[0], agent1.state[1]
        agent2_x, agent2_y = agent2.state[0], agent2.state[1]
        
        # ===== PROCESS AGENT 1 PROBABILITIES =====
        agent1_probabilities = {}
        if node_probabilities:
            # Start with base reachability probabilities
            for node_idx, base_prob in node_probabilities.items():
                agent1_probabilities[node_idx] = base_prob
                result['computation_stats']['agent1_nodes_processed'] += 1
            
            # Merge with extended probabilities from rotating rods if available (optional parameter)
            if propagated_probabilities:
                for node_idx, extended_prob in propagated_probabilities.items():
                    if node_idx in agent1_probabilities:
                        # Use maximum of base and extended probability (union of reachable sets)
                        agent1_probabilities[node_idx] = max(agent1_probabilities[node_idx], extended_prob)
                    else:
                        # Add new extended probability locations
                        agent1_probabilities[node_idx] = extended_prob
                        result['computation_stats']['agent1_nodes_processed'] += 1
        
        result['agent1_probabilities'] = agent1_probabilities
        
        # ===== PROCESS AGENT 2 PROBABILITIES =====
        agent2_probabilities = {}
        if visibility_map and self.map_graph:
            # DYNAMIC RANGE CALCULATION: Agent 1 reachability + inter-agent distance (max 800px)
            agent1_reachability = time_horizon * agent_speed  # Agent 1's reachable distance
            inter_agent_distance = math.dist((agent1_x, agent1_y), (agent2_x, agent2_y))
            
            # Calculate dynamic vision range: reachability + distance between agents, capped at 800
            agent2_vision_range = min(agent1_reachability + inter_agent_distance, DEFAULT_VISION_RANGE)
            
            # Calculate Agent 2 visibility-based probabilities
            agent2_pos = (agent2_x, agent2_y)
            agent2_node_index = find_closest_node(self.map_graph.nodes, agent2_pos)
            
            if agent2_node_index is not None:
                visible_node_indices = set(visibility_map[agent2_node_index])
                
                for i, node in enumerate(self.map_graph.nodes):
                    node_x, node_y = node
                    dist_to_node = math.dist((agent2_x, agent2_y), (node_x, node_y))
                    
                    if dist_to_node <= agent2_vision_range:
                        if i in visible_node_indices:
                            from multitrack.utils.config import AGENT2_BASE_PROBABILITY
                            if AGENT2_BASE_PROBABILITY > 0:
                                agent2_probabilities[i] = AGENT2_BASE_PROBABILITY
                                result['computation_stats']['agent2_nodes_processed'] += 1
            
            # Merge with gap probabilities if available (optional parameter)
            if agent2_gap_probabilities:
                for node_index, gap_prob in agent2_gap_probabilities.items():
                    if node_index not in agent2_probabilities:
                        agent2_probabilities[node_index] = gap_prob
                        result['computation_stats']['agent2_nodes_processed'] += 1
        
        result['agent2_probabilities'] = agent2_probabilities
        
        # ===== CALCULATE COMBINED PROBABILITIES =====
        all_node_indices = set(agent1_probabilities.keys()) | set(agent2_probabilities.keys())
        combined_probabilities = {}
        
        # Initialize threat classification data structures
        rod_colors = [
            (255, 0, 0),    # Red - Rod 1
            (0, 255, 0),    # Green - Rod 2  
            (0, 0, 255),    # Blue - Rod 3
            (255, 165, 0),  # Orange - Rod 4
            (255, 0, 255),  # Magenta - Rod 5
            (0, 255, 255),  # Cyan - Rod 6
            (255, 255, 0),  # Yellow - Rod 7
            (128, 0, 128),  # Purple - Rod 8
        ]
        
        # Track rod-based threats from Agent 2 gap probabilities
        node_rod_sources = {}  # Map node_index -> list of (rod_id, probability)
        if show_threat_classification and agent2_gap_probabilities:
            # Assign rod IDs based on gap processing order
            rod_id = 0
            for node_index, gap_prob in agent2_gap_probabilities.items():
                if node_index not in node_rod_sources:
                    node_rod_sources[node_index] = []
                
                current_rod_id = rod_id % len(rod_colors)
                node_rod_sources[node_index].append((current_rod_id, gap_prob))
                rod_id += 1
        
        # Process all combined probabilities and classification
        for node_idx in all_node_indices:
            prob1 = agent1_probabilities.get(node_idx, 0.0)
            prob2 = agent2_probabilities.get(node_idx, 0.0)
            
            # Multiply probabilities (both must exist and be non-zero)
            if prob1 > 0 and prob2 > 0:
                combined_prob = prob1 * prob2
                result['computation_stats']['combined_nodes_processed'] += 1
                
                # THREAT CLASSIFICATION: Check if above threshold (only when enabled)
                if show_threat_classification and combined_prob >= threat_threshold:
                    # This is a threat node - classify by rod source
                    rod_sources = node_rod_sources.get(node_idx, [])
                    node_x, node_y = self.map_graph.nodes[node_idx]
                    
                    if rod_sources:
                        # Node has rod-based sources - classify by dominant rod
                        dominant_rod = max(rod_sources, key=lambda x: x[1])  # Highest probability rod
                        rod_id = dominant_rod[0]
                        base_color = rod_colors[rod_id % len(rod_colors)]
                        
                        # Intensity based on probability (above threshold)
                        prob_normalized = min(1.0, (combined_prob - threat_threshold) / (1.0 - threat_threshold))
                        intensity = 0.5 + 0.5 * prob_normalized  # 50% to 100% intensity
                        
                        threat_color = tuple(int(c * intensity) for c in base_color)
                        node_size = 6 + int(prob_normalized * 4)  # 6-10 pixels
                        
                        # Store threat node data
                        threat_data = {
                            'position': (node_x, node_y),
                            'color': threat_color,
                            'size': node_size,
                            'type': 'rod_based',
                            'rod_id': rod_id,
                            'all_rods': rod_sources
                        }
                        
                        # Add overlapping rod indicators if multiple rods
                        if len(rod_sources) > 1:
                            threat_data['overlapping_rods'] = []
                            angle_step = 2 * math.pi / len(rod_sources)
                            for i, (rod_id_inner, _) in enumerate(rod_sources):
                                angle = i * angle_step
                                offset_x = int(node_x + (node_size + 3) * math.cos(angle))
                                offset_y = int(node_y + (node_size + 3) * math.sin(angle))
                                rod_color = rod_colors[rod_id_inner % len(rod_colors)]
                                threat_data['overlapping_rods'].append({
                                    'position': (offset_x, offset_y),
                                    'color': rod_color,
                                    'size': 2
                                })
                        
                        result['combined_threat_nodes_data'].append(threat_data)
                        result['computation_stats']['threat_nodes_classified'] += 1
                    else:
                        # Threat from visibility only (no rod involvement) - use white/gray
                        prob_normalized = min(1.0, (combined_prob - threat_threshold) / (1.0 - threat_threshold))
                        intensity = int(128 + 127 * prob_normalized)  # Gray to white
                        threat_color = (intensity, intensity, intensity)
                        
                        node_size = 6 + int(prob_normalized * 4)
                        
                        # Store visibility-only threat data
                        result['combined_threat_nodes_data'].append({
                            'position': (node_x, node_y),
                            'color': threat_color,
                            'size': node_size,
                            'type': 'visibility_only'
                        })
                        result['computation_stats']['threat_nodes_classified'] += 1
                
                # Store for regular combined probability visualization (if enabled)
                # Use consistent threshold for both visualization and threat classification
                if combined_prob >= threat_threshold:
                    combined_probabilities[node_idx] = combined_prob
        
        result['combined_probabilities'] = combined_probabilities
        
        # ===== PROCESS COMBINED PROBABILITY NODES (when threat classification disabled) =====
        if not show_threat_classification:
            for node_idx, combined_prob in combined_probabilities.items():
                node_x, node_y = self.map_graph.nodes[node_idx]
                
                # Purple-yellow color scheme: purple (low) to yellow (high)
                # Low probability: dark purple, High probability: bright yellow
                purple_color = (128, 0, 128)  # Dark purple
                yellow_color = (255, 255, 0)  # Bright yellow
                
                # Normalize combined probability for color blending (0.0 to 1.0)
                # Since we multiply probabilities, the range is typically much smaller
                # Use a scaling factor to make the visualization more visible
                display_prob = min(1.0, combined_prob * 10)  # Scale up for better visibility
                
                # Color blending
                yellow_weight = display_prob
                purple_weight = 1.0 - display_prob
                
                red = int(purple_color[0] * purple_weight + yellow_color[0] * yellow_weight)
                green = int(purple_color[1] * purple_weight + yellow_color[1] * yellow_weight)
                blue = int(purple_color[2] * purple_weight + yellow_color[2] * yellow_weight)
                
                color = (red, green, blue)
                
                # Node size based on combined probability (larger for higher probability)
                min_size, max_size = 3, 7  # Slightly larger than individual modes
                node_size = int(min_size + display_prob * (max_size - min_size))
                
                # Store combined probability node data
                node_data = {
                    'position': (node_x, node_y),
                    'color': color,
                    'size': node_size,
                    'probability': combined_prob,
                    'glow_effect': display_prob > 0.8
                }
                
                # Add glow effect data for very high combined probabilities
                if display_prob > 0.8:
                    node_data['glow_color'] = (255, 255, 150)  # Bright yellow glow
                    node_data['glow_size'] = node_size + 2
                
                result['combined_probability_nodes_data'].append(node_data)
        
        # ===== PROCESS VISIBILITY RANGES DATA =====
        if self.map_graph:
            # Agent 1 reachability circle (faint blue)
            max_reachable_distance = time_horizon * agent_speed
            result['combined_visibility_ranges_data'].append({
                'position': (int(agent1_x), int(agent1_y)),
                'radius': int(max_reachable_distance),
                'color': (100, 150, 255),
                'line_width': 1,
                'agent': 'agent1'
            })
            
            # Agent 2 dynamic visibility circle (faint cyan)
            if 'agent2_vision_range' in locals():
                result['combined_visibility_ranges_data'].append({
                    'position': (int(agent2_x), int(agent2_y)),
                    'radius': int(agent2_vision_range),
                    'color': (0, 150, 150),
                    'line_width': 1,
                    'agent': 'agent2'
                })
        
        # ===== PROCESS INFO DISPLAY DATA =====
        result['combined_info_display_data'] = {
            'info_text': [
                "Combined Probability Mode",
                f"Agent 1 nodes: {len(agent1_probabilities)}",
                f"Agent 2 nodes: {len(agent2_probabilities)}",
                f"Combined nodes: {len(combined_probabilities)}",
                "Purple → Yellow: Low → High"
            ],
            'background_color': (40, 0, 40),  # Dark purple background
            'text_color_primary': (255, 255, 255),
            'text_color_secondary': (200, 200, 200),
            'font_size': 16,
            'position': (250, 10),  # Will be adjusted based on WIDTH in visualization
            'size': (250, 120)
        }
        
        # Record computation time
        computation_end = time.perf_counter()
        result['computation_stats']['computation_time_ms'] = (computation_end - computation_start) * 1000
        
        return result

    def calculate_agent2_visibility_gaps_computation(self, agent2, visibility_map, time_horizon, agent_speed, DEFAULT_VISION_RANGE=800):
        """
        Calculate Agent 2 visibility gaps computation phase (K key functionality - computation part).
        
        This method encapsulates the computation phase that was previously done inline in the 
        agent 2 visibility gaps section, separating computation from visualization for better modularity.
        
        Args:
            agent2: Agent 2 object with state [x, y, theta, ...]
            visibility_map: Dictionary mapping node indices to lists of visible node indices
            time_horizon: Look-ahead time in seconds for dynamic range calculation
            agent_speed: Agent movement speed (typically LEADER_LINEAR_VEL)
            DEFAULT_VISION_RANGE: Maximum vision range cap (default 800px)
            
        Returns:
            dict: {
                'agent2_visible_nodes_data': list of visible node connection data for visualization,
                'agent2_gap_visualization_data': list of gap line visualization data,
                'agent2_visibility_points_data': list of individual visibility point data,
                'agent2_pos': tuple of agent 2 position (x2, y2),
                'agent2_node_index': index of closest node to agent 2,
                'agent2_node': position of closest node to agent 2,
                'computation_stats': dict with performance statistics
            }
        """
        # Initialize computation result structure
        result = {
            'agent2_visible_nodes_data': [],
            'agent2_gap_visualization_data': [],
            'agent2_visibility_points_data': [],
            'agent2_pos': None,
            'agent2_node_index': None,
            'agent2_node': None,
            'computation_stats': {
                'visible_nodes_processed': 0,
                'gap_lines_processed': 0,
                'visibility_points_processed': 0,
                'computation_time_ms': 0
            }
        }
        
        if not self.map_graph or not self.agent1_current_position:
            result['computation_stats']['message'] = 'Insufficient data for computation'
            return result
        
        computation_start = time.perf_counter()
        
        # Calculate agent position and node data
        agent2_x, agent2_y = agent2.state[0], agent2.state[1]
        agent2_pos = (agent2_x, agent2_y)
        result['agent2_pos'] = agent2_pos
        
        # DYNAMIC RANGE CALCULATION: Agent 1 reachability + inter-agent distance (max 800px)
        agent1_x, agent1_y = self.agent1_current_position
        agent1_reachability = time_horizon * agent_speed  # Agent 1's reachable distance
        inter_agent_distance = math.dist((agent1_x, agent1_y), (agent2_x, agent2_y))
        
        # Calculate dynamic vision range: reachability + distance between agents, capped at 800
        dynamic_vision_range = min(agent1_reachability + inter_agent_distance, DEFAULT_VISION_RANGE)
        
        # Get gap lines for agent 2 using the dynamic range
        agent2_node_index = find_closest_node(self.map_graph.nodes, agent2_pos)
        ray_endpoints, gap_lines = self.calculate_visibility_gaps(
            agent2, 
            dynamic_vision_range, 
            agent2_node_index
        )
        
        # Get closest node and visible nodes from preprocessed visibility data
        if visibility_map and self.map_graph:
            agent2_node_index = find_closest_node(self.map_graph.nodes, agent2_pos)
            result['agent2_node_index'] = agent2_node_index
            
            if agent2_node_index is not None:
                agent2_node = self.map_graph.nodes[agent2_node_index]
                result['agent2_node'] = agent2_node
                visible_nodes = visibility_map[agent2_node_index]
                
                # Process all visible nodes and store visualization data
                for visible_index in visible_nodes:
                    if visible_index < len(self.map_graph.nodes):  # Safety check
                        visible_node = self.map_graph.nodes[visible_index]
                        # Store data for later visualization
                        result['agent2_visible_nodes_data'].append({
                            'start': agent2_node,
                            'end': visible_node,
                            'color': (0, 255, 255),  # Cyan
                            'line_width': 1,
                            'circle_size': 3
                        })
                        result['computation_stats']['visible_nodes_processed'] += 1
        
        # Process gap lines and store visualization data
        for start_point, end_point, gap_size in gap_lines:
            # Determine gap orientation relative to clockwise ray casting
            start_dist = math.dist((agent2_x, agent2_y), start_point)
            end_dist = math.dist((agent2_x, agent2_y), end_point)
            
            # Classify gap type based on distance progression
            is_near_to_far = start_dist < end_dist  # Near point first, far point second
            is_far_to_near = start_dist > end_dist  # Far point first, near point second
            
            # Choose base color based on gap orientation - using cyan/green for second agent's gaps
            if is_near_to_far:
                # Cyan for near-to-far transitions (expanding gaps)
                base_color = (0, 200, 255) if gap_size > 150 else (0, 220, 255) if gap_size > 80 else (0, 240, 255)
            elif is_far_to_near:
                # Green-cyan for far-to-near transitions (contracting gaps)
                base_color = (0, 240, 180) if gap_size > 150 else (0, 255, 180) if gap_size > 80 else (0, 255, 220)
            else:
                # Fallback color for equal distances (rare case)
                base_color = (0, 200, 200)
            
            # Determine line width based on gap size
            if gap_size > 150:
                line_width = 3
            elif gap_size > 80:
                line_width = 2
            else:
                line_width = 1
            
            # Calculate circle size
            circle_size = max(2, min(5, int(gap_size / 30)))
            
            # Store gap visualization data
            result['agent2_gap_visualization_data'].append({
                'start_point': start_point,
                'end_point': end_point,
                'color': base_color,
                'line_width': line_width,
                'circle_size': circle_size
            })
            result['computation_stats']['gap_lines_processed'] += 1
        
        # Process individual visibility points data (for when probability overlay is OFF)
        for start_point, end_point, gap_size in gap_lines:
            result['agent2_visibility_points_data'].extend([
                {'point': start_point, 'color': (255, 50, 255), 'size': 1},
                {'point': end_point, 'color': (255, 50, 255), 'size': 1}
            ])
            result['computation_stats']['visibility_points_processed'] += 2
        
        # Record computation time
        computation_end = time.perf_counter()
        result['computation_stats']['computation_time_ms'] = (computation_end - computation_start) * 1000
        
        return result
