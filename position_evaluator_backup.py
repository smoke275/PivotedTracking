#!/usr/bin/env python3
"""
Position Evaluator Module
A standalone module for tracking agent positions and calculating distances between them.
Designed to be extensible for future spatial analysis and coordination algorithms.
"""

import math
import time
import math
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

# Z functionality imports (lazy loaded when needed)
_vision_utils = None
_pathfinding_utils = None

@dataclass
class AgentPosition:
    """Data structure to hold agent position and metadata"""
    agent_id: str
    x: float
    y: float
    heading: float  # in radians
    timestamp: float
    velocity: Optional[Tuple[float, float]] = None  # (linear_vel, angular_vel)

class PositionEvaluator:
    """
    Central position tracking and distance calculation system.
    Tracks multiple agents and provides various spatial analysis functions.
    """
    
    def __init__(self, environment=None, map_graph=None):
        self.agent_positions: Dict[str, AgentPosition] = {}
        self.position_history: Dict[str, List[AgentPosition]] = defaultdict(list)
        self.max_history_length = 100  # Keep last 100 positions per agent
        self.last_update_time = time.time()
        
        # Environment and map graph integration
        self.environment = environment
        self.map_graph = map_graph
        
        # Agent following state
        self.follow_agent_mode = False
        self.followed_agent_id = None
        self.selected_node_index = None
        self.agent_movement_threshold = 20.0  # Minimum distance to consider agent has moved
        
    def update_agent_position(self, agent_id: str, x: float, y: float, heading: float, 
                            velocity: Optional[Tuple[float, float]] = None):
        """
        Update the position of an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            x, y: Position coordinates
            heading: Agent heading in radians
            velocity: Optional tuple of (linear_velocity, angular_velocity)
        """
        timestamp = time.time()
        
        position = AgentPosition(
            agent_id=agent_id,
            x=x,
            y=y,
            heading=heading,
            timestamp=timestamp,
            velocity=velocity
        )
        
        # Update current position
        self.agent_positions[agent_id] = position
        
        # Add to history
        self.position_history[agent_id].append(position)
        
        # Limit history length
        if len(self.position_history[agent_id]) > self.max_history_length:
            self.position_history[agent_id].pop(0)
        
        self.last_update_time = timestamp
    
    def get_distance_between_agents(self, agent1_id: str, agent2_id: str) -> Optional[float]:
        """
        Calculate Euclidean distance between two agents.
        
        Args:
            agent1_id, agent2_id: IDs of the two agents
            
        Returns:
            Distance in pixels, or None if either agent is not found
        """
        if agent1_id not in self.agent_positions or agent2_id not in self.agent_positions:
            return None
        
        pos1 = self.agent_positions[agent1_id]
        pos2 = self.agent_positions[agent2_id]
        
        distance = math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
        return distance
    
    def get_all_distances(self) -> Dict[Tuple[str, str], float]:
        """
        Calculate distances between all pairs of agents.
        
        Returns:
            Dictionary mapping agent pairs to their distances
        """
        distances = {}
        agent_ids = list(self.agent_positions.keys())
        
        for i, agent1_id in enumerate(agent_ids):
            for j, agent2_id in enumerate(agent_ids[i+1:], i+1):
                distance = self.get_distance_between_agents(agent1_id, agent2_id)
                if distance is not None:
                    distances[(agent1_id, agent2_id)] = distance
        
        return distances
    
    def get_agent_position(self, agent_id: str) -> Optional[AgentPosition]:
        """Get the current position of a specific agent."""
        return self.agent_positions.get(agent_id)
    
    def get_all_agent_positions(self) -> Dict[str, AgentPosition]:
        """Get current positions of all agents."""
        return self.agent_positions.copy()
    
    def get_agents_within_range(self, center_agent_id: str, max_distance: float) -> List[Tuple[str, float]]:
        """
        Find all agents within a specified distance of a center agent.
        
        Args:
            center_agent_id: ID of the center agent
            max_distance: Maximum distance threshold
            
        Returns:
            List of (agent_id, distance) tuples for agents within range
        """
        if center_agent_id not in self.agent_positions:
            return []
        
        center_pos = self.agent_positions[center_agent_id]
        nearby_agents = []
        
        for agent_id, position in self.agent_positions.items():
            if agent_id == center_agent_id:
                continue
                
            distance = math.sqrt((center_pos.x - position.x)**2 + (center_pos.y - position.y)**2)
            if distance <= max_distance:
                nearby_agents.append((agent_id, distance))
        
        # Sort by distance
        nearby_agents.sort(key=lambda x: x[1])
        return nearby_agents
    
    def get_relative_bearing(self, from_agent_id: str, to_agent_id: str) -> Optional[float]:
        """
        Calculate the bearing from one agent to another relative to the first agent's heading.
        
        Args:
            from_agent_id: ID of the observing agent
            to_agent_id: ID of the target agent
            
        Returns:
            Relative bearing in radians, or None if either agent is not found
        """
        if from_agent_id not in self.agent_positions or to_agent_id not in self.agent_positions:
            return None
        
        from_pos = self.agent_positions[from_agent_id]
        to_pos = self.agent_positions[to_agent_id]
        
        # Calculate absolute bearing
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        absolute_bearing = math.atan2(dy, dx)
        
        # Calculate relative bearing
        relative_bearing = absolute_bearing - from_pos.heading
        
        # Normalize to [-pi, pi]
        while relative_bearing > math.pi:
            relative_bearing -= 2 * math.pi
        while relative_bearing < -math.pi:
            relative_bearing += 2 * math.pi
        
        return relative_bearing
    
    def get_agent_count(self) -> int:
        """Get the number of tracked agents."""
        return len(self.agent_positions)
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from tracking."""
        if agent_id in self.agent_positions:
            del self.agent_positions[agent_id]
        if agent_id in self.position_history:
            del self.position_history[agent_id]
    
    def clear_all_agents(self):
        """Remove all agents from tracking."""
        self.agent_positions.clear()
        self.position_history.clear()
    
    def set_environment_data(self, environment, map_graph):
        """
        Set or update environment data for advanced spatial analysis.
        
        Args:
            environment: SimulationEnvironment instance with walls and doors
            map_graph: MapGraph instance with nodes and edges
        """
        self.environment = environment
        self.map_graph = map_graph
        print(f"Position Evaluator: Environment data updated")
        if map_graph and hasattr(map_graph, 'nodes'):
            print(f"  Map graph: {len(map_graph.nodes)} nodes, {len(map_graph.edges) if hasattr(map_graph, 'edges') else 0} edges")
    
    def has_environment_data(self) -> bool:
        """Check if environment data is available for advanced features."""
        return self.environment is not None and self.map_graph is not None
    
    def find_closest_graph_node(self, agent_id: str) -> Optional[int]:
        """
        Find the closest map graph node to an agent (F component preparation).
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Index of closest node, or None if not found
        """
        if not self.has_environment_data() or agent_id not in self.agent_positions:
            return None
        
        if not hasattr(self.map_graph, 'nodes') or not self.map_graph.nodes:
            return None
        
        agent_pos = self.agent_positions[agent_id]
        agent_position = (agent_pos.x, agent_pos.y)
        
        # Find closest node (simple distance-based search)
        closest_node_index = None
        closest_distance = float('inf')
        
        for i, node in enumerate(self.map_graph.nodes):
            distance = math.sqrt((node[0] - agent_position[0])**2 + (node[1] - agent_position[1])**2)
            if distance < closest_distance:
                closest_distance = distance
                closest_node_index = i
        
        return closest_node_index
    
    def get_statistics(self) -> Dict:
        """
        Get various statistics about the tracked agents.
        
        Returns:
            Dictionary containing statistics
        """
        if not self.agent_positions:
            return {"agent_count": 0}
        
        distances = list(self.get_all_distances().values())
        
        stats = {
            "agent_count": len(self.agent_positions),
            "last_update": self.last_update_time,
            "tracked_agents": list(self.agent_positions.keys()),
            "has_environment": self.has_environment_data(),
            "follow_mode": self.follow_agent_mode,
            "followed_agent": self.followed_agent_id
        }
        
        if distances:
            stats.update({
                "min_distance": min(distances),
                "max_distance": max(distances),
                "avg_distance": sum(distances) / len(distances),
                "total_pairs": len(distances)
            })
        
        # Add environment data info if available
        if self.has_environment_data():
            stats.update({
                "map_nodes": len(self.map_graph.nodes) if hasattr(self.map_graph, 'nodes') else 0,
                "map_edges": len(self.map_graph.edges) if hasattr(self.map_graph, 'edges') else 0
            })
        
        return stats
    
    # Future extension points (placeholder methods for evolution)
    
    def predict_collision_risk(self, time_horizon: float = 5.0) -> Dict[Tuple[str, str], float]:
        """
        Placeholder for future collision prediction algorithm.
        
        Args:
            time_horizon: Time horizon for prediction in seconds
            
        Returns:
            Dictionary mapping agent pairs to collision risk scores (0-1)
        """
        # TODO: Implement collision prediction based on current velocities and trajectories
        return {}
    
    def calculate_formation_metrics(self) -> Dict:
        """
        Placeholder for future formation analysis.
        
        Returns:
            Dictionary containing formation quality metrics
        """
        # TODO: Implement formation analysis (spread, alignment, etc.)
        return {}
    
    def suggest_coordination_actions(self) -> Dict[str, str]:
        """
        Placeholder for future coordination recommendations.
        
        Returns:
            Dictionary mapping agent IDs to suggested actions
        """
        # TODO: Implement coordination logic
        return {}
    
    # Z Functionality Implementation (F+O+B+Y+H)
    
    def _lazy_load_vision_utils(self):
        """Lazy load vision utilities when needed for Z functionality."""
        global _vision_utils
        if _vision_utils is None:
            try:
                from multitrack.utils.vision import cast_vision_ray
                _vision_utils = {'cast_vision_ray': cast_vision_ray}
            except ImportError as e:
                print(f"Warning: Could not import vision utilities: {e}")
                _vision_utils = {}
        return _vision_utils
    
    def _lazy_load_pathfinding_utils(self):
        """Lazy load pathfinding utilities when needed for Z functionality."""
        global _pathfinding_utils
        if _pathfinding_utils is None:
            try:
                from multitrack.utils.pathfinding import find_closest_node
                _pathfinding_utils = {'find_closest_node': find_closest_node}
            except ImportError as e:
                print(f"Warning: Could not import pathfinding utilities: {e}")
                _pathfinding_utils = {}
        return _pathfinding_utils
    
    def predict_collision_risk(self, time_horizon: float = 5.0) -> Dict[Tuple[str, str], float]:
        """
        Enable Z key functionality (F+O+B+Y+H) for an agent.
        
        Args:
            agent_id: ID of the agent to enable Z functionality for
            
        Returns:
            True if successfully enabled, False if prerequisites not met
        """
        if not self.has_environment_data():
            print("Cannot enable Z functionality: No environment data available")
            return False
        
        if agent_id not in self.agent_positions:
            print(f"Cannot enable Z functionality: Agent {agent_id} not tracked")
            return False
        
        # Load required utilities
        vision_utils = self._lazy_load_vision_utils()
        pathfinding_utils = self._lazy_load_pathfinding_utils()
        
        if not vision_utils or not pathfinding_utils:
            print("Cannot enable Z functionality: Required utilities not available")
            return False
        
        # Enable all Z components
        self.z_mode_enabled = True
        self.follow_agent_mode = True
        self.followed_agent_id = agent_id
        self.probability_overlay_enabled = True
        self.visibility_gaps_enabled = True
        self.rotating_rods_enabled = True
        self.extended_probability_enabled = True
        
        # Update selected node to agent's closest node (F functionality)
        self.selected_node_index = self.find_closest_graph_node(agent_id)
        
        print(f"Z functionality enabled for agent {agent_id}")
        print("  ✓ F: Agent-following mode")
        print("  ✓ O: Probability overlay")
        print("  ✓ B: Visibility gaps")
        print("  ✓ Y: Rotating rods")
        print("  ✓ H: Extended probability set")
        
        return True
    
    def disable_z_functionality(self):
        """Disable Z key functionality."""
        self.z_mode_enabled = False
        self.follow_agent_mode = False
        self.followed_agent_id = None
        self.probability_overlay_enabled = False
        self.visibility_gaps_enabled = False
        self.rotating_rods_enabled = False
        self.extended_probability_enabled = False
        print("Z functionality disabled")
    
    def calculate_probability_overlay(self, agent_id: str) -> Dict[int, float]:
        """
        Calculate probability overlay for O functionality.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary mapping node indices to probability values
        """
        if agent_id not in self.agent_positions:
            return {}
        
        if not self.has_environment_data() or self.selected_node_index is None:
            return {}
        
        # Get agent state
        agent_pos = self.agent_positions[agent_id]
        agent_x, agent_y = agent_pos.x, agent_pos.y
        agent_theta = agent_pos.heading
        
        # Calculate maximum reachable distance based on time horizon
        # Import config value for agent speed
        try:
            from multitrack.utils.config import LEADER_LINEAR_VEL
            agent_speed = LEADER_LINEAR_VEL
        except ImportError:
            agent_speed = 50.0  # Fallback default speed
        max_reachable_distance = self.time_horizon * agent_speed
        
        # Get visibility map if available - handle nested dictionary structure
        if not self.visibility_map:
            return {}
        
        # Extract actual visibility data from nested structure
        actual_visibility_map = self.visibility_map.get('visibility_map', self.visibility_map)
        
        if self.selected_node_index not in actual_visibility_map:
            return {}
        
        visible_node_indices = set(actual_visibility_map[self.selected_node_index])
        node_probabilities = {}
        
        # Calculate probabilities for visible, reachable nodes
        for i, node in enumerate(self.map_graph.nodes):
            if i in visible_node_indices:
                node_x, node_y = node
                distance = math.sqrt((node_x - agent_x)**2 + (node_y - agent_y)**2)
                
                if distance <= max_reachable_distance:
                    # Distance-based probability
                    distance_prob = max(0, 1.0 - (distance / max_reachable_distance))
                    
                    # Heading bias factor
                    if distance > 1.0:
                        node_angle = math.atan2(node_y - agent_y, node_x - agent_x)
                        angle_diff = abs(agent_theta - node_angle)
                        if angle_diff > math.pi:
                            angle_diff = 2 * math.pi - angle_diff
                        heading_bias = max(0, 1.0 - (angle_diff / math.pi))
                        probability = distance_prob * (0.3 + 0.7 * heading_bias)
                    else:
                        probability = distance_prob
                    
                    if probability > 0.05:  # Only store significant probabilities
                        node_probabilities[i] = probability
        
        return node_probabilities
    
    def calculate_visibility_gaps(self, agent_id: str) -> List[Tuple]:
        """
        Calculate visibility gaps for B functionality.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of gap lines (start_point, end_point, gap_size)
        """
        if self.selected_node_index is None:
            return []
        
        vision_utils = self._lazy_load_vision_utils()
        if not vision_utils or 'cast_vision_ray' not in vision_utils:
            return []
        
        cast_vision_ray = vision_utils['cast_vision_ray']
        selected_node = self.map_graph.nodes[self.selected_node_index]
        
        # Cast rays in all directions to find discontinuities
        num_rays = 360  # Every 1 degree
        angle_step = (2 * math.pi) / num_rays
        ray_endpoints = []
        
        # Cast rays in all directions
        for i in range(num_rays):
            angle = i * angle_step
            endpoint = cast_vision_ray(
                selected_node[0], 
                selected_node[1], 
                angle, 
                self.visibility_range,
                self.environment.get_all_walls(),
                self.environment.get_doors()
            )
            ray_endpoints.append(endpoint)
        
        # Find discontinuities in ray distances
        min_gap_distance = 30  # Minimum distance difference to consider a gap
        gap_lines = []
        
        for i in range(num_rays):
            current_endpoint = ray_endpoints[i]
            next_endpoint = ray_endpoints[(i + 1) % num_rays]  # Wrap around
            
            # Calculate distances from selected node
            current_dist = math.dist(selected_node, current_endpoint)
            next_dist = math.dist(selected_node, next_endpoint)
            
            # Check for significant distance change (gap)
            distance_diff = abs(current_dist - next_dist)
            if distance_diff > min_gap_distance:
                gap_lines.append((current_endpoint, next_endpoint, distance_diff))
        
        return gap_lines
    
    def calculate_rotating_rods(self, agent_id: str) -> List[Tuple]:
        """
        Calculate rotating rod patterns for Y functionality.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of rotating rod lines (start_point, end_point, rotation_angle)
        """
        if self.selected_node_index is None:
            return []
        
        if agent_id not in self.agent_positions:
            return []
        
        vision_utils = self._lazy_load_vision_utils()
        if not vision_utils or 'cast_vision_ray' not in vision_utils:
            return []
        
        cast_vision_ray = vision_utils['cast_vision_ray']
        selected_node = self.map_graph.nodes[self.selected_node_index]
        agent_pos = self.agent_positions[agent_id]
        
        # Calculate primary direction from selected node to agent
        primary_angle = math.atan2(
            agent_pos.y - selected_node[1],
            agent_pos.x - selected_node[0]
        )
        
        # Generate rotating rods at specific angular intervals
        rod_angles = [
            primary_angle + offset 
            for offset in [-math.pi/4, 0, math.pi/4, math.pi/2, -math.pi/2]
        ]
        
        rotating_rods = []
        for angle in rod_angles:
            # Cast ray to find rod endpoint
            endpoint = cast_vision_ray(
                selected_node[0],
                selected_node[1],
                angle,
                self.visibility_range,
                self.environment.get_all_walls(),
                self.environment.get_doors()
            )
            
            # Add rotation animation angle (can be time-based)
            rotation_offset = (angle * 180 / math.pi) % 360
            rotating_rods.append((selected_node, endpoint, rotation_offset))
        
        return rotating_rods
    
    def calculate_extended_probability_set(self, agent_id: str) -> Dict[str, Dict[int, float]]:
        """
        Calculate extended probability set for H functionality.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with different probability calculation methods
        """
        if not self.extended_probability_enabled:
            return {}
        
        probability_sets = {}
        
        # Base probability overlay (O functionality)
        base_probabilities = self.calculate_probability_overlay(agent_id)
        probability_sets['base'] = base_probabilities
        
        # Gap-influenced probabilities (B functionality influence)
        gap_probabilities = self._calculate_gap_influenced_probabilities(agent_id, base_probabilities)
        probability_sets['gap_influenced'] = gap_probabilities
        
        # Direction-biased probabilities (Y functionality influence)
        directional_probabilities = self._calculate_directional_probabilities(agent_id, base_probabilities)
        probability_sets['directional'] = directional_probabilities
        
        # Time-decay probabilities (temporal component)
        time_decay_probabilities = self._calculate_time_decay_probabilities(agent_id, base_probabilities)
        probability_sets['time_decay'] = time_decay_probabilities
        
        # Combined weighted probabilities
        combined_probabilities = self._calculate_combined_probabilities(probability_sets)
        probability_sets['combined'] = combined_probabilities
        
        return probability_sets
    
    def _calculate_gap_influenced_probabilities(self, agent_id: str, base_probabilities: Dict[int, float]) -> Dict[int, float]:
        """Calculate probabilities influenced by visibility gaps."""
        gap_lines = self.calculate_visibility_gaps(agent_id)
        if not gap_lines or not base_probabilities:
            return base_probabilities.copy()
        
        gap_influenced = base_probabilities.copy()
        
        # Boost probabilities near gaps (areas of uncertainty)
        for node_idx, prob in base_probabilities.items():
            node = self.map_graph.nodes[node_idx]
            
            # Find distance to nearest gap
            min_gap_distance = float('inf')
            for gap_start, gap_end, gap_size in gap_lines:
                gap_center = ((gap_start[0] + gap_end[0]) / 2, (gap_start[1] + gap_end[1]) / 2)
                gap_distance = math.dist(node, gap_center)
                min_gap_distance = min(min_gap_distance, gap_distance)
            
            # Apply gap influence (closer to gaps = higher uncertainty = higher probability)
            if min_gap_distance < 100:  # Within influence range
                gap_boost = 1.0 + (1.0 - min_gap_distance / 100) * 0.5
                gap_influenced[node_idx] = min(1.0, prob * gap_boost)
        
        return gap_influenced
    
    def _calculate_directional_probabilities(self, agent_id: str, base_probabilities: Dict[int, float]) -> Dict[int, float]:
        """Calculate probabilities biased by rotating rod directions."""
        rotating_rods = self.calculate_rotating_rods(agent_id)
        if not rotating_rods or not base_probabilities:
            return base_probabilities.copy()
        
        directional = base_probabilities.copy()
        
        # Boost probabilities along rotating rod directions
        for node_idx, prob in base_probabilities.items():
            node = self.map_graph.nodes[node_idx]
            
            # Check alignment with any rotating rod
            max_alignment = 0.0
            for rod_start, rod_end, rotation_angle in rotating_rods:
                # Calculate alignment with rod direction
                rod_vector = (rod_end[0] - rod_start[0], rod_end[1] - rod_start[1])
                node_vector = (node[0] - rod_start[0], node[1] - rod_start[1])
                
                # Normalize vectors
                rod_length = math.sqrt(rod_vector[0]**2 + rod_vector[1]**2)
                node_length = math.sqrt(node_vector[0]**2 + node_vector[1]**2)
                
                if rod_length > 0 and node_length > 0:
                    # Dot product for alignment
                    dot_product = (rod_vector[0] * node_vector[0] + rod_vector[1] * node_vector[1])
                    alignment = dot_product / (rod_length * node_length)
                    max_alignment = max(max_alignment, max(0, alignment))
            
            # Apply directional boost
            if max_alignment > 0.7:  # Strong alignment
                directional_boost = 1.0 + max_alignment * 0.3
                directional[node_idx] = min(1.0, prob * directional_boost)
        
        return directional
    
    def _calculate_time_decay_probabilities(self, agent_id: str, base_probabilities: Dict[int, float]) -> Dict[int, float]:
        """Calculate probabilities with time decay factor."""
        if not base_probabilities:
            return {}
        
        # Simple time decay based on time horizon
        decay_factor = max(0.1, 1.0 - (self.time_horizon / 10.0))  # Decay over time
        
        time_decay = {}
        for node_idx, prob in base_probabilities.items():
            time_decay[node_idx] = prob * decay_factor
        
        return time_decay
    
    def _calculate_combined_probabilities(self, probability_sets: Dict[str, Dict[int, float]]) -> Dict[int, float]:
        """Calculate weighted combination of all probability sets."""
        if not probability_sets:
            return {}
        
        # Weights for different probability components
        weights = {
            'base': 0.4,
            'gap_influenced': 0.25,
            'directional': 0.2,
            'time_decay': 0.15
        }
        
        combined = {}
        all_nodes = set()
        
        # Collect all node indices
        for prob_set in probability_sets.values():
            all_nodes.update(prob_set.keys())
        
        # Calculate weighted combination for each node
        for node_idx in all_nodes:
            weighted_sum = 0.0
            weight_total = 0.0
            
            for set_name, prob_set in probability_sets.items():
                if set_name in weights and node_idx in prob_set:
                    weighted_sum += prob_set[node_idx] * weights[set_name]
                    weight_total += weights[set_name]
            
            if weight_total > 0:
                combined[node_idx] = weighted_sum / weight_total
        
        return combined
    
    def get_z_functionality_status(self) -> Dict[str, bool]:
        """Get status of all Z functionality components."""
        return {
            'z_mode_enabled': self.z_mode_enabled,
            'F_agent_following': self.follow_agent_mode,
            'O_probability_overlay': self.probability_overlay_enabled,
            'B_visibility_gaps': self.visibility_gaps_enabled,
            'Y_rotating_rods': self.rotating_rods_enabled,
            'H_extended_probability': self.extended_probability_enabled,
            'prerequisites_met': self.is_z_functionality_available()
        }
    
    def is_z_functionality_available(self) -> bool:
        """Check if Z functionality can be enabled."""
        return (self.has_environment_data() and 
                len(self.agent_positions) > 0 and
                self._lazy_load_vision_utils() and
                self._lazy_load_pathfinding_utils())
    
    def update_z_functionality(self, force_update: bool = False) -> Dict[str, any]:
        """
        Update all Z functionality components and return results.
        
        Args:
            force_update: Force update even if cache is valid
            
        Returns:
            Dictionary containing all Z functionality results
        """
        if not self.z_mode_enabled or not self.followed_agent_id:
            return {}
        
        # Check cache validity
        current_time = time.time()
        if (not force_update and 
            current_time - self._z_cache['last_update'] < self._z_cache_timeout):
            return self._z_cache.copy()
        
        results = {}
        agent_id = self.followed_agent_id
        
        # Update F: Agent following (already handled in enable_z_functionality)
        results['agent_following'] = {
            'enabled': self.follow_agent_mode,
            'followed_agent': self.followed_agent_id,
            'selected_node': self.selected_node_index
        }
        
        # Update O: Probability overlay
        if self.probability_overlay_enabled:
            probabilities = self.calculate_probability_overlay(agent_id)
            results['probability_overlay'] = probabilities
            self._z_cache['probabilities'] = probabilities
        
        # Update B: Visibility gaps
        if self.visibility_gaps_enabled:
            gaps = self.calculate_visibility_gaps(agent_id)
            results['visibility_gaps'] = gaps
            self._z_cache['gaps'] = gaps
        
        # Update Y: Rotating rods
        if self.rotating_rods_enabled:
            rods = self.calculate_rotating_rods(agent_id)
            results['rotating_rods'] = rods
            self._z_cache['rods'] = rods
        
        # Update H: Extended probability set
        if self.extended_probability_enabled:
            extended_sets = self.calculate_extended_probability_set(agent_id)
            results['extended_probability_set'] = extended_sets
            self._z_cache['extended_sets'] = extended_sets
        
        # Update cache timestamp
        self._z_cache['last_update'] = current_time
        
        return results
    
    def get_z_results(self) -> Dict[str, any]:
        """Get current Z functionality results from cache."""
        if not self.z_mode_enabled:
            return {}
        return self._z_cache.copy()
    
    def invalidate_z_cache(self):
        """Invalidate Z functionality cache to force update on next call."""
        self._z_cache['last_update'] = 0
    
    # G Functionality Implementation (Map Graph Visual Display Control)
    
    def enable_g_functionality(self) -> bool:
        """
        Enable G key functionality (Map Graph Visual Display Control).
        
        Returns:
            True if successfully enabled, False if prerequisites not met
        """
        if not self.has_environment_data():
            print("Cannot enable G functionality: No environment data available")
            return False
        
        if not hasattr(self, 'show_map_graph_visuals'):
            self.show_map_graph_visuals = False
        
        self.show_map_graph_visuals = True
        print("G functionality enabled: Map graph visuals activated")
        return True
    
    def disable_g_functionality(self):
        """Disable G key functionality."""
        if hasattr(self, 'show_map_graph_visuals'):
            self.show_map_graph_visuals = False
        print("G functionality disabled: Map graph visuals deactivated")
    
    def toggle_g_functionality(self) -> bool:
        """
        Toggle G key functionality state.
        
        Returns:
            True if now enabled, False if now disabled
        """
        if not hasattr(self, 'show_map_graph_visuals'):
            self.show_map_graph_visuals = False
        
        if self.show_map_graph_visuals:
            self.disable_g_functionality()
            return False
        else:
            return self.enable_g_functionality()
    
    def get_g_functionality_status(self) -> Dict[str, bool]:
        """Get status of G functionality."""
        return {
            'g_mode_enabled': getattr(self, 'show_map_graph_visuals', False),
            'prerequisites_met': self.has_environment_data()
        }
    
    # M Functionality Implementation (Combined Probability Mode)
    
    def enable_m_functionality(self) -> bool:
        """
        Enable M key functionality (Combined Probability Mode for multiple agents).
        
        Returns:
            True if successfully enabled, False if prerequisites not met
        """
        if not self.has_environment_data():
            print("Cannot enable M functionality: No environment data available")
            return False
        
        if len(self.agent_positions) < 2:
            print("Cannot enable M functionality: Need at least 2 agents")
            return False
        
        if not hasattr(self, 'combined_probability_mode'):
            self.combined_probability_mode = False
        
        self.combined_probability_mode = True
        print("M functionality enabled: Combined probability mode activated")
        return True
    
    def disable_m_functionality(self):
        """Disable M key functionality."""
        if hasattr(self, 'combined_probability_mode'):
            self.combined_probability_mode = False
        print("M functionality disabled: Combined probability mode deactivated")
    
    def calculate_combined_probabilities(self) -> Dict[int, float]:
        """
        Calculate combined probabilities for M functionality.
        Multiplies probabilities from multiple agents with purple-yellow color scheme.
        
        Returns:
            Dictionary mapping node indices to combined probability values
        """
        if not getattr(self, 'combined_probability_mode', False):
            return {}
        
        if len(self.agent_positions) < 2:
            return {}
        
        agent_ids = list(self.agent_positions.keys())
        combined_probabilities = {}
        
        # Get probabilities for each agent
        agent_probability_maps = {}
        for agent_id in agent_ids:
            probabilities = self.calculate_probability_overlay(agent_id)
            if probabilities:
                agent_probability_maps[agent_id] = probabilities
        
        if len(agent_probability_maps) < 2:
            return {}
        
        # Find all nodes that appear in any agent's probability map
        all_nodes = set()
        for prob_map in agent_probability_maps.values():
            all_nodes.update(prob_map.keys())
        
        # Calculate combined probabilities by multiplication
        for node_idx in all_nodes:
            combined_prob = 1.0
            valid_agents = 0
            
            for agent_id, prob_map in agent_probability_maps.items():
                if node_idx in prob_map:
                    combined_prob *= prob_map[node_idx]
                    valid_agents += 1
                else:
                    # If node not in agent's probability map, use small default value
                    combined_prob *= 0.1
            
            # Only include nodes with reasonable combined probability
            if valid_agents >= 2 and combined_prob > 0.01:
                combined_probabilities[node_idx] = combined_prob
        
        # Normalize combined probabilities to [0, 1] range
        if combined_probabilities:
            max_prob = max(combined_probabilities.values())
            if max_prob > 0:
                for node_idx in combined_probabilities:
                    combined_probabilities[node_idx] /= max_prob
        
        return combined_probabilities
    
    def get_m_functionality_status(self) -> Dict[str, any]:
        """Get status of M functionality."""
        return {
            'm_mode_enabled': getattr(self, 'combined_probability_mode', False),
            'agent_count': len(self.agent_positions),
            'prerequisites_met': self.has_environment_data() and len(self.agent_positions) >= 2
        }
    
    # X Functionality Implementation (Complete Dual-Agent System)
    
    def enable_x_functionality(self, primary_agent_id: str) -> bool:
        """
        Enable X key functionality (Complete Dual-Agent System: Z + G + M).
        
        Args:
            primary_agent_id: ID of the primary agent for Z functionality
            
        Returns:
            True if successfully enabled, False if prerequisites not met
        """
        if not self.has_environment_data():
            print("Cannot enable X functionality: No environment data available")
            return False
        
        if len(self.agent_positions) < 2:
            print("Cannot enable X functionality: Need at least 2 agents")
            return False
        
        if primary_agent_id not in self.agent_positions:
            print(f"Cannot enable X functionality: Primary agent {primary_agent_id} not found")
            return False
        
        # Enable Z functionality for primary agent
        z_enabled = self.enable_z_functionality(primary_agent_id)
        if not z_enabled:
            print("Cannot enable X functionality: Z functionality failed to enable")
            return False
        
        # Enable G functionality
        g_enabled = self.enable_g_functionality()
        if not g_enabled:
            print("Cannot enable X functionality: G functionality failed to enable")
            return False
        
        # Enable M functionality
        m_enabled = self.enable_m_functionality()
        if not m_enabled:
            print("Cannot enable X functionality: M functionality failed to enable")
            return False
        
        print(f"X functionality enabled for primary agent {primary_agent_id}")
        print("  ✓ Z: Complete dual-agent tracking (F+O+B+Y+H)")
        print("  ✓ G: Map graph visual display control")
        print("  ✓ M: Combined probability mode")
        print("  → Complete dual-agent visualization system activated")
        
        return True
    
    def disable_x_functionality(self):
        """Disable X key functionality (disables Z, G, and M)."""
        self.disable_z_functionality()
        self.disable_g_functionality()
        self.disable_m_functionality()
        print("X functionality disabled: Complete dual-agent system deactivated")
    
    def get_x_functionality_status(self) -> Dict[str, any]:
        """Get comprehensive status of X functionality."""
        z_status = self.get_z_functionality_status()
        g_status = self.get_g_functionality_status()
        m_status = self.get_m_functionality_status()
        
        x_enabled = (z_status['z_mode_enabled'] and 
                    g_status['g_mode_enabled'] and 
                    m_status['m_mode_enabled'])
        
        return {
            'x_mode_enabled': x_enabled,
            'z_component': z_status,
            'g_component': g_status,
            'm_component': m_status,
            'prerequisites_met': (self.has_environment_data() and 
                                len(self.agent_positions) >= 2),
            'tracked_agents': list(self.agent_positions.keys()),
            'primary_agent': self.followed_agent_id if z_status['z_mode_enabled'] else None
        }

# Global instance for easy access
global_position_evaluator = PositionEvaluator()

# Convenience functions for easy integration
def update_position(agent_id: str, x: float, y: float, heading: float, velocity: Optional[Tuple[float, float]] = None):
    """Convenience function to update agent position in the global evaluator."""
    global_position_evaluator.update_agent_position(agent_id, x, y, heading, velocity)

def get_distance(agent1_id: str, agent2_id: str) -> Optional[float]:
    """Convenience function to get distance between two agents."""
    return global_position_evaluator.get_distance_between_agents(agent1_id, agent2_id)

def get_all_positions() -> Dict[str, AgentPosition]:
    """Convenience function to get all agent positions."""
    return global_position_evaluator.get_all_agent_positions()

def get_stats() -> Dict:
    """Convenience function to get position evaluator statistics."""
    return global_position_evaluator.get_statistics()

def set_environment_data(environment, map_graph, visibility_map=None):
    """Convenience function to set environment data in the global evaluator."""
    global_position_evaluator.set_environment_data(environment, map_graph, visibility_map)

def find_closest_node(agent_id: str) -> Optional[int]:
    """Convenience function to find closest graph node to an agent."""
    return global_position_evaluator.find_closest_graph_node(agent_id)

def Z() -> Dict[int, float]:
    """
    Z function - calls F, O, B, Y, H in sequence and returns probability distribution.
    Z = F + O + B + Y + H: Complete agent tracking with probability analysis.
    
    F: Agent Following - Track first agent and set selected node to closest node
    O: Probability Overlay - Calculate probability distribution for reachable nodes  
    B: Visibility Gaps - Analyze visibility discontinuities
    Y: Rotating Rods - Calculate directional scanning patterns
    H: Extended Probability Set - Advanced probability calculations with gap/directional influence
    
    Returns:
        Dict[int, float] mapping node indices to probability values (0.0 to 1.0)
    """
    # Get the first agent that's being tracked
    agents = list(global_position_evaluator.agent_positions.keys())
    if not agents:
        return {}
    
    agent_id = agents[0]  # Use first agent
    
    # F: Agent Following - set selected node to closest node for the agent
    global_position_evaluator.selected_node_index = global_position_evaluator.find_closest_graph_node(agent_id)
    global_position_evaluator.followed_agent_id = agent_id
    
    if global_position_evaluator.selected_node_index is None:
        return {}
    
    # O: Probability Overlay - calculate base probabilities
    base_probabilities = global_position_evaluator.calculate_probability_overlay(agent_id)
    
    # B: Visibility Gaps - calculate gaps for uncertainty analysis
    visibility_gaps = global_position_evaluator.calculate_visibility_gaps(agent_id) 
    
    # Y: Rotating Rods - calculate directional patterns
    rotating_rods = global_position_evaluator.calculate_rotating_rods(agent_id)
    
    # H: Extended Probability Set - combine all probability influences
    if base_probabilities:
        gap_influenced = global_position_evaluator._calculate_gap_influenced_probabilities(agent_id, base_probabilities)
        directional = global_position_evaluator._calculate_directional_probabilities(agent_id, base_probabilities) 
        
        # Return combined probabilities (merge directional and gap influences)
        final_probabilities = {}
        all_nodes = set(base_probabilities.keys()) | set(gap_influenced.keys()) | set(directional.keys())
        
        for node_idx in all_nodes:
            base_prob = base_probabilities.get(node_idx, 0.0)
            gap_prob = gap_influenced.get(node_idx, base_prob)
            dir_prob = directional.get(node_idx, base_prob)
            
            # Take maximum of all probability influences
            final_probabilities[node_idx] = max(base_prob, gap_prob, dir_prob)
        
        return final_probabilities
    
    return base_probabilities

def M() -> Dict[int, float]:
    """
    M function - calculates combined probabilities for multiple agents.
    M = Z + Combined Probability Mode: First calls Z for Agent 1, then calculates Agent 2 probabilities, then multiplies.
    
    This function follows the same pattern as inspect_environment.py where:
    - First get Z probabilities for Agent 1 (complete tracking with F+O+B+Y+H)
    - Then get Z-like probabilities for Agent 2 
    - combined_prob = prob1 * prob2
    
    Purple-yellow color scheme represents low to high combined probability.
    Only stores probabilities above threshold (typically 0.1) to focus on significant intersections.
    
    Returns:
        Dict[int, float] mapping node indices to combined probability values (0.0 to 1.0)
    """
    # Need at least 2 agents for combined probability calculation
    agents = list(global_position_evaluator.agent_positions.keys())
    if len(agents) < 2:
        return {}
    
    # Get Agent 1 and Agent 2 IDs
    agent1_id = agents[0]  # First agent
    agent2_id = agents[1]  # Second agent
    
    # Step 1: Get Z probabilities for Agent 1 (complete F+O+B+Y+H functionality)
    agent1_probabilities = Z()  # This calls the full Z functionality for Agent 1
    
    if not agent1_probabilities:
        return {}
    
    # Step 2: Calculate Agent 2 probabilities using similar methodology to Z
    # Find closest node for Agent 2
    agent2_node_index = global_position_evaluator.find_closest_graph_node(agent2_id)
    if agent2_node_index is None:
        return {}
    
    # Temporarily switch to Agent 2's node for their probability calculation
    original_selected_node = global_position_evaluator.selected_node_index
    original_followed_agent = global_position_evaluator.followed_agent_id
    
    global_position_evaluator.selected_node_index = agent2_node_index
    global_position_evaluator.followed_agent_id = agent2_id
    
    # Calculate Agent 2 probabilities (O: Probability Overlay)
    agent2_base_probabilities = global_position_evaluator.calculate_probability_overlay(agent2_id)
    
    # Apply enhanced probabilities for Agent 2 (similar to Z's H: Extended Probability Set)
    agent2_probabilities = agent2_base_probabilities
    if agent2_base_probabilities:
        # Apply gap-influenced probabilities
        gap_influenced = global_position_evaluator._calculate_gap_influenced_probabilities(agent2_id, agent2_base_probabilities)
        directional = global_position_evaluator._calculate_directional_probabilities(agent2_id, agent2_base_probabilities)
        
        # Merge all Agent 2 probability influences (similar to Z)
        agent2_probabilities = {}
        all_agent2_nodes = set(agent2_base_probabilities.keys()) | set(gap_influenced.keys()) | set(directional.keys())
        
        for node_idx in all_agent2_nodes:
            base_prob = agent2_base_probabilities.get(node_idx, 0.0)
            gap_prob = gap_influenced.get(node_idx, base_prob)
            dir_prob = directional.get(node_idx, base_prob)
            
            # Take maximum of all probability influences for Agent 2
            agent2_probabilities[node_idx] = max(base_prob, gap_prob, dir_prob)
    
    # Restore original selected node and followed agent
    global_position_evaluator.selected_node_index = original_selected_node
    global_position_evaluator.followed_agent_id = original_followed_agent
    
    # Step 3: Calculate combined probabilities using multiplication (as in inspect_environment.py)
    combined_probabilities = {}
    threat_threshold = 0.1  # Same threshold as in environment_inspection_simulation.py
    
    # Get all node indices that appear in either agent's probability map
    all_node_indices = set(agent1_probabilities.keys()) | set(agent2_probabilities.keys())
    
    for node_idx in all_node_indices:
        prob1 = agent1_probabilities.get(node_idx, 0.0)
        prob2 = agent2_probabilities.get(node_idx, 0.0) 
        
        # Multiply probabilities (both must exist and be non-zero)
        if prob1 > 0 and prob2 > 0:
            combined_prob = prob1 * prob2
            
            # Store for combined probability visualization (using consistent threshold)
            if combined_prob >= threat_threshold:
                combined_probabilities[node_idx] = combined_prob
    
    return combined_probabilities

if __name__ == "__main__":
    # Test the position evaluator
    print("Testing Position Evaluator...")
    
    evaluator = PositionEvaluator()
    
    # Add some test agents
    evaluator.update_agent_position("agent1", 100, 100, 0, (50, 0.1))
    evaluator.update_agent_position("agent2", 200, 200, math.pi/4, (30, -0.05))
    evaluator.update_agent_position("agent3", 150, 120, math.pi/2)
    
    # Test distance calculations
    print(f"Distance agent1-agent2: {evaluator.get_distance_between_agents('agent1', 'agent2'):.2f}")
    print(f"Distance agent1-agent3: {evaluator.get_distance_between_agents('agent1', 'agent3'):.2f}")
    
    # Test all distances
    print("\nAll distances:")
    for (a1, a2), dist in evaluator.get_all_distances().items():
        print(f"  {a1} - {a2}: {dist:.2f}")
    
    # Test agents within range
    print(f"\nAgents within 100 pixels of agent1:")
    nearby = evaluator.get_agents_within_range("agent1", 100)
    for agent_id, distance in nearby:
        print(f"  {agent_id}: {distance:.2f}")
    
    # Test relative bearing
    bearing = evaluator.get_relative_bearing("agent1", "agent2")
    if bearing is not None:
        print(f"\nRelative bearing from agent1 to agent2: {math.degrees(bearing):.1f}°")
    
    # Test statistics
    print(f"\nStatistics:")
    stats = evaluator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
