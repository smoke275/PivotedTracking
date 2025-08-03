#!/usr/bin/env python3
"""
Position Evaluator Module
A standalone module for tracking agent positions and calculating distances between them.
Designed to be extensible for future spatial analysis and coordination algorithms.
"""

import math
import time
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class AgentPosition:
    """Data structure to hold agent position and metadata"""
    agent_id: str
    x: float
    y: float
    heading: float  # in radians
    timestamp: float
    velocity: Optional[Tuple[float, float]] = None  # (linear_vel, angular_vel)

@dataclass
class RRTNode:
    """Node in the RRT* tree"""
    x: float
    y: float
    parent: Optional['RRTNode'] = None
    cost: float = 0.0
    children: List['RRTNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

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
        
        # RRT* trees for each agent
        self.agent_rrt_trees: Dict[str, List[RRTNode]] = {}
        self.rrt_enabled = True
        self.rrt_max_nodes = 100
        self.rrt_step_size = 30.0
        self.rrt_search_radius = 50.0
        
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
        current_time = time.time()
        
        # Create new position record
        new_position = AgentPosition(
            agent_id=agent_id,
            x=x, y=y, 
            heading=heading,
            timestamp=current_time,
            velocity=velocity
        )
        
        # Update current position
        self.agent_positions[agent_id] = new_position
        
        # Add to history
        self.position_history[agent_id].append(new_position)
        
        # Trim history if needed
        if len(self.position_history[agent_id]) > self.max_history_length:
            self.position_history[agent_id].pop(0)
        
        # Check if agent moved significantly and update RRT tree if needed
        if self.rrt_enabled:
            should_update_rrt = False
            
            if agent_id not in self.agent_rrt_trees:
                should_update_rrt = True
            else:
                # Check if agent moved beyond threshold
                if len(self.position_history[agent_id]) >= 2:
                    prev_pos = self.position_history[agent_id][-2]
                    movement_dist = math.sqrt(
                        (new_position.x - prev_pos.x)**2 + 
                        (new_position.y - prev_pos.y)**2
                    )
                    if movement_dist > self.agent_movement_threshold:
                        should_update_rrt = True
            
            if should_update_rrt:
                self.update_agent_rrt_tree(agent_id)
        
        self.last_update_time = current_time
    
    def get_agent_position(self, agent_id: str) -> Optional[AgentPosition]:
        """Get current position of an agent."""
        return self.agent_positions.get(agent_id)
    
    def get_agent_history(self, agent_id: str) -> List[AgentPosition]:
        """Get position history for an agent."""
        return self.position_history.get(agent_id, [])
    
    def get_distance_between_agents(self, agent1_id: str, agent2_id: str) -> Optional[float]:
        """
        Calculate Euclidean distance between two agents.
        
        Args:
            agent1_id: ID of first agent
            agent2_id: ID of second agent
            
        Returns:
            Distance in pixels, or None if either agent not found
        """
        if agent1_id not in self.agent_positions or agent2_id not in self.agent_positions:
            return None
        
        pos1 = self.agent_positions[agent1_id]
        pos2 = self.agent_positions[agent2_id]
        
        return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
    
    def get_all_distances(self) -> Dict[Tuple[str, str], float]:
        """
        Calculate distances between all pairs of agents.
        
        Returns:
            Dictionary mapping agent pairs to distances
        """
        distances = {}
        agent_ids = list(self.agent_positions.keys())
        
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent1, agent2 = agent_ids[i], agent_ids[j]
                distance = self.get_distance_between_agents(agent1, agent2)
                if distance is not None:
                    distances[(agent1, agent2)] = distance
        
        return distances
    
    def get_agents_within_range(self, agent_id: str, max_distance: float) -> List[Tuple[str, float]]:
        """
        Find all agents within a specified range of a target agent.
        
        Args:
            agent_id: Target agent ID
            max_distance: Maximum distance threshold
            
        Returns:
            List of (agent_id, distance) tuples for agents within range
        """
        if agent_id not in self.agent_positions:
            return []
        
        nearby_agents = []
        
        for other_agent_id in self.agent_positions:
            if other_agent_id != agent_id:
                distance = self.get_distance_between_agents(agent_id, other_agent_id)
                if distance is not None and distance <= max_distance:
                    nearby_agents.append((other_agent_id, distance))
        
        # Sort by distance
        nearby_agents.sort(key=lambda x: x[1])
        return nearby_agents
    
    def get_relative_bearing(self, observer_id: str, target_id: str) -> Optional[float]:
        """
        Calculate relative bearing from observer to target.
        
        Args:
            observer_id: ID of observing agent
            target_id: ID of target agent
            
        Returns:
            Bearing in radians, or None if either agent not found
        """
        if observer_id not in self.agent_positions or target_id not in self.agent_positions:
            return None
        
        observer = self.agent_positions[observer_id]
        target = self.agent_positions[target_id]
        
        # Calculate angle from observer to target
        dx = target.x - observer.x
        dy = target.y - observer.y
        bearing = math.atan2(dy, dx)
        
        # Make relative to observer's heading
        relative_bearing = bearing - observer.heading
        
        # Normalize to [-pi, pi]
        while relative_bearing > math.pi:
            relative_bearing -= 2 * math.pi
        while relative_bearing < -math.pi:
            relative_bearing += 2 * math.pi
        
        return relative_bearing
    
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
        Find the closest map graph node to an agent.
        
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
        min_distance = float('inf')
        closest_node_index = None
        
        for i, node_pos in enumerate(self.map_graph.nodes):
            distance = math.sqrt(
                (node_pos[0] - agent_position[0])**2 + 
                (node_pos[1] - agent_position[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_node_index = i
        
        return closest_node_index
    
    def _line_intersects_wall(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """
        Check if a line segment intersects with any wall in the environment.
        
        Args:
            start: Start point (x, y)
            end: End point (x, y)
            
        Returns:
            True if line intersects any wall, False otherwise
        """
        if not self.environment or not hasattr(self.environment, 'get_all_walls'):
            return False
        
        walls = self.environment.get_all_walls()
        
        for wall in walls:
            if self._line_segment_intersection(start, end, (wall.x1, wall.y1), (wall.x2, wall.y2)):
                return True
        
        return False
    
    def _line_segment_intersection(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                                 p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """
        Check if two line segments intersect.
        
        Args:
            p1, p2: First line segment endpoints
            p3, p4: Second line segment endpoints
            
        Returns:
            True if segments intersect, False otherwise
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def _is_point_in_environment(self, x: float, y: float) -> bool:
        """
        Check if a point is within the environment bounds and not inside walls.
        
        Args:
            x, y: Point coordinates
            
        Returns:
            True if point is valid, False otherwise
        """
        if not self.environment:
            return True
        
        # Check environment bounds (assuming environment has width/height)
        if hasattr(self.environment, 'width') and hasattr(self.environment, 'height'):
            if x < 0 or x >= self.environment.width or y < 0 or y >= self.environment.height:
                return False
        
        # For simplicity, we'll assume points are valid if they don't intersect walls
        # A more sophisticated implementation would check if point is inside any wall polygon
        return True
    
    def generate_rrt_star_tree(self, agent_id: str) -> List[RRTNode]:
        """
        Generate an RRT* tree from the agent's current position.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of RRT nodes representing the tree
        """
        if agent_id not in self.agent_positions or not self.environment:
            return []
        
        agent_pos = self.agent_positions[agent_id]
        start_pos = (agent_pos.x, agent_pos.y)
        
        # Initialize tree with root node
        root = RRTNode(x=start_pos[0], y=start_pos[1], cost=0.0)
        tree = [root]
        
        # Get environment bounds for sampling
        env_width = getattr(self.environment, 'width', 1280)
        env_height = getattr(self.environment, 'height', 720)
        
        for _ in range(self.rrt_max_nodes - 1):  # -1 because we already have root
            # Sample random point
            rand_x = random.uniform(0, env_width)
            rand_y = random.uniform(0, env_height)
            rand_point = (rand_x, rand_y)
            
            # Find nearest node in tree
            nearest_node = self._find_nearest_node(tree, rand_point)
            if nearest_node is None:
                continue
            
            # Steer towards random point
            new_point = self._steer(
                (nearest_node.x, nearest_node.y), 
                rand_point, 
                self.rrt_step_size
            )
            
            # Check if path is collision-free
            if not self._line_intersects_wall((nearest_node.x, nearest_node.y), new_point):
                if self._is_point_in_environment(new_point[0], new_point[1]):
                    # Create new node
                    new_cost = nearest_node.cost + self._distance(
                        (nearest_node.x, nearest_node.y), new_point
                    )
                    new_node = RRTNode(
                        x=new_point[0], 
                        y=new_point[1], 
                        parent=nearest_node, 
                        cost=new_cost
                    )
                    
                    # RRT* optimization: find nodes within search radius
                    near_nodes = self._find_near_nodes(tree, new_point, self.rrt_search_radius)
                    
                    # Choose parent with minimum cost
                    best_parent = nearest_node
                    best_cost = new_cost
                    
                    for near_node in near_nodes:
                        potential_cost = near_node.cost + self._distance(
                            (near_node.x, near_node.y), new_point
                        )
                        if (potential_cost < best_cost and 
                            not self._line_intersects_wall((near_node.x, near_node.y), new_point)):
                            best_parent = near_node
                            best_cost = potential_cost
                    
                    new_node.parent = best_parent
                    new_node.cost = best_cost
                    best_parent.children.append(new_node)
                    
                    # Rewire tree: check if new node provides better path to near nodes
                    for near_node in near_nodes:
                        if near_node != best_parent:
                            new_cost_via_new = new_node.cost + self._distance(
                                new_point, (near_node.x, near_node.y)
                            )
                            if (new_cost_via_new < near_node.cost and
                                not self._line_intersects_wall(new_point, (near_node.x, near_node.y))):
                                # Rewire
                                if near_node.parent:
                                    near_node.parent.children.remove(near_node)
                                near_node.parent = new_node
                                near_node.cost = new_cost_via_new
                                new_node.children.append(near_node)
                    
                    tree.append(new_node)
        
        return tree
    
    def _find_nearest_node(self, tree: List[RRTNode], point: Tuple[float, float]) -> Optional[RRTNode]:
        """Find the nearest node in the tree to a given point."""
        if not tree:
            return None
        
        min_dist = float('inf')
        nearest = None
        
        for node in tree:
            dist = self._distance((node.x, node.y), point)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _find_near_nodes(self, tree: List[RRTNode], point: Tuple[float, float], radius: float) -> List[RRTNode]:
        """Find all nodes within a given radius of a point."""
        near_nodes = []
        for node in tree:
            if self._distance((node.x, node.y), point) <= radius:
                near_nodes.append(node)
        return near_nodes
    
    def _steer(self, from_point: Tuple[float, float], to_point: Tuple[float, float], max_dist: float) -> Tuple[float, float]:
        """
        Steer from one point towards another, limited by maximum distance.
        
        Args:
            from_point: Starting point
            to_point: Target point
            max_dist: Maximum step distance
            
        Returns:
            New point stepped towards target
        """
        dx = to_point[0] - from_point[0]
        dy = to_point[1] - from_point[1]
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist <= max_dist:
            return to_point
        
        # Normalize and scale
        scale = max_dist / dist
        new_x = from_point[0] + dx * scale
        new_y = from_point[1] + dy * scale
        
        return (new_x, new_y)
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def update_agent_rrt_tree(self, agent_id: str):
        """
        Update the RRT* tree for a specific agent.
        
        Args:
            agent_id: ID of the agent to update tree for
        """
        if self.rrt_enabled and agent_id in self.agent_positions:
            tree = self.generate_rrt_star_tree(agent_id)
            self.agent_rrt_trees[agent_id] = tree
    
    def get_agent_rrt_tree(self, agent_id: str) -> List[RRTNode]:
        """
        Get the RRT* tree for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of RRT nodes, or empty list if no tree exists
        """
        return self.agent_rrt_trees.get(agent_id, [])
    
    def update_all_rrt_trees(self):
        """Update RRT* trees for all tracked agents."""
        for agent_id in self.agent_positions:
            self.update_agent_rrt_tree(agent_id)
    
    def set_rrt_parameters(self, max_nodes: int = 100, step_size: float = 30.0, search_radius: float = 50.0):
        """
        Set RRT* algorithm parameters.
        
        Args:
            max_nodes: Maximum number of nodes in each tree
            step_size: Maximum step size for tree expansion
            search_radius: Radius for finding near nodes during optimization
        """
        self.rrt_max_nodes = max_nodes
        self.rrt_step_size = step_size
        self.rrt_search_radius = search_radius
    
    def enable_rrt(self):
        """Enable RRT* tree generation."""
        self.rrt_enabled = True
    
    def disable_rrt(self):
        """Disable RRT* tree generation."""
        self.rrt_enabled = False
        self.agent_rrt_trees.clear()
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about tracked agents.
        
        Returns:
            Dictionary containing various statistics
        """
        stats = {
            "agent_count": len(self.agent_positions),
            "total_position_updates": sum(len(history) for history in self.position_history.values()),
            "last_update_time": self.last_update_time,
            "has_environment": self.has_environment_data(),
        }
        
        # Calculate distance statistics if we have multiple agents
        distances = list(self.get_all_distances().values())
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
        
        # Add RRT* tree information
        if self.rrt_enabled:
            rrt_stats = {
                "rrt_enabled": True,
                "rrt_max_nodes": self.rrt_max_nodes,
                "rrt_step_size": self.rrt_step_size,
                "rrt_search_radius": self.rrt_search_radius,
                "agents_with_trees": len(self.agent_rrt_trees)
            }
            
            # Add per-agent tree node counts
            for agent_id, tree in self.agent_rrt_trees.items():
                rrt_stats[f"rrt_nodes_{agent_id}"] = len(tree)
            
            stats.update(rrt_stats)
        else:
            stats["rrt_enabled"] = False
        
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
    
    def analyze_movement_patterns(self, agent_id: str, window_size: int = 10) -> Dict:
        """
        Placeholder for future movement pattern analysis.
        
        Args:
            agent_id: Agent to analyze
            window_size: Number of recent positions to analyze
            
        Returns:
            Dictionary containing movement pattern metrics
        """
        # TODO: Implement movement pattern analysis (speed, direction changes, etc.)
        return {}


# Global position evaluator instance
global_position_evaluator = PositionEvaluator()

# Convenience functions for global access
def update_position(agent_id: str, x: float, y: float, heading: float, velocity: Optional[Tuple[float, float]] = None):
    """Convenience function to update agent position in global evaluator."""
    global_position_evaluator.update_agent_position(agent_id, x, y, heading, velocity)

def get_distance(agent1_id: str, agent2_id: str) -> Optional[float]:
    """Convenience function to get distance between two agents."""
    return global_position_evaluator.get_distance_between_agents(agent1_id, agent2_id)

def get_stats() -> Dict:
    """Convenience function to get evaluator statistics."""
    return global_position_evaluator.get_statistics()

def set_environment_data(environment, map_graph):
    """Convenience function to set environment data."""
    global_position_evaluator.set_environment_data(environment, map_graph)

def find_closest_node(agent_id: str) -> Optional[int]:
    """Convenience function to find closest graph node to an agent."""
    return global_position_evaluator.find_closest_graph_node(agent_id)

def get_agent_rrt_tree(agent_id: str) -> List[RRTNode]:
    """Convenience function to get RRT* tree for an agent."""
    return global_position_evaluator.get_agent_rrt_tree(agent_id)

def update_agent_rrt_tree(agent_id: str):
    """Convenience function to update RRT* tree for an agent."""
    global_position_evaluator.update_agent_rrt_tree(agent_id)

def update_all_rrt_trees():
    """Convenience function to update RRT* trees for all agents."""
    global_position_evaluator.update_all_rrt_trees()

def set_rrt_parameters(max_nodes: int = 100, step_size: float = 30.0, search_radius: float = 50.0):
    """Convenience function to set RRT* parameters."""
    global_position_evaluator.set_rrt_parameters(max_nodes, step_size, search_radius)


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
