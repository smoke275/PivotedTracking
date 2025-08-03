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
import numpy as np

# Try to import scipy for KD-tree optimization
try:
    from scipy.spatial import KDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Using linear search for closest node finding.")

# Try to import dubins library, fallback to manual implementation if not available
try:
    import dubins
    DUBINS_AVAILABLE = True
except ImportError:
    DUBINS_AVAILABLE = False
    print("Warning: dubins library not available. Using manual Dubins implementation.")

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
        
        # Closest node optimization infrastructure
        self._closest_node_cache: Dict[str, Tuple[int, float, float, float]] = {}  # agent_id -> (node_idx, x, y, timestamp)
        self._map_graph_kdtree: Optional[KDTree] = None
        self._map_graph_nodes_array: Optional[np.ndarray] = None
        self._cache_movement_threshold = 15.0  # Recalculate if agent moved more than this distance
        self._cache_time_threshold = 2.0  # Recalculate after this many seconds
        self._kdtree_rebuild_needed = True
        
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
        self.rrt_forward_bias = 0.7  # Probability of sampling in forward direction (0.0 = no bias, 1.0 = always forward)
        self.rrt_forward_cone_angle = math.pi / 3  # 60 degrees cone in forward direction
        
        # Dubins path parameters
        self.dubins_enabled = True
        self.dubins_turning_radius = 30.0  # Minimum turning radius for vehicle (pixels)
        self.dubins_step_size = 1.0  # Step size for Dubins path discretization
        self.agent_velocity = 50.0  # Agent velocity for time calculations (pixels/second)
        
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
        Set environment data for advanced position evaluation features.
        
        Args:
            environment: The simulation environment with walls and doors
            map_graph: The map graph for pathfinding
        """
        self.environment = environment
        self.map_graph = map_graph
        
        # Mark KD-tree for rebuilding when map graph changes
        self._kdtree_rebuild_needed = True
        self._closest_node_cache.clear()  # Clear cache when environment changes
        print(f"Environment data updated. Map graph has {len(map_graph.nodes) if map_graph and hasattr(map_graph, 'nodes') else 0} nodes")
    
    def _build_map_graph_kdtree(self):
        """Build KD-tree for fast closest node queries."""
        if not self.has_environment_data() or not hasattr(self.map_graph, 'nodes') or not self.map_graph.nodes:
            self._map_graph_kdtree = None
            self._map_graph_nodes_array = None
            return
        
        if SCIPY_AVAILABLE:
            # Convert nodes to numpy array for KD-tree
            self._map_graph_nodes_array = np.array(self.map_graph.nodes)
            self._map_graph_kdtree = KDTree(self._map_graph_nodes_array)
            print(f"Built KD-tree with {len(self.map_graph.nodes)} nodes for optimized closest node queries")
        else:
            # Fallback: just store nodes as numpy array for faster iteration
            self._map_graph_nodes_array = np.array(self.map_graph.nodes) if self.map_graph.nodes else None
            self._map_graph_kdtree = None
        
        self._kdtree_rebuild_needed = False
    
    def has_environment_data(self) -> bool:
        """Check if environment data is available for advanced features."""
        return self.environment is not None and self.map_graph is not None
    
    def find_closest_graph_node(self, agent_id: str) -> Optional[int]:
        """
        Find the closest map graph node to an agent using optimized search.
        Uses caching and KD-tree for improved performance.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Index of closest node, or None if not found
        """
        if not self.has_environment_data() or agent_id not in self.agent_positions:
            return None
        
        if not hasattr(self.map_graph, 'nodes') or not self.map_graph.nodes:
            return None
        
        # Build KD-tree if needed
        if self._kdtree_rebuild_needed:
            self._build_map_graph_kdtree()
        
        agent_pos = self.agent_positions[agent_id]
        current_time = time.time()
        
        # Check cache validity
        if agent_id in self._closest_node_cache:
            cached_node_idx, cached_x, cached_y, cached_time = self._closest_node_cache[agent_id]
            
            # Check if agent moved significantly or cache is too old
            distance_moved = math.sqrt((agent_pos.x - cached_x)**2 + (agent_pos.y - cached_y)**2)
            time_elapsed = current_time - cached_time
            
            if (distance_moved < self._cache_movement_threshold and 
                time_elapsed < self._cache_time_threshold):
                return cached_node_idx
        
        # Find closest node using optimized search
        closest_node_index = self._find_closest_node_optimized(agent_pos.x, agent_pos.y)
        
        # Update cache
        if closest_node_index is not None:
            self._closest_node_cache[agent_id] = (
                closest_node_index, agent_pos.x, agent_pos.y, current_time
            )
        
        return closest_node_index
    
    def _find_closest_node_optimized(self, x: float, y: float) -> Optional[int]:
        """
        Optimized closest node search using KD-tree or vectorized operations.
        
        Args:
            x, y: Query position
            
        Returns:
            Index of closest node, or None if not found
        """
        if self._map_graph_kdtree is not None:
            # Use KD-tree for O(log n) search
            try:
                distance, index = self._map_graph_kdtree.query([x, y])
                return int(index)
            except Exception as e:
                print(f"KD-tree query failed: {e}, falling back to linear search")
                # Fall through to backup methods
        
        if self._map_graph_nodes_array is not None:
            # Use vectorized numpy operations for faster linear search
            try:
                # Calculate all distances at once
                distances = np.sqrt(
                    (self._map_graph_nodes_array[:, 0] - x)**2 + 
                    (self._map_graph_nodes_array[:, 1] - y)**2
                )
                return int(np.argmin(distances))
            except Exception as e:
                print(f"Vectorized search failed: {e}, falling back to basic search")
        
        # Fallback: basic linear search (original implementation)
        min_distance = float('inf')
        closest_node_index = None
        
        for i, node_pos in enumerate(self.map_graph.nodes):
            distance = math.sqrt(
                (node_pos[0] - x)**2 + 
                (node_pos[1] - y)**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_node_index = i
        
        return closest_node_index
    
    def _line_intersects_wall(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """
        Check if a line segment intersects with any wall in the environment.
        Doors are considered passable and do not block the line.
        
        Args:
            start: Start point (x, y)
            end: End point (x, y)
            
        Returns:
            True if line intersects any wall (but not doors), False otherwise
        """
        if not self.environment or not hasattr(self.environment, 'get_all_walls'):
            return False
        
        walls = self.environment.get_all_walls()
        doors = self.environment.get_doors() if hasattr(self.environment, 'get_doors') else []
        
        for wall in walls:
            # Check if line intersects with any of the four edges of the rectangle
            wall_edges = [
                # Top edge
                ((wall.x, wall.y), (wall.x + wall.width, wall.y)),
                # Bottom edge
                ((wall.x, wall.y + wall.height), (wall.x + wall.width, wall.y + wall.height)),
                # Left edge
                ((wall.x, wall.y), (wall.x, wall.y + wall.height)),
                # Right edge
                ((wall.x + wall.width, wall.y), (wall.x + wall.width, wall.y + wall.height))
            ]
            
            for edge_start, edge_end in wall_edges:
                if self._line_segment_intersection(start, end, edge_start, edge_end):
                    # Check if the intersection point is within a door (doors are passable)
                    # Calculate intersection point
                    intersection_point = self._get_intersection_point(start, end, edge_start, edge_end)
                    if intersection_point:
                        ix, iy = intersection_point
                        
                        # Check if intersection is within any door
                        in_door = False
                        for door in doors:
                            # Check if intersection point is inside the door rectangle
                            if (door.x <= ix <= door.x + door.width and 
                                door.y <= iy <= door.y + door.height):
                                in_door = True
                                break
                        
                        # If intersection is not in a door, it's a blocking wall
                        if not in_door:
                            return True
        
        return False
    
    def _get_intersection_point(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                              p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Calculate the intersection point of two line segments.
        
        Args:
            p1, p2: First line segment endpoints
            p3, p4: Second line segment endpoints
            
        Returns:
            Intersection point (x, y) or None if no intersection
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Calculate intersection point
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (ix, iy)
        
        return None
    
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
            # Sample random point with forward bias
            if random.random() < self.rrt_forward_bias:
                # Biased sampling in the forward direction
                rand_point = self._sample_forward_biased(agent_pos, env_width, env_height)
            else:
                # Uniform random sampling
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
    
    def _sample_forward_biased(self, agent_pos: AgentPosition, env_width: float, env_height: float) -> Tuple[float, float]:
        """
        Sample a point biased towards the agent's forward direction.
        
        Args:
            agent_pos: Current agent position with heading
            env_width: Environment width
            env_height: Environment height
            
        Returns:
            Sampled point (x, y) biased in forward direction
        """
        agent_heading = agent_pos.heading
        
        # Sample distance from agent (exponential distribution favors closer points)
        max_sample_distance = min(env_width, env_height) * 0.6  # Sample within reasonable range
        sample_distance = random.expovariate(1.0 / (max_sample_distance * 0.3))  # Mean distance = 30% of max
        sample_distance = min(sample_distance, max_sample_distance)
        
        # Sample angle within forward cone
        angle_offset = random.uniform(-self.rrt_forward_cone_angle / 2, self.rrt_forward_cone_angle / 2)
        sample_angle = agent_heading + angle_offset
        
        # Calculate sample point
        sample_x = agent_pos.x + sample_distance * math.cos(sample_angle)
        sample_y = agent_pos.y + sample_distance * math.sin(sample_angle)
        
        # Clamp to environment bounds
        sample_x = max(0, min(sample_x, env_width))
        sample_y = max(0, min(sample_y, env_height))
        
        return (sample_x, sample_y)
    
    # ===== DUBINS PATH FUNCTIONALITY =====
    
    def calculate_dubins_path_to_node(self, agent_id: str, target_node: RRTNode) -> Optional[Tuple[float, float, List[Tuple[float, float]]]]:
        """
        Calculate the Dubins shortest path from agent's current position to a target RRT node.
        
        Args:
            agent_id: ID of the agent
            target_node: Target RRT node
            
        Returns:
            Tuple of (path_length, travel_time, path_points) or None if calculation fails
        """
        if agent_id not in self.agent_positions:
            return None
            
        agent_pos = self.agent_positions[agent_id]
        
        # Start configuration: (x, y, heading)
        start_config = (agent_pos.x, agent_pos.y, agent_pos.heading)
        
        # For target, we need to estimate heading. Use direction from parent or current agent heading
        target_heading = self._estimate_node_heading(target_node, agent_pos.heading)
        target_config = (target_node.x, target_node.y, target_heading)
        
        # Calculate Dubins path
        if DUBINS_AVAILABLE:
            try:
                path = dubins.shortest_path(start_config, target_config, self.dubins_turning_radius)
                path_length = path.path_length()
                path_points = self._sample_dubins_path(path, self.dubins_step_size)
            except Exception as e:
                print(f"Dubins library error: {e}")
                return self._calculate_manual_dubins_path(start_config, target_config)
        else:
            return self._calculate_manual_dubins_path(start_config, target_config)
        
        # Calculate travel time
        travel_time = path_length / self.agent_velocity
        
        return (path_length, travel_time, path_points)
    
    def get_dubins_times_to_all_nodes(self, agent_id: str) -> Dict[int, Tuple[float, float]]:
        """
        Calculate Dubins path length and time to all nodes in the agent's RRT tree.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary mapping node index to (path_length, travel_time)
        """
        if agent_id not in self.agent_rrt_trees:
            return {}
        
        tree = self.agent_rrt_trees[agent_id]
        dubins_data = {}
        
        for i, node in enumerate(tree):
            if i == 0:  # Skip root node (agent's current position)
                dubins_data[i] = (0.0, 0.0)
                continue
                
            result = self.calculate_dubins_path_to_node(agent_id, node)
            if result:
                path_length, travel_time, _ = result
                dubins_data[i] = (path_length, travel_time)
            else:
                # Fallback to Euclidean distance if Dubins calculation fails
                if agent_id in self.agent_positions:
                    agent_pos = self.agent_positions[agent_id]
                    euclidean_dist = math.sqrt((node.x - agent_pos.x)**2 + (node.y - agent_pos.y)**2)
                    euclidean_time = euclidean_dist / self.agent_velocity
                    dubins_data[i] = (euclidean_dist, euclidean_time)
        
        return dubins_data
    
    def find_shortest_time_node(self, agent_id: str) -> Optional[Tuple[int, RRTNode, float]]:
        """
        Find the RRT node with the shortest Dubins travel time from agent's current position.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Tuple of (node_index, node, travel_time) or None if no nodes found
        """
        dubins_times = self.get_dubins_times_to_all_nodes(agent_id)
        if not dubins_times:
            return None
        
        # Find node with minimum travel time (excluding root node)
        min_time = float('inf')
        best_node_idx = None
        
        for node_idx, (_, travel_time) in dubins_times.items():
            if node_idx > 0 and travel_time < min_time:  # Skip root node (index 0)
                min_time = travel_time
                best_node_idx = node_idx
        
        if best_node_idx is not None:
            tree = self.agent_rrt_trees[agent_id]
            return (best_node_idx, tree[best_node_idx], min_time)
        
        return None
    
    def _estimate_node_heading(self, node: RRTNode, default_heading: float) -> float:
        """
        Estimate the heading at a node based on its parent relationship.
        
        Args:
            node: The RRT node
            default_heading: Default heading to use if estimation fails
            
        Returns:
            Estimated heading in radians
        """
        if node.parent is None:
            return default_heading
        
        # Calculate heading from parent to this node
        dx = node.x - node.parent.x
        dy = node.y - node.parent.y
        
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return default_heading
        
        return math.atan2(dy, dx)
    
    def _sample_dubins_path(self, dubins_path, step_size: float) -> List[Tuple[float, float]]:
        """
        Sample points along a Dubins path.
        
        Args:
            dubins_path: Dubins path object
            step_size: Distance between sampled points
            
        Returns:
            List of (x, y) points along the path
        """
        if not DUBINS_AVAILABLE:
            return []
        
        path_length = dubins_path.path_length()
        num_samples = max(2, int(path_length / step_size))
        
        points = []
        for i in range(num_samples + 1):
            t = i / num_samples
            distance = t * path_length
            try:
                config = dubins_path.sample(distance)
                points.append((config[0], config[1]))
            except:
                break
        
        return points
    
    def _calculate_manual_dubins_path(self, start_config: Tuple[float, float, float], 
                                    end_config: Tuple[float, float, float]) -> Optional[Tuple[float, float, List[Tuple[float, float]]]]:
        """
        Manual implementation of Dubins path calculation as fallback.
        Simplified implementation - considers only straight line for now.
        
        Args:
            start_config: (x, y, heading) start configuration
            end_config: (x, y, heading) end configuration
            
        Returns:
            Tuple of (path_length, travel_time, path_points) or None
        """
        # Simplified: use straight line distance as approximation
        x1, y1, h1 = start_config
        x2, y2, h2 = end_config
        
        straight_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Add penalty for heading changes (rough approximation)
        heading_diff = abs(h2 - h1)
        while heading_diff > math.pi:
            heading_diff = abs(heading_diff - 2 * math.pi)
        
        # Approximate additional distance for turning
        turn_penalty = heading_diff * self.dubins_turning_radius
        total_distance = straight_distance + turn_penalty
        
        travel_time = total_distance / self.agent_velocity
        
        # Simple straight line path points
        num_points = max(2, int(total_distance / self.dubins_step_size))
        path_points = []
        for i in range(num_points + 1):
            t = i / num_points
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            path_points.append((x, y))
        
        return (total_distance, travel_time, path_points)
    
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
    
    def get_path_to_node(self, agent_id: str, node: RRTNode) -> List[RRTNode]:
        """
        Get the path from root to a specific node in the RRT* tree.
        
        Args:
            agent_id: ID of the agent
            node: The target node to find path to
            
        Returns:
            List of nodes from root to target node (including both endpoints)
        """
        path = []
        current = node
        
        # Traverse back from target to root
        while current is not None:
            path.append(current)
            current = current.parent
        
        # Reverse to get path from root to target
        path.reverse()
        return path
    
    def find_node_at_position(self, agent_id: str, x: float, y: float, tolerance: float = 10.0) -> Optional[RRTNode]:
        """
        Find an RRT node near a given position.
        
        Args:
            agent_id: ID of the agent
            x, y: Position to search near
            tolerance: Maximum distance to consider a match
            
        Returns:
            The closest node within tolerance, or None if no node found
        """
        tree = self.get_agent_rrt_tree(agent_id)
        closest_node = None
        closest_distance = float('inf')
        
        for node in tree:
            distance = math.sqrt((node.x - x) ** 2 + (node.y - y) ** 2)
            if distance <= tolerance and distance < closest_distance:
                closest_node = node
                closest_distance = distance
        
        return closest_node
    
    def update_all_rrt_trees(self):
        """Update RRT* trees for all tracked agents."""
        for agent_id in self.agent_positions:
            self.update_agent_rrt_tree(agent_id)
    
    def set_rrt_parameters(self, max_nodes: int = 100, step_size: float = 30.0, search_radius: float = 50.0,
                         forward_bias: float = 0.7, forward_cone_angle: float = math.pi / 3,
                         dubins_turning_radius: float = 30.0, agent_velocity: float = 50.0):
        """
        Set RRT* algorithm parameters.
        
        Args:
            max_nodes: Maximum number of nodes in each tree
            step_size: Maximum step size for tree expansion
            search_radius: Radius for finding near nodes during optimization
            forward_bias: Probability of sampling in forward direction (0.0 = no bias, 1.0 = always forward)
            forward_cone_angle: Angle of forward sampling cone in radians
            dubins_turning_radius: Minimum turning radius for Dubins paths (pixels)
            agent_velocity: Agent velocity for time calculations (pixels/second)
        """
        self.rrt_max_nodes = max_nodes
        self.rrt_step_size = step_size
        self.rrt_search_radius = search_radius
        self.rrt_forward_bias = forward_bias
        self.rrt_forward_cone_angle = forward_cone_angle
        self.dubins_turning_radius = dubins_turning_radius
        self.agent_velocity = agent_velocity
    
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
                "rrt_forward_bias": self.rrt_forward_bias,
                "rrt_forward_cone_angle": math.degrees(self.rrt_forward_cone_angle),
                "dubins_enabled": self.dubins_enabled,
                "dubins_turning_radius": self.dubins_turning_radius,
                "dubins_available": DUBINS_AVAILABLE,
                "agent_velocity": self.agent_velocity,
                "agents_with_trees": len(self.agent_rrt_trees)
            }
            
            # Add per-agent tree node counts
            for agent_id, tree in self.agent_rrt_trees.items():
                rrt_stats[f"rrt_nodes_{agent_id}"] = len(tree)
            
            stats.update(rrt_stats)
        else:
            stats["rrt_enabled"] = False
        
        return stats
    
    # Closest node optimization methods
    
    def clear_closest_node_cache(self):
        """Clear the closest node cache to force recalculation."""
        self._closest_node_cache.clear()
        print("Closest node cache cleared")
    
    def set_closest_node_cache_parameters(self, movement_threshold: float = 15.0, time_threshold: float = 2.0):
        """
        Configure closest node cache parameters.
        
        Args:
            movement_threshold: Minimum distance agent must move to invalidate cache
            time_threshold: Maximum age of cache entry in seconds
        """
        self._cache_movement_threshold = movement_threshold
        self._cache_time_threshold = time_threshold
        print(f"Closest node cache parameters updated: movement={movement_threshold}, time={time_threshold}s")
    
    def get_closest_node_cache_stats(self) -> Dict:
        """Get statistics about closest node cache performance."""
        cache_size = len(self._closest_node_cache)
        has_kdtree = self._map_graph_kdtree is not None
        optimization_method = "KD-tree" if has_kdtree else ("Vectorized" if self._map_graph_nodes_array is not None else "Linear")
        
        return {
            "cache_size": cache_size,
            "optimization_method": optimization_method,
            "movement_threshold": self._cache_movement_threshold,
            "time_threshold": self._cache_time_threshold,
            "scipy_available": SCIPY_AVAILABLE,
            "kdtree_available": has_kdtree
        }
    
    def force_rebuild_spatial_index(self):
        """Force rebuilding of spatial index (KD-tree) on next query."""
        self._kdtree_rebuild_needed = True
        self.clear_closest_node_cache()
        print("Spatial index marked for rebuilding")
    
    # RRT to Map Graph Mapping for Strategic Analysis
    
    def map_rrt_nodes_to_graph(self, agent_id: str) -> Dict[int, int]:
        """
        Map each RRT node to its closest map graph node.
        
        Args:
            agent_id: ID of the agent whose RRT tree to process
            
        Returns:
            Dictionary mapping RRT node index to closest map graph node index
            Returns empty dict if no RRT tree or map graph available
        """
        if not self.has_environment_data():
            return {}
        
        if agent_id not in self.agent_rrt_trees:
            return {}
        
        rrt_tree = self.agent_rrt_trees[agent_id]
        if not rrt_tree:
            return {}
        
        # Build KD-tree if needed for optimal performance
        if self._kdtree_rebuild_needed:
            self._build_map_graph_kdtree()
        
        mapping = {}
        for i, rrt_node in enumerate(rrt_tree):
            closest_graph_node_idx = self._find_closest_node_optimized(rrt_node.x, rrt_node.y)
            if closest_graph_node_idx is not None:
                mapping[i] = closest_graph_node_idx
        
        return mapping
    
    def map_all_rrt_nodes_to_graph(self) -> Dict[str, Dict[int, int]]:
        """
        Map all RRT nodes to their closest map graph nodes for all agents.
        
        Returns:
            Dictionary mapping agent_id -> {rrt_node_index -> map_graph_node_index}
        """
        all_mappings = {}
        
        for agent_id in self.agent_rrt_trees.keys():
            mapping = self.map_rrt_nodes_to_graph(agent_id)
            if mapping:
                all_mappings[agent_id] = mapping
        
        return all_mappings
    
    def get_rrt_to_graph_mapping_stats(self) -> Dict:
        """Get statistics about RRT to map graph mappings."""
        all_mappings = self.map_all_rrt_nodes_to_graph()
        
        stats = {
            "total_agents": len(all_mappings),
            "mappings_per_agent": {}
        }
        
        for agent_id, mapping in all_mappings.items():
            rrt_tree_size = len(self.agent_rrt_trees.get(agent_id, []))
            stats["mappings_per_agent"][agent_id] = {
                "rrt_nodes": rrt_tree_size,
                "mapped_nodes": len(mapping),
                "mapping_success_rate": len(mapping) / rrt_tree_size if rrt_tree_size > 0 else 0.0
            }
        
        return stats
    
    # Strategic Pursuit-Evasion Analysis
    
    def calculate_pursuit_evasion_advantages(self, pursuer_id: str, evader_id: str) -> Dict[int, float]:
        """
        Calculate strategic time advantages for pursuit-evasion scenario.
        
        For each pursuer RRT node:
        1. Find closest map graph node
        2. Check visibility to all evader RRT nodes from that map graph position
        3. For invisible evader nodes, calculate time advantage (evader_time - pursuer_time)
        4. Store maximum time advantage for the pursuer node
        
        Args:
            pursuer_id: ID of the pursuing agent
            evader_id: ID of the evading agent
            
        Returns:
            Dictionary mapping pursuer RRT node index -> maximum time advantage
        """
        if not self.has_environment_data():
            print("No environment data available for visibility analysis")
            return {}
        
        if pursuer_id not in self.agent_rrt_trees or evader_id not in self.agent_rrt_trees:
            print(f"Missing RRT trees for {pursuer_id} or {evader_id}")
            return {}
        
        # Get RRT-to-map-graph mappings
        pursuer_mapping = self.map_rrt_nodes_to_graph(pursuer_id)
        evader_mapping = self.map_rrt_nodes_to_graph(evader_id)
        
        if not pursuer_mapping or not evader_mapping:
            print("Missing RRT-to-map-graph mappings")
            return {}
        
        # Get travel times (this should be available from previous calculations)
        try:
            # Import the trajectory calculator functions
            from path_trajectory_optimizer import get_trajectory_calculator
            
            trajectory_calc = get_trajectory_calculator()
            pursuer_times = trajectory_calc.get_travel_times_async(pursuer_id)
            evader_times = trajectory_calc.get_travel_times_async(evader_id)
            
            if not pursuer_times or not evader_times:
                print("Travel times not available - run 'U' key first to calculate travel times")
                return {}
            
        except Exception as e:
            print(f"Could not access travel times: {e}")
            return {}
        
        # Convert travel times to dictionaries for faster lookup
        pursuer_time_dict = {}
        for node, travel_time, path_len in pursuer_times:
            # Find the node index in the pursuer tree
            pursuer_tree = self.agent_rrt_trees[pursuer_id]
            for i, tree_node in enumerate(pursuer_tree):
                if tree_node.x == node.x and tree_node.y == node.y:
                    pursuer_time_dict[i] = travel_time
                    break
        
        evader_time_dict = {}
        for node, travel_time, path_len in evader_times:
            # Find the node index in the evader tree
            evader_tree = self.agent_rrt_trees[evader_id]
            for i, tree_node in enumerate(evader_tree):
                if tree_node.x == node.x and tree_node.y == node.y:
                    evader_time_dict[i] = travel_time
                    break
        
        print(f"Found travel times for {len(pursuer_time_dict)} pursuer nodes and {len(evader_time_dict)} evader nodes")
        
        # Calculate time advantages
        time_advantages = {}
        
        for pursuer_idx, pursuer_graph_node in pursuer_mapping.items():
            if pursuer_idx not in pursuer_time_dict:
                continue  # Skip if no travel time available
            
            pursuer_time = pursuer_time_dict[pursuer_idx]
            max_time_advantage = float('-inf')
            invisible_count = 0
            
            # Check visibility from this pursuer's map graph position to all evader nodes
            for evader_idx, evader_graph_node in evader_mapping.items():
                if evader_idx not in evader_time_dict:
                    continue  # Skip if no travel time available
                
                # Check if evader node is visible from pursuer's map graph position
                is_visible = self._check_visibility_between_graph_nodes(pursuer_graph_node, evader_graph_node)
                
                if not is_visible:
                    # Calculate time advantage for this invisible evader node
                    evader_time = evader_time_dict[evader_idx]
                    time_advantage = evader_time - pursuer_time
                    
                    if time_advantage > max_time_advantage:
                        max_time_advantage = time_advantage
                    
                    invisible_count += 1
            
            # Store the maximum time advantage for this pursuer node
            if invisible_count > 0:
                time_advantages[pursuer_idx] = max_time_advantage
                if pursuer_idx < 5:  # Debug: show first few
                    pursuer_pos = self.agent_rrt_trees[pursuer_id][pursuer_idx]
                    print(f"  Pursuer node {pursuer_idx} at ({pursuer_pos.x:.0f}, {pursuer_pos.y:.0f}): "
                          f"max advantage = {max_time_advantage:.2f}s from {invisible_count} invisible evader nodes")
            else:
                time_advantages[pursuer_idx] = 0.0  # No advantage if all evader nodes are visible
        
        print(f"Calculated time advantages for {len(time_advantages)} pursuer nodes")
        return time_advantages
    
    def _check_visibility_between_graph_nodes(self, graph_node1_idx: int, graph_node2_idx: int) -> bool:
        """
        Check if two map graph nodes have line-of-sight visibility to each other.
        
        Args:
            graph_node1_idx: Index of first map graph node
            graph_node2_idx: Index of second map graph node
            
        Returns:
            True if nodes are mutually visible, False if blocked by walls
        """
        if not hasattr(self.map_graph, 'nodes') or not self.map_graph.nodes:
            return True  # Assume visible if no map graph
        
        if graph_node1_idx >= len(self.map_graph.nodes) or graph_node2_idx >= len(self.map_graph.nodes):
            return False
        
        node1_pos = self.map_graph.nodes[graph_node1_idx]
        node2_pos = self.map_graph.nodes[graph_node2_idx]
        
        # Use the existing line intersection method to check for wall occlusion
        return not self._line_intersects_wall(node1_pos, node2_pos)
    
    def get_pursuit_evasion_stats(self, pursuer_id: str, evader_id: str) -> Dict:
        """
        Get statistics about the pursuit-evasion analysis.
        
        Args:
            pursuer_id: ID of the pursuing agent
            evader_id: ID of the evading agent
            
        Returns:
            Dictionary containing analysis statistics
        """
        time_advantages = self.calculate_pursuit_evasion_advantages(pursuer_id, evader_id)
        
        if not time_advantages:
            return {"status": "No data available"}
        
        advantages = list(time_advantages.values())
        
        return {
            "status": "Active",
            "pursuer_nodes_analyzed": len(time_advantages),
            "min_time_advantage": min(advantages),
            "max_time_advantage": max(advantages),
            "avg_time_advantage": sum(advantages) / len(advantages),
            "positive_advantages": sum(1 for adv in advantages if adv > 0),
            "negative_advantages": sum(1 for adv in advantages if adv < 0),
            "zero_advantages": sum(1 for adv in advantages if adv == 0)
        }
    
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

def get_path_to_node(agent_id: str, node) -> List:
    """Convenience function to get path from root to a specific node."""
    return global_position_evaluator.get_path_to_node(agent_id, node)

def find_node_at_position(agent_id: str, x: float, y: float, tolerance: float = 10.0):
    """Convenience function to find a node near a given position."""
    return global_position_evaluator.find_node_at_position(agent_id, x, y, tolerance)

def set_rrt_parameters(max_nodes: int = 100, step_size: float = 30.0, search_radius: float = 50.0,
                     forward_bias: float = 0.7, forward_cone_angle: float = math.pi / 3,
                     dubins_turning_radius: float = 30.0, agent_velocity: float = 50.0):
    """Convenience function to set RRT* parameters."""
    global_position_evaluator.set_rrt_parameters(max_nodes, step_size, search_radius, forward_bias, forward_cone_angle,
                                                dubins_turning_radius, agent_velocity)

# Closest node optimization convenience functions

def clear_closest_node_cache():
    """Convenience function to clear the closest node cache."""
    global_position_evaluator.clear_closest_node_cache()

def set_closest_node_cache_parameters(movement_threshold: float = 15.0, time_threshold: float = 2.0):
    """Convenience function to configure closest node cache parameters."""
    global_position_evaluator.set_closest_node_cache_parameters(movement_threshold, time_threshold)

def get_closest_node_cache_stats() -> Dict:
    """Convenience function to get closest node cache statistics."""
    return global_position_evaluator.get_closest_node_cache_stats()

def force_rebuild_spatial_index():
    """Convenience function to force rebuilding of spatial index."""
    global_position_evaluator.force_rebuild_spatial_index()

# RRT to Map Graph Mapping convenience functions

def map_rrt_nodes_to_graph(agent_id: str) -> Dict[int, int]:
    """Convenience function to map RRT nodes to closest map graph nodes."""
    return global_position_evaluator.map_rrt_nodes_to_graph(agent_id)

def map_all_rrt_nodes_to_graph() -> Dict[str, Dict[int, int]]:
    """Convenience function to map all RRT nodes to map graph nodes for all agents."""
    return global_position_evaluator.map_all_rrt_nodes_to_graph()

def get_rrt_to_graph_mapping_stats() -> Dict:
    """Convenience function to get RRT to map graph mapping statistics."""
    return global_position_evaluator.get_rrt_to_graph_mapping_stats()

# Strategic Pursuit-Evasion Analysis convenience functions

def calculate_pursuit_evasion_advantages(pursuer_id: str, evader_id: str) -> Dict[int, float]:
    """Convenience function to calculate strategic time advantages for pursuit-evasion."""
    return global_position_evaluator.calculate_pursuit_evasion_advantages(pursuer_id, evader_id)

def get_pursuit_evasion_stats(pursuer_id: str, evader_id: str) -> Dict:
    """Convenience function to get pursuit-evasion analysis statistics."""
    return global_position_evaluator.get_pursuit_evasion_stats(pursuer_id, evader_id)

def get_dubins_times_to_all_nodes(agent_id: str) -> Dict[int, Tuple[float, float]]:
    """Convenience function to get Dubins times to all nodes for an agent."""
    return global_position_evaluator.get_dubins_times_to_all_nodes(agent_id)

def find_shortest_time_node(agent_id: str) -> Optional[Tuple[int, 'RRTNode', float]]:
    """Convenience function to find the node with shortest Dubins travel time."""
    return global_position_evaluator.find_shortest_time_node(agent_id)

def calculate_dubins_path_to_node(agent_id: str, target_node: 'RRTNode') -> Optional[Tuple[float, float, List[Tuple[float, float]]]]:
    """Convenience function to calculate Dubins path to a specific node."""
    return global_position_evaluator.calculate_dubins_path_to_node(agent_id, target_node)


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
