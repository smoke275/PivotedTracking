"""
Map graph utility for pathfinding and visualization.
This module creates a navigable graph from the environment map.
"""

import pygame
import numpy as np
from math import sqrt, dist
import time
import multiprocessing
from functools import partial
import os
import pickle
import hashlib
from multitrack.utils.config import (
    MAP_GRAPH_GRID_SIZE, MAP_GRAPH_MAX_EDGE_DISTANCE, 
    MAP_GRAPH_MAX_CONNECTIONS, MAP_GRAPH_NODE_COLOR, MAP_GRAPH_EDGE_COLOR,
    MAP_GRAPH_CACHE_ENABLED, MAP_GRAPH_CACHE_FILE, MAP_GRAPH_MULTICORE_DEFAULT
)

class MapGraph:
    """
    Creates a navigation graph for the environment by sampling 
    collision-free points and connecting them based on line-of-sight visibility.
    """
    
    def __init__(self, width, height, walls, doors, grid_size=None):
        """
        Initialize the map graph with environment dimensions and obstacles.
        
        Args:
            width (int): Environment width
            height (int): Environment height
            walls (list): List of wall rectangles (pygame.Rect)
            doors (list): List of door rectangles (pygame.Rect)
            grid_size (int): Resolution of the sampling grid, defaults to config value
        """
        self.width = width
        self.height = height
        self.walls = walls
        self.doors = doors
        
        # Use provided grid_size or default to config value
        self.grid_size = grid_size if grid_size is not None else MAP_GRAPH_GRID_SIZE
        
        # Grid cell dimensions
        self.cell_width = self.width / self.grid_size
        self.cell_height = self.height / self.grid_size
        
        # Store graph nodes and edges
        self.nodes = []  # List of (x, y) tuples for valid positions
        self.edges = []  # List of (node_idx1, node_idx2) tuples for connected nodes
        self.adjacency = {}  # Dictionary mapping node index to list of connected node indices
        
        # Generation status and progress
        self.is_generated = False
        self.generation_progress = 0.0
    
    def generate(self, status_callback=None):
        """
        Generate the map graph by sampling and connecting nodes.
        
        Args:
            status_callback: Optional callback function to update progress
        
        Returns:
            bool: True if generation was successful
        """
        # Reset any existing graph
        self.nodes = []
        self.edges = []
        self.adjacency = {}
        self.is_generated = False
        self.generation_progress = 0.0
        
        # Sample valid positions (collision-free)
        if status_callback:
            status_callback("Creating graph nodes...", 0.1)
        self._sample_nodes()
        
        # Connect nodes with line-of-sight
        if status_callback:
            status_callback("Connecting nodes...", 0.3)
        self._connect_nodes(status_callback)
        
        # Prune unnecessary connections to keep graph sparse
        if status_callback:
            status_callback("Optimizing graph...", 0.8)
        self._prune_edges()
        
        self.is_generated = True
        self.generation_progress = 1.0
        
        if status_callback:
            status_callback("Graph generation complete", 1.0)
        
        return True
    
    def generate_parallel(self, status_callback=None, num_cores=None):
        """
        Generate the map graph using parallel processing for faster execution.
        
        Args:
            status_callback: Optional callback function to update progress
            num_cores: Number of CPU cores to use (None = all available)
            
        Returns:
            bool: True if generation was successful
        """
        # Reset any existing graph
        self.nodes = []
        self.edges = []
        self.adjacency = {}
        self.is_generated = False
        self.generation_progress = 0.0
        
        # Determine number of cores to use
        if num_cores is None:
            num_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free for UI
        else:
            num_cores = min(num_cores, multiprocessing.cpu_count())
            
        if status_callback:
            status_callback(f"Creating graph nodes using {num_cores} CPU cores...", 0.1)
            # Add a small sleep to allow the UI to update
            time.sleep(0.05)
            
        # Sample valid positions (collision-free) in parallel
        self._sample_nodes_parallel(num_cores, status_callback)
        
        if status_callback:
            status_callback(f"Connecting nodes using {num_cores} CPU cores... ({len(self.nodes)} nodes)", 0.3)
            # Add a small sleep to allow the UI to update
            time.sleep(0.05)
        
        # Connect nodes with line-of-sight in parallel
        self._connect_nodes_parallel(num_cores, status_callback)
        
        # Prune unnecessary connections to keep graph sparse
        if status_callback:
            status_callback("Optimizing graph...", 0.8)
            # Add a small sleep to allow the UI to update
            time.sleep(0.05)
        self._prune_edges()
        
        self.is_generated = True
        self.generation_progress = 1.0
        
        if status_callback:
            status_callback(f"Graph generation complete: {len(self.nodes)} nodes, {len(self.edges)} edges", 1.0)
            # Add a small sleep to allow the UI to update
            time.sleep(0.1)
        
        return True
    
    def _sample_nodes(self):
        """Sample valid nodes in the environment (collision-free positions)"""
        # Create a grid of sample points
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Calculate position at center of grid cell
                x = (i + 0.5) * self.cell_width
                y = (j + 0.5) * self.cell_height
                
                # Check if this position is valid (not colliding with walls)
                if self._is_valid_position(x, y):
                    self.nodes.append((x, y))
        
        # Add nodes at doorway centers for better navigation
        for door in self.doors:
            # Calculate door center
            x = door.x + door.width / 2
            y = door.y + door.height / 2
            self.nodes.append((x, y))
    
    def _is_valid_position(self, x, y):
        """Check if position is valid (not inside a wall, excluding doors)"""
        # Create a small rect representing an agent at the position
        agent_rect = pygame.Rect(
            int(x) - 10,  # x
            int(y) - 10,  # y
            20, 20  # width, height
        )
        
        # Check collision with walls
        for wall in self.walls:
            if agent_rect.colliderect(wall):
                # Check if we're in a door
                in_door = False
                for door in self.doors:
                    if agent_rect.colliderect(door):
                        in_door = True
                        break
                
                if not in_door:
                    return False
        
        return True
    
    def _has_line_of_sight(self, x1, y1, x2, y2):
        """Check if there's a direct line of sight between two points"""
        # Create line segment from (x1,y1) to (x2,y2)
        line_start = (x1, y1)
        line_end = (x2, y2)
        
        # Check intersection with all walls
        for wall in self.walls:
            # Convert wall to line segments
            wall_segments = [
                ((wall.x, wall.y), (wall.x + wall.width, wall.y)),  # Top
                ((wall.x, wall.y), (wall.x, wall.y + wall.height)),  # Left
                ((wall.x + wall.width, wall.y), (wall.x + wall.width, wall.y + wall.height)),  # Right
                ((wall.x, wall.y + wall.height), (wall.x + wall.width, wall.y + wall.height))  # Bottom
            ]
            
            # Check if line intersects with any wall segment
            for segment in wall_segments:
                if self._line_segments_intersect(line_start, line_end, segment[0], segment[1]):
                    # Check if intersection is in a door
                    intersection_point = self._get_line_intersection(
                        line_start[0], line_start[1], line_end[0], line_end[1],
                        segment[0][0], segment[0][1], segment[1][0], segment[1][1]
                    )
                    
                    if intersection_point:
                        in_door = False
                        for door in self.doors:
                            # Use a slightly expanded door rect to ensure lines can pass through
                            door_rect = pygame.Rect(door.x-2, door.y-2, door.width+4, door.height+4)
                            if door_rect.collidepoint(int(intersection_point[0]), int(intersection_point[1])):
                                in_door = True
                                break
                        
                        if not in_door:
                            return False
        
        return True
    
    def _connect_nodes(self, status_callback=None):
        """Connect nodes based on line-of-sight visibility"""
        num_nodes = len(self.nodes)
        connections_checked = 0
        total_connections = (num_nodes * (num_nodes - 1)) // 2
        
        # Initialize adjacency lists
        for i in range(num_nodes):
            self.adjacency[i] = []
        
        # Check visibility between each pair of nodes
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Update progress occasionally
                connections_checked += 1
                if connections_checked % 1000 == 0 and status_callback:
                    progress = 0.3 + 0.5 * (connections_checked / total_connections)
                    status_callback(f"Connecting nodes: {connections_checked}/{total_connections}", progress)
                
                # Get node positions
                node1 = self.nodes[i]
                node2 = self.nodes[j]
                
                # Check distance first (don't connect nodes that are too far apart)
                distance = sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
                if distance > MAP_GRAPH_MAX_EDGE_DISTANCE:  # Use configured max distance
                    continue
                
                # Check line of sight
                if self._has_line_of_sight(node1[0], node1[1], node2[0], node2[1]):
                    # Add edge
                    self.edges.append((i, j))
                    
                    # Update adjacency lists (bidirectional)
                    self.adjacency[i].append(j)
                    self.adjacency[j].append(i)
    
    def _prune_edges(self):
        """Prune unnecessary edges to keep the graph sparse"""
        # Simple pruning - remove edges if nodes have too many connections
        pruned_edges = []
        for i in range(len(self.nodes)):
            # Sort neighbors by distance
            neighbors = sorted(self.adjacency[i], 
                              key=lambda j: dist(self.nodes[i], self.nodes[j]))
            
            # Keep only the closest N neighbors, using the configured max connections
            if len(neighbors) > MAP_GRAPH_MAX_CONNECTIONS:
                # Keep the closest neighbors
                kept_neighbors = neighbors[:MAP_GRAPH_MAX_CONNECTIONS]
                
                # Update adjacency list
                self.adjacency[i] = kept_neighbors
                
                # Add kept edges to pruned_edges
                for j in kept_neighbors:
                    # Ensure we only add each edge once
                    if i < j:
                        pruned_edges.append((i, j))
                    else:
                        pruned_edges.append((j, i))
            else:
                # Keep all existing edges for this node
                for j in neighbors:
                    if i < j:
                        pruned_edges.append((i, j))
                    else:
                        pruned_edges.append((j, i))
        
        # Remove duplicates and update edges
        self.edges = list(set(pruned_edges))
    
    def draw(self, surface, show_nodes=True, node_color=None, edge_color=None):
        """
        Draw the map graph onto a pygame surface
        
        Args:
            surface: Pygame surface to draw on
            show_nodes: Whether to show node points
            node_color: RGB color tuple for nodes, defaults to config value
            edge_color: RGB color tuple for edges, defaults to config value
        """
        if not self.is_generated:
            return
        
        # Use provided colors or default to config values
        node_color = node_color if node_color is not None else MAP_GRAPH_NODE_COLOR
        edge_color = edge_color if edge_color is not None else MAP_GRAPH_EDGE_COLOR
        
        # Create transparent surface
        graph_surface = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
        
        # Draw edges first
        for edge in self.edges:
            node1 = self.nodes[edge[0]]
            node2 = self.nodes[edge[1]]
            pygame.draw.line(graph_surface, edge_color, 
                            (int(node1[0]), int(node1[1])), 
                            (int(node2[0]), int(node2[1])), 2)
        
        # Draw nodes
        if show_nodes:
            for node in self.nodes:
                pygame.draw.circle(graph_surface, node_color, 
                                 (int(node[0]), int(node[1])), 3)
        
        # Blit graph surface onto main surface
        surface.blit(graph_surface, (0, 0))
    
    def _line_segments_intersect(self, p1, p2, p3, p4):
        """
        Check if line segments (p1,p2) and (p3,p4) intersect
        
        Args:
            p1, p2: Start and end points of first line segment
            p3, p4: Start and end points of second line segment
            
        Returns:
            bool: True if the segments intersect
        """
        return self._get_line_intersection(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]) is not None
    
    def _get_line_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Calculate the intersection point of two line segments
        
        Args:
            x1, y1, x2, y2: First line segment coordinates
            x3, y3, x4, y4: Second line segment coordinates
            
        Returns:
            (x, y) tuple of intersection point, or None if no intersection
        """
        # Calculate determinants
        den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        
        # If denominator is zero, lines are parallel
        if den == 0:
            return None
        
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
        
        # Check if intersection point is within both line segments
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            # Calculate intersection point
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (x, y)
            
        return None
    
    def _sample_nodes_parallel(self, num_cores, status_callback=None):
        """Sample valid nodes in the environment using parallel processing"""
        # Create grid cell coordinates
        grid_coords = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        
        # Split work across cores
        chunk_size = max(1, len(grid_coords) // num_cores)
        chunks = [grid_coords[i:i + chunk_size] for i in range(0, len(grid_coords), chunk_size)]
        
        # Create a pool of workers
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Create a partial function with the instance variables needed
            worker_func = partial(self._process_positions_chunk, 
                                 cell_width=self.cell_width,
                                 cell_height=self.cell_height,
                                 walls=self.walls,
                                 doors=self.doors)
            
            # Process chunks in parallel with progress updates
            valid_positions = []
            for i, chunk_result in enumerate(pool.imap(worker_func, chunks)):
                valid_positions.extend(chunk_result)
                if status_callback:
                    progress = 0.1 + 0.2 * ((i + 1) / len(chunks))
                    status_callback(f"Sampling nodes: {len(valid_positions)} valid nodes found...", progress)
        
        # Store valid positions as nodes
        self.nodes = valid_positions
        
        # Add nodes at doorway centers for better navigation
        for door in self.doors:
            # Calculate door center
            x = door.x + door.width / 2
            y = door.y + door.height / 2
            self.nodes.append((x, y))
    
    @staticmethod
    def _process_positions_chunk(coords_chunk, cell_width, cell_height, walls, doors):
        """Process a chunk of grid coordinates in a worker process"""
        valid_positions = []
        for i, j in coords_chunk:
            # Calculate position at center of grid cell
            x = (i + 0.5) * cell_width
            y = (j + 0.5) * cell_height
            
            # Check if position is valid
            agent_rect = pygame.Rect(
                int(x) - 10,  # x
                int(y) - 10,  # y
                20, 20  # width, height
            )
            
            # Check collision with walls
            valid = True
            for wall in walls:
                if agent_rect.colliderect(wall):
                    # Check if we're in a door
                    in_door = False
                    for door in doors:
                        if agent_rect.colliderect(door):
                            in_door = True
                            break
                    
                    if not in_door:
                        valid = False
                        break
            
            if valid:
                valid_positions.append((x, y))
                
        return valid_positions
    
    def _connect_nodes_parallel(self, num_cores, status_callback=None):
        """Connect nodes based on line-of-sight visibility using parallel processing"""
        num_nodes = len(self.nodes)
        
        # Create node pairs for visibility checking
        node_pairs = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Get node positions
                node1 = self.nodes[i]
                node2 = self.nodes[j]
                
                # Check distance first (don't connect nodes that are too far apart)
                distance = sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
                if distance <= MAP_GRAPH_MAX_EDGE_DISTANCE:
                    node_pairs.append((i, j, node1, node2))
        
        # Split work across cores
        chunk_size = max(1, len(node_pairs) // num_cores)
        chunks = [node_pairs[i:i + chunk_size] for i in range(0, len(node_pairs), chunk_size)]
        
        # Initialize adjacency lists
        for i in range(num_nodes):
            self.adjacency[i] = []
        
        # Create a pool of workers
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Create a partial function with the instance variables needed
            worker_func = partial(self._process_connections_chunk, 
                                 walls=self.walls,
                                 doors=self.doors)
            
            # Process chunks in parallel with progress updates
            all_edges = []
            total_chunks = len(chunks)
            for i, chunk_result in enumerate(pool.imap(worker_func, chunks)):
                all_edges.extend(chunk_result)
                if status_callback:
                    progress = 0.3 + 0.5 * ((i + 1) / total_chunks)
                    status_callback(f"Connecting nodes: {i+1}/{total_chunks} chunks processed", progress)
        
        # Update edges and adjacency lists
        self.edges = all_edges
        for i, j in all_edges:
            self.adjacency[i].append(j)
            self.adjacency[j].append(i)
    
    @staticmethod
    def _process_connections_chunk(pairs_chunk, walls, doors):
        """Process a chunk of node pairs for connections in a worker process"""
        valid_edges = []
        for i, j, node1, node2 in pairs_chunk:
            # Create line segment from node1 to node2
            line_start = (node1[0], node1[1])
            line_end = (node2[0], node2[1])
            
            # Check intersection with walls
            has_line_of_sight = True
            for wall in walls:
                # Convert wall to line segments
                wall_segments = [
                    ((wall.x, wall.y), (wall.x + wall.width, wall.y)),  # Top
                    ((wall.x, wall.y), (wall.x, wall.y + wall.height)),  # Left
                    ((wall.x + wall.width, wall.y), (wall.x + wall.width, wall.y + wall.height)),  # Right
                    ((wall.x, wall.y + wall.height), (wall.x + wall.width, wall.y + wall.height))  # Bottom
                ]
                
                # Check if line intersects with any wall segment
                for segment in wall_segments:
                    intersection_point = MapGraph._get_line_intersection_static(
                        line_start[0], line_start[1], line_end[0], line_end[1],
                        segment[0][0], segment[0][1], segment[1][0], segment[1][1]
                    )
                    
                    if intersection_point:
                        # Check if intersection is in a door
                        in_door = False
                        for door in doors:
                            # Use a slightly expanded door rect
                            door_rect = pygame.Rect(door.x-2, door.y-2, door.width+4, door.height+4)
                            if door_rect.collidepoint(int(intersection_point[0]), int(intersection_point[1])):
                                in_door = True
                                break
                        
                        if not in_door:
                            has_line_of_sight = False
                            break
                
                if not has_line_of_sight:
                    break
            
            if has_line_of_sight:
                valid_edges.append((i, j))
                
        return valid_edges
    
    @staticmethod
    def _get_line_intersection_static(x1, y1, x2, y2, x3, y3, x4, y4):
        """Static version of line intersection calculation for multiprocessing"""
        # Calculate determinants
        den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        
        # If denominator is zero, lines are parallel
        if den == 0:
            return None
        
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
        
        # Check if intersection point is within both line segments
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            # Calculate intersection point
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (x, y)
            
        return None
    
    def save_to_cache(self):
        """
        Save the map graph to cache file.
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if not self.is_generated:
            print("Cannot save: Graph not yet generated")
            return False
            
        try:
            # Create a dictionary with all relevant data and metadata
            cache_data = {
                'nodes': self.nodes,
                'edges': self.edges,
                'adjacency': self.adjacency,
                'width': self.width,
                'height': self.height,
                'grid_size': self.grid_size,
                'max_edge_distance': MAP_GRAPH_MAX_EDGE_DISTANCE,
                'max_connections': MAP_GRAPH_MAX_CONNECTIONS,
                'environment_hash': self._compute_environment_hash()
            }
            
            # Save to cache file
            cache_dir = os.path.dirname(os.path.abspath(MAP_GRAPH_CACHE_FILE))
            if not os.path.exists(cache_dir) and cache_dir:
                os.makedirs(cache_dir)
                
            with open(MAP_GRAPH_CACHE_FILE, 'wb') as f:
                pickle.dump(cache_data, f)
                
            print(f"Map graph saved to cache: {len(self.nodes)} nodes, {len(self.edges)} edges")
            return True
        except Exception as e:
            print(f"Error saving map graph to cache: {e}")
            return False
    
    def load_from_cache(self, validate=True):
        """
        Load the map graph from cache file.
        
        Args:
            validate (bool): Whether to validate the cached graph against the current environment
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not MAP_GRAPH_CACHE_ENABLED or not os.path.exists(MAP_GRAPH_CACHE_FILE):
            return False
            
        try:
            with open(MAP_GRAPH_CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Check basic validation criteria
            if (cache_data['width'] != self.width or 
                cache_data['height'] != self.height or
                cache_data['grid_size'] != self.grid_size or
                cache_data['max_edge_distance'] != MAP_GRAPH_MAX_EDGE_DISTANCE or
                cache_data['max_connections'] != MAP_GRAPH_MAX_CONNECTIONS):
                print("Cached map graph parameters don't match current configuration")
                return False
                
            # Check environment hash for changes
            current_hash = self._compute_environment_hash()
            if validate and cache_data['environment_hash'] != current_hash:
                print("Environment has changed since map graph was cached")
                return False
                
            # If validation passes, load the data
            self.nodes = cache_data['nodes']
            self.edges = cache_data['edges']
            self.adjacency = cache_data['adjacency']
            self.is_generated = True
            self.generation_progress = 1.0
            
            print(f"Loaded map graph from cache: {len(self.nodes)} nodes, {len(self.edges)} edges")
            return True
        except Exception as e:
            print(f"Error loading map graph from cache: {e}")
            return False
    
    def validate_cached_graph(self, sample_count=20):
        """
        Validate that the cached graph is still valid for the current environment.
        Tests a sample of nodes and edges for validity.
        
        Args:
            sample_count: Number of nodes/edges to sample for validation
            
        Returns:
            bool: True if graph appears valid, False otherwise
        """
        if not self.is_generated or len(self.nodes) == 0:
            return False
            
        # 1. Validate a sample of nodes to make sure they're still valid positions
        node_indices = np.random.choice(len(self.nodes), 
                                       min(sample_count, len(self.nodes)), 
                                       replace=False)
        
        for idx in node_indices:
            node = self.nodes[idx]
            if not self._is_valid_position(node[0], node[1]):
                print(f"Invalid node at {node} - environment may have changed")
                return False
        
        # 2. Validate a sample of edges to make sure they still have line of sight
        if len(self.edges) > 0:
            edge_indices = np.random.choice(len(self.edges), 
                                           min(sample_count, len(self.edges)), 
                                           replace=False)
            
            for idx in edge_indices:
                edge = self.edges[idx]
                node1 = self.nodes[edge[0]]
                node2 = self.nodes[edge[1]]
                
                if not self._has_line_of_sight(node1[0], node1[1], node2[0], node2[1]):
                    print(f"Edge between {node1} and {node2} no longer valid - environment may have changed")
                    return False
        
        # All sampled nodes and edges are valid
        return True
    
    def _compute_environment_hash(self):
        """
        Compute a hash of the environment (walls and doors) to detect changes.
        
        Returns:
            str: Hash string representing the environment
        """
        # Create a string representation of all walls and doors
        env_str = ""
        
        # Add walls to string
        for wall in sorted(self.walls, key=lambda w: (w.x, w.y, w.width, w.height)):
            env_str += f"W:{wall.x},{wall.y},{wall.width},{wall.height};"
            
        # Add doors to string
        for door in sorted(self.doors, key=lambda d: (d.x, d.y, d.width, d.height)):
            env_str += f"D:{door.x},{door.y},{door.width},{door.height};"
            
        # Add configuration parameters
        env_str += f"G:{self.grid_size};E:{MAP_GRAPH_MAX_EDGE_DISTANCE};C:{MAP_GRAPH_MAX_CONNECTIONS};"
        
        # Calculate SHA-256 hash
        return hashlib.sha256(env_str.encode()).hexdigest()