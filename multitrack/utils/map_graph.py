"""
Map graph utility for pathfinding and visualization.
This module creates a navigable graph from the environment map.
"""

import pygame
import numpy as np
import math
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
    MAP_GRAPH_CACHE_ENABLED, MAP_GRAPH_CACHE_FILE, MAP_GRAPH_INSPECTION_CACHE_FILE, MAP_GRAPH_MULTICORE_DEFAULT
)

# Standalone function for multiprocessing
def process_visibility_batch(batch_data):
    """
    Process a batch of nodes for visibility analysis.
    This is a standalone function so it can be pickled for multiprocessing.
    
    Args:
        batch_data: A tuple containing (batch_index, node_batch, nodes_list, walls, doors, max_range)
        
    Returns:
        A tuple of (batch_index, batch_results, batch_stats)
    """
    batch_index, node_batch, nodes_list, walls, doors, max_range = batch_data
    batch_results = {}
    batch_stats = {
        'total_checks': 0,
        'visible_connections': 0,
        'start_time': time.time()
    }
    
    # Line of sight helper function
    def check_line_of_sight(from_pos, to_pos, walls, doors):
        # Create line segments from the walls
        wall_segments = []
        for wall in walls:
            # Get wall coordinates 
            if isinstance(wall, tuple) or isinstance(wall, list):
                x1, y1, w, h = wall
            else:  # pygame.Rect
                x1, y1, w, h = wall.x, wall.y, wall.width, wall.height
            
            x2, y2 = x1 + w, y1 + h
            
            # Add the four wall segments
            wall_segments.append((x1, y1, x2, y1))  # Top
            wall_segments.append((x2, y1, x2, y2))  # Right
            wall_segments.append((x2, y2, x1, y2))  # Bottom
            wall_segments.append((x1, y2, x1, y1))  # Left
        
        # Define the sight line
        x1, y1 = from_pos
        x2, y2 = to_pos
        
        # Check intersection with each wall segment
        for wx1, wy1, wx2, wy2 in wall_segments:
            # Check if the lines intersect using a more robust algorithm
            denom = (wy2 - wy1) * (x2 - x1) - (wx2 - wx1) * (y2 - y1)
            
            # If lines are parallel
            if abs(denom) < 1e-9:
                continue
                
            ua = ((wx2 - wx1) * (y1 - wy1) - (wy2 - wy1) * (x1 - wx1)) / denom
            ub = ((x2 - x1) * (y1 - wy1) - (y2 - y1) * (x1 - wx1)) / denom
            
            # If intersection point is within both line segments
            if 0.0 <= ua <= 1.0 and 0.0 <= ub <= 1.0:
                # Calculate the intersection point
                ix = x1 + ua * (x2 - x1)
                iy = y1 + ua * (y2 - y1)
                
                # Check if this intersection is with a door (doors are passable)
                is_door = False
                for door in doors:
                    if isinstance(door, tuple) or isinstance(door, list):
                        dx, dy, dw, dh = door
                    else:  # pygame.Rect
                        dx, dy, dw, dh = door.x, door.y, door.width, door.height
                        
                    if dx <= ix <= dx + dw and dy <= iy <= dy + dh:
                        is_door = True
                        break
                
                # If intersection is not at a door, there's no line of sight
                if not is_door:
                    return False
        
        # If we've checked all wall segments and found no blocking intersection, there's line of sight
        return True
    
    # Helper function to check line intersection
    def do_lines_intersect(line1_p1, line1_p2, line2_p1, line2_p2):
        # Extract coordinates
        x1, y1 = line1_p1
        x2, y2 = line1_p2
        x3, y3 = line2_p1
        x4, y4 = line2_p2
        
        # Calculate directions
        d1x = x2 - x1
        d1y = y2 - y1
        d2x = x4 - x3
        d2y = y4 - y3
        
        # Calculate the determinant
        det = d1x * d2y - d1y * d2x
        
        # If determinant is zero, lines are parallel
        if abs(det) < 1e-9:
            return False
            
        # Calculate the parameters
        t = ((x3 - x1) * d2y - (y3 - y1) * d2x) / det
        u = ((x3 - x1) * d1y - (y3 - y1) * d1x) / det
        
        # Check if intersection is within both line segments
        return 0 <= t <= 1 and 0 <= u <= 1
    
    # Process each node in the batch
    for node_index in node_batch:
        node_from = nodes_list[node_index]
        visible_nodes = []
        
        for j, node_to in enumerate(nodes_list):
            # Skip self
            if node_index == j:
                continue
            
            batch_stats['total_checks'] += 1
            
            # Check if within range
            distance = math.sqrt((node_from[0] - node_to[0])**2 + (node_from[1] - node_to[1])**2)
            if distance > max_range:
                continue
            
            # Check line of sight
            has_sight = check_line_of_sight(node_from, node_to, walls, doors)
            if has_sight:
                visible_nodes.append(j)
                batch_stats['visible_connections'] += 1
        
        batch_results[node_index] = visible_nodes
    
    batch_stats['duration'] = time.time() - batch_stats['start_time']
    batch_stats['nodes_processed'] = len(node_batch)
    return batch_index, batch_results, batch_stats

class MapGraph:
    """
    Creates a navigation graph for the environment by sampling 
    collision-free points and connecting them based on line-of-sight visibility.
    """
    
    def __init__(self, width, height, walls, doors, grid_size=None, cache_file=None):
        """
        Initialize the map graph with environment dimensions and obstacles.
        
        Args:
            width (int): Environment width
            height (int): Environment height
            walls (list): List of wall rectangles (pygame.Rect)
            doors (list): List of door rectangles (pygame.Rect)
            grid_size (int): Resolution of the sampling grid, defaults to config value
            cache_file (str): Custom cache file name, defaults to MAP_GRAPH_CACHE_FILE
        """
        self.width = width
        self.height = height
        self.walls = walls
        self.doors = doors
        
        # Use provided grid_size or default to config value
        self.grid_size = grid_size if grid_size is not None else MAP_GRAPH_GRID_SIZE
        
        # Store custom cache file if provided
        self.cache_file = cache_file if cache_file is not None else MAP_GRAPH_CACHE_FILE
        
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
        
        # Process events to keep UI responsive before heavy processing
        pygame.event.pump()
        
        # Calculate optimal chunk size - smaller chunks for better responsiveness
        # Use more chunks than cores for better load balancing and responsiveness
        total_cells = len(grid_coords)
        chunks_per_core = 8  # Use multiple smaller chunks per core for better UI responsiveness
        ideal_chunks = num_cores * chunks_per_core
        chunk_size = max(1, total_cells // ideal_chunks)
        chunks = [grid_coords[i:i + chunk_size] for i in range(0, total_cells, chunk_size)]
        
        if status_callback:
            status_callback(f"Preparing to sample {total_cells} positions across {len(chunks)} chunks...", 0.1)
            # Process events to keep UI responsive
            pygame.event.pump()
            time.sleep(0.05)  # Short sleep to allow UI updates
        
        # Create a pool of workers
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Create a partial function with the instance variables needed
            worker_func = partial(self._process_positions_chunk, 
                                 cell_width=self.cell_width,
                                 cell_height=self.cell_height,
                                 walls=self.walls,
                                 doors=self.doors)
            
            # Process chunks in parallel with more frequent progress updates
            valid_positions = []
            total_chunks = len(chunks)
            
            # More frequent updates for better user feedback
            update_interval = max(1, total_chunks // 50)  # Update about 50 times during processing
            
            for i, chunk_result in enumerate(pool.imap_unordered(worker_func, chunks)):
                valid_positions.extend(chunk_result)
                
                # More frequent status updates with UI event processing
                if status_callback and (i % update_interval == 0 or i == total_chunks - 1):
                    progress = 0.1 + 0.2 * ((i + 1) / total_chunks)
                    status_callback(f"Sampling nodes: {len(valid_positions)} valid nodes found ({i+1}/{total_chunks} chunks)...", progress)
                    # Process events to keep UI responsive - CRITICAL for preventing freezing
                    pygame.event.pump()
                    
                    # Brief pause every few chunks to allow the main thread to process
                    if i % (update_interval * 4) == 0:
                        time.sleep(0.01)  # Very brief sleep to yield to main thread
        
        # Process events again after intensive processing
        pygame.event.pump()
        
        if status_callback:
            status_callback(f"Sampling complete: Found {len(valid_positions)} valid nodes", 0.3)
            # Process events to keep UI responsive
            pygame.event.pump()
        
        # Store valid positions as nodes
        self.nodes = valid_positions
        
        # Add nodes at doorway centers for better navigation
        for door in self.doors:
            # Calculate door center
            x = door.x + door.width / 2
            y = door.y + door.height / 2
            self.nodes.append((x, y))
            
        # Process events one final time before returning
        pygame.event.pump()
    
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
        
        if status_callback:
            status_callback(f"Preparing to connect {num_nodes} nodes...", 0.3)
            # Process events to keep UI responsive
            pygame.event.pump()
            time.sleep(0.05)
        
        # Create node pairs for visibility checking
        node_pairs = []
        pair_count = 0
        
        # Show progress during node pair generation which can be slow for large maps
        total_possible_pairs = (num_nodes * (num_nodes - 1)) // 2
        status_update_interval = max(1, total_possible_pairs // 50)  # More frequent updates (50 instead of 20)
        
        # Process events frequently during the pair generation phase
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Get node positions
                node1 = self.nodes[i]
                node2 = self.nodes[j]
                
                # Show progress more frequently during this potentially time-consuming step
                pair_count += 1
                if status_callback and pair_count % status_update_interval == 0:
                    progress_percent = pair_count / total_possible_pairs
                    status_callback(f"Preparing node pairs: {pair_count:,}/{total_possible_pairs:,} ({progress_percent:.1%})", 0.3)
                    # Process events to keep UI responsive - critical for preventing freezes
                    pygame.event.pump()
                    
                    # Add a very brief sleep periodically to allow the main thread to process
                    if pair_count % (status_update_interval * 5) == 0:
                        time.sleep(0.01)
                    
                # Check distance first (don't connect nodes that are too far apart)
                distance = sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
                if distance <= MAP_GRAPH_MAX_EDGE_DISTANCE:
                    node_pairs.append((i, j, node1, node2))
        
        if status_callback:
            status_callback(f"Found {len(node_pairs):,} potential connections to check...", 0.35)
            # Process events to keep UI responsive
            pygame.event.pump()
            time.sleep(0.05)
        
        # Split work into smaller chunks for better load balancing and UI responsiveness
        # More chunks = more frequent updates = more responsive UI
        chunks_per_core = 8  # Multiple chunks per core for better load balancing
        ideal_chunks = num_cores * chunks_per_core
        chunk_size = max(1, len(node_pairs) // ideal_chunks)
        chunks = [node_pairs[i:i + chunk_size] for i in range(0, len(node_pairs), chunk_size)]
        
        # Initialize adjacency lists
        for i in range(num_nodes):
            self.adjacency[i] = []
        
        if status_callback:
            status_callback(f"Processing {len(chunks)} chunks across {num_cores} cores...", 0.4)
            # Process events to keep UI responsive
            pygame.event.pump()
            time.sleep(0.05)
        
        # Create a pool of workers
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Create a partial function with the instance variables needed
            worker_func = partial(self._process_connections_chunk, 
                                 walls=self.walls,
                                 doors=self.doors)
            
            # Process chunks in parallel with more frequent progress updates
            all_edges = []
            total_chunks = len(chunks)
            
            # More frequent updates (using imap_unordered for better load balancing)
            update_interval = max(1, total_chunks // 100)  # Update more frequently
            
            for i, chunk_result in enumerate(pool.imap_unordered(worker_func, chunks)):
                all_edges.extend(chunk_result)
                
                # More frequent status updates with UI event processing
                if status_callback and (i % update_interval == 0 or i == total_chunks - 1):
                    progress = 0.4 + 0.4 * ((i + 1) / total_chunks)
                    status_callback(f"Connecting nodes: {i+1}/{total_chunks} chunks ({((i+1)/total_chunks):.1%})", progress)
                    # Process events to keep UI responsive
                    pygame.event.pump()
                    
                    # Brief pause periodically to allow main thread to process
                    if i % (update_interval * 5) == 0:
                        time.sleep(0.01)
        
        # Process events again after intensive processing
        pygame.event.pump()
        
        if status_callback:
            status_callback(f"Found {len(all_edges):,} valid connections...", 0.8)
            # Process events to keep UI responsive
            pygame.event.pump()
        
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
            cache_dir = os.path.dirname(os.path.abspath(self.cache_file))
            if not os.path.exists(cache_dir) and cache_dir:
                os.makedirs(cache_dir)
            
            # Print the full path of the cache file
            full_cache_path = os.path.abspath(self.cache_file)
            print(f"Saving map graph to: {full_cache_path}")
                
            with open(self.cache_file, 'wb') as f:
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
        if not MAP_GRAPH_CACHE_ENABLED:
            return False
            
        # Print the full path of the cache file being checked
        full_cache_path = os.path.abspath(self.cache_file)
        print(f"Checking for map graph cache at: {full_cache_path}")
        
        if not os.path.exists(self.cache_file):
            print(f"Cache file not found at: {full_cache_path}")
            return False
            
        try:
            print(f"Loading map graph from: {full_cache_path}")
            with open(self.cache_file, 'rb') as f:
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
        
    def analyze_node_visibility(self, max_range=None, status_callback=None, num_cores=None, visibility_cache_file=None):
        """
        Analyzes which nodes are visible from each node and saves the result to a cache file.
        This is useful for simulations that need visibility information.
        Uses parallel processing for faster computation.
        
        Args:
            max_range (float): Maximum visibility range in pixels, defaults to MAP_GRAPH_VISIBILITY_RANGE
            status_callback (callable): Optional callback for progress updates
            num_cores (int): Number of CPU cores to use (None = auto-detect)
            visibility_cache_file (str): Optional custom path for the visibility cache file
            
        Returns:
            dict: Visibility information mapping each node index to a list of visible node indices
        """
        from multitrack.utils.config import MAP_GRAPH_VISIBILITY_CACHE_FILE, MAP_GRAPH_VISIBILITY_RANGE
        
        if not self.is_generated or len(self.nodes) == 0:
            print("Cannot analyze visibility: Graph not generated or empty")
            return None
            
        if max_range is None:
            max_range = MAP_GRAPH_VISIBILITY_RANGE
            
        print(f"Analyzing node visibility with max range of {max_range} pixels...")
        
        start_time = time.time()
        total_nodes = len(self.nodes)
        visibility_map = {}
        
        # Use multiprocessing to speed up visibility analysis
        try:
            # Determine number of cores to use
            available_cores = multiprocessing.cpu_count()
            if num_cores is None:
                # Leave one core free for UI responsiveness
                num_cores = max(1, available_cores - 1)
            else:
                num_cores = min(num_cores, available_cores)
                
            print(f"Using {num_cores} of {available_cores} CPU cores for visibility analysis...")
            
            # Create batches of nodes to process - we want more batches than cores for better load balancing
            node_indices = list(range(total_nodes))
            
            # Many smaller batches provides better progress reporting and load balancing
            batches_per_core = 4  # Multiple batches per core
            ideal_batch_count = num_cores * batches_per_core
            batch_size = max(1, total_nodes // ideal_batch_count)
            
            # Create batches with their index
            raw_batches = [node_indices[i:i + batch_size] for i in range(0, total_nodes, batch_size)]
            
            # Prepare data for the standalone function
            # We need to include all nodes, walls, and doors so the function can access them
            wall_data = [
                (w.x, w.y, w.width, w.height) if hasattr(w, 'x') else w for w in self.walls
            ]
            door_data = [
                (d.x, d.y, d.width, d.height) if hasattr(d, 'x') else d for d in self.doors
            ]
            
            # Create complete batch data with all required information for processing
            batches = [
                (i, batch, self.nodes, wall_data, door_data, max_range) 
                for i, batch in enumerate(raw_batches)
            ]
            
            # Initialize progress tracking
            completed_batches = 0
            total_batches = len(batches)
            total_connections_found = 0
            
            # Initial status update
            if status_callback:
                status_callback(f"Starting visibility analysis with {total_nodes} nodes...", 0.01)
                # Process events to keep UI responsive
                pygame.event.pump()
            
            # Create a pool of worker processes
            with multiprocessing.Pool(processes=num_cores) as pool:
                # Process batches in parallel and collect results with additional stats
                for batch_index, batch_result, batch_stats in pool.imap_unordered(process_visibility_batch, batches):
                    # Update visibility map with batch results
                    visibility_map.update(batch_result)
                    
                    # Update progress counters
                    completed_batches += 1
                    progress = completed_batches / total_batches
                    total_connections_found += batch_stats['visible_connections']
                    
                    # Calculate batch performance metrics
                    nodes_per_second = batch_stats['nodes_processed'] / batch_stats['duration'] if batch_stats['duration'] > 0 else 0
                    
                    # More detailed status updates with performance metrics
                    status_message = (
                        f"Analyzing visibility: {int(progress * 100)}% complete | "
                        f"Found {total_connections_found:,} sight lines | "
                        f"Speed: {nodes_per_second:.1f} nodes/sec"
                    )
                    
                    # Update status with performance metrics if callback provided
                    if status_callback:
                        additional_text = [
                            f"Batch {completed_batches}/{total_batches}",
                            f"Processing {nodes_per_second:.1f} nodes/sec"
                        ]
                        status_callback(status_message, progress, additional_text)
                    
                    # Print progress periodically with more detailed statistics
                    if completed_batches % max(1, total_batches // 20) == 0 or completed_batches == total_batches:
                        elapsed = time.time() - start_time
                        eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
                        avg_connections = total_connections_found / len(visibility_map) if visibility_map else 0
                        
                        print(f"Visibility analysis: {int(progress * 100)}% complete | " +
                              f"ETA: {eta:.1f}s | " + 
                              f"Avg visible nodes: {avg_connections:.1f}")
                        
        except Exception as e:
            print(f"Error during parallel visibility analysis: {e}")
            print("Falling back to single-core processing...")
            
            # Fallback to single-core processing with more status updates
            start_fallback_time = time.time()
            total_connections = 0
            
            for i, node_from in enumerate(self.nodes):
                # More frequent status updates
                if i % 5 == 0:
                    progress = i / total_nodes
                    elapsed = time.time() - start_fallback_time
                    eta = (elapsed / (i + 1)) * (total_nodes - i - 1) if i > 0 else 0
                    
                    status_message = f"Analyzing visibility (single-core): {int(progress * 100)}% complete - ETA: {eta:.1f}s"
                    if status_callback:
                        status_callback(status_message, progress)
                    
                    # Process events to keep UI responsive during fallback
                    pygame.event.pump()
                    
                visible_nodes = []
                
                for j, node_to in enumerate(self.nodes):
                    # Skip self
                    if i == j:
                        continue
                        
                    # Check if within range
                    distance = math.sqrt((node_from[0] - node_to[0])**2 + (node_from[1] - node_to[1])**2)
                    if distance > max_range:
                        continue
                        
                    # Check line of sight
                    has_sight = self._check_line_of_sight(node_from, node_to, self.walls, self.doors)
                    if has_sight:
                        visible_nodes.append(j)
                        total_connections += 1
                        
                visibility_map[i] = visible_nodes
                
                # Print progress periodically
                if i % 50 == 0 and i > 0:
                    elapsed = time.time() - start_fallback_time
                    eta = (elapsed / (i + 1)) * (total_nodes - i - 1)
                    avg_connections = total_connections / (i + 1)
                    print(f"Processed {i}/{total_nodes} nodes | ETA: {eta:.1f}s | Avg connections: {avg_connections:.1f}")
        
        # Final progress update
        if status_callback:
            status_callback("Analysis complete - Preparing to save results...", 0.95)
        
        # Calculate final statistics for reporting
        total_connections = sum(len(nodes) for nodes in visibility_map.values())
        avg_connections = total_connections / len(visibility_map) if visibility_map else 0
        
        # Include metadata in the saved file
        visibility_data = {
            'visibility_map': visibility_map,
            'max_range': max_range,
            'grid_size': self.grid_size,
            'node_count': len(self.nodes),
            'environment_hash': self._compute_environment_hash(),
            'timestamp': time.time(),
            'total_connections': total_connections,
            'avg_connections': avg_connections
        }
        
        # Save to cache file
        try:
            # Use custom cache file if provided, otherwise use the default
            cache_file = visibility_cache_file or MAP_GRAPH_VISIBILITY_CACHE_FILE
            
            # Hide the full cache path in user-facing messages
            cache_type = "custom" if visibility_cache_file else "default"
            print(f"Saving visibility information to {cache_type} cache")
            if status_callback:
                status_callback("Saving visibility data to cache...", 0.97)
                
            with open(cache_file, 'wb') as f:
                pickle.dump(visibility_data, f)
                
            analysis_time = time.time() - start_time
            print(f"Visibility analysis completed in {analysis_time:.2f} seconds")
            print(f"Found {total_connections:,} sight lines, average {avg_connections:.1f} visible nodes per position")
            
            if status_callback:
                status_callback("Visibility analysis complete!", 1.0, [
                    f"Found {total_connections:,} sight lines",
                    f"Average {avg_connections:.1f} visible nodes per position",
                    f"Completed in {analysis_time:.2f} seconds"
                ])
                # Allow time to see the completion message
                pygame.event.pump()
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Error saving visibility data: {e}")
            if status_callback:
                status_callback(f"Error saving data: {e}", 1.0)
                pygame.event.pump()
                time.sleep(0.5)
            
        return visibility_map
        
    def load_visibility_data(self, status_callback=None, visibility_cache_file=None):
        """
        Loads the visibility data from cache if available.
        
        Args:
            status_callback (callable): Optional callback for progress updates
            visibility_cache_file (str): Optional custom path for the visibility cache file
        
        Returns:
            dict: Visibility map or None if not available
        """
        from multitrack.utils.config import MAP_GRAPH_VISIBILITY_CACHE_FILE
        
        # Use custom cache file if provided, otherwise use the default
        cache_file = visibility_cache_file or MAP_GRAPH_VISIBILITY_CACHE_FILE
        
        if not os.path.exists(cache_file):
            # Hide the full cache path in user-facing messages
            cache_type = "custom" if visibility_cache_file else "default"
            print(f"{cache_type.capitalize()} visibility cache file not found")
            if status_callback:
                status_callback("Visibility cache file not found", 1.0, ["No cached data available"])
                pygame.event.pump()
                time.sleep(0.5)
            return None
            
        try:
            # Initial status
            if status_callback:
                status_callback("Loading visibility data from cache...", 0.1)
                pygame.event.pump()
            
            # Hide the full cache path in user-facing messages
            cache_type = "custom" if visibility_cache_file else "default"
            print(f"Loading visibility data from {cache_type} cache")
            with open(cache_file, 'rb') as f:
                if status_callback:
                    status_callback("Reading cache file...", 0.3)
                visibility_data = pickle.load(f)
                
            # Update status
            if status_callback:
                status_callback("Validating visibility data...", 0.5)
                pygame.event.pump()
                
            # Validate against current environment
            current_hash = self._compute_environment_hash()
            if visibility_data['environment_hash'] != current_hash:
                print("Environment has changed since visibility data was cached")
                if status_callback:
                    status_callback("Cache invalid - environment has changed", 1.0, 
                                   ["Environment differs from when cache was created",
                                    "Press V to generate new visibility data"])
                    pygame.event.pump()
                    time.sleep(1.0)
                return None
                
            if visibility_data['node_count'] != len(self.nodes):
                print(f"Node count mismatch: cached {visibility_data['node_count']}, current {len(self.nodes)}")
                if status_callback:
                    status_callback("Cache invalid - node count mismatch", 1.0, 
                                   [f"Cached nodes: {visibility_data['node_count']}", 
                                    f"Current nodes: {len(self.nodes)}",
                                    "Press V to generate new visibility data"])
                    pygame.event.pump()
                    time.sleep(1.0)
                return None
            
            # Extract statistics
            total_connections = visibility_data.get('total_connections', 0)
            avg_connections = visibility_data.get('avg_connections', 0)
            timestamp = visibility_data.get('timestamp', 0)
            max_range = visibility_data.get('max_range', 0)
            
            # Format the timestamp if available
            cache_date = "unknown date"
            if timestamp:
                from datetime import datetime
                cache_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            # Final status update
            visibility_map = visibility_data['visibility_map']
            node_count = len(visibility_map)
            
            print(f"Loaded visibility data for {node_count} nodes")
            print(f"Total sight lines: {total_connections:,}, average {avg_connections:.1f} per node")
            print(f"Cache created on: {cache_date}, visibility range: {max_range} pixels")
            
            if status_callback:
                status_callback("Successfully loaded visibility data", 1.0, [
                    f"Loaded data for {node_count:,} nodes",
                    f"Total sight lines: {total_connections:,}",
                    f"Average: {avg_connections:.1f} visible nodes per position",
                    f"Cache date: {cache_date}"
                ])
                pygame.event.pump()
                time.sleep(0.5)
                
            return visibility_map
            
        except Exception as e:
            print(f"Error loading visibility data: {e}")
            if status_callback:
                status_callback(f"Error loading data: {str(e)}", 1.0, ["Try generating new data with V key"])
                pygame.event.pump()
                time.sleep(1.0)
            return None
            
    def _check_line_of_sight(self, from_pos, to_pos, walls, doors):
        """
        Checks if there is a clear line of sight between two positions.
        
        Args:
            from_pos: Starting position (x, y)
            to_pos: Target position (x, y)
            walls: List of wall rectangles
            doors: List of door rectangles (considered passable)
            
        Returns:
            bool: True if there is a clear line of sight
        """
        # Create line segments from the walls
        wall_segments = []
        for wall in walls:
            # Get wall coordinates 
            if isinstance(wall, tuple) or isinstance(wall, list):
                x1, y1, w, h = wall
            else:  # pygame.Rect
                x1, y1, w, h = wall.x, wall.y, wall.width, wall.height
            
            x2, y2 = x1 + w, y1 + h
            
            # Add the four wall segments
            wall_segments.append((x1, y1, x2, y1))  # Top
            wall_segments.append((x2, y1, x2, y2))  # Right
            wall_segments.append((x2, y2, x1, y2))  # Bottom
            wall_segments.append((x1, y2, x1, y1))  # Left
            
        # Line of sight from from_pos to to_pos
        x1, y1 = from_pos
        x2, y2 = to_pos
        
        # Check intersection with each wall segment
        for wx1, wy1, wx2, wy2 in wall_segments:
            # Get intersection parameters
            intersection = self._get_line_intersection_static(x1, y1, x2, y2, wx1, wy1, wx2, wy2)
            if intersection is not None:
                ua, ub = intersection
                # Check if the intersection is within the segments (not just the lines)
                if 0.0 <= ua <= 1.0 and 0.0 <= ub <= 1.0:
                    # Calculate the intersection point
                    ix = x1 + ua * (x2 - x1)
                    iy = y1 + ua * (y2 - y1)
                    
                    # Check if this intersection is with a door (doors are passable)
                    is_door = False
                    for door in doors:
                        if isinstance(door, tuple) or isinstance(door, list):
                            dx, dy, dw, dh = door
                        else:  # pygame.Rect
                            dx, dy, dw, dh = door.x, door.y, door.width, door.height
                            
                        if dx <= ix <= dx + dw and dy <= iy <= dy + dh:
                            is_door = True
                            break
                    
                    # If intersection is not at a door, there's no line of sight
                    if not is_door:
                        return False
        
        return True