"""
Environment Inspection Simulation
This script renders the simulation environment without any agents for inspection purposes.
"""

import pygame
import sys
import os
import multiprocessing
import math
import time
import numpy as np
import pickle

from multitrack.utils.config import *
from multitrack.models.simulation_environment import SimulationEnvironment
from multitrack.utils.map_graph import MapGraph

def run_environment_inspection(multicore=True, num_cores=None, auto_analyze=False, load_visibility=False, visibility_cache_file=None):
    pygame.init()

    # Import dimensions from the simulation module
    from multitrack.simulation.unicycle_reachability_simulation import (
        ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, WIDTH, HEIGHT, SIDEBAR_WIDTH,
        MAP_GRAPH_GRID_SIZE, MAP_GRAPH_MAX_EDGE_DISTANCE, MAP_GRAPH_MAX_CONNECTIONS
    )
    
    # Import cache settings
    from multitrack.utils.config import (
        MAP_GRAPH_CACHE_ENABLED, 
        MAP_GRAPH_CACHE_FILE,
        MAP_GRAPH_INSPECTION_CACHE_FILE,
        MAP_GRAPH_VISIBILITY_CACHE_FILE,
        MAP_GRAPH_VISIBILITY_RANGE
    )

    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Environment Inspection")

    # Initialize the environment
    environment = SimulationEnvironment(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT)

    # --- AGENT SETUP ---
    from multitrack.models.agents.visitor_agent import UnicycleModel
    
    AGENT_STATE_FILE = os.path.join(os.path.dirname(__file__), 'agent_state.pkl')
    
    # Try to load agent state from file
    agent_state = None
    if os.path.exists(AGENT_STATE_FILE):
        try:
            with open(AGENT_STATE_FILE, 'rb') as f:
                agent_state = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load agent state: {e}")
    if agent_state is not None:
        agent = UnicycleModel(initial_position=agent_state[:2], walls=environment.get_all_walls(), doors=environment.get_doors())
        agent.state = agent_state.copy()
    else:
        agent = UnicycleModel(walls=environment.get_all_walls(), doors=environment.get_doors())
    
    AGENT_LINEAR_VEL = LEADER_LINEAR_VEL
    AGENT_ANGULAR_VEL = LEADER_ANGULAR_VEL
    AGENT_COLOR = (255, 0, 255)  # Bright magenta for maximum visibility
    AGENT_RADIUS = 16

    # Initialize font for display
    font = pygame.font.SysFont('Arial', 18)

    # Clock for controlling the frame rate
    clock = pygame.time.Clock()
    
    # Loading status display function
    def update_loading_status(message, progress):
        screen.fill((0, 0, 0))
        
        # Create loading text
        loading_font = pygame.font.SysFont('Arial', 24)
        loading_text = loading_font.render(f"{message} {progress*100:.1f}%", True, (255, 255, 255))
        
        # Position text in center of screen
        text_rect = loading_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(loading_text, text_rect)
        
        # Draw progress bar
        bar_width = WIDTH // 2
        bar_height = 20
        bar_x = (WIDTH - bar_width) // 2
        bar_y = HEIGHT // 2 + 30
        
        # Outline
        pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height), 2)
        # Fill
        pygame.draw.rect(screen, (0, 255, 0), (bar_x, bar_y, int(bar_width * progress), bar_height))
        
        pygame.display.flip()
    
    # Function to display an animated loading screen with more details
    def render_animated_loading_screen(message, progress, additional_text=None):
        screen.fill((20, 20, 30))  # Dark blue background
        
        # Create title
        title_font = pygame.font.SysFont('Arial', 30)
        title_text = title_font.render("Environment Inspection Tool", True, (200, 200, 255))
        title_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 4))
        screen.blit(title_text, title_rect)
        
        # Create loading text
        loading_font = pygame.font.SysFont('Arial', 24)
        loading_text = loading_font.render(f"{message} {progress*100:.1f}%", True, (255, 255, 255))
        text_rect = loading_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(loading_text, text_rect)
        
        # Draw progress bar
        bar_width = WIDTH // 2
        bar_height = 20
        bar_x = (WIDTH - bar_width) // 2
        bar_y = HEIGHT // 2 + 30
        
        # Outline
        pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height), 2)
        # Fill with gradient
        for i in range(int(bar_width * progress)):
            # Create a gradient from green to blue
            color_r = int(max(0, 100 - (i / bar_width) * 100))
            color_g = int(min(255, 100 + (i / bar_width) * 155))
            color_b = int(min(255, 100 + (i / bar_width) * 155))
            pygame.draw.line(screen, (color_r, color_g, color_b), (bar_x + i, bar_y), (bar_x + i, bar_y + bar_height))
        
        # Display additional text if provided
        if additional_text:
            info_font = pygame.font.SysFont('Arial', 18)
            y_offset = HEIGHT // 2 + 70
            for line in additional_text:
                info_text = info_font.render(line, True, (200, 200, 200))
                info_rect = info_text.get_rect(center=(WIDTH // 2, y_offset))
                screen.blit(info_text, info_rect)
                y_offset += 25
        
        # Add animated spinner
        current_time = pygame.time.get_ticks()
        spinner_radius = 15
        spinner_pos = (WIDTH // 2, HEIGHT // 2 + 100)
        spinner_angle = (current_time / 30) % 360  # Rotate based on time
        
        # Draw spinner segments
        for i in range(8):
            angle = spinner_angle + i * 45
            color_intensity = 100 + (i * 20)
            end_pos = (
                spinner_pos[0] + spinner_radius * math.cos(math.radians(angle)),
                spinner_pos[1] + spinner_radius * math.sin(math.radians(angle))
            )
            pygame.draw.line(
                screen, 
                (min(255, color_intensity), min(255, color_intensity), 255), 
                spinner_pos, 
                end_pos, 
                3
            )
        
        pygame.display.flip()
        pygame.event.pump()  # Process events to prevent "not responding"
    
    # Check if we can load from cache first
    update_loading_status("Checking for cached map graph...", 0.0)
    cache_loaded = False
    
    # Initialize map graph with inspection-specific cache file
    map_graph = MapGraph(
        ENVIRONMENT_WIDTH, 
        ENVIRONMENT_HEIGHT, 
        environment.get_all_walls(), 
        environment.get_doors(),
        cache_file=MAP_GRAPH_INSPECTION_CACHE_FILE  # Use inspection-specific cache file
    )
    
    # Initialize visibility map as empty dict
    visibility_map = {}
    selected_node_index = None
    
    # Try to load from cache if enabled
    if MAP_GRAPH_CACHE_ENABLED:
        print("Attempting to load inspection map graph from cache...")
        cache_loaded = map_graph.load_from_cache()
        if cache_loaded:
            print(f"Successfully loaded map graph from cache with {len(map_graph.nodes)} nodes and {len(map_graph.edges)} edges.")
        else:
            print("Cache loading failed or no valid cache found.")
    
    # Generate map graph if not loaded from cache
    if not cache_loaded:
        print("Generating new map graph...")
        if multicore:
            cores_to_use = num_cores if num_cores else multiprocessing.cpu_count()
            print(f"Generating map graph using {cores_to_use} CPU cores...")
            map_graph.generate_parallel(update_loading_status, cores_to_use)
        else:
            print("Generating map graph using single core...")
            map_graph.generate(update_loading_status)
        
        print(f"Map graph generated with {len(map_graph.nodes)} nodes and {len(map_graph.edges)} edges.")
        
        # Save to cache for future use if caching is enabled
        if MAP_GRAPH_CACHE_ENABLED:
            update_loading_status("Saving inspection map graph to cache...", 0.95)
            map_graph.save_to_cache()

    running = True
    show_map_graph = True  # Start with map graph visible
    
    # Handle automatic loading and analysis of visibility data if requested
    if load_visibility:
        print("Auto-loading visibility data requested...")
        visibility_map = map_graph.load_visibility_data(
            render_animated_loading_screen,
            visibility_cache_file or MAP_GRAPH_VISIBILITY_CACHE_FILE
        )
        if visibility_map:
            total_connections = sum(len(nodes) for nodes in visibility_map.values())
            print(f"Loaded visibility data: {total_connections:,} total sight lines")
            # Set initial selected node
            if selected_node_index is None and map_graph.nodes:
                selected_node_index = 0
        else:
            print("Could not load visibility data - continuing without it")
            
    # Handle automatic visibility analysis if requested
    if auto_analyze and not visibility_map:
        print("Auto-analyzing visibility requested...")
        # Use the animated loading screen for better feedback
        cores_to_use = num_cores if num_cores else None
        visibility_map = map_graph.analyze_node_visibility(
            MAP_GRAPH_VISIBILITY_RANGE,
            render_animated_loading_screen,
            cores_to_use,
            visibility_cache_file or MAP_GRAPH_VISIBILITY_CACHE_FILE
        )
        if visibility_map:
            # Set initial selected node after analysis
            if selected_node_index is None and map_graph.nodes:
                selected_node_index = 0
    
    # Function to calculate and display detailed stats during visibility analysis
    def calculate_visibility_stats(visibility_map, total_nodes):
        """Calculate statistics about the visibility data"""
        if not visibility_map:
            return []
            
        # Calculate statistics
        total_connections = sum(len(nodes) for nodes in visibility_map.values())
        avg_connections = total_connections / len(visibility_map) if visibility_map else 0
        max_connections = max(len(nodes) for nodes in visibility_map.values()) if visibility_map else 0
        visibility_pct = 100 * avg_connections / total_nodes if total_nodes > 0 else 0
        
        # Max visibility node
        max_visibility_node = max(visibility_map.items(), key=lambda x: len(x[1]))[0] if visibility_map else None
        
        return [
            f"Visibility Statistics:",
            f"Total sight lines: {total_connections:,}",
            f"Average visible nodes: {avg_connections:.1f}",
            f"Visibility percentage: {visibility_pct:.1f}%",
            f"Maximum connections: {max_connections}",
            f"Most visible node: {max_visibility_node}"
        ]
    
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_g:
                        # Toggle map graph display
                        show_map_graph = not show_map_graph
                        print(f"Map graph display: {'On' if show_map_graph else 'Off'}")
                    elif event.key == pygame.K_r and show_map_graph:
                        # Regenerate map graph
                        print("Regenerating map graph...")
                        # Create new map graph with the current configuration parameters
                        map_graph = MapGraph(
                            ENVIRONMENT_WIDTH, 
                            ENVIRONMENT_HEIGHT, 
                            environment.get_all_walls(), 
                            environment.get_doors(),
                            cache_file=MAP_GRAPH_INSPECTION_CACHE_FILE  # Use inspection-specific cache file
                        )
                        if multicore:
                            cores_to_use = num_cores if num_cores else multiprocessing.cpu_count()
                            map_graph.generate_parallel(update_loading_status, cores_to_use)
                        else:
                            map_graph.generate(update_loading_status)
                        print(f"Map graph regenerated with {len(map_graph.nodes)} nodes and {len(map_graph.edges)} edges.")
                        
                        # Save to cache if enabled
                        if MAP_GRAPH_CACHE_ENABLED:
                            print(f"Saving regenerated map graph to inspection cache file")
                            try:
                                map_graph.save_to_cache()
                                print("Inspection map graph successfully saved to cache.")
                            except Exception as e:
                                print(f"Error saving inspection map graph to cache: {e}")
                    elif event.key == pygame.K_v:
                        # Generate and save visibility information
                        print(f"Analyzing node visibility (range: {MAP_GRAPH_VISIBILITY_RANGE} pixels)...")
                        # Show a message on screen
                        message_font = pygame.font.SysFont('Arial', 30)
                        message = message_font.render("Preparing visibility analysis...", True, (255, 255, 255))
                        screen.blit(message, (WIDTH // 2 - message.get_width() // 2, HEIGHT // 2 - message.get_height() // 2))
                        pygame.display.flip()
                        
                        # Give the UI a moment to update before starting intensive processing
                        pygame.event.pump()
                        time.sleep(0.1)
                        
                        # Start the visibility analysis with the improved animated loading screen
                        # Pass num_cores to allow command-line control of parallelism
                        cores_to_use = num_cores if num_cores else None
                        visibility_map = map_graph.analyze_node_visibility(
                            MAP_GRAPH_VISIBILITY_RANGE,
                            render_animated_loading_screen,
                            cores_to_use,
                            visibility_cache_file or MAP_GRAPH_VISIBILITY_CACHE_FILE
                        )
                        if visibility_map:
                            total_connections = sum(len(nodes) for nodes in visibility_map.values())
                            print(f"Visibility analysis complete: {total_connections} total sight lines")
                            
                            # Calculate and display visibility statistics
                            visibility_stats = calculate_visibility_stats(visibility_map, len(map_graph.nodes))
                            for stat in visibility_stats:
                                print(stat)
                            
                            # Select first node if none selected
                            if selected_node_index is None and map_graph.nodes:
                                selected_node_index = 0
                    
                    elif event.key == pygame.K_l:
                        # Load visibility data from cache
                        print("Loading visibility data from cache...")
                        message_font = pygame.font.SysFont('Arial', 30)
                        message = message_font.render("Loading visibility data...", True, (255, 255, 255))
                        screen.blit(message, (WIDTH // 2 - message.get_width() // 2, HEIGHT // 2 - message.get_height() // 2))
                        pygame.display.flip()
                        
                        # Pass the animated loading screen function for better feedback
                        visibility_map = map_graph.load_visibility_data(
                            render_animated_loading_screen,
                            visibility_cache_file or MAP_GRAPH_VISIBILITY_CACHE_FILE
                        )
                        if visibility_map:
                            total_connections = sum(len(nodes) for nodes in visibility_map.values())
                            print(f"Loaded visibility data: {total_connections} total sight lines")
                            
                            # Calculate and display visibility statistics
                            visibility_stats = calculate_visibility_stats(visibility_map, len(map_graph.nodes))
                            for stat in visibility_stats:
                                print(stat)
                            
                            # Select first node if none selected
                            if selected_node_index is None and map_graph.nodes:
                                selected_node_index = 0
                        else:
                            print("No visibility data found. Press V to analyze visibility.")
                        
                    elif event.key == pygame.K_n:
                        # Go to next node in map graph
                        if map_graph.nodes:
                            if selected_node_index is None:
                                selected_node_index = 0
                            else:
                                selected_node_index = (selected_node_index + 1) % len(map_graph.nodes)
                            
                            if visibility_map and selected_node_index in visibility_map:
                                visible_count = len(visibility_map[selected_node_index])
                                print(f"Selected node {selected_node_index} with {visible_count} visible nodes")
                            else:
                                print(f"Selected node {selected_node_index} (no visibility data)")
                    
                    elif event.key == pygame.K_p:
                        # Go to previous node in map graph
                        if map_graph.nodes:
                            if selected_node_index is None:
                                selected_node_index = len(map_graph.nodes) - 1
                            else:
                                selected_node_index = (selected_node_index - 1) % len(map_graph.nodes)
                            
                            if visibility_map and selected_node_index in visibility_map:
                                visible_count = len(visibility_map[selected_node_index])
                                print(f"Selected node {selected_node_index} with {visible_count} visible nodes")
                            else:
                                print(f"Selected node {selected_node_index} (no visibility data)")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and show_map_graph:  # Left mouse button
                        # Get mouse position
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        
                        # Find the closest node to the mouse click
                        closest_node_index = None
                        closest_distance = float('inf')
                        
                        # Only search within a limited area for better performance
                        click_search_radius = 20  # pixels
                        
                        for i, node in enumerate(map_graph.nodes):
                            node_x, node_y = node
                            # Quick bounds check before calculating exact distance
                            if abs(node_x - mouse_x) > click_search_radius or abs(node_y - mouse_y) > click_search_radius:
                                continue
                                
                            # Calculate exact distance
                            distance = ((node_x - mouse_x) ** 2 + (node_y - mouse_y) ** 2) ** 0.5
                            
                            # Only consider nodes within the click radius
                            if distance < click_search_radius and distance < closest_distance:
                                closest_distance = distance
                                closest_node_index = i
                        
                        # If a node was clicked, select it
                        if closest_node_index is not None:
                            # Store previous selection to show changes
                            previous_node_index = selected_node_index
                            selected_node_index = closest_node_index
                            
                            # Get node coordinates for better feedback
                            node_x, node_y = map_graph.nodes[selected_node_index]
                            
                            # Get visibility information
                            visible_nodes = []
                            if visibility_map and selected_node_index in visibility_map:
                                visible_nodes = visibility_map[selected_node_index]
                                visible_count = len(visible_nodes)
                                
                                # Calculate percentage of nodes visible
                                visibility_percentage = (visible_count / len(map_graph.nodes)) * 100
                                
                                print(f"Selected node {selected_node_index} at ({int(node_x)}, {int(node_y)})")
                                print(f"Visibility: {visible_count:,} nodes ({visibility_percentage:.1f}% of all nodes)")
                                
                                # If this is a new selection, play a subtle sound effect for feedback
                                if previous_node_index != selected_node_index:
                                    # Simple feedback with pygame.mixer if available
                                    try:
                                        pygame.mixer.init()
                                        pygame.mixer.music.stop()  # Stop any playing sound
                                        
                                        # Generate a quick beep
                                        frequency = 1000  # Hz
                                        duration = 100  # ms
                                        sample_rate = 44100  # Hz
                                        
                                        # Generate square wave (simple beep)
                                        buf = np.sin(2 * np.pi * np.arange(sample_rate * duration / 1000) * frequency / sample_rate)
                                        sound = pygame.sndarray.make_sound((buf * 32767).astype(np.int16))
                                        sound.set_volume(0.2)  # Keep volume low
                                        sound.play()
                                    except:
                                        # If sound fails, just continue silently
                                        pass
                            else:
                                print(f"Selected node {selected_node_index} at ({int(node_x)}, {int(node_y)})")
                                print("No visibility data available for this node.")
                                print("Press V to analyze visibility or L to load cached data.")
                        else:
                            print("No node found near the click position")

            # --- AGENT CONTROL (after event handling, before drawing) ---
            keys = pygame.key.get_pressed()
            linear_vel = 0
            angular_vel = 0
            if keys[pygame.K_UP]:
                linear_vel = AGENT_LINEAR_VEL
            if keys[pygame.K_DOWN]:
                linear_vel = -AGENT_LINEAR_VEL
            if keys[pygame.K_LEFT]:
                angular_vel = -AGENT_ANGULAR_VEL
            if keys[pygame.K_RIGHT]:
                angular_vel = AGENT_ANGULAR_VEL
            agent.set_controls(linear_vel, angular_vel)
            agent.update(dt=0.1, walls=environment.get_all_walls(), doors=environment.get_doors())

            # Debug: print agent state
            # print(f"Agent state: {agent.state}")

            # Clear the screen
            screen.fill((0, 0, 0))

            # Draw the environment
            environment.draw(screen, font)
            if show_map_graph:
                # Draw the graph edges first
                for edge in map_graph.edges:
                    i, j = edge
                    start = map_graph.nodes[i]
                    end = map_graph.nodes[j]
                    pygame.draw.line(screen, MAP_GRAPH_EDGE_COLOR, start, end, 1)
                
                # Draw the nodes last to make them more visible
                for i, node in enumerate(map_graph.nodes):
                    # Determine if this is the node under the mouse for hover effect
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    distance = ((node[0] - mouse_x) ** 2 + (node[1] - mouse_y) ** 2) ** 0.5
                    
                    if distance < 15:  # Mouse hover highlight
                        pygame.draw.circle(screen, (255, 165, 0), node, 6)  # Orange highlight for hover
                    else:
                        # Regular node
                        pygame.draw.circle(screen, MAP_GRAPH_NODE_COLOR, node, 4)
                
                # Display map graph stats
                stats_font = pygame.font.SysFont('Arial', 20)
                stats_text = [
                    f"Map Graph Parameters:",
                    f"Grid Size: {MAP_GRAPH_GRID_SIZE}",
                    f"Nodes: {len(map_graph.nodes)}",
                    f"Edges: {len(map_graph.edges)}",
                    f"Max Edge Distance: {MAP_GRAPH_MAX_EDGE_DISTANCE}",
                    f"Max Connections: {MAP_GRAPH_MAX_CONNECTIONS}"
                ]
                
                # Draw stats background
                stats_bg = pygame.Surface((300, len(stats_text) * 25 + 10))
                stats_bg.fill((0, 0, 0))
                stats_bg.set_alpha(180)
                
                # Position at bottom right of screen
                stats_x = WIDTH - 310
                stats_y = HEIGHT - len(stats_text) * 25 - 20
                
                # Draw background and text
                screen.blit(stats_bg, (stats_x, stats_y))
                for i, text in enumerate(stats_text):
                    text_surf = stats_font.render(text, True, (200, 200, 255))
                    screen.blit(text_surf, (stats_x + 10, stats_y + 10 + i * 25))

            # Display info text
            info_text = [
                "Environment Inspection Mode",
                f"Size: {ENVIRONMENT_WIDTH}x{ENVIRONMENT_HEIGHT}",
                "Controls:",
                "G: Toggle map graph display",
                "R: Regenerate map graph (when visible)",
                "V: Analyze node visibility",
                "L: Load visibility data",
                "N: Next node (visibility mode)",
                "P: Previous node (visibility mode)",
                "ESC: Exit"
            ]
            
            # Add cache info
            if MAP_GRAPH_CACHE_ENABLED:
                cache_status = "Enabled" if MAP_GRAPH_CACHE_ENABLED else "Disabled"
                cache_info = f"Map cache: {cache_status}"
                
                # For visibility cache, just indicate if custom or default without showing the full path
                if visibility_cache_file and visibility_cache_file != MAP_GRAPH_VISIBILITY_CACHE_FILE:
                    visibility_cache = "Visibility cache: Custom"
                else:
                    visibility_cache = "Visibility cache: Default"
                    
                info_text.append(cache_info)
                info_text.append(visibility_cache)
            
            # Display the info on the screen
            info_bg = pygame.Surface((SIDEBAR_WIDTH - 20, len(info_text) * 25 + 10))
            info_bg.fill((0, 0, 0))
            info_bg.set_alpha(180)
            
            # Position at top right in sidebar area
            screen.blit(info_bg, (ENVIRONMENT_WIDTH + 10, 10))
            
            for i, text in enumerate(info_text):
                text_surface = font.render(text, True, (255, 255, 255))
                screen.blit(text_surface, (ENVIRONMENT_WIDTH + 20, 20 + i * 25))

            # Draw visibility data if selected node and visibility map exist
            if show_map_graph and selected_node_index is not None:
                if selected_node_index < len(map_graph.nodes):
                    # Get selected node's position
                    selected_node = map_graph.nodes[selected_node_index]
                    
                    # If visibility data is available, draw the connections
                    if visibility_map and selected_node_index in visibility_map:
                        # Draw a large circle showing the visibility range
                        # Surface needs to be larger than the range to fit the entire circle
                        range_size = MAP_GRAPH_VISIBILITY_RANGE * 2
                        range_surface = pygame.Surface((range_size, range_size), pygame.SRCALPHA)
                        
                        # Create pulsing effect for the range indicator
                        pulse_intensity = (math.sin(pygame.time.get_ticks() / 500) + 1) / 4 + 0.25  # Value between 0.25 and 0.75
                        range_alpha = int(40 * pulse_intensity)  # Pulsing alpha between 10 and 30
                        
                        # Draw the visibility range with pulsing effect
                        pygame.draw.circle(range_surface, (0, 255, 100, range_alpha), (range_size//2, range_size//2), MAP_GRAPH_VISIBILITY_RANGE)
                        
                        # Draw an additional thin circle at the exact visibility range
                        pygame.draw.circle(range_surface, (0, 255, 120, 80), (range_size//2, range_size//2), MAP_GRAPH_VISIBILITY_RANGE, 1)
                        
                        # Position the surface so the circle is centered on the selected node
                        range_x = selected_node[0] - range_size//2
                        range_y = selected_node[1] - range_size//2
                        screen.blit(range_surface, (range_x, range_y))
                        
                        # Get visible nodes and prepare for visualization
                        visible_nodes = visibility_map[selected_node_index]
                        visible_count = len(visible_nodes)
                        
                        # Group nodes by distance for better visualization (close, medium, far)
                        node_distance_groups = {
                            'close': [],
                            'medium': [],
                            'far': []
                        }
                        
                        # Calculate distance thresholds based on visibility range
                        close_threshold = MAP_GRAPH_VISIBILITY_RANGE * 0.33
                        medium_threshold = MAP_GRAPH_VISIBILITY_RANGE * 0.66
                        
                        # Categorize visible nodes by distance
                        for visible_index in visible_nodes:
                            if visible_index < len(map_graph.nodes):  # Safety check
                                visible_node = map_graph.nodes[visible_index]
                                node_distance = math.dist(selected_node, visible_node)
                                
                                if node_distance <= close_threshold:
                                    node_distance_groups['close'].append(visible_node)
                                elif node_distance <= medium_threshold:
                                    node_distance_groups['medium'].append(visible_node)
                                else:
                                    node_distance_groups['far'].append(visible_node)
                        
                        # Draw lines to close nodes (bright green)
                        for visible_node in node_distance_groups['close']:
                            pygame.draw.line(screen, (50, 255, 50), selected_node, visible_node, 2)
                            pygame.draw.circle(screen, (50, 255, 50), visible_node, 5)
                        
                        # Draw lines to medium-range nodes (yellow-green)
                        for visible_node in node_distance_groups['medium']:
                            pygame.draw.line(screen, (150, 255, 30), selected_node, visible_node, 1)
                            pygame.draw.circle(screen, (150, 255, 30), visible_node, 4)
                        
                        # Draw lines to far nodes (faded green)
                        for visible_node in node_distance_groups['far']:
                            pygame.draw.line(screen, (100, 180, 30, 150), selected_node, visible_node, 1)
                            pygame.draw.circle(screen, (100, 180, 30), visible_node, 3)
                    
                    # Always draw the selected node with a different color and larger size (on top of everything)
                    # Add pulsing effect to selected node
                    pulse = (math.sin(pygame.time.get_ticks() / 300) + 1) / 4 + 0.75  # Value between 0.75 and 1.25
                    
                    # Draw outer glow/halo
                    glow_size = int(12 * pulse)
                    pygame.draw.circle(screen, (255, 255, 100, 100), selected_node, glow_size)
                    
                    # Draw the selected node itself
                    pygame.draw.circle(screen, (255, 255, 0), selected_node, 8)
                    pygame.draw.circle(screen, (255, 200, 0), selected_node, 4)
                    
                    # Draw selection information with enhanced details
                    visibility_info = [
                        f"Selected Node: {selected_node_index}",
                        f"Position: ({int(selected_node[0])}, {int(selected_node[1])})"
                    ]
                    
                    # Add visibility info if available
                    if visibility_map and selected_node_index in visibility_map:
                        visible_nodes = visibility_map[selected_node_index]
                        visible_count = len(visible_nodes)
                        visibility_percentage = (visible_count / len(map_graph.nodes)) * 100 if map_graph.nodes else 0
                        
                        # Count nodes by distance category
                        close_count = len(node_distance_groups['close'])
                        medium_count = len(node_distance_groups['medium'])
                        far_count = len(node_distance_groups['far'])
                        
                        visibility_info.extend([
                            f"Visible Nodes: {visible_count:,} ({visibility_percentage:.1f}% of map)",
                            f"Distance Groups:",
                            f"  Close: {close_count} nodes (≤{int(close_threshold)}px)",
                            f"  Medium: {medium_count} nodes (≤{int(medium_threshold)}px)",
                            f"  Far: {far_count} nodes (≤{MAP_GRAPH_VISIBILITY_RANGE}px)",
                            f"Visibility Range: {MAP_GRAPH_VISIBILITY_RANGE} pixels"
                        ])
                    else:
                        visibility_info.extend([
                            "No visibility data available",
                            "Press V to analyze visibility",
                            "Press L to load cached data"
                        ])
                    
                    visibility_info.append("Click on nodes or use N/P to navigate")
                    
                    # Create background for visibility info
                    info_width = 350
                    info_bg = pygame.Surface((info_width, len(visibility_info) * 25 + 10))
                    info_bg.fill((0, 0, 30))  # Dark blue background
                    info_bg.set_alpha(200)
                    
                    # Position at top left
                    info_x = 10
                    info_y = 10
                    
                    # Draw background with rounded corners effect
                    screen.blit(info_bg, (info_x, info_y))
                    
                    # Add a title bar effect
                    pygame.draw.rect(screen, (0, 100, 200), (info_x, info_y, info_width, 30))
                    pygame.draw.rect(screen, (0, 70, 150), (info_x, info_y, info_width, 30), 1)
                    
                    # Draw a thin border around the entire info panel
                    pygame.draw.rect(screen, (100, 100, 200), (info_x, info_y, info_width, len(visibility_info) * 25 + 10), 1)
                    
                    # Render each line of text with appropriate styling
                    title_font = pygame.font.SysFont('Arial', 22, bold=True)
                    regular_font = pygame.font.SysFont('Arial', 20)
                    small_font = pygame.font.SysFont('Arial', 18)
                    
                    # Draw title
                    title_text = title_font.render("Node Visibility Analysis", True, (255, 255, 255))
                    screen.blit(title_text, (info_x + (info_width - title_text.get_width()) // 2, info_y + 5))
                    
                    # Draw content with different colors based on type
                    y_offset = info_y + 35
                    
                    for i, text in enumerate(visibility_info):
                        # Skip the first item (handled as title)
                        if i == 0:
                            # Draw node ID with highlighted color
                            node_text = regular_font.render(text, True, (255, 255, 100))
                            screen.blit(node_text, (info_x + 10, y_offset))
                            y_offset += 25
                        elif "Position:" in text:
                            # Position info
                            pos_text = regular_font.render(text, True, (200, 200, 255))
                            screen.blit(pos_text, (info_x + 10, y_offset))
                            y_offset += 25
                        elif "Visible Nodes:" in text:
                            # Visibility count (important info in bright color)
                            vis_text = regular_font.render(text, True, (50, 255, 50))
                            screen.blit(vis_text, (info_x + 10, y_offset))
                            y_offset += 25
                        elif "Distance Groups:" in text:
                            # Section header
                            header_text = regular_font.render(text, True, (255, 200, 100))
                            screen.blit(header_text, (info_x + 10, y_offset))
                            y_offset += 25
                        elif text.startswith("  Close:"):
                            # Close distance nodes (bright green)
                            close_text = small_font.render(text, True, (50, 255, 50))
                            screen.blit(close_text, (info_x + 10, y_offset))
                            y_offset += 20
                        elif text.startswith("  Medium:"):
                            # Medium distance nodes (yellow-green)
                            medium_text = small_font.render(text, True, (150, 255, 30))
                            screen.blit(medium_text, (info_x + 10, y_offset))
                            y_offset += 20
                        elif text.startswith("  Far:"):
                            # Far distance nodes (faded green)
                            far_text = small_font.render(text, True, (100, 180, 30))
                            screen.blit(far_text, (info_x + 10, y_offset))
                            y_offset += 20
                        elif "No visibility data" in text:
                            # Warning message
                            warning_text = regular_font.render(text, True, (255, 100, 100))
                            screen.blit(warning_text, (info_x + 10, y_offset))
                            y_offset += 25
                        elif "Press V" in text or "Press L" in text:
                            # Instruction text
                            instruction_text = small_font.render(text, True, (200, 200, 200))
                            screen.blit(instruction_text, (info_x + 20, y_offset))
                            y_offset += 20
                        elif "Click on nodes" in text:
                            # Help text at the bottom
                            help_text = small_font.render(text, True, (180, 180, 220))
                            screen.blit(help_text, (info_x + 10, y_offset + 5))
                            y_offset += 25
                        else:
                            # Default styling
                            default_text = regular_font.render(text, True, (200, 200, 255))
                            screen.blit(default_text, (info_x + 10, y_offset))
                            y_offset += 25

            # Draw the agent last, always on top
            x, y, theta, _ = agent.state
            pygame.draw.circle(screen, AGENT_COLOR, (int(x), int(y)), AGENT_RADIUS)
            end_x = x + AGENT_RADIUS * 1.2 * math.cos(theta)
            end_y = y + AGENT_RADIUS * 1.2 * math.sin(theta)
            pygame.draw.line(screen, (255,255,255), (x, y), (end_x, end_y), 3)

            # Update the display
            pygame.display.flip()

            # Cap the frame rate
            clock.tick(60)
    finally:
        # Save agent state on exit
        try:
            with open(AGENT_STATE_FILE, 'wb') as f:
                pickle.dump(agent.state, f)
        except Exception as e:
            print(f"Warning: Could not save agent state: {e}")

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_environment_inspection()
