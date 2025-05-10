"""
Unicycle model simulation with Kalman filter monitoring system for state estimation and prediction.
Also includes an MPPI-controlled follower agent that tracks the main agent.
"""
import sys
import pygame
import numpy as np
from math import sin, cos, pi, sqrt
from multitrack.models.agents.visitor_agent import UnicycleModel
from multitrack.models.agents.escort_agent import FollowerAgent
from multitrack.controllers.mppi_controller import DEVICE_INFO
from multitrack.models.simulation_environment import SimulationEnvironment
from multitrack.visualizations.enhanced_rendering import (
    draw_enhanced_agent, generate_particles, update_particles, draw_particles
)
from multitrack.visualizations.information_overlay import InformationSidebarThread
from multitrack.utils.config import *
from multitrack.utils.vision import is_agent_in_vision_cone
from multitrack.filters.kalman_filter import UnicycleKalmanFilter
from multitrack.utils.map_graph import MapGraph  # Importing MapGraph
import multiprocessing
import time

# Initialize pygame
pygame.init()

# Constants
SIDEBAR_WIDTH = 250  # Width of the information sidebar
ENVIRONMENT_WIDTH = 1280  # Width of the environment area
ENVIRONMENT_HEIGHT = 720  # Height of the environment area
WIDTH = ENVIRONMENT_WIDTH + SIDEBAR_WIDTH  # Total window width including sidebar
HEIGHT = ENVIRONMENT_HEIGHT  # Window height

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)  # Color for the follower agent

# Monitoring system settings
SHOW_PREDICTIONS = True  # Show Kalman filter predictions
PREDICTION_STEPS = 20    # Number of steps to predict into the future
SHOW_UNCERTAINTY = True  # Show uncertainty ellipse
PREDICTION_COLOR = CYAN  # Color for predictions
UNCERTAINTY_COLOR = (100, 100, 255, 100)  # Light blue with transparency

# Follower agent settings
SHOW_MPPI_PREDICTIONS = True  # Show MPPI predictions
MPPI_PREDICTION_COLOR = (255, 100, 100)  # Light red
FOLLOWER_ENABLED = True  # Enable/disable follower agent

# Main simulation loop
def run_simulation(multicore=None, num_cores=None):
    """
    Run the main simulation loop.
    
    Args:
        multicore: Whether to use multicore processing for map generation (None = use config value)
        num_cores: Number of CPU cores to use (None = all available)
    """
    # Determine whether to use multicore processing
    if multicore is None:
        multicore = MAP_GRAPH_MULTICORE_DEFAULT
        
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Visitor and Escort Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)
    
    # Create and start the information sidebar thread
    overlay = InformationSidebarThread(WIDTH, HEIGHT, SIDEBAR_WIDTH)
    overlay.start()
    
    # Monitoring options
    global SHOW_PREDICTIONS, SHOW_UNCERTAINTY, SHOW_MPPI_PREDICTIONS, FOLLOWER_ENABLED
    
    # Create environment
    environment = SimulationEnvironment(width=ENVIRONMENT_WIDTH, height=ENVIRONMENT_HEIGHT)
    
    # Create map graph with loading screen - use environment dimensions only
    map_graph = MapGraph(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, environment.get_all_walls(), environment.get_doors())
    
    # Display loading screen and generate map graph
    def update_loading_status(message, progress):
        render_loading_screen(screen, message, progress)
        pygame.event.pump()  # Process events to prevent "not responding"
    
    # Check if we can load from cache first
    update_loading_status("Checking for cached map graph...", 0.0)
    cache_loaded = False
    
    # Try to load from cache if enabled
    if MAP_GRAPH_CACHE_ENABLED:
        cache_loaded = map_graph.load_from_cache()
        
        if cache_loaded:
            # Validate the cached graph against the current environment
            update_loading_status("Validating cached map graph...", 0.3)
            if not map_graph.validate_cached_graph():
                print("Cached map graph validation failed. Generating new graph...")
                cache_loaded = False
            else:
                update_loading_status("Cached map graph is valid", 1.0)
                time.sleep(0.5)  # Brief pause to show completion
    
    # Generate map graph if not loaded from cache
    if not cache_loaded:
        update_loading_status("No valid cache found. Generating new map graph...", 0.0)
        
        # Use parallel processing if enabled
        start_time = time.time()
        cores_to_use = num_cores if num_cores else multiprocessing.cpu_count()
        
        if multicore:
            print(f"Generating map graph using {cores_to_use} CPU cores...")
            map_graph.generate_parallel(update_loading_status, cores_to_use)
        else:
            print("Generating map graph using single core...")
            map_graph.generate(update_loading_status)
        
        generation_time = time.time() - start_time
        print(f"Map graph generation completed in {generation_time:.2f} seconds")
        print(f"Generated {len(map_graph.nodes)} nodes and {len(map_graph.edges)} edges")
        
        # Save to cache for future use if caching is enabled
        if MAP_GRAPH_CACHE_ENABLED:
            update_loading_status("Saving map graph to cache...", 0.95)
            map_graph.save_to_cache()
    
    # Initialize models with environment information to ensure valid starting positions
    model = UnicycleModel(walls=environment.get_all_walls(), doors=environment.get_doors())
    
    if FOLLOWER_ENABLED:
        follower = FollowerAgent(target_distance=100.0, 
                               walls=environment.get_all_walls(), 
                               doors=environment.get_doors())
    else:
        follower = None
    
    # Display options
    show_fps = True
    debug_mode = False  # For visualizing turning radius
    enhanced_visuals = True  # Toggle for human-like rendering
    
    # Map graph display flag
    show_map_graph = False
    paused = False
    
    # Particle effects
    visitor_particles = []
    escort_particles = []
    
    # Performance monitoring
    frame_times = []
    last_mppi_update_time = 0
    mppi_update_interval = 1.0  # Update MPPI stats every second
    
    # Time tracking
    start_time_ms = pygame.time.get_ticks()
    
    # Follower control mode
    manual_escort_control = False
    
    # Key debugging
    key_debug = False  # Toggle for key debugging display on screen
    debug_key_messages = False  # Toggle for key debug messages in console
    
    running = True
    while running:
        frame_start_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Print which key was pressed for debugging (only if debug_key_messages is True)
                if debug_key_messages:
                    print(f"Key pressed: {pygame.key.name(event.key)} (keycode: {event.key})")
                
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    show_fps = not show_fps
                elif event.key == pygame.K_k:
                    # Toggle Kalman filter prediction display
                    SHOW_PREDICTIONS = not SHOW_PREDICTIONS
                    if debug_key_messages:
                        print(f"Kalman filter predictions {'enabled' if SHOW_PREDICTIONS else 'disabled'}")
                elif event.key == pygame.K_u:
                    # Toggle uncertainty ellipse display
                    SHOW_UNCERTAINTY = not SHOW_UNCERTAINTY
                    if debug_key_messages:
                        print(f"Uncertainty ellipses {'enabled' if SHOW_UNCERTAINTY else 'disabled'}")
                elif event.key == pygame.K_m:
                    # Toggle MPPI predictions display
                    SHOW_MPPI_PREDICTIONS = not SHOW_MPPI_PREDICTIONS
                    if debug_key_messages:
                        print(f"MPPI predictions {'enabled' if SHOW_MPPI_PREDICTIONS else 'disabled'}")
                elif event.key == pygame.K_t:
                    # Toggle follower agent
                    FOLLOWER_ENABLED = not FOLLOWER_ENABLED
                    if FOLLOWER_ENABLED and follower is None:
                        follower = FollowerAgent(target_distance=100.0)
                    if not FOLLOWER_ENABLED:
                        follower = None
                # Toggle controller type
                elif event.key == pygame.K_p:
                    # Toggle controller type (only if follower is enabled)
                    if follower is not None:
                        # Switch between MPPI and PID controllers
                        new_controller_type = "pid" if follower.controller_type == "mppi" else "mppi"
                        follower.set_controller_type(new_controller_type)
                        if debug_key_messages:
                            print(f"Switched to {new_controller_type.upper()} controller")
                elif event.key == pygame.K_a:
                    # Toggle camera auto-tracking
                    if follower:
                        auto_track_enabled = follower.toggle_camera_auto_track()
                        if debug_key_messages:
                            print(f"Camera auto-tracking {'enabled' if auto_track_enabled else 'disabled'}")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    # For measurement rate control with Shift, follower distance without
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT and follower:
                        # Increase measurement frequency (decrease interval)
                        new_interval = follower.adjust_measurement_interval(-0.1)
                        if debug_key_messages:
                            print(f"Measurement interval decreased to {new_interval:.1f}s")
                    elif follower:
                        # Increase following distance
                        follower.target_distance = min(FOLLOWER_MAX_DISTANCE, follower.target_distance + 10)
                        if debug_key_messages:
                            print(f"Follower distance increased to {follower.target_distance}")
                elif event.key == pygame.K_MINUS:
                    # For measurement rate control with Shift, follower distance without
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT and follower:
                        # Decrease measurement frequency (increase interval)
                        new_interval = follower.adjust_measurement_interval(0.1)
                        if debug_key_messages:
                            print(f"Measurement interval increased to {new_interval:.1f}s")
                    elif follower:
                        # Decrease following distance
                        follower.target_distance = max(FOLLOWER_MIN_DISTANCE, follower.target_distance - 10)
                        if debug_key_messages:
                            print(f"Follower distance decreased to {follower.target_distance}")
                elif event.key == pygame.K_r:
                    # Reset follower position (random)
                    if follower:
                        follower = FollowerAgent(
                            target_distance=follower.target_distance,
                            walls=environment.get_all_walls(), 
                            doors=environment.get_doors()
                        )
                elif event.key == pygame.K_d:
                    # Toggle debug mode
                    debug_mode = not debug_mode
                elif event.key == pygame.K_v:
                    # Toggle enhanced visuals
                    enhanced_visuals = not enhanced_visuals
                elif event.key == pygame.K_c:
                    # Toggle manual control mode for escort
                    if follower:
                        manual_escort_control = not manual_escort_control
                        follower.set_auto_mode(not manual_escort_control)
                elif event.key == pygame.K_h:
                    # Toggle key debugging on screen
                    key_debug = not key_debug
                    if debug_key_messages:
                        print(f"Key debugging display {'enabled' if key_debug else 'disabled'}")
                elif event.key == pygame.K_j:
                    # Toggle debug key messages in console
                    debug_key_messages = not debug_key_messages
                    print(f"Key message debugging {'enabled' if debug_key_messages else 'disabled'}")
                elif event.key == pygame.K_g:
                    # Toggle map graph display and pause the simulation
                    show_map_graph = not show_map_graph
                    paused = show_map_graph  # Pause when showing graph, unpause when hiding
                    if show_map_graph:
                        print("Simulation paused. Showing map graph.")
                    else:
                        print("Simulation resumed. Map graph hidden.")
                elif event.key == pygame.K_r and show_map_graph:
                    # Regenerate map graph with current config parameters when map is displayed
                    print(f"Regenerating map graph with grid size: {MAP_GRAPH_GRID_SIZE}...")
                    # Create new map graph with the current configuration parameters
                    map_graph = MapGraph(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, environment.get_all_walls(), environment.get_doors())
                    # Display loading screen and generate graph
                    if multicore:
                        cores_to_use = num_cores if num_cores else multiprocessing.cpu_count()
                        print(f"Regenerating map graph using {cores_to_use} CPU cores...")
                        map_graph.generate_parallel(update_loading_status, cores_to_use)
                    else:
                        print("Regenerating map graph using single core...")
                        map_graph.generate(update_loading_status)
                    print(f"Map graph regenerated with {len(map_graph.nodes)} nodes and {len(map_graph.edges)} edges.")
        
        # Get keyboard input to control the unicycle
        keys = pygame.key.get_pressed()
        
        # Debug key states - print which arrow keys are pressed (only if debug_key_messages is True)
        if debug_key_messages and (keys[pygame.K_UP] or keys[pygame.K_DOWN] or keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]):
            print(f"Active arrow keys: UP={keys[pygame.K_UP]} DOWN={keys[pygame.K_DOWN]} LEFT={keys[pygame.K_LEFT]} RIGHT={keys[pygame.K_RIGHT]}")
        
        # Debug WASD keys when in manual escort control (only if debug_key_messages is True)
        if debug_key_messages and manual_escort_control and (keys[pygame.K_w] or keys[pygame.K_a] or keys[pygame.K_s] or keys[pygame.K_d]):
            print(f"Active WASD keys: W={keys[pygame.K_w]} A={keys[pygame.K_a]} S={keys[pygame.K_s]} D={keys[pygame.K_d]}")
            
        linear_vel = 0
        angular_vel = 0
        
        if keys[pygame.K_UP]:
            linear_vel = LEADER_LINEAR_VEL
        if keys[pygame.K_DOWN]:
            linear_vel = -LEADER_LINEAR_VEL
        # Reversed controls for left and right
        if keys[pygame.K_RIGHT]:
            angular_vel = LEADER_ANGULAR_VEL
        if keys[pygame.K_LEFT]:
            angular_vel = -LEADER_ANGULAR_VEL
            
        model.set_controls(linear_vel, angular_vel)
        
        # Handle manual escort controls using WASD if in manual mode
        if follower and manual_escort_control:
            escort_linear_vel = 0
            escort_angular_vel = 0
            
            if keys[pygame.K_w]:
                escort_linear_vel = FOLLOWER_LINEAR_VEL_MAX
            if keys[pygame.K_s]:
                escort_linear_vel = -FOLLOWER_LINEAR_VEL_MAX
            if keys[pygame.K_d]:
                escort_angular_vel = FOLLOWER_ANGULAR_VEL_MAX
            if keys[pygame.K_a]:
                escort_angular_vel = FOLLOWER_ANGULAR_VEL_MIN
                
            follower.set_manual_controls(escort_linear_vel, escort_angular_vel)
        
        # Calculate elapsed time in seconds
        elapsed_time = (pygame.time.get_ticks() - start_time_ms) / 1000.0
        
        # Check for Q and E keys for camera rotation
        camera_rotation_input = 0
        if keys[pygame.K_q]:
            camera_rotation_input = -1  # Counter-clockwise
        elif keys[pygame.K_e]:
            camera_rotation_input = 1   # Clockwise
            
        # Update secondary camera with continuous rotation if follower exists
        if follower:
            # Calculate time delta for physics-based camera rotation (in seconds)
            dt = clock.get_time() / 1000.0
            
            # First check if auto-tracking is enabled, which overrides manual control
            if follower.camera_auto_track:
                # Update camera position using PID controller to track visitor
                follower.update_camera_auto_tracking(model.noisy_position, dt)
            else:
                # When auto-tracking is off, always use manual control (Q/E keys)
                # No auto-search in manual mode - only use Q/E keys to control the camera
                follower.update_secondary_camera(camera_rotation_input, dt)
        
        # Determine if visitor is visible to the escort agent
        is_visitor_visible = False
        is_visitor_visible_secondary = False
        if follower:
            # Create a proper leader object with both state and noisy_position
            leader = type('Leader', (), {
                'state': model.state,  # True state (red circle)
                'noisy_position': model.noisy_position  # Noisy measurement (what would be detected)
            })
            
            # Check if the noisy measurement is visible with the primary camera (fixed forward)
            is_visitor_visible = is_agent_in_vision_cone(
                observer=follower,
                target=leader,
                vision_range=DEFAULT_VISION_RANGE,
                vision_angle=VISION_ANGLE,
                walls=environment.get_all_walls(),
                doors=environment.get_doors()
            )
            
            # Update and check the secondary camera's vision cone
            is_visitor_visible_secondary = follower.update_secondary_vision(
                leader,
                environment.get_all_walls(),
                environment.get_doors()
            )
            
            # Combine visibility results - visitor is visible if seen by either camera
            is_visitor_visible_combined = is_visitor_visible or is_visitor_visible_secondary
        
        # Update model with elapsed time - no Kalman filter here anymore
        if not paused:
            model.update(elapsed_time=elapsed_time, 
                         walls=environment.get_all_walls(),
                         doors=environment.get_doors(),
                         is_visible=is_visitor_visible_combined)
            
            # Update follower agent with Kalman filter
            if follower:
                # Get the noisy measurement from the visitor
                noisy_measurement = model.noisy_position  # [x, y, theta]
                
                # Update the Kalman filter in the escort agent - using combined visibility
                # This allows the Kalman filter to be updated if visitor is seen by either camera
                follower.update_kalman_filter(noisy_measurement, elapsed_time, is_visitor_visible_combined)
                
                # Determine which state to use for MPPI control
                if is_visitor_visible_combined:
                    # When visible by either camera, directly track the noisy measurement
                    tracking_state = np.array([
                        noisy_measurement[0],  # x from noisy measurement
                        noisy_measurement[1],  # y from noisy measurement
                        noisy_measurement[2],  # theta from noisy measurement
                        0.0                   # Default velocity
                    ])
                    
                    # If visitor was previously lost and is now found again, update the info panel
                    if follower.search_timer >= follower.search_duration or not follower.kalman_filter_active:
                        # This condition now handles:
                        # 1. When search timer exceeded duration (standard case)
                        # 2. When Kalman filter is inactive but we just regained vision (secondary camera case)
                        follower.kalman_filter_active = True
                        
                        # If the Kalman filter was reset, we need to reinitialize it with the new measurement
                        if not follower.kalman_filter_active or follower.kalman_filter is None:
                            # Reset with current measurement
                            follower.kalman_filter = UnicycleKalmanFilter(
                                np.array([
                                    noisy_measurement[0],
                                    noisy_measurement[1], 
                                    noisy_measurement[2],
                                    0.0  # Reset velocity
                                ]), dt=0.1)
                            follower.kalman_filter_active = True
                            print("Kalman filter reinitialized from secondary camera visibility")
                            
                        # Restart MPPI when visitor is found by either camera
                        # This is essential to ensure tracking resumes properly after a period without measurements
                        print("Visitor detected by camera - reactivating MPPI controller.")
                        follower.search_timer = 0  # Reset search timer
                else:
                    # If not visible, use the Kalman estimate as best guess of where visitor might be,
                    # but ONLY if the Kalman filter is still active (not reset)
                    if follower.kalman_filter_active and follower.kalman_filter is not None:
                        tracking_state = np.array([
                            follower.kalman_filter.state[0],  # x from Kalman filter
                            follower.kalman_filter.state[1],  # y from Kalman filter
                            follower.kalman_filter.state[2],  # theta from Kalman filter
                            follower.kalman_filter.state[3]   # v from Kalman filter
                        ])
                    else:
                        # If Kalman filter has been deactivated, MPPI has no position estimate to use
                        # Just use the last known position from the follower's memory
                        if follower.last_seen_position is not None:
                            # Use last seen position with default orientation
                            tracking_state = np.array([
                                follower.last_seen_position[0],  # x from last seen position
                                follower.last_seen_position[1],  # y from last seen position
                                follower.state[2],              # Use escort's current orientation
                                0.0                             # Zero velocity (unknown)
                            ])
                        else:
                            # If no history at all, just use the escort's current position (shouldn't happen)
                            tracking_state = follower.state.copy()
                    
                    # Check if search duration has just expired or continues to be expired
                    # This ensures the Kalman filter is reset and stays reset during extended periods without sight
                    if follower.search_timer >= follower.search_duration:
                        # Only print message when first crossing the threshold
                        if follower.search_timer == follower.search_duration:
                            print("Search duration expired. Kalman filter reset and deactivated.")
                            # Reset the controller when search duration expires
                            if hasattr(follower, 'controller'):
                                follower.controller.reset()
                                print(f"{follower.controller_type.upper()} controller reset due to search duration expiration.")
                        
                        # Reset and deactivate the Kalman filter when search duration expires or continues to be expired
                        follower.reset_kalman_filter()
                
                # Update the follower agent with the appropriate tracking state
                follower.update(dt=0.1, leader_state=tracking_state,
                              walls=environment.get_all_walls(),
                              doors=environment.get_doors())
        
        # Get current time to update MPPI stats at regular intervals
        current_time = elapsed_time
        
        # Update MPPI performance statistics
        mppi_stats = None
        if follower and hasattr(follower, 'controller'):
            if current_time - last_mppi_update_time >= mppi_update_interval:
                # Check controller type and access stats accordingly
                if follower.controller_type == "mppi":
                    mppi_stats = follower.controller.get_computation_stats()
                last_mppi_update_time = current_time
        
        # Draw environment first (instead of just filling with black)
        environment.draw(screen, font)
        
        # Draw map graph if enabled
        if show_map_graph:
            # Draw the map graph with a message indicating the paused state
            map_graph.draw(screen, show_nodes=True)
            
            # Display "PAUSED" message
            pause_font = pygame.font.SysFont('Arial', 36)
            pause_text = pause_font.render("SIMULATION PAUSED - Press G to Resume", True, (255, 255, 255))
            pause_bg = pygame.Surface((pause_text.get_width() + 20, pause_text.get_height() + 10))
            pause_bg.fill((0, 0, 0))
            pause_bg.set_alpha(180)
            
            # Position at top center of screen
            pause_x = (WIDTH - pause_text.get_width()) // 2
            pause_y = 50
            
            # Draw background and text
            screen.blit(pause_bg, (pause_x - 10, pause_y - 5))
            screen.blit(pause_text, (pause_x, pause_y))
            
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
        
        # Draw standard representations for monitoring and predictions
        # (Keep the original drawing for predictions, uncertainty ellipses, etc.)
        model.draw(screen)
        
        # Generate particles for visitor agent
        if abs(model.state[3]) > 1.0:  # Only generate particles when moving
            visitor_particles.extend(generate_particles(model))
        
        # Update and draw visitor particles
        visitor_particles = update_particles(visitor_particles)
        draw_particles(screen, visitor_particles, RED)
        
        # Draw enhanced human-like visitor agent
        draw_enhanced_agent(screen, model, RED, debug_mode)
        
        # Draw follower agent if enabled
        if follower:
            # Draw standard representation (for predictions)
            follower.draw(screen)
            
            # Generate particles for escort agent
            if abs(follower.state[3]) > 1.0:  # Only generate particles when moving
                escort_particles.extend(generate_particles(follower))
            
            # Update and draw escort particles
            escort_particles = update_particles(escort_particles)
            draw_particles(screen, escort_particles, ORANGE)
            
            # Draw enhanced human-like escort agent
            draw_enhanced_agent(screen, follower, ORANGE, debug_mode)
            
            # Draw line from leader to follower
            leader_pos = (int(model.state[0]), int(model.state[1]))
            follower_pos = (int(follower.state[0]), int(follower.state[1]))
            pygame.draw.line(screen, (50, 50, 50), leader_pos, follower_pos, 1)
        
        # Calculate FPS
        frame_end_time = pygame.time.get_ticks()
        frame_time = frame_end_time - frame_start_time
        frame_times.append(frame_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_frame_time = sum(frame_times) / len(frame_times)
        fps = int(1000 / max(1, avg_frame_time))
        
        # Prepare info text for overlay
        info_text = [
            f"Controls: Arrow keys to move, ESC to quit",
            f"Visitor: ({int(model.state[0])}, {int(model.state[1])}), Heading: {model.state[2]:.2f}",
            f"F: Toggle FPS | K: Kalman viz | U: Uncertainty | M: MPPI viz",
            f"T: Toggle escort | +/-: Adjust escort distance | R: Reset escort",
            f"C: Toggle manual escort control (WASD to control when manual)",
            f"D: Toggle debug view | V: Toggle enhanced visuals",
            f"Computing device: {DEVICE_INFO}"  # Add device info to display
        ]
        
        if follower:
            info_text.append(f"Escort: ({int(follower.state[0])}, {int(follower.state[1])}), Target dist: {follower.target_distance:.1f}")
            info_text.append(f"Escort mode: {'MANUAL (WASD)' if manual_escort_control else 'AUTO-TRACKING'}")
            # Add camera rotation info
            camera_mode = "AUTO-TRACKING" if follower.camera_auto_track else "MANUAL (Q/E)"
            info_text.append(f"Camera: {camera_mode} | Press A to toggle auto-tracking")
            info_text.append(f"Camera angular vel: {follower.secondary_camera_angular_vel:.2f}")
            # Add measurement interval info if it exists
            if hasattr(follower, 'measurement_interval'):
                info_text.append(f"Shift+/- or Shift++: Change measurement rate ({follower.measurement_interval:.1f}s)")
            
            # Display MPPI performance statistics if available
            if mppi_stats:
                info_text.append(f"MPPI compute: {mppi_stats['last_time']:.1f}ms (avg: {mppi_stats['avg_time']:.1f}ms)")
                # Safely check for MPPI controller and computation cache
                if follower.controller_type == "mppi" and hasattr(follower.controller, 'computation_cache'):
                    cache_hits = sum(1 for entry in follower.controller.computation_cache if entry is not None)
                    info_text.append(f"Cache: {cache_hits}/{MPPI_CACHE_SIZE} | Batch size: {follower.controller.batch_size}")
                
        # Add entropy information if Kalman filter is active
        if follower and follower.kalman_filter_active:
            info_text.append(f"KF Entropy: {follower.current_entropy:.2f} (uncertainty measure)")
        
        if show_fps:
            info_text.append(f"FPS: {fps} (Avg frame time: {avg_frame_time:.1f}ms)")
        
        # Update the information overlay with current data
        overlay.update_data(
            info_text=info_text,
            title_text="Visitor with Escort Simulation",
            mppi_stats=mppi_stats,
            key_debug=key_debug,
            keys=keys,
            show_fps=show_fps,
            fps=fps,
            avg_frame_time=avg_frame_time
        )
        
        # Get and draw the overlay surface
        overlay_surface = overlay.get_surface()
        if overlay_surface:
            # Position the sidebar on the right side of the screen in its dedicated space
            screen.blit(overlay_surface, (ENVIRONMENT_WIDTH, 0))
        
        # Update display
        pygame.display.flip()
        
        # Control framerate
        clock.tick(60)
    
    # Cleanup - stop the overlay thread before quitting
    overlay.stop()
    
    pygame.quit()
    sys.exit()

def render_loading_screen(screen, message, progress=0.0):
    """
    Render a loading screen with a progress bar.
    
    Args:
        screen: Pygame surface to draw on
        message: Message to display
        progress: Progress value between 0.0 and 1.0
    """
    # Clear screen with a dark background
    screen.fill((30, 30, 40))
    
    # Set up fonts
    title_font = pygame.font.SysFont('Arial', 32)
    message_font = pygame.font.SysFont('Arial', 24)
    
    # Calculate center position of the environment area (not including sidebar)
    env_center_x = ENVIRONMENT_WIDTH // 2
    
    # Draw title
    title_text = "MultiTrack Simulation"
    title_surf = title_font.render(title_text, True, (200, 200, 255))
    screen.blit(title_surf, (env_center_x - title_surf.get_width()//2, HEIGHT//4))
    
    # Draw message
    message_surf = message_font.render(message, True, (255, 255, 255))
    screen.blit(message_surf, (env_center_x - message_surf.get_width()//2, HEIGHT//2 - 50))
    
    # Draw progress bar border - centered in environment area
    bar_width = ENVIRONMENT_WIDTH // 2
    bar_height = 20
    bar_x = ENVIRONMENT_WIDTH // 4
    bar_y = HEIGHT // 2
    pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height), 2)
    
    # Draw progress bar fill
    fill_width = int(bar_width * progress)
    pygame.draw.rect(screen, (100, 200, 100), (bar_x, bar_y, fill_width, bar_height))
    
    # Draw percentage text
    percent_text = f"{int(progress * 100)}%"
    percent_font = pygame.font.SysFont('Arial', 18)
    percent_surf = percent_font.render(percent_text, True, (255, 255, 255))
    screen.blit(percent_surf, (env_center_x - percent_surf.get_width()//2, bar_y + bar_height + 10))
    
    # Update display
    pygame.display.flip()

if __name__ == "__main__":
    run_simulation()
