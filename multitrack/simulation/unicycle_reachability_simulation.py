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
from multitrack.utils.config import *
from multitrack.utils.vision import is_agent_in_vision_cone

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1280, 720  # Increased from 800x600 to 1280x720
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
def run_simulation():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Visitor and Escort Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)
    
    # Monitoring options
    global SHOW_PREDICTIONS, SHOW_UNCERTAINTY, SHOW_MPPI_PREDICTIONS, FOLLOWER_ENABLED
    
    # Create environment
    environment = SimulationEnvironment(width=WIDTH, height=HEIGHT)
    
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
        
        # Determine if visitor is visible to the escort agent
        is_visitor_visible = False
        if follower:
            # First, create a proper leader object with both state and noisy_position
            leader = type('Leader', (), {
                'state': model.state,  # True state (red circle)
                'noisy_position': model.noisy_position  # Noisy measurement (what would be detected)
            })
            
            # Check if the noisy measurement (not the Kalman estimate) is in the vision cone
            is_visitor_visible = is_agent_in_vision_cone(
                observer=follower,
                target=leader,
                vision_range=DEFAULT_VISION_RANGE,
                vision_angle=VISION_ANGLE,
                walls=environment.get_all_walls(),
                doors=environment.get_doors()
            )
        
        # Update model with elapsed time - no Kalman filter here anymore
        model.update(elapsed_time=elapsed_time, 
                     walls=environment.get_all_walls(),
                     doors=environment.get_doors(),
                     is_visible=is_visitor_visible)
        
        # Update follower agent with Kalman filter
        if follower:
            # Get the noisy measurement from the visitor
            noisy_measurement = model.noisy_position  # [x, y, theta]
            
            # Update the Kalman filter in the escort agent
            follower.update_kalman_filter(noisy_measurement, elapsed_time, is_visitor_visible)
            
            # Determine which state to use for MPPI control
            if is_visitor_visible:
                # When visible, directly track the noisy measurement
                tracking_state = np.array([
                    noisy_measurement[0],  # x from noisy measurement
                    noisy_measurement[1],  # y from noisy measurement
                    noisy_measurement[2],  # theta from noisy measurement
                    0.0                   # Default velocity
                ])
                
                # If visitor was previously lost and is now found again, update the info panel
                if follower.search_timer >= follower.search_duration:
                    info_text.append("Visitor found! Kalman filter reactivated with new measurement.")
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
        if follower and current_time - last_mppi_update_time >= mppi_update_interval:
            mppi_stats = follower.mppi.get_computation_stats()
            last_mppi_update_time = current_time
        
        # Draw environment first (instead of just filling with black)
        environment.draw(screen, font)
        
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
        
        # Display info
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
            # Add measurement interval info if it exists
            if hasattr(follower, 'measurement_interval'):
                info_text.append(f"Shift+/- or Shift++: Change measurement rate ({follower.measurement_interval:.1f}s)")
            
            # Display MPPI performance statistics if available
            if mppi_stats:
                info_text.append(f"MPPI compute: {mppi_stats['last_time']:.1f}ms (avg: {mppi_stats['avg_time']:.1f}ms)")
                cache_hits = sum(1 for entry in follower.mppi.computation_cache if entry is not None)
                info_text.append(f"Cache: {cache_hits}/{MPPI_CACHE_SIZE} | Batch size: {follower.mppi.batch_size}")
                
        # Add entropy information if Kalman filter is active
        if follower and follower.kalman_filter_active:
            info_text.append(f"KF Entropy: {follower.current_entropy:.2f} (uncertainty measure)")
        
        if show_fps:
            info_text.append(f"FPS: {fps} (Avg frame time: {avg_frame_time:.1f}ms)")
        
        # Calculate text panel height to ensure all text is visible
        text_panel_height = len(info_text) * 20 + 10  # 20px per line + 10px padding
        
        # Draw text background for better readability (with higher transparency)
        info_bg = pygame.Surface((WIDTH, text_panel_height))
        info_bg.fill(BLACK)
        info_bg.set_alpha(120)  # Make much more transparent (lower alpha value)
        screen.blit(info_bg, (0, HEIGHT - text_panel_height))
        
        # Display text with adjusted starting position to ensure all lines are visible
        for i, text in enumerate(info_text):
            text_surf = font.render(text, True, WHITE)
            screen.blit(text_surf, (10, HEIGHT - text_panel_height + 5 + i*20))
        
        # Draw title
        title_text = "Visitor with Escort Simulation"
        title = font.render(title_text, True, WHITE)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 10))
        
        # Display key state monitoring if enabled
        if key_debug:
            key_monitor_text = [
                "KEY STATE MONITOR:",
                f"Arrow keys: UP={keys[pygame.K_UP]} DOWN={keys[pygame.K_DOWN]} LEFT={keys[pygame.K_LEFT]} RIGHT={keys[pygame.K_RIGHT]}",
                f"WASD keys: W={keys[pygame.K_w]} A={keys[pygame.K_a]} S={keys[pygame.K_s]} D={keys[pygame.K_d]}",
                f"Function keys: K={keys[pygame.K_k]} U={keys[pygame.K_u]} T={keys[pygame.K_t]} M={keys[pygame.K_m]} C={keys[pygame.K_c]} V={keys[pygame.K_v]} F={keys[pygame.K_f]}",
                f"Special keys: PLUS={keys[pygame.K_PLUS]} EQUALS={keys[pygame.K_EQUALS]} MINUS={keys[pygame.K_MINUS]} SHIFT={bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)}"
            ]
            
            # Draw background for key monitor
            key_bg = pygame.Surface((400, 120))
            key_bg.fill((50, 50, 50))
            key_bg.set_alpha(200)
            screen.blit(key_bg, (20, 20))
            
            # Display key state text
            for i, text in enumerate(key_monitor_text):
                key_text = font.render(text, True, (255, 255, 255))
                screen.blit(key_text, (30, 30 + i*20))
        
        # Update display
        pygame.display.flip()
        
        # Control framerate
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_simulation()
