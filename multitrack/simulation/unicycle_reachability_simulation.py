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
    
    model = UnicycleModel()
    follower = FollowerAgent(target_distance=100.0) if FOLLOWER_ENABLED else None
    
    # Create environment
    environment = SimulationEnvironment(width=WIDTH, height=HEIGHT)
    
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
    
    running = True
    while running:
        frame_start_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    show_fps = not show_fps
                elif event.key == pygame.K_k:
                    # Toggle Kalman filter prediction display
                    SHOW_PREDICTIONS = not SHOW_PREDICTIONS
                elif event.key == pygame.K_u:
                    # Toggle uncertainty ellipse display
                    SHOW_UNCERTAINTY = not SHOW_UNCERTAINTY
                elif event.key == pygame.K_m:
                    # Toggle MPPI predictions display
                    SHOW_MPPI_PREDICTIONS = not SHOW_MPPI_PREDICTIONS
                elif event.key == pygame.K_t:
                    # Toggle follower agent
                    FOLLOWER_ENABLED = not FOLLOWER_ENABLED
                    if FOLLOWER_ENABLED and follower is None:
                        follower = FollowerAgent(target_distance=100.0)
                    if not FOLLOWER_ENABLED:
                        follower = None
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    # For measurement rate control with Shift, follower distance without
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        # Increase measurement frequency (decrease interval)
                        new_interval = model.adjust_measurement_interval(-0.1)
                        print(f"Measurement interval decreased to {new_interval:.1f}s")
                    elif follower:
                        # Increase following distance
                        follower.target_distance = min(FOLLOWER_MAX_DISTANCE, follower.target_distance + 10)
                elif event.key == pygame.K_MINUS:
                    # For measurement rate control with Shift, follower distance without
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        # Decrease measurement frequency (increase interval)
                        new_interval = model.adjust_measurement_interval(0.1)
                        print(f"Measurement interval increased to {new_interval:.1f}s")
                    elif follower:
                        # Decrease following distance
                        follower.target_distance = max(FOLLOWER_MIN_DISTANCE, follower.target_distance - 10)
                elif event.key == pygame.K_r:
                    # Reset follower position (random)
                    if follower:
                        follower = FollowerAgent(target_distance=follower.target_distance)
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
        
        # Get keyboard input to control the unicycle
        keys = pygame.key.get_pressed()
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
        
        # Update model with elapsed time for Kalman filter timing
        model.update(elapsed_time=elapsed_time, 
                     walls=environment.get_all_walls(),
                     doors=environment.get_doors())
        
        # Update follower agent
        if follower:
            # Use Kalman filter estimate instead of actual state
            kalman_state = np.array([
                model.kalman_filter.state[0],  # x from Kalman filter
                model.kalman_filter.state[1],  # y from Kalman filter
                model.kalman_filter.state[2],  # theta from Kalman filter
                model.state[3]                 # v (keep original velocity)
            ])
            follower.update(dt=0.1, leader_state=kalman_state,
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
            f"Shift+/- or Shift++: Change measurement rate ({model.measurement_interval:.1f}s)",
            f"Computing device: {DEVICE_INFO}"  # Add device info to display
        ]
        
        if follower:
            info_text.append(f"Escort: ({int(follower.state[0])}, {int(follower.state[1])}), Target dist: {follower.target_distance:.1f}")
            info_text.append(f"Escort mode: {'MANUAL (WASD)' if manual_escort_control else 'AUTO-TRACKING'}")
            
            # Display MPPI performance statistics if available
            if mppi_stats:
                info_text.append(f"MPPI compute: {mppi_stats['last_time']:.1f}ms (avg: {mppi_stats['avg_time']:.1f}ms)")
                cache_hits = sum(1 for entry in follower.mppi.computation_cache if entry is not None)
                info_text.append(f"Cache: {cache_hits}/{MPPI_CACHE_SIZE} | Batch size: {follower.mppi.batch_size}")
                
        # Add entropy information
        info_text.append(f"KF Entropy: {model.current_entropy:.2f} (uncertainty measure)")
        
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
        
        # Update display
        pygame.display.flip()
        
        # Control framerate
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_simulation()
