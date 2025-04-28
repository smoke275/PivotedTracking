"""
Enhanced rendering module for drawing more human-like agents.
Adapted from the original rendering.py file.
"""

import math
import random
import pygame
from multitrack.utils.config import *

def draw_enhanced_agent(surface, agent, color, debug_mode=False):
    """
    Draw an agent with human-like appearance (body, head, eyes, arms).
    
    Args:
        surface: Pygame surface to draw on
        agent: Agent instance (visitor or escort)
        color: RGB color tuple for the agent
        debug_mode: Whether to draw debug information
    """
    # Get agent position and orientation
    center_x, center_y = int(agent.state[0]), int(agent.state[1])
    orientation = agent.state[2]
    
    # Calculate sizes based on agent_size
    head_radius = AGENT_SIZE // 2
    body_radius = AGENT_SIZE // 1.3
    
    # Calculate direction vector for orientation
    forward_x = math.cos(orientation)
    forward_y = math.sin(orientation)
    
    # Draw body (circle)
    pygame.draw.circle(surface, color, (center_x, center_y), body_radius)
    
    # Draw head (smaller circle) in front of the body
    head_x = center_x + forward_x * body_radius * 0.8
    head_y = center_y + forward_y * body_radius * 0.8
    pygame.draw.circle(surface, color, (int(head_x), int(head_y)), head_radius)
    
    # Draw eyes (to show direction clearly) - from top down they appear as dots
    eye_radius = head_radius // 3
    right_x = math.cos(orientation + math.pi/2)
    right_y = math.sin(orientation + math.pi/2)
    
    left_eye_x = head_x + right_x * head_radius * 0.4
    left_eye_y = head_y + right_y * head_radius * 0.4
    right_eye_x = head_x - right_x * head_radius * 0.4
    right_eye_y = head_y - right_y * head_radius * 0.4
    
    pygame.draw.circle(surface, BLACK, (int(left_eye_x), int(left_eye_y)), eye_radius)
    pygame.draw.circle(surface, BLACK, (int(right_eye_x), int(right_eye_y)), eye_radius)
    
    # Animation - add a slight bobbing effect for movement
    velocity = agent.state[3]  # Get velocity from agent state
    if abs(velocity) > 0.2:
        anim_factor = min(abs(velocity) / MAX_VELOCITY, 1.0)
        bob_amplitude = anim_factor * 2
        bob_speed = pygame.time.get_ticks() * 0.02
        
        # Draw "arms" (small circles) that move slightly as the agent moves
        for i in range(2):
            angle = orientation + math.pi/2 * (1 if i == 0 else -1)
            arm_x = center_x + math.cos(angle) * body_radius * 0.9
            arm_y = center_y + math.sin(angle) * body_radius * 0.9
            
            # Add bobbing effect
            bob_offset = math.sin(bob_speed + i * math.pi) * bob_amplitude
            arm_x += math.cos(orientation + math.pi/2) * bob_offset
            arm_y += math.sin(orientation + math.pi/2) * bob_offset
            
            pygame.draw.circle(surface, color, (int(arm_x), int(arm_y)), head_radius // 1.5)
    
    # Visualize turning radius if debugging
    if debug_mode and abs(agent.controls[1]) > 0.01:  # controls[1] is the angular velocity (steering)
        # Calculate turning circle center
        steering = agent.controls[1]
        radius = TURNING_RADIUS / max(0.01, math.sin(abs(steering)))  # Avoid division by zero
        turn_direction = 1 if steering > 0 else -1
        circle_center_x = center_x - (radius * math.sin(orientation) * turn_direction)
        circle_center_y = center_y + (radius * math.cos(orientation) * turn_direction)
        
        # Draw turning circle (partially transparent)
        circle_surface = pygame.Surface((int(radius*2), int(radius*2)), pygame.SRCALPHA)
        pygame.draw.circle(circle_surface, (*color, 30), (int(radius), int(radius)), int(radius), 1)
        surface.blit(circle_surface, (int(circle_center_x-radius), int(circle_center_y-radius)))

def generate_particles(agent, num_particles=5):
    """
    Generate particle effects behind an agent when it's moving.
    
    Args:
        agent: Agent instance (visitor or escort)
        num_particles: Number of particles to generate
        
    Returns:
        List of particle dictionaries
    """
    particles = []
    
    # Only generate particles if the agent is moving at a reasonable speed
    velocity = agent.state[3]
    if abs(velocity) > 1.0:
        center_x, center_y = agent.state[0], agent.state[1]
        orientation = agent.state[2]
        
        # Generate particles behind the agent
        back_x = center_x - math.cos(orientation) * AGENT_SIZE * 0.75
        back_y = center_y - math.sin(orientation) * AGENT_SIZE * 0.75
        
        for _ in range(num_particles):
            # Random offset from the back point
            offset_x = (0.5 - random.random()) * AGENT_SIZE * 0.5
            offset_y = (0.5 - random.random()) * AGENT_SIZE * 0.5
            
            # Calculate particle position
            pos_x = back_x + offset_x
            pos_y = back_y + offset_y
            
            # Create particle
            particle = {
                'pos': [pos_x, pos_y],
                'velocity': [
                    -math.cos(orientation) * velocity * PARTICLE_VELOCITY_FACTOR,
                    -math.sin(orientation) * velocity * PARTICLE_VELOCITY_FACTOR
                ],
                'timer': PARTICLE_LIFETIME
            }
            
            particles.append(particle)
    
    return particles

def update_particles(particles, dt=0.1):
    """
    Update particle positions and lifetimes.
    
    Args:
        particles: List of particle dictionaries
        dt: Time step
        
    Returns:
        Updated list of particles
    """
    updated_particles = []
    
    for particle in particles:
        # Update position based on velocity
        particle['pos'][0] += particle['velocity'][0] * dt
        particle['pos'][1] += particle['velocity'][1] * dt
        
        # Decrease timer
        particle['timer'] -= 1
        
        # Keep particle if it's still alive
        if particle['timer'] > 0:
            updated_particles.append(particle)
    
    return updated_particles

def draw_particles(surface, particles, color):
    """
    Draw particle effects.
    
    Args:
        surface: Pygame surface to draw on
        particles: List of particle dictionaries
        color: RGB color tuple for the particles
    """
    for particle in particles:
        size = particle['timer'] // 2
        pygame.draw.circle(
            surface, 
            color, 
            (int(particle['pos'][0]), int(particle['pos'][1])), 
            size
        )