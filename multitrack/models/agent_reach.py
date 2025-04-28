"""
Simplified environment using pygame to show the probability distribution and segmented entropy 
for a unicycle model, with 30 equal segments.
"""
import pygame
import numpy as np
import sys
from math import sin, cos, pi, log, atan2

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)

class UnicycleModel:
    def __init__(self):
        # State: [x, y, theta, v]
        self.state = np.array([WIDTH//2, HEIGHT//2, 0.0, 0.0])
        # Control inputs: [v, omega] (linear and angular velocity)
        self.controls = np.array([0.0, 0.0])
        # Process noise parameters
        self.noise_pos = 2.0
        self.noise_angle = 0.1
        
        # Reachability set representation (minimal)
        self.num_trajectories = 200
        self.trajectory_length = 20
        self.prediction_dt = 0.1
        
        # Pre-compute grid - use a small resolution
        self.grid_size = 20  # 20x20 grid
        self.grid_cell_width = WIDTH / self.grid_size
        self.grid_cell_height = HEIGHT / self.grid_size
        
        # Store the reachability grid for probability
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        # For segmented entropy, divide trajectories into 30 equal segments
        self.num_segments = 30
        # Store particles for each segment
        self.segment_particles = [[] for _ in range(self.num_segments)]
        # Store entropy for each segment
        self.segment_entropy = np.zeros(self.num_segments)
        # Track if segment has particles
        self.segment_has_particles = np.zeros(self.num_segments, dtype=bool)
        
        # Store particles for visualization
        self.particles = []
        self.initialize_particles()
    
    def get_segment_index(self, angle):
        """Convert an angle to a segment index from 0-29"""
        # Normalize angle to [0, 2π]
        normalized_angle = angle % (2 * pi)
        # Map to segment index
        segment_size = 2 * pi / self.num_segments
        segment_idx = int(normalized_angle / segment_size)
        # Ensure it's within range
        return min(self.num_segments - 1, max(0, segment_idx))
    
    def initialize_particles(self):
        """Initialize particles for reachability set visualization"""
        self.particles = []
        # Clear segment particles
        for i in range(self.num_segments):
            self.segment_particles[i] = []
            self.segment_has_particles[i] = False
        
        for _ in range(self.num_trajectories):
            # Each particle is a list of positions (x, y) forming a trajectory
            particle = []
            
            # Start at current position with small noise
            x = self.state[0] + np.random.normal(0, 3)
            y = self.state[1] + np.random.normal(0, 3)
            theta = self.state[2] + np.random.normal(0, 0.1)
            v = max(0, self.state[3] + np.random.normal(0, 5))
            
            particle.append((x, y))
            
            # Generate the rest of the trajectory based on unicycle dynamics
            for _ in range(self.trajectory_length):
                # Add control variation
                v_ctrl = v + np.random.normal(0, 10)
                omega = self.controls[1] + np.random.normal(0, 0.2)
                
                # Update position with unicycle dynamics
                x += v_ctrl * cos(theta) * self.prediction_dt + np.random.normal(0, self.noise_pos)
                y += v_ctrl * sin(theta) * self.prediction_dt + np.random.normal(0, self.noise_pos)
                theta += omega * self.prediction_dt + np.random.normal(0, self.noise_angle)
                
                # Normalize angle
                theta = (theta + pi) % (2 * pi) - pi
                
                # Apply boundaries
                x = max(0, min(WIDTH, x))
                y = max(0, min(HEIGHT, y))
                
                # Add to trajectory
                particle.append((x, y))
            
            self.particles.append(particle)
            
            # Calculate steering angle from final point to starting point
            if len(particle) > 1:
                start_x, start_y = particle[0]
                final_x, final_y = particle[-1]
                
                # Angle between start and final position
                dx = final_x - start_x
                dy = final_y - start_y
                steering_angle = atan2(dy, dx)
                
                # Get segment index
                segment_idx = self.get_segment_index(steering_angle)
                
                # Add to segment particles
                self.segment_particles[segment_idx].append(particle)
                self.segment_has_particles[segment_idx] = True
        
        # Update the grid
        self.update_grid()
    
    def update_grid(self):
        """Update the probability grid and segment entropy"""
        # Clear grid
        self.grid.fill(0)
        
        # Count particles in each grid cell
        for particle in self.particles:
            for x, y in particle:
                # Get grid cell indices
                grid_x = min(self.grid_size - 1, int(x / self.grid_cell_width))
                grid_y = min(self.grid_size - 1, int(y / self.grid_cell_height))
                self.grid[grid_x, grid_y] += 1
        
        # Normalize to get probability
        total = np.sum(self.grid)
        if total > 0:
            self.grid = self.grid / total
        
        # Calculate entropy for each segment
        for segment_idx in range(self.num_segments):
            # Skip if no particles in this segment
            if not self.segment_has_particles[segment_idx]:
                self.segment_entropy[segment_idx] = 0
                continue
            
            # Create a grid for this segment
            segment_grid = np.zeros((self.grid_size, self.grid_size))
            
            # Count particles in each grid cell for this segment
            for particle in self.segment_particles[segment_idx]:
                for x, y in particle:
                    # Get grid cell indices
                    grid_x = min(self.grid_size - 1, int(x / self.grid_cell_width))
                    grid_y = min(self.grid_size - 1, int(y / self.grid_cell_height))
                    segment_grid[grid_x, grid_y] += 1
            
            # Normalize to get probability for this segment
            segment_total = np.sum(segment_grid)
            if segment_total > 0:
                segment_grid = segment_grid / segment_total
                
                # Calculate entropy for this segment
                epsilon = 1e-10  # Small constant to avoid log(0)
                segment_entropy = 0
                
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        p = segment_grid[i, j]
                        if p > epsilon:
                            segment_entropy -= p * log(p, 2)
                
                self.segment_entropy[segment_idx] = segment_entropy
            else:
                self.segment_entropy[segment_idx] = 0
    
    def update(self, dt=0.1):
        v, omega = self.controls
        theta = self.state[2]
        
        # Unicycle model dynamics
        self.state[0] += v * cos(theta) * dt
        self.state[1] += v * sin(theta) * dt
        self.state[2] += omega * dt
        self.state[3] = v  # Update velocity state
        
        # Normalize angle to [-pi, pi]
        self.state[2] = (self.state[2] + pi) % (2 * pi) - pi
        
        # Boundary conditions
        self.state[0] = np.clip(self.state[0], 0, WIDTH)
        self.state[1] = np.clip(self.state[1], 0, HEIGHT)
        
        # Re-initialize particles if controls have changed significantly
        if abs(v) > 5 or abs(omega) > 0.1:
            self.initialize_particles()
    
    def set_controls(self, linear_vel, angular_vel):
        old_v, old_omega = self.controls
        self.controls = np.array([linear_vel, angular_vel])
        
        # If controls changed significantly, update particles
        if abs(old_v - linear_vel) > 10 or abs(old_omega - angular_vel) > 0.2:
            self.initialize_particles()
    
    def draw(self, screen, show_entropy=False):
        if not show_entropy:
            # Regular probability distribution visualization
            # Get maximum value for normalization
            max_val = np.max(self.grid) if np.max(self.grid) > 0 else 1.0
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Skip cells with very low values
                    if self.grid[i, j] < 0.0001:
                        continue
                    
                    # Calculate color based on value
                    intensity = int(255 * self.grid[i, j] / max_val)
                    color = (0, intensity, intensity)  # Cyan for probability
                    
                    # Calculate rectangle position and size
                    rect_x = i * self.grid_cell_width
                    rect_y = j * self.grid_cell_height
                    
                    # Draw rectangle
                    pygame.draw.rect(screen, color, 
                                    (rect_x, rect_y, self.grid_cell_width, self.grid_cell_height))
            
            # Draw all particles with alpha
            for particle in self.particles:
                # Draw trajectory with fading alpha
                for i in range(1, len(particle)):
                    # Fade color based on position in trajectory
                    alpha = 50 + int(150 * i / len(particle))
                    pygame.draw.line(screen, (0, 0, alpha), 
                                    particle[i-1], particle[i], 1)
        else:
            # Segmented entropy visualization - only show non-zero segments
            # Find maximum entropy for normalization
            max_entropy = max(self.segment_entropy) if max(self.segment_entropy) > 0 else 1.0
            
            # Agent's current position
            center_x, center_y = self.state[0], self.state[1]
            
            # Draw segments with equal angular distribution (30 segments)
            segment_radius = 200  # Radius of segments
            inner_radius = 30    # Inner radius (near agent)
            
            # Define segment angular boundaries (30 equal segments)
            segment_size = 2 * pi / self.num_segments
            
            # Draw each segment with non-zero probability
            for i in range(self.num_segments):
                # Skip segments with no particles
                if not self.segment_has_particles[i]:
                    continue
                    
                # Calculate segment angles
                angle_start = i * segment_size
                angle_end = (i + 1) * segment_size
                
                # Get entropy value for this segment
                entropy_val = self.segment_entropy[i]
                
                # Skip segments with zero entropy
                if entropy_val < 0.0001:
                    continue
                
                # Normalize and calculate color intensity
                intensity = int(255 * entropy_val / max_entropy) if max_entropy > 0 else 0
                color = (intensity, 0, intensity)  # Purple for entropy
                
                # Draw segment as a polygon
                points = []
                # Add center point
                points.append((center_x, center_y))
                
                # Add inner arc points
                for angle_step in range(6):  # Use 6 points for smooth arc
                    angle = angle_start + angle_step * (angle_end - angle_start) / 5
                    x = center_x + inner_radius * cos(angle)
                    y = center_y + inner_radius * sin(angle)
                    points.append((x, y))
                
                # Add outer arc points (in reverse)
                for angle_step in range(6):
                    angle = angle_end - angle_step * (angle_end - angle_start) / 5
                    x = center_x + segment_radius * cos(angle)
                    y = center_y + segment_radius * sin(angle)
                    points.append((x, y))
                
                # Draw segment
                pygame.draw.polygon(screen, color, points)
                
                # Draw segment outline
                pygame.draw.polygon(screen, (128, 128, 128), points, 1)
                
                # Draw particles in this segment
                for particle in self.segment_particles[i]:
                    # Draw trajectory with fading alpha
                    for j in range(1, len(particle)):
                        # Fade color based on position in trajectory
                        alpha = 50 + int(150 * j / len(particle))
                        color = (50, 0, alpha)  # Purple-blue for entropy trajectories
                        pygame.draw.line(screen, color, 
                                        particle[j-1], particle[j], 1)
                
                # Add entropy text for each segment
                mid_angle = (angle_start + angle_end) / 2
                text_x = center_x + (segment_radius * 0.7) * cos(mid_angle)
                text_y = center_y + (segment_radius * 0.7) * sin(mid_angle)
                
                # Create text surface
                font = pygame.font.SysFont('Arial', 12)
                text = font.render(f"{entropy_val:.2f}", True, WHITE)
                text_rect = text.get_rect(center=(text_x, text_y))
                screen.blit(text, text_rect)
        
        # Draw unicycle agent
        x, y, theta, _ = self.state
        radius = 10
        pygame.draw.circle(screen, RED, (int(x), int(y)), radius)
        end_x = x + radius * cos(theta)
        end_y = y + radius * sin(theta)
        pygame.draw.line(screen, WHITE, (x, y), (end_x, end_y), 2)

# Main simulation loop
def run_simulation():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Unicycle Model - Reachability Distribution")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)
    
    model = UnicycleModel()
    
    # Display options
    show_fps = True
    show_entropy = False  # Toggle between probability and entropy
    
    # Performance monitoring
    frame_times = []
    
    running = True
    while running:
        start_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    show_fps = not show_fps
                elif event.key == pygame.K_r:
                    # Reset particles
                    model.initialize_particles()
                elif event.key == pygame.K_e:
                    # Toggle between probability and entropy display
                    show_entropy = not show_entropy
        
        # Get keyboard input to control the unicycle
        keys = pygame.key.get_pressed()
        linear_vel = 0
        angular_vel = 0
        
        if keys[pygame.K_UP]:
            linear_vel = 50
        if keys[pygame.K_DOWN]:
            linear_vel = -50
        # Reversed controls for left and right
        if keys[pygame.K_RIGHT]:
            angular_vel = 1.0
        if keys[pygame.K_LEFT]:
            angular_vel = -1.0
            
        model.set_controls(linear_vel, angular_vel)
        model.update()
        
        # Clear screen
        screen.fill(BLACK)
        
        # Draw model with current display mode
        model.draw(screen, show_entropy)
        
        # Calculate FPS
        end_time = pygame.time.get_ticks()
        frame_time = end_time - start_time
        frame_times.append(frame_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_frame_time = sum(frame_times) / len(frame_times)
        fps = int(1000 / max(1, avg_frame_time))
        
        # Display info
        info_text = [
            f"Controls: Arrow keys to move, ESC to quit",
            f"Position: ({int(model.state[0])}, {int(model.state[1])}), Heading: {model.state[2]:.2f}",
            f"F: Toggle FPS | R: Reset distribution | E: Toggle entropy",
            f"Currently showing: {'Segmented Entropy' if show_entropy else 'Probability'} distribution"
        ]
        
        if show_fps:
            info_text.append(f"FPS: {fps} (Avg frame time: {avg_frame_time:.1f}ms)")
        
        for i, text in enumerate(info_text):
            text_surf = font.render(text, True, WHITE)
            screen.blit(text_surf, (10, HEIGHT - 110 + i*20))
        
        # Draw title
        title_text = "Unicycle Reachability Set - " + ("Segmented Entropy" if show_entropy else "Probability") + " Distribution"
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
