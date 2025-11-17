import pygame
import numpy as np
from shapely.geometry import Polygon, Point
from collections import deque, defaultdict
import sys

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
WORLD_WIDTH = 12.0  # meters
WORLD_HEIGHT = 12.0  # meters
SCALE = min(WINDOW_WIDTH / WORLD_WIDTH, WINDOW_HEIGHT / WORLD_HEIGHT) * 0.9
OFFSET_X = 50
OFFSET_Y = 50

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (80, 80, 80)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)

# Agent parameters
AGENT_RADIUS = 0.3
v_max = 2.0
omega_max = 1.0
time_horizon = 3.0

class ProgressiveStateExpansion:
    """Progressive state expansion for reachability"""
    
    def __init__(self, v_max, omega_max, time_horizon, obstacles, 
                 grid_resolution=0.3, theta_resolution=8):
        self.v_max = v_max
        self.omega_max = omega_max
        self.T = time_horizon
        self.obstacles = obstacles
        
        self.grid_res = grid_resolution
        self.theta_bins = theta_resolution
        self.theta_res = 2 * np.pi / theta_resolution
        
        self.dt = 0.2
        self.max_steps = int(time_horizon / self.dt)
        
        self.primitives = self.generate_motion_primitives()
    
    def generate_motion_primitives(self):
        primitives = []
        
        # Forward motions
        for v in [self.v_max, self.v_max * 0.7, self.v_max * 0.4]:
            primitives.append(('forward', v, 0, self.dt))
            for omega in [self.omega_max, self.omega_max * 0.5, -self.omega_max * 0.5, -self.omega_max]:
                primitives.append(('arc', v, omega, self.dt))
        
        # Turn in place
        for omega in [self.omega_max, -self.omega_max]:
            primitives.append(('turn', 0, omega, self.dt))
        
        return primitives
    
    def discretize_state(self, x, y, theta):
        ix = int(round(x / self.grid_res))
        iy = int(round(y / self.grid_res))
        itheta = int(round(theta / self.theta_res)) % self.theta_bins
        return (ix, iy, itheta)
    
    def apply_primitive(self, state, primitive):
        x, y, theta = state
        prim_type, v, omega, dt = primitive
        
        v_noisy = v + np.random.normal(0, 0.05)
        omega_noisy = omega + np.random.normal(0, 0.02)
        
        if prim_type == 'turn':
            new_x = x
            new_y = y
            new_theta = theta + omega_noisy * dt
        else:
            if abs(omega_noisy) < 0.01:
                new_x = x + v_noisy * np.cos(theta) * dt
                new_y = y + v_noisy * np.sin(theta) * dt
                new_theta = theta
            else:
                radius = v_noisy / omega_noisy
                new_theta = theta + omega_noisy * dt
                new_x = x + radius * (np.sin(new_theta) - np.sin(theta))
                new_y = y - radius * (np.cos(new_theta) - np.cos(theta))
        
        new_theta = self.normalize_angle(new_theta)
        return (new_x, new_y, new_theta)
    
    def is_valid_state(self, state):
        x, y, _ = state
        point = Point(x, y)
        
        for obstacle in self.obstacles:
            if obstacle.contains(point) or obstacle.distance(point) < 0.2:
                return False
        return True
    
    def compute_reachability(self, start_state):
        start_discrete = self.discretize_state(*start_state)
        queue = deque([(0, start_discrete, start_state, 1.0)])
        
        visited = {start_discrete: 1.0}
        state_info = {start_discrete: {
            'continuous': start_state,
            'probability': 1.0,
            'time_step': 0,
            'parent': None
        }}
        
        while queue:
            time_step, discrete_state, cont_state, probability = queue.popleft()
            
            if time_step >= self.max_steps:
                continue
            
            for primitive in self.primitives:
                new_cont_state = self.apply_primitive(cont_state, primitive)
                
                if not self.is_valid_state(new_cont_state):
                    continue
                
                new_discrete = self.discretize_state(*new_cont_state)
                
                transition_prob = self.compute_transition_probability(
                    cont_state, new_cont_state, primitive
                )
                
                new_probability = probability * transition_prob
                
                if new_discrete not in visited or new_probability > visited[new_discrete]:
                    visited[new_discrete] = new_probability
                    
                    state_info[new_discrete] = {
                        'continuous': new_cont_state,
                        'probability': new_probability,
                        'time_step': time_step + 1,
                        'parent': discrete_state,
                        'primitive': primitive
                    }
                    
                    queue.append((time_step + 1, new_discrete, new_cont_state, new_probability))
        
        # Convert to (x, y) reachability scores
        reachability_scores = {}
        for discrete_state, info in state_info.items():
            x, y, _ = info['continuous']
            key = (round(x, 2), round(y, 2))
            
            if key not in reachability_scores:
                reachability_scores[key] = 0.0
            reachability_scores[key] += info['probability']
        
        # Normalize
        total = sum(reachability_scores.values())
        if total > 0:
            reachability_scores = {k: v/total for k, v in reachability_scores.items()}
        
        return reachability_scores, state_info
    
    def compute_transition_probability(self, from_state, to_state, primitive):
        prim_type, v, omega, dt = primitive
        control_cost = 0.1 * (v**2 + omega**2) * dt
        
        _, _, theta = from_state
        x1, y1, _ = from_state
        x2, y2, theta2 = to_state
        
        direction = np.arctan2(y2 - y1, x2 - x1)
        angle_diff = abs(self.normalize_angle(direction - theta))
        heading_cost = angle_diff / np.pi
        
        total_cost = control_cost + heading_cost
        probability = np.exp(-total_cost)
        
        return probability
    
    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def world_to_screen(x, y):
    """Convert world coordinates to screen coordinates"""
    screen_x = int(x * SCALE + OFFSET_X)
    screen_y = int(WINDOW_HEIGHT - (y * SCALE + OFFSET_Y))
    return screen_x, screen_y


def screen_to_world(screen_x, screen_y):
    """Convert screen coordinates to world coordinates"""
    x = (screen_x - OFFSET_X) / SCALE
    y = (WINDOW_HEIGHT - screen_y - OFFSET_Y) / SCALE
    return x, y


def create_environment():
    """Create obstacles"""
    obstacles = [
        Polygon([(3, 2), (6, 2), (6, 6), (3, 6)]),
        Polygon([(8, 4), (10, 5), (9, 7)]),
        Polygon([(1, 8), (3, 8), (3, 9), (2, 9), (2, 10), (1, 10)]),
        Polygon([(7, 0.5), (8, 0.5), (8, 1.5), (7, 1.5)]),
    ]
    return obstacles


def draw_obstacles(screen, obstacles):
    """Draw obstacles on screen"""
    for obs in obstacles:
        coords = list(obs.exterior.coords)
        screen_coords = [world_to_screen(x, y) for x, y in coords]
        pygame.draw.polygon(screen, GRAY, screen_coords)
        pygame.draw.polygon(screen, BLACK, screen_coords, 2)


def draw_agent(screen, x, y, theta):
    """Draw the agent (circle with direction indicator)"""
    screen_pos = world_to_screen(x, y)
    
    # Draw circle
    radius_pixels = int(AGENT_RADIUS * SCALE)
    pygame.draw.circle(screen, GREEN, screen_pos, radius_pixels)
    pygame.draw.circle(screen, DARK_GREEN, screen_pos, radius_pixels, 3)
    
    # Draw direction arrow
    arrow_length = AGENT_RADIUS * 1.5 * SCALE
    end_x = screen_pos[0] + int(arrow_length * np.cos(theta))
    end_y = screen_pos[1] - int(arrow_length * np.sin(theta))
    pygame.draw.line(screen, DARK_GREEN, screen_pos, (end_x, end_y), 4)
    
    # Draw arrowhead
    arrow_size = 8
    angle1 = theta + 2.5
    angle2 = theta - 2.5
    p1 = (end_x - int(arrow_size * np.cos(angle1)), end_y + int(arrow_size * np.sin(angle1)))
    p2 = (end_x - int(arrow_size * np.cos(angle2)), end_y + int(arrow_size * np.sin(angle2)))
    pygame.draw.polygon(screen, DARK_GREEN, [(end_x, end_y), p1, p2])


def draw_reachability(screen, reachability_scores, max_prob):
    """Draw reachability heatmap"""
    if not reachability_scores:
        return
    
    for (x, y), prob in reachability_scores.items():
        screen_pos = world_to_screen(x, y)
        
        # Color based on probability (hot colormap)
        intensity = int(255 * (prob / max_prob))
        color = (intensity, 0, max(0, 255 - intensity * 2))
        
        size = max(3, int(6 * (prob / max_prob)))
        pygame.draw.circle(screen, color, screen_pos, size)


def draw_state_info(screen, state_info, max_time, dt):
    """Draw states colored by time"""
    if not state_info:
        return
    
    for discrete_state, info in state_info.items():
        x, y, theta = info['continuous']
        time_step = info['time_step']
        
        screen_pos = world_to_screen(x, y)
        
        # Color based on time (viridis-like)
        t_norm = time_step / max_time
        # Simple blue to yellow gradient
        color = (
            int(255 * t_norm),
            int(255 * (0.5 + 0.5 * t_norm)),
            int(255 * (1.0 - t_norm))
        )
        
        pygame.draw.circle(screen, color, screen_pos, 3)


def draw_ui(screen, agent_state, mode, computing, auto_compute, font):
    """Draw UI information"""
    x, y, theta = agent_state
    
    # Background panel
    panel_rect = pygame.Rect(WINDOW_WIDTH - 300, 0, 300, 220)
    pygame.draw.rect(screen, (50, 50, 50, 200), panel_rect)
    pygame.draw.rect(screen, WHITE, panel_rect, 2)
    
    # Text
    texts = [
        f"Position: ({x:.2f}, {y:.2f})",
        f"Heading: {np.degrees(theta):.1f}°",
        f"",
        f"Mode: {mode}",
        f"Auto-compute: {'ON' if auto_compute else 'OFF'}",
        f"",
        "Controls:",
        "WASD - Move agent",
        "Q/E - Rotate left/right",
        "SPACE - Toggle auto-compute",
        "C - Manual compute",
        "1/2/3 - Change view mode",
        "R - Reset position"
    ]
    
    if computing:
        texts.insert(5, "Computing...")
    
    y_offset = 10
    for text in texts:
        surface = font.render(text, True, WHITE)
        screen.blit(surface, (WINDOW_WIDTH - 290, y_offset))
        y_offset += 18


def draw_grid(screen):
    """Draw background grid"""
    for i in range(int(WORLD_WIDTH) + 1):
        start = world_to_screen(i, 0)
        end = world_to_screen(i, WORLD_HEIGHT)
        pygame.draw.line(screen, (200, 200, 200), start, end, 1)
    
    for i in range(int(WORLD_HEIGHT) + 1):
        start = world_to_screen(0, i)
        end = world_to_screen(WORLD_WIDTH, i)
        pygame.draw.line(screen, (200, 200, 200), start, end, 1)


def main():
    # Setup
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Interactive Reachability - WASD to move, SPACE to compute")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Environment
    obstacles = create_environment()
    
    # Agent state
    agent_x = 0.5
    agent_y = 0.5
    agent_theta = np.pi / 4
    
    # Reachability planner
    planner = ProgressiveStateExpansion(
        v_max=v_max,
        omega_max=omega_max,
        time_horizon=time_horizon,
        obstacles=obstacles,
        grid_resolution=0.3,
        theta_resolution=8
    )
    
    # Results
    reachability_scores = {}
    state_info = {}
    max_prob = 1.0
    max_time = 0
    
    # UI state
    view_mode = "reachability"  # "reachability", "wavefront", "both"
    computing = False
    auto_compute = True  # Continuous computation mode
    last_compute_pos = (agent_x, agent_y, agent_theta)
    compute_threshold = 0.3  # Recompute when moved this far
    
    # Movement parameters
    move_speed = 0.1
    rotate_speed = 0.1
    
    # Frame counter for periodic updates
    frame_count = 0
    compute_interval = 15  # Compute every N frames when in auto mode
    
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Toggle auto-compute mode
                    auto_compute = not auto_compute
                    if auto_compute:
                        print("Auto-compute: ON (continuous reachability updates)")
                    else:
                        print("Auto-compute: OFF (press SPACE to toggle)")
                
                elif event.key == pygame.K_c:
                    # Manual compute
                    computing = True
                    print(f"\nManual reachability computation from ({agent_x:.2f}, {agent_y:.2f}, {np.degrees(agent_theta):.1f}°)")
                    
                    start_state = (agent_x, agent_y, agent_theta)
                    reachability_scores, state_info = planner.compute_reachability(start_state)
                    
                    if reachability_scores:
                        max_prob = max(reachability_scores.values())
                    if state_info:
                        max_time = max(info['time_step'] for info in state_info.values())
                    
                    last_compute_pos = start_state
                    computing = False
                    print(f"Done! Found {len(reachability_scores)} reachable locations")
                
                elif event.key == pygame.K_1:
                    view_mode = "reachability"
                    print("View mode: Reachability heatmap")
                
                elif event.key == pygame.K_2:
                    view_mode = "wavefront"
                    print("View mode: Expansion wavefront")
                
                elif event.key == pygame.K_3:
                    view_mode = "both"
                    print("View mode: Both")
                
                elif event.key == pygame.K_r:
                    agent_x = 0.5
                    agent_y = 0.5
                    agent_theta = np.pi / 4
                    reachability_scores = {}
                    state_info = {}
                    last_compute_pos = (agent_x, agent_y, agent_theta)
                    print("Reset agent position")
        
        # Continuous key handling
        keys = pygame.key.get_pressed()
        
        # Track if agent moved
        old_pos = (agent_x, agent_y, agent_theta)
        
        # Movement
        if keys[pygame.K_w]:
            new_x = agent_x + move_speed * np.cos(agent_theta)
            new_y = agent_y + move_speed * np.sin(agent_theta)
            if planner.is_valid_state((new_x, new_y, agent_theta)):
                agent_x = new_x
                agent_y = new_y
        
        if keys[pygame.K_s]:
            new_x = agent_x - move_speed * np.cos(agent_theta)
            new_y = agent_y - move_speed * np.sin(agent_theta)
            if planner.is_valid_state((new_x, new_y, agent_theta)):
                agent_x = new_x
                agent_y = new_y
        
        if keys[pygame.K_a]:
            new_x = agent_x - move_speed * np.sin(agent_theta)
            new_y = agent_y + move_speed * np.cos(agent_theta)
            if planner.is_valid_state((new_x, new_y, agent_theta)):
                agent_x = new_x
                agent_y = new_y
        
        if keys[pygame.K_d]:
            new_x = agent_x + move_speed * np.sin(agent_theta)
            new_y = agent_y - move_speed * np.cos(agent_theta)
            if planner.is_valid_state((new_x, new_y, agent_theta)):
                agent_x = new_x
                agent_y = new_y
        
        # Rotation
        if keys[pygame.K_q]:
            agent_theta += rotate_speed
        
        if keys[pygame.K_e]:
            agent_theta -= rotate_speed
        
        # Normalize angle
        agent_theta = planner.normalize_angle(agent_theta)
        
        # Auto-compute reachability if enabled and agent has moved
        frame_count += 1
        if auto_compute and frame_count % compute_interval == 0:
            # Check if agent has moved significantly
            dx = agent_x - last_compute_pos[0]
            dy = agent_y - last_compute_pos[1]
            dtheta = abs(planner.normalize_angle(agent_theta - last_compute_pos[2]))
            distance_moved = np.sqrt(dx**2 + dy**2)
            
            if distance_moved > compute_threshold or dtheta > 0.5:
                start_state = (agent_x, agent_y, agent_theta)
                reachability_scores, state_info = planner.compute_reachability(start_state)
                
                if reachability_scores:
                    max_prob = max(reachability_scores.values())
                if state_info:
                    max_time = max(info['time_step'] for info in state_info.values())
                
                last_compute_pos = start_state
        
        # Drawing
        screen.fill(WHITE)
        
        # Draw grid
        draw_grid(screen)
        
        # Draw reachability visualization
        if view_mode in ["reachability", "both"] and reachability_scores:
            draw_reachability(screen, reachability_scores, max_prob)
        
        if view_mode in ["wavefront", "both"] and state_info:
            draw_state_info(screen, state_info, max_time, planner.dt)
        
        # Draw obstacles
        draw_obstacles(screen, obstacles)
        
        # Draw agent
        draw_agent(screen, agent_x, agent_y, agent_theta)
        
        # Draw UI
        draw_ui(screen, (agent_x, agent_y, agent_theta), view_mode, computing, auto_compute, font)
        
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    print("="*60)
    print("INTERACTIVE REACHABILITY WITH PYGAME")
    print("="*60)
    print("\nControls:")
    print("  WASD - Move agent (forward/backward/strafe)")
    print("  Q/E - Rotate left/right")
    print("  SPACE - Toggle auto-compute (continuous updates)")
    print("  C - Manual compute once")
    print("  1/2/3 - Switch view modes")
    print("  R - Reset agent position")
    print("\nView Modes:")
    print("  1 - Reachability heatmap (probability)")
    print("  2 - Expansion wavefront (time-colored)")
    print("  3 - Both combined")
    print("\nAuto-compute is ON by default - reachability updates")
    print("automatically as you move the agent around!")
    print("="*60)
    
    main()
