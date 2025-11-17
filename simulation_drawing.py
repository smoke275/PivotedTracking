#!/usr/bin/env python3
"""
Simulation Drawing Helper
Contains drawing functions for the simple agent simulation.
This helps keep the main simulation file clean and organized.
"""

import pygame
import math
import numpy as np
from simulation_config import *
from risk_calculator import detect_visibility_breakoff_points, get_reachability_probabilities_for_fixed_grid

def draw_visibility_boundary_polygon(screen, polygon_points, color=(255, 255, 0), line_width=2):
    """
    Draw the visibility boundary polygon connecting all ray endpoints.
    
    Args:
        screen: Pygame screen to draw on
        polygon_points: List of (x, y) coordinates forming the polygon
        color: RGB color tuple for the polygon outline (default: yellow)
        line_width: Width of the polygon outline (default: 2)
    """
    if not polygon_points or len(polygon_points) < 3:
        return
    
    # Convert points to integers for pygame
    int_points = [(int(x), int(y)) for x, y in polygon_points]
    
    # Draw the polygon outline
    if len(int_points) >= 3:
        pygame.draw.polygon(screen, color, int_points, line_width)

def draw_evader_visibility(screen, agent_x, agent_y, visibility_data, show_rays=True, show_visibility_area=True, agent_theta=None):
    """
    Draw the evader's visibility on screen with 4-category breakoff point coloring and concentric circles.
    
    Args:
        screen: Pygame screen to draw on
        agent_x, agent_y: Evader position
        visibility_data: List from calculate_evader_visibility()
        show_rays: Whether to show individual rays
        show_visibility_area: Whether to show the visibility polygon
        agent_theta: Agent orientation in radians (for breakoff point orientation coloring)
    """
    if not visibility_data:
        return
    
    # Detect visibility breakoff points using the API from risk_calculator with orientation
    breakoff_points, breakoff_lines = detect_visibility_breakoff_points(
        visibility_data, 
        min_gap_distance=MIN_GAP_DISTANCE,
        agent_x=agent_x,
        agent_y=agent_y, 
        agent_theta=agent_theta
    )
    
    # Collect unique breakoff distances for concentric circles - only the near point of each segment
    breakoff_distances = set()
    for line_data in breakoff_lines:
        if len(line_data) >= 3:  # Has start_point, end_point, gap_size (and maybe category)
            start_point, end_point = line_data[0], line_data[1]
            
            # Calculate distances from agent to both endpoints
            dist_to_start = math.sqrt((start_point[0] - agent_x)**2 + (start_point[1] - agent_y)**2)
            dist_to_end = math.sqrt((end_point[0] - agent_x)**2 + (end_point[1] - agent_y)**2)
            
            # Only add the nearer distance (the near point of the segment)
            near_distance = min(dist_to_start, dist_to_end)
            breakoff_distances.add(near_distance)
    
    # Draw concentric circles at breakoff distances
    for distance in sorted(breakoff_distances):
        if distance > 5:  # Only draw circles for reasonable distances
            # Create surface for transparent circle
            circle_surface = pygame.Surface((int(distance * 2 + 4), int(distance * 2 + 4)), pygame.SRCALPHA)
            circle_center = (int(distance + 2), int(distance + 2))
            
            # Draw circle with breakoff circle color and transparency
            pygame.draw.circle(circle_surface, BREAKOFF_CIRCLE_COLOR, circle_center, int(distance), BREAKOFF_CIRCLE_WIDTH)
            
            # Blit to main screen centered on agent
            screen.blit(circle_surface, (int(agent_x - distance - 2), int(agent_y - distance - 2)))
    
    # Draw visibility polygon
    if show_visibility_area and len(visibility_data) > 2:
        # Create a surface for transparent drawing
        visibility_surface = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
        
        # Create polygon points from ray endpoints
        polygon_points = [(int(agent_x), int(agent_y))]  # Start at agent
        for angle, endpoint, distance, blocked in visibility_data:
            polygon_points.append((int(endpoint[0]), int(endpoint[1])))
        
        # Draw filled polygon
        if len(polygon_points) > 3:
            pygame.draw.polygon(visibility_surface, VISIBILITY_COLORS['visibility_area'], polygon_points)
            screen.blit(visibility_surface, (0, 0))
    
    # Draw individual rays
    if show_rays:
        for angle, endpoint, distance, blocked in visibility_data:
            color = VISIBILITY_COLORS['blocked_ray'] if blocked else VISIBILITY_COLORS['ray']
            
            # Create surface for transparent line
            line_surface = pygame.Surface((abs(int(endpoint[0] - agent_x)) + 2, 
                                         abs(int(endpoint[1] - agent_y)) + 2), pygame.SRCALPHA)
            
            # Calculate line position on surface
            start_pos = (max(1, int(agent_x - min(agent_x, endpoint[0]))), 
                        max(1, int(agent_y - min(agent_y, endpoint[1]))))
            end_pos = (max(1, int(endpoint[0] - min(agent_x, endpoint[0]))), 
                      max(1, int(endpoint[1] - min(agent_y, endpoint[1]))))
            
            # Draw line
            pygame.draw.line(line_surface, color, start_pos, end_pos, 1)
            
            # Blit to main screen
            screen.blit(line_surface, (min(int(agent_x), int(endpoint[0])), 
                                     min(int(agent_y), int(endpoint[1]))))
    
    # Draw breakoff/gap lines connecting points where visibility changes abruptly with category-based coloring
    for line_data in breakoff_lines:
        if len(line_data) == 4:  # New format with category
            start_point, end_point, gap_size, category = line_data
            # Choose color based on category
            if category in BREAKOFF_CATEGORY_COLORS:
                line_color = BREAKOFF_CATEGORY_COLORS[category]['line']
            else:
                line_color = VISIBILITY_COLORS['breakoff_line']  # Fallback to default orange
        else:  # Old format without category (fallback compatibility)
            start_point, end_point, gap_size = line_data[:3]
            line_color = VISIBILITY_COLORS['breakoff_line']
        
        # Create surface for transparent gap line
        line_width = min(int(gap_size / 10), 5) + 2  # Thicker lines for larger gaps
        gap_surface = pygame.Surface((abs(int(end_point[0] - start_point[0])) + line_width * 2, 
                                     abs(int(end_point[1] - start_point[1])) + line_width * 2), pygame.SRCALPHA)
        
        # Calculate line position on surface
        start_pos = (max(line_width, int(start_point[0] - min(start_point[0], end_point[0]))), 
                    max(line_width, int(start_point[1] - min(start_point[1], end_point[1]))))
        end_pos = (max(line_width, int(end_point[0] - min(start_point[0], end_point[0]))), 
                  max(line_width, int(end_point[1] - min(start_point[1], end_point[1]))))
        
        # Draw gap line with category-based color
        pygame.draw.line(gap_surface, line_color, start_pos, end_pos, line_width)
        
        # Blit to main screen
        screen.blit(gap_surface, (min(int(start_point[0]), int(end_point[0])) - line_width, 
                                 min(int(start_point[1]), int(end_point[1])) - line_width))
    
    # Draw breakoff points as highlighted circles with category-based coloring
    for point_data in breakoff_points:
        if len(point_data) == 3:  # New format with category
            x, y, category = point_data
            # Choose colors based on category
            if category in BREAKOFF_CATEGORY_COLORS:
                outer_color = BREAKOFF_CATEGORY_COLORS[category]['point']
                middle_color = BREAKOFF_CATEGORY_COLORS[category]['middle']
            else:
                outer_color = (255, 255, 0)   # Yellow fallback
                middle_color = (255, 165, 0)  # Orange fallback
        else:  # Old format without category (fallback compatibility)
            x, y = point_data[:2]
            outer_color = (255, 255, 0)   # Yellow
            middle_color = (255, 165, 0)  # Orange
        
        # Draw a small circle at each breakoff point with category-based coloring
        pygame.draw.circle(screen, outer_color, (int(x), int(y)), 6)      # Colored outer
        pygame.draw.circle(screen, middle_color, (int(x), int(y)), 4)     # Darker middle
        pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 2) # White center


def draw_breakoff_midpoints(screen, breakoff_midpoints):
    """
    Draw the midpoints of breakoff lines with distinct visual styling.
    
    Args:
        screen: Pygame screen to draw on
        breakoff_midpoints: List of (midpoint_x, midpoint_y, gap_size, category) tuples
    """
    if not breakoff_midpoints:
        return
    
    for midpoint_x, midpoint_y, gap_size, category in breakoff_midpoints:
        # Choose colors based on category
        if category in BREAKOFF_CATEGORY_COLORS:
            outer_color = BREAKOFF_CATEGORY_COLORS[category]['point']
            middle_color = BREAKOFF_CATEGORY_COLORS[category]['middle']
        else:
            outer_color = (255, 255, 0)   # Yellow fallback
            middle_color = (255, 165, 0)  # Orange fallback
        
        # Scale point size based on gap size
        base_size = 8
        size_modifier = min(int(gap_size / 30), 6)  # Scale with gap size, max +6 pixels
        point_size = base_size + size_modifier
        
        # Draw midpoint with distinctive diamond/square shape to differentiate from breakoff points
        int_x, int_y = int(midpoint_x), int(midpoint_y)
        
        # Create diamond shape points
        diamond_points = [
            (int_x, int_y - point_size),      # Top
            (int_x + point_size, int_y),      # Right
            (int_x, int_y + point_size),      # Bottom
            (int_x - point_size, int_y)       # Left
        ]
        
        # Draw filled diamond with category-based color
        pygame.draw.polygon(screen, outer_color, diamond_points)
        
        # Draw smaller inner diamond
        inner_size = max(2, point_size - 3)
        inner_diamond_points = [
            (int_x, int_y - inner_size),      # Top
            (int_x + inner_size, int_y),      # Right
            (int_x, int_y + inner_size),      # Bottom
            (int_x - inner_size, int_y)       # Left
        ]
        pygame.draw.polygon(screen, middle_color, inner_diamond_points)
        
        # Draw tiny center dot
        pygame.draw.circle(screen, (255, 255, 255), (int_x, int_y), 2)


def draw_exploration_offset_points(screen, exploration_offset_points):
    """
    Draw exploration offset points that indicate directions toward unexplored areas.
    
    Args:
        screen: Pygame screen to draw on
        exploration_offset_points: List of (offset_x, offset_y, gap_size, category, direction, nearest_node_info) tuples
    """
    if not exploration_offset_points:
        return
    
    for exploration_point in exploration_offset_points:
        # Handle both old and new formats for backward compatibility
        if len(exploration_point) >= 6:
            offset_x, offset_y, gap_size, category, direction, nearest_node_info = exploration_point[:6]
        else:
            offset_x, offset_y, gap_size, category, direction = exploration_point[:5]
            nearest_node_info = None
        # Choose colors based on category
        if category in BREAKOFF_CATEGORY_COLORS:
            outer_color = BREAKOFF_CATEGORY_COLORS[category]['point']
            middle_color = BREAKOFF_CATEGORY_COLORS[category]['middle']
        else:
            outer_color = (255, 255, 0)   # Yellow fallback
            middle_color = (255, 165, 0)  # Orange fallback
        
        # Scale point size based on gap size (smaller than midpoints to show hierarchy)
        base_size = 6
        size_modifier = min(int(gap_size / 40), 4)  # Smaller scaling than midpoints
        point_size = base_size + size_modifier
        
        int_x, int_y = int(offset_x), int(offset_y)
        
        # Create arrow/triangle shape pointing toward the unexplored direction
        # The triangle points in the direction of exploration potential
        
        if direction == 'right':
            # Right-pointing triangle
            triangle_points = [
                (int_x - point_size, int_y - point_size//2),    # Left bottom
                (int_x - point_size, int_y + point_size//2),    # Left top
                (int_x + point_size, int_y)                     # Right point
            ]
        else:  # direction == 'left'
            # Left-pointing triangle
            triangle_points = [
                (int_x + point_size, int_y - point_size//2),    # Right bottom
                (int_x + point_size, int_y + point_size//2),    # Right top
                (int_x - point_size, int_y)                     # Left point
            ]
        
        # Draw filled triangle with category-based color
        pygame.draw.polygon(screen, outer_color, triangle_points)
        
        # Draw smaller inner triangle
        inner_size = max(2, point_size - 2)
        if direction == 'right':
            inner_triangle_points = [
                (int_x - inner_size, int_y - inner_size//2),
                (int_x - inner_size, int_y + inner_size//2),
                (int_x + inner_size, int_y)
            ]
        else:  # direction == 'left'
            inner_triangle_points = [
                (int_x + inner_size, int_y - inner_size//2),
                (int_x + inner_size, int_y + inner_size//2),
                (int_x - inner_size, int_y)
            ]
        
        pygame.draw.polygon(screen, middle_color, inner_triangle_points)
        
        # Draw tiny center dot
        pygame.draw.circle(screen, (255, 255, 255), (int_x, int_y), 1)


def draw_exploration_nearest_nodes(screen, exploration_nearest_nodes, map_graph):
    """
    Draw highlighted map graph nodes that are nearest to exploration offset points.
    
    Args:
        screen: Pygame screen to draw on
        exploration_nearest_nodes: List of nearest node info dictionaries
        map_graph: Map graph object containing node positions
    """
    import math
    
    if not exploration_nearest_nodes or not map_graph:
        return
    
    for node_info in exploration_nearest_nodes:
        node_index = node_info['node_index']
        distance = node_info['distance']
        exploration_point = node_info['exploration_point']
        category = node_info['category']
        direction = node_info['direction']
        
        # Get node position from map graph
        if node_index < len(map_graph.nodes):
            node_pos = map_graph.nodes[node_index]
            int_node_x, int_node_y = int(node_pos[0]), int(node_pos[1])
            
            # Choose colors based on category (same as exploration points)
            if category in BREAKOFF_CATEGORY_COLORS:
                outer_color = BREAKOFF_CATEGORY_COLORS[category]['point']
                middle_color = BREAKOFF_CATEGORY_COLORS[category]['middle']
            else:
                outer_color = (255, 255, 0)   # Yellow fallback
                middle_color = (255, 165, 0)  # Orange fallback
            
            # Draw highlighted node as a larger circle with multiple rings
            # Outer ring (largest)
            pygame.draw.circle(screen, outer_color, (int_node_x, int_node_y), 12, 3)
            
            # Middle ring
            pygame.draw.circle(screen, middle_color, (int_node_x, int_node_y), 8, 2)
            
            # Inner filled circle
            pygame.draw.circle(screen, (255, 255, 255), (int_node_x, int_node_y), 4)
            
            # Draw connection line from exploration point to nearest node
            exploration_x, exploration_y = int(exploration_point[0]), int(exploration_point[1])
            
            # Create a dashed line effect by drawing short segments
            line_dx = int_node_x - exploration_x
            line_dy = int_node_y - exploration_y
            line_length = math.sqrt(line_dx**2 + line_dy**2)
            
            if line_length > 0:
                # Normalize the direction
                unit_dx = line_dx / line_length
                unit_dy = line_dy / line_length
                
                # Draw dashed line with 5-pixel segments and 3-pixel gaps
                segment_length = 5
                gap_length = 3
                total_step = segment_length + gap_length
                
                current_distance = 0
                while current_distance < line_length:
                    # Start of current segment
                    start_x = int(exploration_x + current_distance * unit_dx)
                    start_y = int(exploration_y + current_distance * unit_dy)
                    
                    # End of current segment
                    end_distance = min(current_distance + segment_length, line_length)
                    end_x = int(exploration_x + end_distance * unit_dx)
                    end_y = int(exploration_y + end_distance * unit_dy)
                    
                    # Draw the segment
                    pygame.draw.line(screen, middle_color, (start_x, start_y), (end_x, end_y), 2)
                    
                    current_distance += total_step


def draw_reachability_grid_overlay(screen, mask_data, agent_x, agent_y, agent_theta, alpha=None):
    """Draw the reachability grid overlay around the agent (optimized for performance)."""
    if mask_data is None:
        return
    
    if alpha is None:
        alpha = REACHABILITY_OVERLAY_ALPHA
    
    # Get grid properties
    grid_size = mask_data.get('grid_size', 0)
    cell_size = mask_data.get('cell_size_px', 1)
    
    if grid_size == 0:
        return
    
    # Get reachability probabilities for the fixed grid using the API
    fixed_grid_probabilities = get_reachability_probabilities_for_fixed_grid(
        agent_x, agent_y, agent_theta, mask_data)
    
    # Calculate extent in world coordinates centered on agent
    half_extent_px = (grid_size * cell_size) / 2
    
    # Determine display scaling to fit nicely on screen
    display_cell_size = max(1, min(REACHABILITY_MAX_DISPLAY_SIZE // grid_size, int(cell_size)))
    
    # Create a surface for the grid visualization
    surface_size = grid_size * display_cell_size
    grid_surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
    
    # Find max intensity for normalization
    max_intensity = fixed_grid_probabilities.max() if fixed_grid_probabilities.max() > 0 else 1.0
    
    # Flatten the probability grid for vectorized processing
    flat_probabilities = fixed_grid_probabilities.flatten()
    norm_intensities = flat_probabilities / max_intensity
    
    # Initialize color arrays for all cells
    total_cells = grid_size * grid_size
    r_values = np.zeros(total_cells, dtype=np.uint8)
    g_values = np.zeros(total_cells, dtype=np.uint8)
    b_values = np.zeros(total_cells, dtype=np.uint8)
    
    # Vectorized 'hot' colormap calculation for transparent-to-red-to-yellow-to-white
    # Transparent to red (norm_intensity < 0.33) - red increases with intensity
    mask1 = norm_intensities < 0.33
    r_values[mask1] = 255  # Always full red for consistency with previous implementation
    
    # Red to yellow (0.33 <= norm_intensity < 0.66)
    mask2 = (norm_intensities >= 0.33) & (norm_intensities < 0.66)
    r_values[mask2] = 255
    g_values[mask2] = ((norm_intensities[mask2] - 0.33) * 3 * 255).astype(np.uint8)
    
    # Yellow to white (norm_intensity >= 0.66)
    mask3 = norm_intensities >= 0.66
    r_values[mask3] = 255
    g_values[mask3] = 255
    b_values[mask3] = ((norm_intensities[mask3] - 0.66) * 3 * 255).astype(np.uint8)
    
    # Vectorized alpha calculation for transparent-to-red transition
    alpha_values = np.full(total_cells, alpha, dtype=np.uint8)
    low_intensity_mask = norm_intensities < 0.33
    alpha_values[low_intensity_mask] = (norm_intensities[low_intensity_mask] * 3 * alpha).astype(np.uint8)
    
    # Optimized drawing using direct array indexing
    if display_cell_size == 1:
        # For single-pixel cells, use direct pixel access (fastest method)
        try:
            pixel_array = pygame.surfarray.pixels_alpha(grid_surface)
            color_array = pygame.surfarray.pixels3d(grid_surface)
            
            # Set colors and alpha values directly using reshaped arrays
            color_array[:, :, 0] = r_values.reshape(grid_size, grid_size).T
            color_array[:, :, 1] = g_values.reshape(grid_size, grid_size).T
            color_array[:, :, 2] = b_values.reshape(grid_size, grid_size).T
            pixel_array[:, :] = alpha_values.reshape(grid_size, grid_size).T
            
            # Clean up array references
            del pixel_array, color_array
        except:
            # Fallback to rect drawing if pixel array access fails
            for i in range(total_cells):
                row = i // grid_size
                col = i % grid_size
                display_x = col * display_cell_size
                display_y = row * display_cell_size
                color = (r_values[i], g_values[i], b_values[i], alpha_values[i])
                cell_rect = pygame.Rect(display_x, display_y, display_cell_size, display_cell_size)
                pygame.draw.rect(grid_surface, color, cell_rect)
    else:
        # For larger cells, batch the rect operations for better performance
        rects_and_colors = []
        for i in range(total_cells):
            row = i // grid_size
            col = i % grid_size
            display_x = col * display_cell_size
            display_y = row * display_cell_size
            color = (r_values[i], g_values[i], b_values[i], alpha_values[i])
            cell_rect = pygame.Rect(display_x, display_y, display_cell_size, display_cell_size)
            rects_and_colors.append((cell_rect, color))
        
        # Draw all rects in batches to reduce function call overhead
        for batch_start in range(0, len(rects_and_colors), BATCH_SIZE_RECTS):
            batch_end = min(batch_start + BATCH_SIZE_RECTS, len(rects_and_colors))
            for cell_rect, color in rects_and_colors[batch_start:batch_end]:
                pygame.draw.rect(grid_surface, color, cell_rect)
    
    # Calculate position to center the grid on the agent in screen coordinates
    screen_x = int(agent_x - surface_size // 2)
    screen_y = int(agent_y - surface_size // 2)
    
    # Blit the grid surface to the main screen centered on agent
    screen.blit(grid_surface, (screen_x, screen_y))
    
    # Draw border around the reachability grid
    border_rect = pygame.Rect(screen_x, screen_y, surface_size, surface_size)
    
    # Create surface for transparent border
    border_surface = pygame.Surface((surface_size + REACHABILITY_BORDER_WIDTH * 2, 
                                   surface_size + REACHABILITY_BORDER_WIDTH * 2), pygame.SRCALPHA)
    border_color = REACHABILITY_BORDER_COLOR
    
    # Draw border rectangle with transparency
    pygame.draw.rect(border_surface, border_color, 
                    (0, 0, surface_size + REACHABILITY_BORDER_WIDTH * 2, 
                     surface_size + REACHABILITY_BORDER_WIDTH * 2), 
                    REACHABILITY_BORDER_WIDTH)
    
    # Blit border surface to screen
    screen.blit(border_surface, (screen_x - REACHABILITY_BORDER_WIDTH, 
                               screen_y - REACHABILITY_BORDER_WIDTH))


def draw_agent(screen, agent, color, label, font):
    """
    Draw an agent with direction indicator and label.
    
    Args:
        screen: Pygame screen to draw on
        agent: Agent object with state [x, y, theta]
        color: Color tuple for the agent
        label: Text label for the agent
        font: Pygame font for rendering text
    """
    x, y, theta = agent.state[0], agent.state[1], agent.state[2]
    
    # Draw agent body (circle)
    pygame.draw.circle(screen, color, (int(x), int(y)), AGENT_RADIUS)
    
    # Draw direction indicator (line showing heading)
    end_x = x + AGENT_RADIUS * 1.5 * math.cos(theta)
    end_y = y + AGENT_RADIUS * 1.5 * math.sin(theta)
    pygame.draw.line(screen, (255, 255, 255), (int(x), int(y)), (int(end_x), int(end_y)), 3)
    
    # Draw agent label
    label_surface = font.render(label, True, (255, 255, 255))
    screen.blit(label_surface, (int(x) - 20, int(y) - AGENT_RADIUS - 25))


def draw_trajectory(screen, trajectory):
    """
    Draw the optimized trajectory path.
    
    Args:
        screen: Pygame screen to draw on
        trajectory: Trajectory dictionary with 'path' key
    """
    if not trajectory:
        return
        
    path = trajectory.get('path', [])
    if len(path) >= 2:
        # Draw the optimized path as a smooth curve
        for i in range(len(path) - 1):
            start_pos = (int(path[i][0]), int(path[i][1]))
            end_pos = (int(path[i+1][0]), int(path[i+1][1]))
            pygame.draw.line(screen, TRAJECTORY_COLOR, start_pos, end_pos, TRAJECTORY_LINE_WIDTH)


def draw_rrt_node(screen, pos, node_size, color, alpha, is_invalid=False):
    """
    Draw a single RRT node with transparency and optional invalid indicator.
    
    Args:
        screen: Pygame screen to draw on
        pos: (x, y) position tuple
        node_size: Radius of the node
        color: RGB color tuple
        alpha: Alpha transparency value (0-255)
        is_invalid: Whether to draw the invalid 'X' indicator
    """
    if alpha < 255:
        # Draw transparent node
        node_surface = pygame.Surface((node_size * 2 + 2, node_size * 2 + 2), pygame.SRCALPHA)
        transparent_color = (*color, alpha)
        pygame.draw.circle(node_surface, transparent_color, 
                         (node_size + 1, node_size + 1), node_size)
        
        if is_invalid:
            # Add a small 'X' to indicate invalid
            pygame.draw.line(node_surface, (255, 255, 255, alpha + 50), 
                           (1, 1), (node_size * 2 + 1, node_size * 2 + 1), 1)
            pygame.draw.line(node_surface, (255, 255, 255, alpha + 50), 
                           (node_size * 2 + 1, 1), (1, node_size * 2 + 1), 1)
        
        screen.blit(node_surface, (pos[0] - node_size - 1, pos[1] - node_size - 1))
    else:
        # Draw normal opaque node
        pygame.draw.circle(screen, color, pos, node_size)


def draw_rrt_edge(screen, start_pos, end_pos, color, alpha):
    """
    Draw a single RRT edge with transparency.
    
    Args:
        screen: Pygame screen to draw on
        start_pos: (x, y) start position tuple
        end_pos: (x, y) end position tuple
        color: RGB color tuple
        alpha: Alpha transparency value (0-255)
    """
    if alpha < 255:
        # Create surface for transparent drawing
        edge_surface = pygame.Surface((abs(int(end_pos[0] - start_pos[0])) + 2, 
                                     abs(int(end_pos[1] - start_pos[1])) + 2), pygame.SRCALPHA)
        surface_start_pos = (max(0, int(start_pos[0] - min(start_pos[0], end_pos[0]))), 
                           max(0, int(start_pos[1] - min(start_pos[1], end_pos[1]))))
        surface_end_pos = (max(0, int(end_pos[0] - min(start_pos[0], end_pos[0]))), 
                         max(0, int(end_pos[1] - min(start_pos[1], end_pos[1]))))
        edge_color_with_alpha = (*color, alpha)
        pygame.draw.line(edge_surface, edge_color_with_alpha, surface_start_pos, surface_end_pos, 1)
        screen.blit(edge_surface, (min(int(start_pos[0]), int(end_pos[0])), 
                                 min(int(start_pos[1]), int(end_pos[1]))))
    else:
        pygame.draw.line(screen, color, start_pos, end_pos, 1)


def draw_strategic_node_highlight(screen, pos, highlight_type):
    """
    Draw strategic highlighting for worst pursuit nodes.
    
    Args:
        screen: Pygame screen to draw on
        pos: (x, y) position tuple
        highlight_type: 'worst' or 'worst_10'
    """
    if highlight_type == 'worst':
        # Highlight the single worst pursuit node with red color and larger size
        pygame.draw.circle(screen, STRATEGIC_COLORS['worst_node'], pos, 8)  # Red highlight
        pygame.draw.circle(screen, STRATEGIC_COLORS['inner_circle'], pos, 6)  # White inner circle
    elif highlight_type == 'worst_10':
        # Highlight nodes in worst 10 with orange color and medium size
        pygame.draw.circle(screen, STRATEGIC_COLORS['worst_10_nodes'], pos, 6)  # Orange highlight
        pygame.draw.circle(screen, STRATEGIC_COLORS['inner_circle'], pos, 4)  # White inner circle


def draw_exploration_flood_regions(screen, exploration_flood_regions, map_graph):
    """
    Draw flood fill regions showing connected unexplored areas around exploration points.
    Each region is color-coded by category and shows the connected network of nodes.
    
    Args:
        screen: Pygame screen to draw on
        exploration_flood_regions: List of flood region dictionaries from EvaderAnalysis
        map_graph: MapGraph object containing node positions and adjacency information
    """
    if not exploration_flood_regions or not map_graph or not hasattr(map_graph, 'nodes'):
        return
    
    # Color mapping for different exploration categories
    flood_colors = {
        'clockwise_near_far': (100, 255, 100, 80),      # Light green (semi-transparent)
        'clockwise_far_near': (255, 100, 100, 80),      # Light red (semi-transparent)
        'counterclockwise_near_far': (100, 100, 255, 80), # Light blue (semi-transparent)
        'counterclockwise_far_near': (255, 255, 100, 80)  # Light yellow (semi-transparent)
    }
    
    # Create a surface for alpha blending of flood regions
    flood_surface = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
    
    for region in exploration_flood_regions:
        category = region.get('exploration_category', 'unknown')
        flood_nodes = region.get('flood_nodes', [])
        
        # Get color for this category
        flood_color = flood_colors.get(category, (128, 128, 128, 80))  # Default gray
        
        if not flood_nodes:
            continue
            
        # Draw connections between flood nodes to show the connected region
        processed_edges = set()
        
        for node_idx in flood_nodes:
            if node_idx >= len(map_graph.nodes):
                continue
                
            node_pos = map_graph.nodes[node_idx]
            
            # Draw connections to other flood nodes in this region
            if hasattr(map_graph, 'adjacency') and node_idx in map_graph.adjacency:
                for neighbor_idx in map_graph.adjacency[node_idx]:
                    if neighbor_idx in flood_nodes:
                        # Create a unique edge identifier to avoid drawing the same edge twice
                        edge_key = tuple(sorted([node_idx, neighbor_idx]))
                        if edge_key not in processed_edges:
                            processed_edges.add(edge_key)
                            
                            # Draw connection line
                            neighbor_pos = map_graph.nodes[neighbor_idx]
                            pygame.draw.line(
                                flood_surface, 
                                flood_color, 
                                (int(node_pos[0]), int(node_pos[1])), 
                                (int(neighbor_pos[0]), int(neighbor_pos[1])), 
                                2
                            )
        
        # Draw nodes in the flood region
        for node_idx in flood_nodes:
            if node_idx >= len(map_graph.nodes):
                continue
                
            node_pos = map_graph.nodes[node_idx]
            
            # Draw larger circle for flood region nodes
            pygame.draw.circle(
                flood_surface, 
                flood_color, 
                (int(node_pos[0]), int(node_pos[1])), 
                6
            )
            
            # Draw inner circle with higher opacity
            inner_color = (*flood_color[:3], min(255, flood_color[3] * 2))
            pygame.draw.circle(
                flood_surface, 
                inner_color, 
                (int(node_pos[0]), int(node_pos[1])), 
                3
            )
    
    # Blit the flood regions surface onto the main screen
    screen.blit(flood_surface, (0, 0))

def draw_polygon_exploration_paths(screen, polygon_exploration_paths):
    """
    Draw polygon exploration paths for breakpoints into unknown areas.
    Uses pygame.draw.arc for circle segments and lines for straight segments.
    Each complete path gets a different color for easy identification.
    
    Args:
        screen: Pygame screen to draw on
        polygon_exploration_paths: List of polygon path dictionaries from calculate_polygon_exploration_paths()
    """
    if not polygon_exploration_paths:
        return
    
    # Define a palette of distinct colors for complete paths
    completed_path_colors = [
        (0, 255, 0),      # Bright green
        (0, 255, 255),    # Bright cyan
        (255, 0, 255),    # Bright magenta
        (255, 255, 0),    # Bright yellow
        (255, 128, 0),    # Bright orange
        (128, 255, 0),    # Lime green
        (0, 128, 255),    # Sky blue
        (255, 0, 128),    # Hot pink
        (128, 0, 255),    # Purple
        (255, 255, 128),  # Light yellow
        (128, 255, 255),  # Light cyan
        (255, 128, 255),  # Light magenta
    ]
    
    completed_path_index = 0  # Track which completed path we're drawing
    
    for path_data in polygon_exploration_paths:
        path_points = path_data.get('path_points', [])
        path_segments = path_data.get('path_segments', [])
        completed = path_data.get('completed', False)
        breakoff_line = path_data.get('breakoff_line', None)
        is_merged_path = path_data.get('is_merged_path', False)
        represented_paths = path_data.get('represented_paths', [])
        
        if len(path_points) < 2:
            continue
        
        # Choose color based on completion status with much higher visibility
        if completed:
            # Use different colors for each completed path
            color_index = completed_path_index % len(completed_path_colors)
            path_color = completed_path_colors[color_index]
            completed_path_index += 1
            line_width = 10  # Much thicker for better visibility
            
            # If this is a merged path representing multiple breakpoints, make it extra thick
            if is_merged_path and len(represented_paths) > 1:
                line_width = 14  # Even thicker for merged paths
        else:
            path_color = (255, 100, 0)  # Bright orange-red for incomplete paths
            line_width = 8  # Thicker for incomplete paths
            
            # If this is a merged path representing multiple breakpoints, make it extra thick
            if is_merged_path and len(represented_paths) > 1:
                line_width = 12  # Even thicker for merged paths
        
        # Draw each segment individually based on its type
        for segment in path_segments:
            segment_type = segment.get('type', 'line')
            start_point = segment['start']
            end_point = segment['end']
            
            if segment_type == 'arc':
                # Draw arc segment - use edge_data that comes from intersection graph
                edge_data = segment.get('edge_data', {})
                
                if 'center' in edge_data and 'radius' in edge_data:
                    # Arc data from polygon exploration intersection graph
                    circle_center = edge_data['center']
                    circle_radius = edge_data['radius']
                    start_angle = edge_data.get('start_angle', 0)
                    end_angle = edge_data.get('end_angle', 0)
                    
                    # Trust the polygon exploration angle computation - no additional wrapping needed
                    # The polygon exploration module already handles angle differences correctly
                    original_end_angle = end_angle
                    
                    # Create rectangle for arc drawing
                    arc_rect = pygame.Rect(
                        int(circle_center[0] - circle_radius),
                        int(circle_center[1] - circle_radius),
                        int(2 * circle_radius),
                        int(2 * circle_radius)
                    )
                    
                    # Draw the arc with proper angle handling and enhanced visibility
                    try:
                        # Convert angles for pygame's flipped Y-axis coordinate system
                        # In pygame, Y increases downward, so we need to negate the angles
                        pygame_start_angle = -start_angle
                        pygame_end_angle = -end_angle
                        
                        # Ensure angles are in the correct order for pygame.draw.arc
                        if pygame_start_angle > pygame_end_angle:
                            pygame_start_angle, pygame_end_angle = pygame_end_angle, pygame_start_angle
                        
                        # Normalize angles to [0, 2π] range
                        pygame_start_angle = pygame_start_angle % (2 * math.pi)
                        pygame_end_angle = pygame_end_angle % (2 * math.pi)
                        
                        # Handle angle wrapping for pygame
                        if pygame_end_angle < pygame_start_angle:
                            pygame_end_angle += 2 * math.pi
                        
                        angle_span = pygame_end_angle - pygame_start_angle
                        
                        if abs(angle_span) > 0.01:  # Only draw if there's a meaningful arc
                            # Draw multiple arc lines for better visibility
                            pygame.draw.arc(screen, path_color, arc_rect, pygame_start_angle, pygame_end_angle, line_width)
                            # Add inner arc for contrast
                            inner_rect = pygame.Rect(
                                int(circle_center[0] - circle_radius + 2),
                                int(circle_center[1] - circle_radius + 2),
                                int(2 * (circle_radius - 2)),
                                int(2 * (circle_radius - 2))
                            )
                            if circle_radius > 5:  # Only if radius is large enough
                                pygame.draw.arc(screen, (255, 255, 255), inner_rect, pygame_start_angle, pygame_end_angle, max(2, line_width//3))
                    except:
                        # Fallback to line if arc drawing fails
                        int_start = (int(start_point[0]), int(start_point[1]))
                        int_end = (int(end_point[0]), int(end_point[1]))
                        pygame.draw.line(screen, path_color, int_start, int_end, line_width)
                else:
                    # Fallback: draw as line if arc data is missing with enhanced visibility
                    int_start = (int(start_point[0]), int(start_point[1]))
                    int_end = (int(end_point[0]), int(end_point[1]))
                    pygame.draw.line(screen, path_color, int_start, int_end, line_width)
                    # Add white outline for better contrast
                    pygame.draw.line(screen, (255, 255, 255), int_start, int_end, max(2, line_width//3))
            elif segment_type == 'circle_arc' and 'circle_center' in segment and 'circle_radius' in segment:
                # Legacy format support - draw arc segment with old format
                circle_center = segment['circle_center']
                circle_radius = segment['circle_radius']
                
                # Calculate start and end angles with pygame's flipped Y-axis in mind
                # In pygame, Y increases downward, so we need to adjust angle calculations
                start_angle = math.atan2(-(start_point[1] - circle_center[1]), start_point[0] - circle_center[0])
                end_angle = math.atan2(-(end_point[1] - circle_center[1]), end_point[0] - circle_center[0])
                
                # Normalize angles to [0, 2π]
                start_angle = start_angle % (2 * math.pi)
                end_angle = end_angle % (2 * math.pi)
                
                # Determine arc direction and angle span
                angle_diff = end_angle - start_angle
                
                # Handle angle wrapping - choose the shorter arc direction
                if angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                elif angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                
                # Create rectangle for arc drawing
                arc_rect = pygame.Rect(
                    int(circle_center[0] - circle_radius),
                    int(circle_center[1] - circle_radius),
                    int(2 * circle_radius),
                    int(2 * circle_radius)
                )
                
                # Draw the arc
                try:
                    if abs(angle_diff) > 0.01:  # Only draw if there's a meaningful arc
                        pygame.draw.arc(screen, path_color, arc_rect, start_angle, start_angle + angle_diff, line_width)
                except:
                    # Fallback to line if arc drawing fails
                    int_start = (int(start_point[0]), int(start_point[1]))
                    int_end = (int(end_point[0]), int(end_point[1]))
                    pygame.draw.line(screen, path_color, int_start, int_end, line_width)
            else:
                # Draw straight line segment with enhanced visibility
                int_start = (int(start_point[0]), int(start_point[1]))
                int_end = (int(end_point[0]), int(end_point[1]))
                # Draw main line
                pygame.draw.line(screen, path_color, int_start, int_end, line_width)
                # Add white center line for better contrast
                pygame.draw.line(screen, (255, 255, 255), int_start, int_end, max(2, line_width//3))
        
        # Draw enhanced start and end point markers
        if path_points:
            # Start point (bright yellow circle with multiple rings)
            start_pos = (int(path_points[0][0]), int(path_points[0][1]))
            pygame.draw.circle(screen, (255, 255, 0), start_pos, 15, 4)  # Large yellow ring
            pygame.draw.circle(screen, (0, 0, 0), start_pos, 11, 2)      # Black middle ring
            pygame.draw.circle(screen, (255, 255, 255), start_pos, 8)    # White center
            
            # End point - use same color as path for completed paths, red for incomplete
            end_pos = (int(path_points[-1][0]), int(path_points[-1][1]))
            end_color = path_color if completed else (255, 0, 0)
            pygame.draw.circle(screen, end_color, end_pos, 15, 4)        # Large colored ring
            pygame.draw.circle(screen, (0, 0, 0), end_pos, 11, 2)       # Black middle ring
            pygame.draw.circle(screen, (255, 255, 255), end_pos, 8)     # White center
        
        # Draw path points as larger, more visible circles
        for i, point in enumerate(path_points):
            int_point = (int(point[0]), int(point[1]))
            
            if i == 0:
                # Starting point - much larger circle with high contrast
                pygame.draw.circle(screen, (255, 255, 0), int_point, 12)  # Larger yellow circle
                pygame.draw.circle(screen, (0, 0, 0), int_point, 12, 3)  # Thick black border
                pygame.draw.circle(screen, (255, 255, 255), int_point, 6)  # White center
            elif i == len(path_points) - 1 and completed:
                # End point (same as start for completed polygons) - use path color
                pygame.draw.circle(screen, path_color, int_point, 11)  # Larger circle in path color
                pygame.draw.circle(screen, (0, 0, 0), int_point, 11, 3)  # Thick black border
                pygame.draw.circle(screen, (255, 255, 255), int_point, 5)  # White center
            else:
                # Intermediate points - more visible, use path color
                pygame.draw.circle(screen, path_color, int_point, 9)  # Larger intermediate points
                pygame.draw.circle(screen, (0, 0, 0), int_point, 9, 2)  # Black border
                pygame.draw.circle(screen, (255, 255, 255), int_point, 4)  # White center
        
        # Draw the breakoff point marker based on category
        if breakoff_line:
            start_point, end_point, gap_size, category = breakoff_line
            
            # For this function, we don't have direct access to agent position,
            # but we can use the path_points[0] as a reference since it should be
            # the starting point which is typically the outer breakoff point
            # Alternatively, we can make assumptions based on category naming
            
            if category in ['clockwise_near_far', 'counterclockwise_near_far']:
                # For near-to-far transitions, mark the far (outer) point
                # Assume end_point is the outer point based on algorithm design
                marker_point = end_point
            elif category in ['clockwise_far_near', 'counterclockwise_far_near']:
                # For far-to-near transitions, mark the near (inner) point
                # Assume start_point is the inner point based on algorithm design
                marker_point = start_point
            else:
                # Default: mark the outer point (end_point)
                marker_point = end_point
            
            marker_int = (int(marker_point[0]), int(marker_point[1]))
            
            # Draw a distinctive marker for the breakoff point
            # Use a different shape - a square marker
            breakoff_marker_color = (255, 100, 255)  # Light magenta
            square_size = 6
            square_rect = pygame.Rect(
                marker_int[0] - square_size, 
                marker_int[1] - square_size, 
                square_size * 2, 
                square_size * 2
            )
            pygame.draw.rect(screen, breakoff_marker_color, square_rect)
            pygame.draw.rect(screen, (0, 0, 0), square_rect, 2)  # Black border


def draw_path_links(screen, path_links):
    """
    Draw connections between linked exploration paths.
    Shows when one breakpoint's path contains another breakpoint.
    
    Args:
        screen: Pygame screen to draw on
        path_links: List of link dictionaries from detect_and_link_overlapping_paths()
    """
    if not path_links:
        return
    
    for link in path_links:
        from_point = link.get('from_breakpoint')
        to_point = link.get('to_breakpoint')
        link_type = link.get('link_type', 'containment')
        
        if not from_point or not to_point:
            continue
        
        # Convert to integer coordinates
        from_int = (int(from_point[0]), int(from_point[1]))
        to_int = (int(to_point[0]), int(to_point[1]))
        
        # Draw the connection line
        if link_type == 'mutual_containment':
            # Double dashed line for mutual containment links (stronger connection)
            link_color = (255, 255, 100)  # Bright yellow
            line_width = 5
            
            # Draw double dashed line by drawing multiple small segments with different offsets
            import math
            distance = math.sqrt((to_int[0] - from_int[0])**2 + (to_int[1] - from_int[1])**2)
            if distance > 0:
                dash_length = 10
                gap_length = 5
                total_dash_cycle = dash_length + gap_length
                num_cycles = int(distance / total_dash_cycle)
                
                dx = (to_int[0] - from_int[0]) / distance
                dy = (to_int[1] - from_int[1]) / distance
                
                # Draw main dashed line
                for i in range(num_cycles + 1):
                    dash_start_dist = i * total_dash_cycle
                    dash_end_dist = min(dash_start_dist + dash_length, distance)
                    
                    if dash_end_dist > dash_start_dist:
                        dash_start = (
                            int(from_int[0] + dx * dash_start_dist),
                            int(from_int[1] + dy * dash_start_dist)
                        )
                        dash_end = (
                            int(from_int[0] + dx * dash_end_dist),
                            int(from_int[1] + dy * dash_end_dist)
                        )
                        pygame.draw.line(screen, link_color, dash_start, dash_end, line_width)
        elif link_type == 'containment':
            # Single dashed line for one-way containment links
            link_color = (255, 255, 100)  # Bright yellow
            line_width = 4
            
            # Draw dashed line by drawing multiple small segments
            import math
            distance = math.sqrt((to_int[0] - from_int[0])**2 + (to_int[1] - from_int[1])**2)
            if distance > 0:
                dash_length = 8
                gap_length = 6
                total_dash_cycle = dash_length + gap_length
                num_cycles = int(distance / total_dash_cycle)
                
                dx = (to_int[0] - from_int[0]) / distance
                dy = (to_int[1] - from_int[1]) / distance
                
                for i in range(num_cycles + 1):
                    dash_start_dist = i * total_dash_cycle
                    dash_end_dist = min(dash_start_dist + dash_length, distance)
                    
                    if dash_end_dist > dash_start_dist:
                        dash_start = (
                            int(from_int[0] + dx * dash_start_dist),
                            int(from_int[1] + dy * dash_start_dist)
                        )
                        dash_end = (
                            int(from_int[0] + dx * dash_end_dist),
                            int(from_int[1] + dy * dash_end_dist)
                        )
                        pygame.draw.line(screen, link_color, dash_start, dash_end, line_width)
        
        # Draw connection markers at both ends
        marker_radius = 8
        marker_color = (255, 255, 100)  # Bright yellow
        border_color = (0, 0, 0)  # Black border
        
        # From point marker (circle)
        pygame.draw.circle(screen, marker_color, from_int, marker_radius)
        pygame.draw.circle(screen, border_color, from_int, marker_radius, 2)
        
        # To point marker (triangle)
        triangle_size = marker_radius
        triangle_points = [
            (to_int[0], to_int[1] - triangle_size),  # Top
            (to_int[0] - triangle_size, to_int[1] + triangle_size),  # Bottom left
            (to_int[0] + triangle_size, to_int[1] + triangle_size)   # Bottom right
        ]
        pygame.draw.polygon(screen, marker_color, triangle_points)
        pygame.draw.polygon(screen, border_color, triangle_points, 2)


def draw_measurement_line(screen, start_pos, end_pos, font, distance_px=None):
    """
    Draw a measurement line with distance information.
    
    Args:
        screen: Pygame screen to draw on
        start_pos: (x, y) starting position
        end_pos: (x, y) ending position  
        font: Font for text rendering
        distance_px: Optional precalculated distance in pixels
    """
    if start_pos is None or end_pos is None:
        return
    
    # Calculate distance if not provided
    if distance_px is None:
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance_px = math.sqrt(dx * dx + dy * dy)
    
    # Draw the measurement line
    line_color = (255, 255, 0)  # Yellow
    line_width = 2
    pygame.draw.line(screen, line_color, start_pos, end_pos, line_width)
    
    # Draw start and end markers
    marker_radius = 6
    start_color = (0, 255, 0)  # Green
    end_color = (255, 0, 0)    # Red
    
    pygame.draw.circle(screen, start_color, start_pos, marker_radius)
    pygame.draw.circle(screen, (0, 0, 0), start_pos, marker_radius, 2)  # Black border
    
    pygame.draw.circle(screen, end_color, end_pos, marker_radius)
    pygame.draw.circle(screen, (0, 0, 0), end_pos, marker_radius, 2)  # Black border
    
    # Calculate midpoint for text placement
    mid_x = (start_pos[0] + end_pos[0]) // 2
    mid_y = (start_pos[1] + end_pos[1]) // 2
    
    # Format distance text
    distance_text = f"{distance_px:.1f} px"
    
    # Estimate world units (assuming roughly 1 pixel = 0.1 world units based on typical game scales)
    world_distance = distance_px * 0.1
    world_text = f"({world_distance:.1f} units)"
    
    # Render text with background for visibility
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)         # Black background
    
    # Create text surfaces
    dist_surface = font.render(distance_text, True, text_color)
    world_surface = font.render(world_text, True, text_color)
    
    # Calculate text positions (offset from midpoint to avoid line)
    text_offset_y = -25
    dist_rect = dist_surface.get_rect(center=(mid_x, mid_y + text_offset_y))
    world_rect = world_surface.get_rect(center=(mid_x, mid_y + text_offset_y + 15))
    
    # Draw text backgrounds
    bg_margin = 2
    dist_bg_rect = dist_rect.inflate(bg_margin * 2, bg_margin * 2)
    world_bg_rect = world_rect.inflate(bg_margin * 2, bg_margin * 2)
    
    pygame.draw.rect(screen, bg_color, dist_bg_rect)
    pygame.draw.rect(screen, bg_color, world_bg_rect)
    
    # Draw text
    screen.blit(dist_surface, dist_rect)
    screen.blit(world_surface, world_rect)

def draw_reachability_overlay(screen, reachability_data, agent_x, agent_y, alpha=0.6):
    """
    Draw the reachability heatmap overlay as a transparent background.
    
    Args:
        screen: Pygame screen to draw on
        reachability_data: Tuple of (reachability_mask, world_bounds) from overlay API
        agent_x, agent_y: Agent position (center of the reachability data)
        alpha: Transparency level (default: 0.6)
    """
    if not reachability_data:
        return
    
    reachability_mask, world_bounds = reachability_data
    
    # Extract world coordinate bounds
    if isinstance(world_bounds, dict):
        # Transform bounds from reachability grid coordinates to world coordinates
        grid_x_min = world_bounds['x_min']
        grid_x_max = world_bounds['x_max'] 
        grid_y_min = world_bounds['y_min']
        grid_y_max = world_bounds['y_max']
        
        # Translate the bounds to be centered at the agent position
        world_x_min = agent_x + grid_x_min
        world_x_max = agent_x + grid_x_max
        world_y_min = agent_y + grid_y_min
        world_y_max = agent_y + grid_y_max
    else:
        # Fallback if bounds are in list/tuple format
        world_x_min, world_y_min, world_x_max, world_y_max = world_bounds
    
    # Create a surface for the heatmap
    width = int(world_x_max - world_x_min)
    height = int(world_y_max - world_y_min)
    
    if width <= 0 or height <= 0:
        return
    
    # Create the heatmap surface
    heatmap_surface = pygame.Surface((width, height))
    heatmap_surface.set_alpha(int(alpha * 255))
    
    # Convert numpy array to pygame surface with hot colormap
    # Scale values from 0-1 to 0-255
    scaled_mask = (reachability_mask * 255).astype(np.uint8)
    
    # Apply hot colormap (red-yellow for high reachability)
    colored_mask = np.zeros((scaled_mask.shape[0], scaled_mask.shape[1], 3), dtype=np.uint8)
    
    for i in range(scaled_mask.shape[0]):
        for j in range(scaled_mask.shape[1]):
            value = scaled_mask[i, j]
            if value > 0:
                # Hot colormap: black -> red -> yellow -> white
                if value < 85:  # Black to red
                    colored_mask[i, j] = [value * 3, 0, 0]
                elif value < 170:  # Red to yellow
                    colored_mask[i, j] = [255, (value - 85) * 3, 0]
                else:  # Yellow to white
                    colored_mask[i, j] = [255, 255, (value - 170) * 3]
    
    # Create surface from the colored array
    pygame.surfarray.blit_array(heatmap_surface, colored_mask.swapaxes(0, 1))
    
    # Draw the heatmap on screen
    screen.blit(heatmap_surface, (int(world_x_min), int(world_y_min)))

def draw_path_analysis_data(screen, path_analysis_data):
    """
    Draw path analysis data including first edges and orientations.
    
    Args:
        screen: Pygame screen to draw on
        path_analysis_data: List of path analysis dictionaries from overlay API
    """
    if not path_analysis_data:
        return
    
    # Colors for different path analysis elements
    first_edge_colors = ['cyan', 'gold', 'lime', 'hotpink', 'turquoise', 'yellow', 'lightgreen', 'orange']
    
    for i, path_info in enumerate(path_analysis_data):
        first_edge = path_info.get('first_edge')
        orientation = path_info.get('orientation')
        target_point = path_info.get('target_point')
        
        if first_edge is None:
            continue
            
        edge_color_name = first_edge_colors[i % len(first_edge_colors)]
        # Convert color name to RGB
        color_map = {
            'cyan': (0, 255, 255),
            'gold': (255, 215, 0),
            'lime': (0, 255, 0),
            'hotpink': (255, 105, 180),
            'turquoise': (64, 224, 208),
            'yellow': (255, 255, 0),
            'lightgreen': (144, 238, 144),
            'orange': (255, 165, 0)
        }
        edge_color = color_map.get(edge_color_name, (255, 255, 255))
        
        # Draw first edge
        if first_edge['type'] == 'line':
            start = first_edge['start']
            end = first_edge['end']
            pygame.draw.line(screen, edge_color, 
                           (int(start[0]), int(start[1])), 
                           (int(end[0]), int(end[1])), 6)
            
            # Add start and end markers
            pygame.draw.circle(screen, edge_color, (int(start[0]), int(start[1])), 8)
            pygame.draw.circle(screen, (0, 0, 0), (int(start[0]), int(start[1])), 8, 2)
            pygame.draw.circle(screen, edge_color, (int(end[0]), int(end[1])), 8)
            pygame.draw.circle(screen, (0, 0, 0), (int(end[0]), int(end[1])), 8, 2)
                   
        elif first_edge['type'] == 'arc':
            # Draw arc edge (simplified as line between start and end points)
            start = first_edge.get('start')
            end = first_edge.get('end')
            if start and end:
                pygame.draw.line(screen, edge_color, 
                               (int(start[0]), int(start[1])), 
                               (int(end[0]), int(end[1])), 6)
                
                # Add start and end markers
                pygame.draw.circle(screen, edge_color, (int(start[0]), int(start[1])), 8)
                pygame.draw.circle(screen, (0, 0, 0), (int(start[0]), int(start[1])), 8, 2)
                pygame.draw.circle(screen, edge_color, (int(end[0]), int(end[1])), 8)
                pygame.draw.circle(screen, (0, 0, 0), (int(end[0]), int(end[1])), 8, 2)
        
        # Draw orientation arrow if available
        if orientation is not None and target_point is not None:
            arrow_start_x = target_point[0]
            arrow_start_y = target_point[1]
            
            # Calculate arrow end position using orientation
            arrow_length = 40
            arrow_end_x = arrow_start_x + arrow_length * math.cos(orientation)
            arrow_end_y = arrow_start_y + arrow_length * math.sin(orientation)
            
            # Draw orientation arrow
            pygame.draw.line(screen, edge_color, 
                           (int(arrow_start_x), int(arrow_start_y)),
                           (int(arrow_end_x), int(arrow_end_y)), 4)
            
            # Draw arrowhead
            arrowhead_length = 10
            arrowhead_angle = math.pi / 6  # 30 degrees
            
            # Calculate arrowhead points
            arrowhead1_x = arrow_end_x - arrowhead_length * math.cos(orientation - arrowhead_angle)
            arrowhead1_y = arrow_end_y - arrowhead_length * math.sin(orientation - arrowhead_angle)
            arrowhead2_x = arrow_end_x - arrowhead_length * math.cos(orientation + arrowhead_angle)
            arrowhead2_y = arrow_end_y - arrowhead_length * math.sin(orientation + arrowhead_angle)
            
            # Draw arrowhead lines
            pygame.draw.line(screen, edge_color, 
                           (int(arrow_end_x), int(arrow_end_y)),
                           (int(arrowhead1_x), int(arrowhead1_y)), 4)
            pygame.draw.line(screen, edge_color, 
                           (int(arrow_end_x), int(arrow_end_y)),
                           (int(arrowhead2_x), int(arrowhead2_y)), 4)
            
            # Add small circle at arrow start
            pygame.draw.circle(screen, edge_color, (int(arrow_start_x), int(arrow_start_y)), 6)
            pygame.draw.circle(screen, (0, 0, 0), (int(arrow_start_x), int(arrow_start_y)), 6, 1)
        
        # Add text annotation for first edge identification
        if first_edge.get('start') and first_edge.get('end'):
            mid_x = (first_edge['start'][0] + first_edge['end'][0]) / 2
            mid_y = (first_edge['start'][1] + first_edge['end'][1]) / 2
            
            # Create text surface
            font = pygame.font.SysFont('Arial', 12, bold=True)
            text = f'1st-{i+1}'
            text_surface = font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(int(mid_x), int(mid_y)))
            
            # Draw background
            bg_rect = text_rect.inflate(6, 4)
            pygame.draw.rect(screen, edge_color, bg_rect)
            pygame.draw.rect(screen, (0, 0, 0), bg_rect, 1)
            
            # Draw text
            screen.blit(text_surface, text_rect)

def draw_sample_points(screen, sample_points_data):
    """
    Draw sample points where reachability values were evaluated.
    
    Args:
        screen: Pygame screen to draw on
        sample_points_data: List of sample point dictionaries from overlay API
    """
    if not sample_points_data:
        return
    
    for i, sample_info in enumerate(sample_points_data):
        sample_x, sample_y = sample_info['sample_point']
        target_value = sample_info['target_value']
        
        # Calculate boosted value (50% increase, capped at 1.0) to match processing
        boosted_value = min(1.0, target_value * 1.5)
        
        # Choose color based on boosted value (heat map style)
        if boosted_value > 0.7:
            sample_color = (255, 0, 0)  # Red
        elif boosted_value > 0.4:
            sample_color = (255, 165, 0)  # Orange
        elif boosted_value > 0.1:
            sample_color = (255, 255, 0)  # Yellow
        else:
            sample_color = (173, 216, 230)  # Light blue
        
        # Draw sample point with distinctive diamond marker
        points = [
            (int(sample_x), int(sample_y - 12)),  # Top
            (int(sample_x + 12), int(sample_y)),  # Right
            (int(sample_x), int(sample_y + 12)),  # Bottom
            (int(sample_x - 12), int(sample_y))   # Left
        ]
        pygame.draw.polygon(screen, sample_color, points)
        pygame.draw.polygon(screen, (0, 0, 0), points, 2)
        
        # Add sample value annotation
        font = pygame.font.SysFont('Arial', 10, bold=True)
        text = f'{target_value:.2f}→{boosted_value:.2f}'
        text_surface = font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(int(sample_x + 20), int(sample_y - 15)))
        
        # Draw background
        bg_rect = text_rect.inflate(4, 2)
        pygame.draw.rect(screen, (255, 255, 255), bg_rect)
        pygame.draw.rect(screen, (0, 0, 0), bg_rect, 1)
        
        # Draw text
        screen.blit(text_surface, text_rect)

