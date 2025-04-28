"""
Environment module for managing walls, doors, windows, and other elements
for the unicycle reachability simulation.
"""

import pygame
from multitrack.utils.config import *

class SimulationEnvironment:
    """Class to manage the indoor environment with walls, doors, and windows."""
    
    def __init__(self, width=WIDTH, height=HEIGHT):
        self.width = width
        self.height = height
        
        # Scale factor compared to original environment (1400x1000)
        scale_x = width / 1400
        scale_y = height / 1000
        
        # Room layout: outer walls
        self.outer_walls = [
            pygame.Rect(int(50 * scale_x), int(50 * scale_y), 
                       int(1300 * scale_x), int(10 * scale_y)),     # Top wall
            pygame.Rect(int(50 * scale_x), int(50 * scale_y), 
                       int(10 * scale_x), int(900 * scale_y)),      # Left wall
            pygame.Rect(int(1340 * scale_x), int(50 * scale_y), 
                       int(10 * scale_x), int(900 * scale_y)),    # Right wall
            pygame.Rect(int(50 * scale_x), int(940 * scale_y), 
                       int(1300 * scale_x), int(10 * scale_y))     # Bottom wall
        ]
        
        # Inner walls to create rooms - original area preserved
        self.inner_walls = [
            # Original house section (unchanged rooms)
            pygame.Rect(int(50 * scale_x), int(300 * scale_y), 
                       int(400 * scale_x), int(10 * scale_y)),     # Upper left room divider
            pygame.Rect(int(550 * scale_x), int(300 * scale_y), 
                       int(400 * scale_x), int(10 * scale_y)),    # Upper right room divider
            pygame.Rect(int(300 * scale_x), int(500 * scale_y), 
                       int(500 * scale_x), int(10 * scale_y)),    # Lower middle room divider
            pygame.Rect(int(450 * scale_x), int(50 * scale_y), 
                       int(10 * scale_x), int(250 * scale_y)),     # Upper room divider
            pygame.Rect(int(700 * scale_x), int(50 * scale_y), 
                       int(10 * scale_x), int(250 * scale_y)),     # Upper right room divider
            pygame.Rect(int(300 * scale_x), int(300 * scale_y), 
                       int(10 * scale_x), int(200 * scale_y)),     # Middle left room divider
            pygame.Rect(int(550 * scale_x), int(500 * scale_y), 
                       int(10 * scale_x), int(240 * scale_y)),    # Lower right room divider
            
            # New wing dividers
            pygame.Rect(int(950 * scale_x), int(50 * scale_y), 
                       int(10 * scale_x), int(450 * scale_y)),     # Main vertical divider - upper section
            pygame.Rect(int(950 * scale_x), int(600 * scale_y), 
                       int(10 * scale_x), int(340 * scale_y)),    # Main vertical divider - lower section
            pygame.Rect(int(950 * scale_x), int(300 * scale_y), 
                       int(400 * scale_x), int(10 * scale_y)),    # Upper divider in new wing
            pygame.Rect(int(950 * scale_x), int(500 * scale_y), 
                       int(400 * scale_x), int(10 * scale_y)),    # Middle divider in new wing
            pygame.Rect(int(950 * scale_x), int(700 * scale_y), 
                       int(400 * scale_x), int(10 * scale_y)),    # Lower divider in new wing
            pygame.Rect(int(1150 * scale_x), int(300 * scale_y), 
                       int(10 * scale_x), int(200 * scale_y)),   # Upper vertical divider in new wing
            pygame.Rect(int(1150 * scale_x), int(500 * scale_y), 
                       int(10 * scale_x), int(200 * scale_y)),   # Lower vertical divider in new wing
        ]
        
        # Doors (gaps in walls)
        self.doors = [
            # Original house doors (unchanged)
            pygame.Rect(int(480 * scale_x), int(300 * scale_y), 
                       int(70 * scale_x), int(10 * scale_y)),     # Door between bedrooms
            pygame.Rect(int(300 * scale_x), int(380 * scale_y), 
                       int(10 * scale_x), int(70 * scale_y)),     # Door to living room
            pygame.Rect(int(550 * scale_x), int(600 * scale_y), 
                       int(10 * scale_x), int(70 * scale_y)),     # Door to bathroom
            pygame.Rect(int(200 * scale_x), int(300 * scale_y), 
                       int(70 * scale_x), int(10 * scale_y)),     # Door in upper left
            pygame.Rect(int(650 * scale_x), int(300 * scale_y), 
                       int(70 * scale_x), int(10 * scale_y)),     # Door in upper right
            pygame.Rect(int(420 * scale_x), int(500 * scale_y), 
                       int(70 * scale_x), int(10 * scale_y)),     # Door in lower middle
            pygame.Rect(int(650 * scale_x), int(500 * scale_y), 
                       int(70 * scale_x), int(10 * scale_y)),     # Door connecting rooms
            pygame.Rect(int(450 * scale_x), int(150 * scale_y), 
                       int(10 * scale_x), int(70 * scale_y)),     # Door in upper vertical
            pygame.Rect(int(700 * scale_x), int(150 * scale_y), 
                       int(10 * scale_x), int(70 * scale_y)),     # Door in upper right vertical
            
            # Connection doors between original house and new wing
            pygame.Rect(int(950 * scale_x), int(150 * scale_y), 
                       int(10 * scale_x), int(70 * scale_y)),     # Door to upper new wing from original house
            pygame.Rect(int(950 * scale_x), int(400 * scale_y), 
                       int(10 * scale_x), int(70 * scale_y)),     # Door to middle new wing from original house
            pygame.Rect(int(950 * scale_x), int(800 * scale_y), 
                       int(10 * scale_x), int(70 * scale_y)),     # Door to lower new wing
            
            # New wing doors
            pygame.Rect(int(1050 * scale_x), int(300 * scale_y), 
                       int(70 * scale_x), int(10 * scale_y)),    # Door in upper new wing
            pygame.Rect(int(1050 * scale_x), int(500 * scale_y), 
                       int(70 * scale_x), int(10 * scale_y)),    # Door in middle new wing
            pygame.Rect(int(1050 * scale_x), int(700 * scale_y), 
                       int(70 * scale_x), int(10 * scale_y)),    # Door in lower new wing
            pygame.Rect(int(1150 * scale_x), int(400 * scale_y), 
                       int(10 * scale_x), int(70 * scale_y)),    # Door in upper vertical new wing
            pygame.Rect(int(1150 * scale_x), int(600 * scale_y), 
                       int(10 * scale_x), int(70 * scale_y)),    # Door in lower vertical new wing
        ]
        
        # Windows on outer walls
        self.windows = [
            # Original windows
            pygame.Rect(int(250 * scale_x), int(50 * scale_y), 
                       int(80 * scale_x), int(10 * scale_y)),      # Top wall window
            pygame.Rect(int(750 * scale_x), int(50 * scale_y), 
                       int(80 * scale_x), int(10 * scale_y)),      # Top wall window
            pygame.Rect(int(50 * scale_x), int(200 * scale_y), 
                       int(10 * scale_x), int(80 * scale_y)),      # Left wall window
            pygame.Rect(int(50 * scale_x), int(600 * scale_y), 
                       int(10 * scale_x), int(80 * scale_y)),      # Left wall window
            
            # New windows
            pygame.Rect(int(1100 * scale_x), int(50 * scale_y), 
                       int(80 * scale_x), int(10 * scale_y)),     # Top wall window in new wing
            pygame.Rect(int(1340 * scale_x), int(200 * scale_y), 
                       int(10 * scale_x), int(80 * scale_y)),    # Right wall window
            pygame.Rect(int(1340 * scale_x), int(500 * scale_y), 
                       int(10 * scale_x), int(80 * scale_y)),    # Right wall window
            pygame.Rect(int(1340 * scale_x), int(800 * scale_y), 
                       int(10 * scale_x), int(80 * scale_y)),    # Right wall window
            pygame.Rect(int(300 * scale_x), int(940 * scale_y), 
                       int(80 * scale_x), int(10 * scale_y)),     # Bottom wall window
            pygame.Rect(int(800 * scale_x), int(940 * scale_y), 
                       int(80 * scale_x), int(10 * scale_y)),     # Bottom wall window
            pygame.Rect(int(1200 * scale_x), int(940 * scale_y), 
                       int(80 * scale_x), int(10 * scale_y)),    # Bottom wall window
        ]
        
        # Combined walls list for collision detection
        self.all_walls = self.outer_walls + self.inner_walls
        
        # Room labels and positions
        self.room_labels = [
            # Original rooms (unchanged)
            {"text": "Bedroom", "pos": (int(150 * scale_x), int(120 * scale_y))},
            {"text": "Kitchen", "pos": (int(600 * scale_x), int(120 * scale_y))},
            {"text": "Living Room", "pos": (int(310 * scale_x), int(380 * scale_y))},
            {"text": "Study", "pos": (int(120 * scale_x), int(450 * scale_y))},
            {"text": "Bathroom", "pos": (int(520 * scale_x), int(600 * scale_y))},
            
            # New rooms in expanded wing
            {"text": "Dining Room", "pos": (int(800 * scale_x), int(150 * scale_y))},
            {"text": "Master Bedroom", "pos": (int(1050 * scale_x), int(150 * scale_y))},
            {"text": "Office", "pos": (int(1250 * scale_x), int(150 * scale_y))},
            {"text": "Game Room", "pos": (int(1050 * scale_x), int(400 * scale_y))},
            {"text": "Garage", "pos": (int(1250 * scale_x), int(400 * scale_y))},
            {"text": "Library", "pos": (int(1050 * scale_x), int(600 * scale_y))},
            {"text": "Laundry", "pos": (int(1250 * scale_x), int(600 * scale_y))},
            {"text": "Storage", "pos": (int(1050 * scale_x), int(800 * scale_y))}
        ]
    
    def draw(self, surface, font=None):
        """Draw all environment elements to the provided surface."""
        # Set default font if none provided
        if font is None:
            font = pygame.font.SysFont(None, 20)
        
        # Draw floor base color
        floor_color = (240, 230, 210)  # Beige floor
        surface.fill(floor_color)
        
        # Draw inner walls
        wall_color = (120, 120, 120)  # Gray walls
        for wall in self.inner_walls:
            pygame.draw.rect(surface, wall_color, wall)
            
        # Draw outer walls
        for wall in self.outer_walls:
            pygame.draw.rect(surface, wall_color, wall)
            
        # Draw windows
        window_color = (173, 216, 230)  # Light blue windows
        for window in self.windows:
            pygame.draw.rect(surface, window_color, window)
            
        # Draw doors
        door_color = (210, 180, 140)  # Tan door
        for door in self.doors:
            pygame.draw.rect(surface, door_color, door)
        
        # Draw room labels if the font is provided
        black = (0, 0, 0)
        for label in self.room_labels:
            text_surface = font.render(label["text"], True, black)
            surface.blit(text_surface, label["pos"])
    
    def get_all_walls(self):
        """Get list of all walls for collision detection."""
        return self.all_walls

    def get_doors(self):
        """Get list of all doors."""
        return self.doors

    def get_windows(self):
        """Get list of all windows."""
        return self.windows