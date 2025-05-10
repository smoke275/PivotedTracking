"""
Information overlay module for displaying text and statistics independently.
This module provides a threaded approach to displaying information overlays
and statistics without blocking the main simulation.
"""

import pygame
import threading
import time
import queue
from multitrack.utils.config import *

class InformationOverlayThread:
    """
    A thread-based information overlay system for displaying information
    without blocking the main simulation thread.
    """
    def __init__(self, screen_width, screen_height):
        """
        Initialize the information overlay thread.
        
        Args:
            screen_width: Width of the screen
            screen_height: Height of the screen
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.running = False
        self.thread = None
        self.data_queue = queue.Queue()
        self.overlay_surface = None
        self.font_normal = None
        self.font_title = None
        self.font_stats = None
        self.last_update_time = 0
        self.update_interval = 1/30  # 30 updates per second limit
        
        # Default data - will be updated by main thread
        self.info_text = []
        self.title_text = "Visitor with Escort Simulation"
        self.mppi_stats = None
        self.key_debug = False
        self.keys = None  # Will store key state
        self.show_fps = True
        self.fps = 0
        self.avg_frame_time = 0
        
    def start(self):
        """Start the overlay thread"""
        if self.thread is not None and self.thread.is_alive():
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the overlay thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            
    def _run(self):
        """Main thread function"""
        # Initialize pygame font module if needed
        if not pygame.font.get_init():
            pygame.font.init()
            
        # Create fonts
        self.font_normal = pygame.font.SysFont('Arial', 16)
        self.font_title = pygame.font.SysFont('Arial', 20)
        self.font_stats = pygame.font.SysFont('Arial', 20)
        
        # Create overlay surface
        self.overlay_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        
        # Main thread loop
        while self.running:
            # Process any pending data updates
            self._process_queue()
            
            # Check if it's time to update the overlay
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                self._render_overlay()
                self.last_update_time = current_time
                
            # Small sleep to prevent 100% CPU usage
            time.sleep(0.005)
            
    def _process_queue(self):
        """Process data updates from the queue"""
        try:
            # Don't block, just check if there's any data
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                
                # Update local data based on the received data
                for key, value in data.items():
                    setattr(self, key, value)
                    
                self.data_queue.task_done()
        except queue.Empty:
            pass
            
    def _render_overlay(self):
        """Render the overlay information"""
        # Clear overlay surface with a completely transparent background
        self.overlay_surface.fill((0, 0, 0, 0))
        
        # Draw text information at the bottom
        if self.info_text:
            # Calculate text panel height
            text_panel_height = len(self.info_text) * 20 + 10  # 20px per line + 10px padding
            
            # Create a semi-transparent background surface with proper alpha channel
            info_bg = pygame.Surface((self.screen_width, text_panel_height), pygame.SRCALPHA)
            info_bg.fill((0, 0, 0, 120))  # Black with 120/255 alpha (semi-transparent)
            self.overlay_surface.blit(info_bg, (0, self.screen_height - text_panel_height))
            
            # Draw text lines
            for i, text in enumerate(self.info_text):
                text_surf = self.font_normal.render(text, True, (255, 255, 255))
                self.overlay_surface.blit(text_surf, (10, self.screen_height - text_panel_height + 5 + i*20))
        
        # Draw title at the top with semi-transparent background
        if self.title_text:
            title = self.font_title.render(self.title_text, True, (255, 255, 255))
            # Create small semi-transparent background for the title
            title_bg_width = title.get_width() + 20
            title_bg = pygame.Surface((title_bg_width, 30), pygame.SRCALPHA)
            title_bg.fill((0, 0, 0, 80))  # Very light black background
            title_bg_x = self.screen_width//2 - title_bg_width//2
            self.overlay_surface.blit(title_bg, (title_bg_x, 5))
            self.overlay_surface.blit(title, (self.screen_width//2 - title.get_width()//2, 10))
        
        # Draw key state monitoring if enabled
        if self.key_debug and self.keys:
            key_monitor_text = [
                "KEY STATE MONITOR:",
                f"Arrow keys: UP={self.keys[pygame.K_UP]} DOWN={self.keys[pygame.K_DOWN]} LEFT={self.keys[pygame.K_LEFT]} RIGHT={self.keys[pygame.K_RIGHT]}",
                f"WASD keys: W={self.keys[pygame.K_w]} A={self.keys[pygame.K_a]} S={self.keys[pygame.K_s]} D={self.keys[pygame.K_d]}",
                f"Function keys: K={self.keys[pygame.K_k]} U={self.keys[pygame.K_u]} T={self.keys[pygame.K_t]} M={self.keys[pygame.K_m]} C={self.keys[pygame.K_c]} V={self.keys[pygame.K_v]} F={self.keys[pygame.K_f]}",
                f"Special keys: PLUS={self.keys[pygame.K_PLUS] if pygame.K_PLUS in self.keys else self.keys[pygame.K_EQUALS]} EQUALS={self.keys[pygame.K_EQUALS]} MINUS={self.keys[pygame.K_MINUS]} SHIFT={bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)}"
            ]
            
            # Create semi-transparent background for key monitor with proper alpha
            key_bg = pygame.Surface((400, 120), pygame.SRCALPHA)
            key_bg.fill((50, 50, 50, 180))  # Semi-transparent gray
            self.overlay_surface.blit(key_bg, (20, 20))
            
            # Draw key state text
            for i, text in enumerate(key_monitor_text):
                key_text = self.font_normal.render(text, True, (255, 255, 255))
                self.overlay_surface.blit(key_text, (30, 30 + i*20))
                
    def update_data(self, **kwargs):
        """
        Update overlay data from the main thread.
        
        Args:
            **kwargs: Keyword arguments containing data to update
        """
        # Put data in the queue for the overlay thread to process
        self.data_queue.put(kwargs)
        
    def get_surface(self):
        """
        Get the current overlay surface to blit onto the main screen.
        
        Returns:
            The overlay surface, or None if not initialized
        """
        return self.overlay_surface