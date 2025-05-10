"""
Information sidebar module for displaying text and statistics independently.
This module provides a threaded approach to displaying information on a sidebar
and statistics without blocking the main simulation.
"""

import pygame
import threading
import time
import queue
from multitrack.utils.config import *

class InformationSidebarThread:
    """
    A thread-based information sidebar system for displaying information
    without blocking the main simulation thread.
    """
    def __init__(self, screen_width, screen_height, sidebar_width=250):
        """
        Initialize the information sidebar thread.
        
        Args:
            screen_width: Width of the main screen (environment)
            screen_height: Height of the screen
            sidebar_width: Width of the sidebar (default: 250 pixels)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sidebar_width = sidebar_width
        self.running = False
        self.thread = None
        self.data_queue = queue.Queue()
        self.sidebar_surface = None
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
        """Start the sidebar thread"""
        if self.thread is not None and self.thread.is_alive():
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the sidebar thread"""
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
        
        # Create sidebar surface - only for the sidebar area
        self.sidebar_surface = pygame.Surface((self.sidebar_width, self.screen_height), pygame.SRCALPHA)
        
        # Main thread loop
        while self.running:
            # Process any pending data updates
            self._process_queue()
            
            # Check if it's time to update the sidebar
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                self._render_sidebar()
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
            
    def _render_sidebar(self):
        """Render the sidebar information"""
        # Fill sidebar with a semi-transparent dark background
        self.sidebar_surface.fill((30, 30, 30, 230))
        
        # Add a thin border on the left side
        pygame.draw.line(self.sidebar_surface, (80, 80, 80), (0, 0), (0, self.screen_height), 2)
        
        current_y = 10  # Starting y position
        
        # Draw title at the top
        if self.title_text:
            title = self.font_title.render(self.title_text, True, (255, 255, 255))
            title_x = self.sidebar_width // 2 - title.get_width() // 2
            self.sidebar_surface.blit(title, (title_x, current_y))
            current_y += 35  # Move down for next element
            
            # Add separator line
            pygame.draw.line(self.sidebar_surface, (100, 100, 100), 
                            (10, current_y), (self.sidebar_width - 10, current_y), 1)
            current_y += 15
        
        # Draw key state monitoring if enabled
        if self.key_debug and self.keys:
            key_monitor_text = [
                "KEY STATE MONITOR:",
                f"Arrow keys: UP={self.keys[pygame.K_UP]} DOWN={self.keys[pygame.K_DOWN]} LEFT={self.keys[pygame.K_LEFT]} RIGHT={self.keys[pygame.K_RIGHT]}",
                f"WASD keys: W={self.keys[pygame.K_w]} A={self.keys[pygame.K_a]} S={self.keys[pygame.K_s]} D={self.keys[pygame.K_d]}",
                f"Function keys: K={self.keys[pygame.K_k]} U={self.keys[pygame.K_u]} T={self.keys[pygame.K_t]} M={self.keys[pygame.K_m]} C={self.keys[pygame.K_c]} V={self.keys[pygame.K_v]} F={self.keys[pygame.K_f]}",
                f"Special keys: PLUS={self.keys[pygame.K_PLUS] if pygame.K_PLUS in self.keys else self.keys[pygame.K_EQUALS]} EQUALS={self.keys[pygame.K_EQUALS]} MINUS={self.keys[pygame.K_MINUS]} SHIFT={bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)}"
            ]
            
            # Draw key state text
            for i, text in enumerate(key_monitor_text):
                key_text = self.font_normal.render(text, True, (255, 255, 255))
                
                # Handle text that's too wide for the sidebar by wrapping or clipping
                if key_text.get_width() > self.sidebar_width - 20:
                    # Simple clipping solution - you could implement text wrapping here
                    key_text = self.font_normal.render(text[:30] + "...", True, (255, 255, 255))
                
                self.sidebar_surface.blit(key_text, (10, current_y))
                current_y += 20
            
            current_y += 10  # Add some space after the key monitor
            # Add separator line
            pygame.draw.line(self.sidebar_surface, (100, 100, 100), 
                           (10, current_y), (self.sidebar_width - 10, current_y), 1)
            current_y += 15
        
        # Display FPS if enabled
        if self.show_fps:
            fps_text = f"FPS: {self.fps:.1f} | Frame time: {self.avg_frame_time*1000:.1f}ms"
            fps_surf = self.font_normal.render(fps_text, True, (200, 200, 100))
            self.sidebar_surface.blit(fps_surf, (10, current_y))
            current_y += 25
        
        # Draw text information 
        if self.info_text:
            for i, text in enumerate(self.info_text):
                text_surf = self.font_normal.render(text, True, (255, 255, 255))
                
                # Handle text that's too wide for the sidebar
                if text_surf.get_width() > self.sidebar_width - 20:
                    # Simple clipping solution
                    text_surf = self.font_normal.render(text[:30] + "...", True, (255, 255, 255))
                
                self.sidebar_surface.blit(text_surf, (10, current_y))
                current_y += 20
                
                # If we're running out of space in the sidebar, stop rendering
                if current_y > self.screen_height - 10:
                    break
                
    def update_data(self, **kwargs):
        """
        Update sidebar data from the main thread.
        
        Args:
            **kwargs: Keyword arguments containing data to update
        """
        # Put data in the queue for the sidebar thread to process
        self.data_queue.put(kwargs)
        
    def get_surface(self):
        """
        Get the current sidebar surface to blit onto the main screen.
        
        Returns:
            The sidebar surface, or None if not initialized
        """
        return self.sidebar_surface
        
    def get_width(self):
        """
        Get the width of the sidebar.
        
        Returns:
            The width of the sidebar in pixels
        """
        return self.sidebar_width