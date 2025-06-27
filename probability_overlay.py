#!/usr/bin/env python3
"""
Probability Distribution Overlay for Pygame

This module creates a transparent overlay showing the distribution of combined probabilities
directly on the pygame window. It displays a simple histogram-like visualization that updates
in real-time as the agents move.
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Optional

class ProbabilityOverlay:
    """Creates a probability distribution overlay on the pygame screen"""
    
    def __init__(self, width: int = 300, height: int = 200, position: Tuple[int, int] = (20, 20)):
        """
        Initialize the probability overlay
        
        Args:
            width: Width of the overlay in pixels
            height: Height of the overlay in pixels
            position: (x, y) position of the overlay on screen
        """
        self.width = width
        self.height = height
        self.position = position
        self.enabled = False
        
        # Visual settings
        self.background_color = (0, 0, 0, 128)  # Semi-transparent black
        self.bar_color = (255, 100, 255, 200)   # Purple bars
        self.text_color = (255, 255, 255)       # White text
        self.border_color = (255, 255, 255, 100) # Semi-transparent white border
        
        # Data settings
        self.num_bins = 20
        self.min_prob = 0.0
        self.max_prob = 1.0
        
        # Font for text
        self.font_small = None
        self.font_medium = None
        self._init_fonts()
        
    def _init_fonts(self):
        """Initialize fonts for text rendering"""
        try:
            self.font_small = pygame.font.Font(None, 16)
            self.font_medium = pygame.font.Font(None, 20)
        except:
            # Fallback if font loading fails
            self.font_small = pygame.font.SysFont('arial', 14)
            self.font_medium = pygame.font.SysFont('arial', 18)
    
    def toggle(self):
        """Toggle the overlay on/off"""
        self.enabled = not self.enabled
        return self.enabled
    
    def is_enabled(self):
        """Check if overlay is enabled"""
        return self.enabled
    
    def update_and_draw(self, screen: pygame.Surface, combined_probabilities: Dict[int, float]):
        """
        Update the overlay with new probability data and draw it
        
        Args:
            screen: Pygame surface to draw on
            combined_probabilities: Dictionary of node_id -> probability
        """
        if not self.enabled or not combined_probabilities:
            return
        
        # Extract probability values
        prob_values = list(combined_probabilities.values())
        if not prob_values:
            return
        
        # Create overlay surface with per-pixel alpha
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw background
        pygame.draw.rect(overlay, self.background_color, (0, 0, self.width, self.height))
        pygame.draw.rect(overlay, self.border_color, (0, 0, self.width, self.height), 2)
        
        # Calculate histogram
        hist, bin_edges = np.histogram(prob_values, bins=self.num_bins, 
                                     range=(self.min_prob, self.max_prob))
        
        # Draw histogram bars
        max_count = max(hist) if max(hist) > 0 else 1
        bar_width = (self.width - 40) // self.num_bins
        max_bar_height = self.height - 60  # Leave space for text
        
        for i, count in enumerate(hist):
            if count > 0:
                bar_height = int((count / max_count) * max_bar_height)
                bar_x = 20 + i * bar_width
                bar_y = self.height - 40 - bar_height
                
                # Draw bar
                bar_rect = pygame.Rect(bar_x, bar_y, bar_width - 1, bar_height)
                pygame.draw.rect(overlay, self.bar_color, bar_rect)
        
        # Draw statistics text
        stats_text = self._format_statistics(prob_values)
        y_offset = 5
        for line in stats_text:
            text_surface = self.font_small.render(line, True, self.text_color)
            overlay.blit(text_surface, (10, y_offset))
            y_offset += 15
        
        # Draw title
        title = f"Combined Probabilities (n={len(prob_values)})"
        title_surface = self.font_medium.render(title, True, self.text_color)
        title_rect = title_surface.get_rect()
        overlay.blit(title_surface, ((self.width - title_rect.width) // 2, self.height - 20))
        
        # Blit overlay to main screen
        screen.blit(overlay, self.position)
    
    def _format_statistics(self, prob_values: List[float]) -> List[str]:
        """Format statistics for display"""
        if not prob_values:
            return ["No data"]
        
        mean_val = np.mean(prob_values)
        std_val = np.std(prob_values)
        min_val = np.min(prob_values)
        max_val = np.max(prob_values)
        
        return [
            f"Mean: {mean_val:.4f}",
            f"Std:  {std_val:.4f}",
            f"Min:  {min_val:.4f}",
            f"Max:  {max_val:.4f}",
            f"Range: {max_val - min_val:.4f}"
        ]
    
    def set_position(self, x: int, y: int):
        """Set the position of the overlay"""
        self.position = (x, y)
    
    def set_size(self, width: int, height: int):
        """Set the size of the overlay"""
        self.width = width
        self.height = height

# Global overlay instance
_probability_overlay = ProbabilityOverlay()

def toggle_probability_overlay() -> bool:
    """Toggle the probability overlay on/off"""
    return _probability_overlay.toggle()

def is_probability_overlay_enabled() -> bool:
    """Check if the probability overlay is enabled"""
    return _probability_overlay.is_enabled()

def draw_probability_overlay(screen: pygame.Surface, combined_probabilities: Dict[int, float]):
    """Draw the probability overlay on the screen"""
    _probability_overlay.update_and_draw(screen, combined_probabilities)

def set_overlay_position(x: int, y: int):
    """Set the position of the overlay"""
    _probability_overlay.set_position(x, y)

def set_overlay_size(width: int, height: int):
    """Set the size of the overlay"""
    _probability_overlay.set_size(width, height)

if __name__ == "__main__":
    # Test the overlay
    import random
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Probability Overlay Test")
    clock = pygame.time.Clock()
    
    # Enable overlay
    toggle_probability_overlay()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    toggle_probability_overlay()
        
        # Generate sample data
        sample_probs = {i: random.random() * 0.8 for i in range(50)}
        
        # Clear screen
        screen.fill((50, 50, 50))
        
        # Draw overlay
        draw_probability_overlay(screen, sample_probs)
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()
