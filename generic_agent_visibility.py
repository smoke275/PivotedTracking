#!/usr/bin/env python3
"""
Generic Agent Visibility Calculator
Demonstrates that the visibility system works for any agent type (evader, pursuer, or any other agent).
"""

import math
from fast_visibility_calculator import calculate_visibility_optimized, FastVisibilityCalculator

def calculate_agent_visibility(agent_x, agent_y, visibility_range, walls, doors, num_rays=100):
    """
    Generic visibility calculation for ANY agent type.
    
    This function demonstrates that visibility calculations are not specific to evaders.
    It can be used for:
    - Evaders (original use case)
    - Pursuers 
    - Any other agent type in the system
    
    Args:
        agent_x, agent_y: Agent position (any agent type)
        visibility_range: Maximum visibility distance
        walls: List of wall rectangles that block vision
        doors: List of door rectangles that allow vision through
        num_rays: Number of rays to cast
        
    Returns:
        List of (angle, endpoint, distance, blocked) tuples
    """
    return calculate_visibility_optimized(agent_x, agent_y, visibility_range, walls, doors, num_rays)

def calculate_pursuer_visibility(pursuer_x, pursuer_y, visibility_range, walls, doors, num_rays=100):
    """
    Calculate 360-degree visibility specifically for a pursuer agent.
    
    This is functionally identical to evader visibility - the algorithm doesn't
    distinguish between agent types. The naming is just for clarity.
    """
    return calculate_agent_visibility(pursuer_x, pursuer_y, visibility_range, walls, doors, num_rays)

def calculate_escort_visibility(escort_x, escort_y, visibility_range, walls, doors, num_rays=100):
    """
    Calculate 360-degree visibility for an escort agent.
    Same algorithm, different semantic naming.
    """
    return calculate_agent_visibility(escort_x, escort_y, visibility_range, walls, doors, num_rays)

def calculate_visitor_visibility(visitor_x, visitor_y, visibility_range, walls, doors, num_rays=100):
    """
    Calculate 360-degree visibility for a visitor agent.
    Same algorithm, different semantic naming.
    """
    return calculate_agent_visibility(visitor_x, visitor_y, visibility_range, walls, doors, num_rays)

class MultiAgentVisibilitySystem:
    """
    Visibility system that can handle multiple agent types efficiently.
    
    This demonstrates that the visibility calculation is agent-type agnostic.
    The same underlying algorithm works for all agent types.
    """
    
    def __init__(self, walls=None, doors=None):
        self.calculator = FastVisibilityCalculator(walls, doors)
        self.visibility_cache = {}  # Cache results by agent ID
    
    def calculate_visibility(self, agent_id, agent_type, agent_x, agent_y, visibility_range, num_rays=100):
        """
        Calculate visibility for any agent type.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent ('evader', 'pursuer', 'escort', 'visitor', etc.)
            agent_x, agent_y: Agent position
            visibility_range: Maximum visibility distance  
            num_rays: Number of rays to cast
            
        Returns:
            Dict containing visibility data and metadata
        """
        
        # The core calculation is the same regardless of agent type
        visibility_data = self.calculator.calculate_visibility(
            agent_x, agent_y, visibility_range, num_rays
        )
        
        # Calculate statistics (same for all agent types)
        total_rays = len(visibility_data)
        blocked_rays = sum(1 for _, _, _, blocked in visibility_data if blocked)
        clear_rays = total_rays - blocked_rays
        
        distances = [distance for _, _, distance, _ in visibility_data]
        avg_distance = sum(distances) / len(distances) if distances else 0.0
        
        result = {
            'agent_id': agent_id,
            'agent_type': agent_type,
            'position': (agent_x, agent_y),
            'visibility_data': visibility_data,
            'statistics': {
                'total_rays': total_rays,
                'clear_rays': clear_rays,
                'blocked_rays': blocked_rays,
                'visibility_percentage': (clear_rays / total_rays * 100) if total_rays > 0 else 0.0,
                'average_visibility_distance': avg_distance,
                'max_visibility_distance': max(distances) if distances else 0.0,
                'min_visibility_distance': min(distances) if distances else 0.0
            }
        }
        
        # Cache the result
        self.visibility_cache[agent_id] = result
        
        return result
    
    def get_mutual_visibility(self, agent1_id, agent1_type, agent1_pos, 
                            agent2_id, agent2_type, agent2_pos, visibility_range):
        """
        Check if two agents can see each other.
        
        This demonstrates that visibility works symmetrically between any agent types.
        """
        
        # Calculate visibility for both agents
        vis1 = self.calculate_visibility(agent1_id, agent1_type, agent1_pos[0], agent1_pos[1], visibility_range)
        vis2 = self.calculate_visibility(agent2_id, agent2_type, agent2_pos[0], agent2_pos[1], visibility_range)
        
        # Check if each agent can see the other's position
        def can_see_position(visibility_data, target_pos):
            for angle, endpoint, distance, blocked in visibility_data:
                if not blocked:
                    # Check if the ray passes near the target position
                    ray_end_x, ray_end_y = endpoint
                    dist_to_target = math.sqrt((target_pos[0] - ray_end_x)**2 + (target_pos[1] - ray_end_y)**2)
                    if dist_to_target < 20:  # Within detection radius
                        return True
            return False
        
        agent1_sees_agent2 = can_see_position(vis1['visibility_data'], agent2_pos)
        agent2_sees_agent1 = can_see_position(vis2['visibility_data'], agent1_pos)
        
        return {
            f'{agent1_type}_sees_{agent2_type}': agent1_sees_agent2,
            f'{agent2_type}_sees_{agent1_type}': agent2_sees_agent1,
            'mutual_visibility': agent1_sees_agent2 and agent2_sees_agent1
        }

def demo_multi_agent_visibility():
    """
    Demonstration showing that visibility works for all agent types.
    """
    import pygame
    
    print("🔍 Multi-Agent Visibility System Demo")
    print("=" * 50)
    
    # Create some test environment
    walls = [
        pygame.Rect(100, 100, 200, 20),
        pygame.Rect(200, 200, 100, 100),
    ]
    doors = [pygame.Rect(150, 100, 30, 20)]
    
    # Create visibility system
    vis_system = MultiAgentVisibilitySystem(walls, doors)
    
    # Test different agent types at different positions
    agents = [
        ('agent1', 'evader', 50, 50),
        ('agent2', 'pursuer', 250, 150), 
        ('agent3', 'escort', 300, 250),
        ('agent4', 'visitor', 150, 300),
    ]
    
    visibility_range = 200
    
    print(f"Environment: {len(walls)} walls, {len(doors)} doors")
    print(f"Visibility range: {visibility_range}")
    print()
    
    # Calculate visibility for each agent
    for agent_id, agent_type, x, y in agents:
        result = vis_system.calculate_visibility(agent_id, agent_type, x, y, visibility_range)
        stats = result['statistics']
        
        print(f"🤖 {agent_type.upper()} at ({x}, {y}):")
        print(f"   👁️  Visibility: {stats['visibility_percentage']:.1f}% clear")
        print(f"   📏 Avg distance: {stats['average_visibility_distance']:.1f}")
        print(f"   🚫 Blocked rays: {stats['blocked_rays']}/{stats['total_rays']}")
        print()
    
    # Test mutual visibility between different agent types
    print("🔄 Mutual Visibility Tests:")
    test_pairs = [
        (('agent1', 'evader', (50, 50)), ('agent2', 'pursuer', (250, 150))),
        (('agent2', 'pursuer', (250, 150)), ('agent3', 'escort', (300, 250))),
        (('agent3', 'escort', (300, 250)), ('agent4', 'visitor', (150, 300))),
    ]
    
    for (id1, type1, pos1), (id2, type2, pos2) in test_pairs:
        mutual = vis_system.get_mutual_visibility(id1, type1, pos1, id2, type2, pos2, visibility_range)
        print(f"   {type1} ↔ {type2}: {mutual['mutual_visibility']}")
    
    print()
    print("✅ Demonstration complete!")
    print("💡 Key insight: Visibility calculations work identically for ALL agent types.")
    print("🎯 The algorithm is agent-type agnostic - only position and environment matter.")

if __name__ == "__main__":
    demo_multi_agent_visibility()
