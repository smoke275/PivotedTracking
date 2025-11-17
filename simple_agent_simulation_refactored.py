#!/usr/bin/env python3
"""
Simple Agent Simulation (Refactored)
A standalone simulation that loads environment and agents for basic movement control.
Independent of the inspection tools - creates its own simulation environment.

This file has been refactored to use helper modules for better organization:
- simulation_config.py: All configuration constants and colors
- simulation_drawing.py: All drawing functions
- simulation_ui.py: UI text generation and info panel functions
"""

# Debug visualization flags
DRAW_CLIPPED_WALLS = False  # Set to True to enable clipped walls visualization

import pygame
import sys
import os
import math
import pickle
import time

# Make sure the project directory is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import configuration and helper modules
from simulation_config import *
from simulation_drawing import *
from simulation_ui import *

# Import required modules
from multitrack.models.simulation_environment import SimulationEnvironment
from multitrack.models.agents.visitor_agent import UnicycleModel
from multitrack.utils.map_graph import MapGraph
from multitrack.utils.config import (MAP_GRAPH_INSPECTION_CACHE_FILE, MAP_GRAPH_CACHE_ENABLED,
                                   MAP_GRAPH_VISIBILITY_CACHE_FILE, LEADER_LINEAR_VEL, LEADER_ANGULAR_VEL,
                                   MAP_GRAPH_NODE_COLOR, MAP_GRAPH_EDGE_COLOR)
from position_evaluator import (global_position_evaluator, update_position, get_distance, get_stats, 
                              set_environment_data, find_closest_node, get_agent_rrt_tree, 
                              update_agent_rrt_tree, update_all_rrt_trees, set_rrt_parameters,
                              get_path_to_node, find_node_at_position, clear_closest_node_cache,
                              set_closest_node_cache_parameters, get_closest_node_cache_stats,
                              force_rebuild_spatial_index, map_rrt_nodes_to_graph, 
                              map_all_rrt_nodes_to_graph, get_rrt_to_graph_mapping_stats,
                              calculate_pursuit_evasion_advantages, get_pursuit_evasion_stats)

# Import trajectory optimization system
from path_trajectory_optimizer import (initialize_trajectory_integrator, generate_trajectory_for_path,
                                     get_current_trajectory, get_trajectory_info, clear_trajectory,
                                     initialize_trajectory_calculator, get_trajectory_calculator,
                                     calculate_all_travel_times)

# Import risk calculator for reachability analysis and visibility calculations
from risk_calculator import (load_reachability_mask, calculate_evader_analysis, EvaderAnalysis,
                            get_reachability_probabilities_for_fixed_grid, calculate_evader_visibility, 
                            get_visibility_statistics, calculate_visibility_sectors, detect_visibility_breakoff_points)

# Import reachability mask API for overlay processing
from reachability_mask_min_api import ReachabilityMaskAPI

# Global variables to cache strategic analysis results
cached_worst_nodes = {}

# Global overlay API instance for reachability processing
overlay_api = None
overlay_configured = False

def get_worst_pursuit_nodes():
    """Get the RRT nodes with the worst (lowest) time advantages for Agent 1 as the pursuer."""
    global cached_worst_nodes
    return cached_worst_nodes.copy()

def initialize_overlay_api():
    """Initialize and configure the reachability overlay API."""
    global overlay_api, overlay_configured
    
    # Configuration constants for overlay (based on reachability_path_viewer.py)
    CLIP_PIXELS = 32.0
    RESIZE_TARGET = (120, 120)
    UPSCALE_TARGET = (400, 400)
    DOWNSAMPLE_METHOD = 'max_pool'
    # Probability weighting: None for no weighting, or (alpha, beta) for Prelect weighting
    # Example: (0.88, 0.88) for typical Prelect parameters, (0.5, 1.5) for different risk attitudes
    PROBABILITY_WEIGHTING = (0.55, 0.8)  # None to disable, (alpha, beta) to enable
    
    try:
        print("🔧 Initializing reachability overlay API...")
        
        # Create overlay API instance
        overlay_api = ReachabilityMaskAPI(filename_base="unicycle_grid")
        
        if overlay_api.is_loaded():
            print("✅ Overlay API loaded successfully")
            
            # STEP 1: Configure overlay (one-time setup) - TIMED
            print("🔧 Configuring reachability overlay (one-time setup)...")
            config_start_time = time.perf_counter()
            overlay_configured = overlay_api.setup_overlay_configuration(
                clip_pixels=CLIP_PIXELS,
                resize_target=RESIZE_TARGET,
                upscale_target=UPSCALE_TARGET,
                downsample_method=DOWNSAMPLE_METHOD,
                probability_weighting=PROBABILITY_WEIGHTING
            )
            config_end_time = time.perf_counter()
            config_duration = config_end_time - config_start_time
            print(f"⏱️ Overlay configuration took: {config_duration:.4f} seconds")
            
            if overlay_configured:
                print("✅ Overlay API configured successfully and ready for path processing")
                print(f"   📐 Clip pixels: {CLIP_PIXELS}")
                print(f"   📏 Resize target: {RESIZE_TARGET}")
                print(f"   📈 Upscale target: {UPSCALE_TARGET}")
                print(f"   🔧 Downsample method: {DOWNSAMPLE_METHOD}")
                print(f"   ⚖️ Probability weighting: {PROBABILITY_WEIGHTING}")
            else:
                print("⚠️ Failed to configure reachability overlay")
                overlay_api = None
        else:
            print("❌ Failed to load reachability data from unicycle_grid.pkl")
            overlay_api = None
            
    except ImportError:
        print("⚠️ Could not import ReachabilityMaskAPI - reachability overlay not available")
        overlay_api = None
        overlay_configured = False
    except Exception as e:
        print(f"⚠️ Error initializing overlay API: {e}")
        overlay_api = None
        overlay_configured = False
    
    return overlay_api, overlay_configured

def get_overlay_api():
    """Get the global overlay API instance."""
    global overlay_api
    return overlay_api

def is_overlay_configured():
    """Check if the overlay API is configured and ready for use."""
    global overlay_configured
    return overlay_configured

def update_worst_pursuit_nodes():
    """Update the cached worst pursuit nodes by running the analysis."""
    global cached_worst_nodes
    cached_worst_nodes = {}
    
    try:
        # Check if strategic analysis has been run by checking if stats are available
        agent1_stats = get_pursuit_evasion_stats("agent1", "agent2")
        
        # Only proceed if we have active analysis results for Agent 1 as pursuer
        if agent1_stats.get("status") != "Active":
            return
        
        # Get travel times for Agent 1 to filter out nodes that exceed time threshold
        travel_times_data = get_trajectory_calculator().get_travel_times_async("agent1")
        valid_node_indices = set()
        
        if travel_times_data:
            # Get the actual RRT tree for node index mapping
            agent1_tree = get_agent_rrt_tree("agent1")
            if agent1_tree:
                # Build a set of valid node indices (those below time threshold)
                for node, travel_time, _ in travel_times_data:
                    if travel_time <= TIME_THRESHOLD:
                        # Find the node index in the tree
                        for idx, tree_node in enumerate(agent1_tree):
                            if tree_node.x == node.x and tree_node.y == node.y:
                                valid_node_indices.add(idx)
                                break
        
        # Agent1 as pursuer vs Agent2 as evader (only calculate for the pursuer)
        agent1_advantages = calculate_pursuit_evasion_advantages("agent1", "agent2")
        if agent1_advantages:
            # Filter advantages to only include nodes below time threshold
            if valid_node_indices:
                filtered_advantages = {
                    node_idx: advantage for node_idx, advantage in agent1_advantages.items()
                    if node_idx in valid_node_indices
                }
                print(f"Strategic analysis: Filtered {len(agent1_advantages)} total nodes to {len(filtered_advantages)} valid nodes (≤{TIME_THRESHOLD:.1f}s)")
            else:
                # If no travel time data available, use all nodes (fallback behavior)
                filtered_advantages = agent1_advantages
                print(f"Strategic analysis: No travel time data available, using all {len(agent1_advantages)} nodes")
            
            if not filtered_advantages:
                print("Strategic analysis: No valid nodes found below time threshold")
                return
            
            # Sort filtered nodes by time advantage (lowest first = worst for pursuer)
            sorted_nodes = sorted(filtered_advantages.items(), key=lambda x: x[1])
            
            # Get the actual RRT node objects
            agent1_tree = get_agent_rrt_tree("agent1")
            if agent1_tree:
                # Cache the single worst node from valid nodes
                worst_node_idx, worst_advantage = sorted_nodes[0]
                if worst_node_idx < len(agent1_tree):
                    worst_node = agent1_tree[worst_node_idx]
                    
                    # Cache the worst 10 nodes (or fewer if less than 10 valid nodes available)
                    worst_10_nodes = []
                    for i, (node_idx, advantage) in enumerate(sorted_nodes[:10]):
                        if node_idx < len(agent1_tree):
                            node = agent1_tree[node_idx]
                            worst_10_nodes.append((node_idx, advantage, node))
                    
                    cached_worst_nodes["agent1"] = {
                        'worst': (worst_node_idx, worst_advantage, worst_node),
                        'worst_10': worst_10_nodes
                    }
        
        worst_count = len(cached_worst_nodes.get("agent1", {}).get('worst_10', []))
        valid_count = len(valid_node_indices) if valid_node_indices else "all"
        print(f"Updated worst pursuit nodes cache: Agent 1 (pursuer) has 1 worst node + {worst_count} worst nodes cached from {valid_count} valid nodes")
        
    except Exception as e:
        print(f"Error updating worst pursuit nodes: {e}")

def handle_key_events(event, show_info, show_map_graph, show_rrt_trees, show_reachability_mask, 
                     show_evader_visibility, mask_data, map_graph_loaded, selected_node, 
                     selected_agent_id, path_to_selected, visibility_range, evader_analysis=None, 
                     agent2=None, environment=None, show_measurement_tool=False):
    """Handle keyboard events and return updated state variables."""
    global cached_worst_nodes
    
    if event.key == pygame.K_ESCAPE:
        return {'quit': True}
    elif event.key == pygame.K_i:
        show_info = not show_info
        print(f"Info display: {'ON' if show_info else 'OFF'}")
    elif event.key == pygame.K_g:
        show_map_graph = not show_map_graph
        print(f"Map graph display: {'ON' if show_map_graph else 'OFF'}")
    elif event.key == pygame.K_r:
        show_rrt_trees = not show_rrt_trees
        print(f"RRT* trees display: {'ON' if show_rrt_trees else 'OFF'}")
    elif event.key == pygame.K_m:
        if mask_data is not None:
            show_reachability_mask = not show_reachability_mask
            print(f"🎯 Reachability grid overlay: {'ON' if show_reachability_mask else 'OFF'}")
            if show_reachability_mask:
                print("🟢 Showing reachability grid around Agent 2 (evader)")
                print("📍 Use WASD to move Agent 2 and see the grid follow")
                print("🔄 Grid rotates with Agent 2's orientation")
                print("🔴 Small colored circles on map graph nodes = projected reachability values")
                print("   Green circles = low reachability, Red circles = high reachability")
                print("   Circle size = proportional to reachability value")
                print("   Tiny dots (1px) = very low reachability, no dots = zero reachability")
                print("   ✅ ALL non-zero values preserved (no matter how small!)")
                print("   Only shows nodes within visibility range with non-zero values")
        else:
            print("❌ Reachability mask not available - run heatmap.py first")
    elif event.key == pygame.K_v:
        show_evader_visibility = not show_evader_visibility
        print(f"👁️  Evader visibility system: {'ON' if show_evader_visibility else 'OFF'}")
        if show_evader_visibility:
            print("🔍 Showing Agent 2 (evader) 360° visibility")
            print("🚫 Red rays = blocked by walls")
            print("✅ Cyan rays = clear line of sight")
            print("� Yellow polygon = visibility boundary (connects all ray endpoints)")
            print("�🔄 Use WASD to move Agent 2 and see visibility change")
            print(f"📏 Visibility range: {visibility_range:.0f} pixels")
            if mask_data is not None:
                print("🔴 Small colored circles on map graph nodes = projected reachability values")
                print("   Green circles = low reachability, Red circles = high reachability")
                print("   Circle size = proportional to reachability value")
                print("   Tiny dots (1px) = very low reachability, no dots = zero reachability")
                print("   ✅ ALL non-zero values preserved (no matter how small!)")
                print("   Rotates with Agent 2's orientation, shows within visibility range")
            print("⚡ Environment clipping enabled for performance optimization")
            print("🟢 Green outlines = clipped walls being processed")
            print("⚪ White pixels = visibility rays converted to 2px walls")
            print("🔵 Cyan pixels = visibility boundary circle converted to walls")
            print("🟡 Yellow box = visibility bounding area")
            print("◆ Diamond shapes = midpoints of breakoff lines (scaled by gap size)")
            print("▶ Triangle shapes = exploration points toward less visible areas")
            print("🎯 Colors indicate orientation: clockwise vs counterclockwise visibility transitions")
            print("🧭 Triangle direction shows optimal exploration direction (left/right of breakoff line)")
            print("⭕ Large ringed circles = map graph nodes nearest to exploration points")
            print("--- Dashed lines connect exploration points to their nearest map graph nodes")
            print("🔺 Polygon exploration paths = traced polygons around breakpoints into unknown areas")
            print("    🌈 Each completed path uses a distinct color (green, cyan, magenta, yellow, etc.)")
            print("    🟠 Orange paths = incomplete exploration paths (hit iteration limit)")
            print("    🟡 Yellow circles = starting points of all exploration paths")
            print("    ⚪ White-centered circles = ending points, colored to match their path")
            print("🎯 REACHABILITY OVERLAY (if available):")
            print("    🔥 Hot colormap heatmap = reachability values (red=high, yellow=medium, black=low)")
            print("    🎨 Thick colored lines = first edges of exploration paths (cyan, gold, lime, etc.)")
            print("    🧭 Colored arrows = path orientations (45° rotation away from circle center)")
            print("    💎 Diamond markers = sample points where reachability was evaluated")
            print("    📊 Value annotations = original→boosted reachability values (50% boost, capped at 1.0)")
            print("    ⚡ Processed through unified overlay API for accurate reachability analysis")
            
            # Show clipping statistics if evader analysis is available
            if evader_analysis and evader_analysis.clipping_statistics:
                stats = evader_analysis.clipping_statistics
                print(f"📊 Environment Clipping Performance:")
                print(f"   🧱 Walls: {stats['clipped_walls']}/{stats['original_walls']} ({stats['wall_reduction_percent']:.1f}% reduction)")
                if 'ray_walls_added' in stats:
                    print(f"   ⚪ Ray walls: {stats['ray_walls_added']} visibility rays converted to walls")
                if 'circle_walls_added' in stats:
                    print(f"   🔵 Circle walls: {stats['circle_walls_added']} visibility boundary segments")
                print(f"   🎯 Total: {stats['total_clipped_objects']}/{stats['total_original_objects']} objects ({stats['total_reduction_percent']:.1f}% fewer to process)")
                bbox = evader_analysis.visibility_bounding_box
                if bbox:
                    print(f"   📐 Visibility area: {bbox[2]-bbox[0]:.0f}×{bbox[3]-bbox[1]:.0f} pixels")
            else:
                print("📊 Clipping statistics will appear once analysis is running")
    elif event.key == pygame.K_t:
        # Clear cached worst nodes since trees will change
        cached_worst_nodes = {}
        
        update_all_rrt_trees()
        print("RRT* trees regenerated for all agents")
        # Clear selected path and trajectory since trees changed
        selected_node = None
        selected_agent_id = None
        path_to_selected = []
        clear_trajectory()
        # Invalidate trajectory calculation caches
        get_trajectory_calculator().invalidate_cache()
        print("Selected path, trajectory, and calculation caches cleared due to tree regeneration")
        
        # Immediately start calculating travel times for the new trees
        handle_tree_regeneration(map_graph_loaded)
    elif event.key == pygame.K_u:
        handle_travel_time_update(map_graph_loaded)
    elif event.key == pygame.K_p:
        handle_strategic_analysis(map_graph_loaded)
    elif event.key == pygame.K_c:
        # Clear selected path and trajectory
        selected_node = None
        selected_agent_id = None
        path_to_selected = []
        clear_trajectory()
        print("Selected path and trajectory cleared")
    elif event.key == pygame.K_s:
        save_agent_states()
    elif event.key == pygame.K_x:
        clear_closest_node_cache()
        print("Closest node cache cleared")
    elif event.key == pygame.K_z:
        force_rebuild_spatial_index()
        print("Spatial index (KD-tree) rebuilt")
    elif event.key == pygame.K_l:
        # Save current evader analysis lines to file
        if evader_analysis and hasattr(evader_analysis, 'clipped_walls'):
            from risk_calculator import save_clipped_environment_to_file
            import time
            
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"environment_lines_{timestamp}.txt"
            
            save_clipped_environment_to_file(evader_analysis, filename)
            print(f"📄 Saved environment lines to: {filename}")
        else:
            print("❌ No evader analysis data available. Enable visibility (V key) first.")
    elif event.key == pygame.K_k:
        # Save current visibility polygon to file
        if evader_analysis and hasattr(evader_analysis, 'visibility_boundary_polygon'):
            from risk_calculator import save_visibility_polygon_to_file
            import time
            
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"visibility_polygon_{timestamp}.txt"
            
            save_visibility_polygon_to_file(evader_analysis, filename)
            print(f"📐 Saved visibility polygon to: {filename}")
        else:
            print("❌ No visibility polygon data available. Enable visibility (V key) first.")
    elif event.key == pygame.K_q:
        show_measurement_tool = not show_measurement_tool
        print(f"📏 Measurement tool: {'ON' if show_measurement_tool else 'OFF'}")
        if show_measurement_tool:
            print("📐 Click and drag to measure distances")
            print("   🖱️  Left click: Start measurement")
            print("   🖱️  Drag: Show real-time distance")
            print("   🖱️  Release: Complete measurement and show final distance")
            print("   📊 Distance shown in pixels and approximate world units")
            print("   ❌ Press Q again to disable measurement tool")
        return {
            'show_info': show_info,
            'show_map_graph': show_map_graph,
            'show_rrt_trees': show_rrt_trees,
            'show_reachability_mask': show_reachability_mask,
            'show_evader_visibility': show_evader_visibility,
            'show_measurement_tool': show_measurement_tool,
            'selected_node': selected_node,
            'selected_agent_id': selected_agent_id,
            'path_to_selected': path_to_selected
        }
    
    return {
        'show_info': show_info,
        'show_map_graph': show_map_graph,
        'show_rrt_trees': show_rrt_trees,
        'show_reachability_mask': show_reachability_mask,
        'show_evader_visibility': show_evader_visibility,
        'selected_node': selected_node,
        'selected_agent_id': selected_agent_id,
        'path_to_selected': path_to_selected
    }

def handle_tree_regeneration(map_graph_loaded):
    """Handle RRT tree regeneration and related operations."""
    stats = get_stats()
    if stats.get('rrt_enabled', False):
        agent1_nodes = stats.get('rrt_nodes_agent1', 0)
        agent2_nodes = stats.get('rrt_nodes_agent2', 0)
        
        if agent1_nodes > 0:
            print(f"Starting travel time calculation for Agent 1 ({agent1_nodes} nodes)...")
            get_trajectory_calculator().get_travel_times_async("agent1")
        
        if agent2_nodes > 0:
            print(f"Starting travel time calculation for Agent 2 ({agent2_nodes} nodes)...")
            get_trajectory_calculator().get_travel_times_async("agent2")
        
        # Automatically perform RRT-to-map-graph mapping after regenerating trees
        if map_graph_loaded and (agent1_nodes > 0 or agent2_nodes > 0):
            print("\nAutomatically mapping RRT nodes to closest map graph nodes...")
            all_mappings = map_all_rrt_nodes_to_graph()
            
            for agent_id, mapping in all_mappings.items():
                print(f"  {agent_id}: Mapped {len(mapping)} RRT nodes to map graph nodes")
            
            # Show mapping statistics
            mapping_stats = get_rrt_to_graph_mapping_stats()
            print(f"  Mapping Statistics:")
            for agent_id, agent_stats in mapping_stats['mappings_per_agent'].items():
                success_rate = agent_stats['mapping_success_rate'] * 100
                print(f"    {agent_id}: {agent_stats['mapped_nodes']}/{agent_stats['rrt_nodes']} nodes mapped ({success_rate:.1f}% success)")

def handle_travel_time_update(map_graph_loaded):
    """Handle forced travel time updates."""
    # Force update travel times by invalidating cache and recalculating
    get_trajectory_calculator().invalidate_cache()
    print("Travel time cache cleared - forcing recalculation...")
    
    stats = get_stats()
    if stats.get('rrt_enabled', False):
        agent1_nodes = stats.get('rrt_nodes_agent1', 0)
        agent2_nodes = stats.get('rrt_nodes_agent2', 0)
        
        if agent1_nodes > 0:
            print(f"Forcing travel time calculation for Agent 1 ({agent1_nodes} nodes)...")
            get_trajectory_calculator().get_travel_times_async("agent1")
        
        if agent2_nodes > 0:
            print(f"Forcing travel time calculation for Agent 2 ({agent2_nodes} nodes)...")
            get_trajectory_calculator().get_travel_times_async("agent2")
        
        # Automatically perform RRT-to-map-graph mapping after travel time calculation
        if map_graph_loaded and (agent1_nodes > 0 or agent2_nodes > 0):
            print("\nAutomatically mapping RRT nodes to closest map graph nodes...")
            all_mappings = map_all_rrt_nodes_to_graph()
            
            for agent_id, mapping in all_mappings.items():
                print(f"  {agent_id}: Mapped {len(mapping)} RRT nodes to map graph nodes")
            
            # Show mapping statistics
            mapping_stats = get_rrt_to_graph_mapping_stats()
            print(f"  Mapping Statistics:")
            for agent_id, agent_stats in mapping_stats['mappings_per_agent'].items():
                success_rate = agent_stats['mapping_success_rate'] * 100
                print(f"    {agent_id}: {agent_stats['mapped_nodes']}/{agent_stats['rrt_nodes']} nodes mapped ({success_rate:.1f}% success)")
    else:
        print("No RRT trees available for travel time calculation")

def handle_strategic_analysis(map_graph_loaded):
    """Handle strategic pursuit-evasion analysis."""
    print("Starting strategic pursuit-evasion analysis...")
    
    stats = get_stats()
    if stats.get('rrt_enabled', False):
        agent1_nodes = stats.get('rrt_nodes_agent1', 0)
        agent2_nodes = stats.get('rrt_nodes_agent2', 0)
        
        if agent1_nodes > 0 and agent2_nodes > 0:
            # Ensure RRT-to-map-graph mapping is done first
            if map_graph_loaded:
                print("Ensuring RRT-to-map-graph mapping is complete...")
                all_mappings = map_all_rrt_nodes_to_graph()
                
                for agent_id, mapping in all_mappings.items():
                    print(f"  {agent_id}: Mapped {len(mapping)} RRT nodes to map graph nodes")
            
            print("\nPerforming strategic pursuit-evasion analysis...")
            print("  Agent 1 (pursuer) vs Agent 2 (evader):")
            
            # Agent1 as pursuer, Agent2 as evader
            agent1_advantages = calculate_pursuit_evasion_advantages("agent1", "agent2")
            agent1_stats = get_pursuit_evasion_stats("agent1", "agent2")
            
            if agent1_stats.get("status") == "Active":
                print(f"    Analyzed {agent1_stats['pursuer_nodes_analyzed']} pursuer positions")
                print(f"    Time advantage range: {agent1_stats['min_time_advantage']:.2f}s to {agent1_stats['max_time_advantage']:.2f}s")
                print(f"    Average advantage: {agent1_stats['avg_time_advantage']:.2f}s")
                print(f"    Positions with advantage: {agent1_stats['positive_advantages']}/{agent1_stats['pursuer_nodes_analyzed']}")
            
            print("Strategic analysis completed!")
            
            # Update the cached worst pursuit nodes for highlighting
            update_worst_pursuit_nodes()
        else:
            print("Strategic analysis requires both agents to have RRT trees. Press T to generate trees first.")
    else:
        print("No RRT trees available for strategic analysis. Press T to generate trees first.")

def save_agent_states():
    """Save agent states to files."""
    try:
        # This will be passed in from main function when we refactor
        # For now, this is a placeholder
        print("Agent states saved")
    except Exception as e:
        print(f"Error saving agent states: {e}")

def find_closest_map_graph_node(spatial_index, x, y, tolerance=20.0):
    """
    Find the closest map graph node using the spatial index.
    
    Args:
        spatial_index: The SpatialIndex instance (can be None for fallback)
        x, y: World coordinates to search near
        tolerance: Maximum distance to consider (in pixels)
    
    Returns:
        Tuple of (node_index, distance) if found, None otherwise
    """
    if spatial_index is None:
        return None
    
    try:
        # Use spatial index for fast nearest neighbor search
        result = spatial_index.find_nearest_node(x, y)
        if result:
            node_idx, distance = result
            if distance <= tolerance:
                return (node_idx, distance)
        return None
    except Exception as e:
        print(f"Error using spatial index for closest node search: {e}")
        return None

def draw_rrt_trees(screen, worst_pursuit_nodes, agent_positions):
    """Draw RRT trees with highlighting and transparency."""
    for agent_id in ["agent1", "agent2"]:
        tree = get_agent_rrt_tree(agent_id)
        if tree:
            color = RRT_TREE_COLORS.get(agent_id, (128, 128, 128))
            current_agent_pos = agent_positions[agent_id]
            
            # Get travel times for this agent if available
            travel_times_data = get_trajectory_calculator().get_travel_times_async(agent_id)
            travel_times_dict = {}
            if travel_times_data:
                # Convert travel times list to dictionary indexed by node index
                for node, travel_time, path_len in travel_times_data:
                    # Find the node index in the tree
                    for idx, tree_node in enumerate(tree):
                        if tree_node.x == node.x and tree_node.y == node.y:
                            travel_times_dict[idx] = travel_time
                            break
            
            # Draw tree edges with transparency based on node conditions
            for node in tree:
                if node.parent:
                    # Calculate edge transparency based on child node conditions
                    node_pos = (node.x, node.y)
                    distance_to_agent = math.sqrt((node_pos[0] - current_agent_pos[0])**2 + 
                                                (node_pos[1] - current_agent_pos[1])**2)
                    
                    # Check if node index exists and get timing data
                    node_idx = None
                    for idx, tree_node in enumerate(tree):
                        if tree_node.x == node.x and tree_node.y == node.y:
                            node_idx = idx
                            break
                    
                    # Determine transparency based on thresholds
                    edge_alpha = NODE_TRANSPARENCY['full_opacity']
                    if distance_to_agent < DISTANCE_THRESHOLD:
                        edge_alpha = NODE_TRANSPARENCY['edge_close']
                    elif node_idx is not None and node_idx in travel_times_dict:
                        travel_time = travel_times_dict[node_idx]
                        if travel_time > TIME_THRESHOLD:
                            edge_alpha = NODE_TRANSPARENCY['edge_invalid']
                    
                    # Draw edge
                    start_pos = (int(node.parent.x), int(node.parent.y))
                    end_pos = (int(node.x), int(node.y))
                    draw_rrt_edge(screen, start_pos, end_pos, color, edge_alpha)
            
            # Draw tree nodes with highlighting and transparency
            for i, node in enumerate(tree):
                pos = (int(node.x), int(node.y))
                node_pos = (node.x, node.y)
                
                # Calculate distance to current agent
                distance_to_agent = math.sqrt((node_pos[0] - current_agent_pos[0])**2 + 
                                            (node_pos[1] - current_agent_pos[1])**2)
                
                # Check if this is the worst pursuit node or in the worst 10
                is_worst_node = False
                is_worst_10_node = False
                
                if agent_id in worst_pursuit_nodes:
                    # Check if this is the single worst node
                    worst_data = worst_pursuit_nodes[agent_id].get('worst')
                    if worst_data and worst_data[0] == i:
                        is_worst_node = True
                    
                    # Check if this is in the worst 10 nodes
                    worst_10_data = worst_pursuit_nodes[agent_id].get('worst_10', [])
                    for node_idx, _, _ in worst_10_data:
                        if node_idx == i:
                            is_worst_10_node = True
                            break
                
                # Draw strategic highlighting first
                if is_worst_node:
                    draw_strategic_node_highlight(screen, pos, 'worst')
                    # Draw original color center
                    pygame.draw.circle(screen, color, pos, 4)
                elif is_worst_10_node:
                    draw_strategic_node_highlight(screen, pos, 'worst_10')
                    # Draw original color center
                    pygame.draw.circle(screen, color, pos, 2)
                else:
                    # Determine node transparency and validity
                    node_alpha = NODE_TRANSPARENCY['full_opacity']
                    is_invalid = False
                    
                    # Apply distance threshold
                    if distance_to_agent < DISTANCE_THRESHOLD:
                        node_alpha = NODE_TRANSPARENCY['close_nodes']
                    
                    # Apply timing threshold if timing data is available
                    if i in travel_times_dict:
                        travel_time = travel_times_dict[i]
                        if travel_time > TIME_THRESHOLD:
                            node_alpha = NODE_TRANSPARENCY['invalid_nodes']
                            is_invalid = True
                    
                    # Draw node
                    node_size = 4 if node.parent is None else 2
                    draw_rrt_node(screen, pos, node_size, color, node_alpha, is_invalid)

def draw_clipped_environment(screen, evader_analysis):
    """Draw the clipped environment objects with distinct colors to show what's being processed."""
    if not evader_analysis or not hasattr(evader_analysis, 'clipped_walls'):
        return
    
    # Separate regular walls from ray walls and breakoff walls for different visualization
    regular_walls = []
    ray_walls = []
    breakoff_walls = []
    
    # Get the number of ray walls and breakoff walls if available from statistics
    num_ray_walls = 0
    num_breakoff_walls = 0
    if hasattr(evader_analysis, 'clipping_statistics') and evader_analysis.clipping_statistics:
        num_ray_walls = evader_analysis.clipping_statistics.get('ray_walls_added', 0)
        num_breakoff_walls = evader_analysis.clipping_statistics.get('breakoff_walls_added', 0)
    
    # Split walls into different types
    total_added_walls = num_ray_walls + num_breakoff_walls
    if total_added_walls > 0 and len(evader_analysis.clipped_walls) >= total_added_walls:
        regular_walls = evader_analysis.clipped_walls[:-total_added_walls]
        
        # Added walls are at the end in order: ray_walls, breakoff_walls
        added_walls = evader_analysis.clipped_walls[-total_added_walls:]
        wall_idx = 0
        
        if num_ray_walls > 0:
            ray_walls = added_walls[wall_idx:wall_idx + num_ray_walls]
            wall_idx += num_ray_walls
            
        if num_breakoff_walls > 0:
            breakoff_walls = added_walls[wall_idx:wall_idx + num_breakoff_walls]
    else:
        regular_walls = evader_analysis.clipped_walls
    
    # Draw regular clipped walls with a bright green outline
    clipped_wall_color = (0, 255, 0)  # Bright green
    clipped_wall_width = 3
    
    for wall in regular_walls:
        # Draw the wall with green outline (these should still be rectangles)
        if hasattr(wall, 'x') and hasattr(wall, 'y'):  # pygame.Rect object
            pygame.draw.rect(screen, clipped_wall_color, wall, clipped_wall_width)
    
    # Draw ray walls as white lines
    ray_wall_color = (255, 255, 255)  # White
    ray_wall_width = 2
    
    for ray_wall in ray_walls:
        if isinstance(ray_wall, dict) and 'start' in ray_wall and 'end' in ray_wall:
            # Ray wall is a line dictionary
            start_pos = (int(ray_wall['start'][0]), int(ray_wall['start'][1]))
            end_pos = (int(ray_wall['end'][0]), int(ray_wall['end'][1]))
            pygame.draw.line(screen, ray_wall_color, start_pos, end_pos, ray_wall_width)
        elif hasattr(ray_wall, 'x') and hasattr(ray_wall, 'y'):  # pygame.Rect object (fallback)
            pygame.draw.rect(screen, ray_wall_color, ray_wall)
    
    # Draw visibility circle as a proper circle (cyan outline)
    if hasattr(evader_analysis, 'visibility_circle') and evader_analysis.visibility_circle:
        circle_info = evader_analysis.visibility_circle
        center_x = int(circle_info['center_x'])
        center_y = int(circle_info['center_y'])
        radius = int(circle_info['radius'])
        circle_color = (0, 255, 255)  # Cyan
        circle_width = 2
        pygame.draw.circle(screen, circle_color, (center_x, center_y), radius, circle_width)
    
    # Draw breakoff walls as magenta lines
    breakoff_wall_color = (255, 0, 255)  # Magenta
    breakoff_wall_width = 3
    
    for breakoff_wall in breakoff_walls:
        if isinstance(breakoff_wall, dict) and 'start' in breakoff_wall and 'end' in breakoff_wall:
            # Breakoff wall is a line dictionary
            start_pos = (int(breakoff_wall['start'][0]), int(breakoff_wall['start'][1]))
            end_pos = (int(breakoff_wall['end'][0]), int(breakoff_wall['end'][1]))
            pygame.draw.line(screen, breakoff_wall_color, start_pos, end_pos, breakoff_wall_width)
        elif hasattr(breakoff_wall, 'x') and hasattr(breakoff_wall, 'y'):  # pygame.Rect object (fallback)
            pygame.draw.rect(screen, breakoff_wall_color, breakoff_wall)
    
    # Draw the visibility bounding box if available
    if hasattr(evader_analysis, 'visibility_bounding_box') and evader_analysis.visibility_bounding_box:
        bbox = evader_analysis.visibility_bounding_box
        bbox_rect = pygame.Rect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
        bbox_color = (255, 255, 0)  # Yellow
        pygame.draw.rect(screen, bbox_color, bbox_rect, 2)

def main():
    global trajectory_calculator
    
    # Initialize Pygame
    pygame.init()
    
    # Set up the display
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Simple Agent Simulation")
    
    # Initialize font
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
    
    # Create the environment
    print("Creating simulation environment...")
    environment = SimulationEnvironment(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT)
    
    # Initialize map graph
    print("Initializing map graph...")
    map_graph = MapGraph(
        ENVIRONMENT_WIDTH, 
        ENVIRONMENT_HEIGHT, 
        environment.get_all_walls(), 
        environment.get_doors(),
        cache_file=MAP_GRAPH_INSPECTION_CACHE_FILE
    )
    
    # Try to load map graph from cache
    map_graph_loaded = False
    spatial_index = None
    if MAP_GRAPH_CACHE_ENABLED:
        print("Attempting to load map graph from cache...")
        map_graph_loaded = map_graph.load_from_cache()
        if map_graph_loaded:
            print(f"Successfully loaded map graph from cache with {len(map_graph.nodes)} nodes and {len(map_graph.edges)} edges.")
            
            # Create spatial index for fast O(1) spatial queries
            try:
                from multitrack.utils.spatial_index import SpatialIndex
                print("Creating spatial index for fast spatial queries...")
                spatial_index = SpatialIndex(map_graph)
                stats = spatial_index.get_statistics()
                print(f"✅ Spatial index created successfully!")
                print(f"   📊 Grid: {stats['grid_size'][0]}x{stats['grid_size'][1]} cells")
                print(f"   📍 Nodes: {stats['total_nodes']} indexed in {stats['occupied_cells']} occupied cells")
                print(f"   ⚡ Cell size: {stats['cell_size']:.1f}px for O(1) lookups")
                print(f"   🎯 Ready for real-time agent position tracking!")
            except ImportError:
                print("Warning: Could not import SpatialIndex - spatial queries will use fallback methods")
            except Exception as e:
                print(f"Warning: Could not create spatial index: {e}")
        else:
            print("No cached map graph found. Map graph will be empty (use inspect_environment.py to generate one).")
    else:
        print("Map graph caching is disabled. Map graph will be empty.")
    
    # Set up position evaluator with environment data and spatial index
    print("Configuring position evaluator with environment data...")
    set_environment_data(environment, map_graph if map_graph_loaded else None, spatial_index)
    if spatial_index is not None:
        print("✅ Position evaluator configured with spatial index for O(1) node lookups")
    else:
        print("Position evaluator configured with environment and map graph data (no spatial index)")
    
    # Configure systems
    print("Configuring systems...")
    set_closest_node_cache_parameters(
        movement_threshold=CLOSEST_NODE_MOVEMENT_THRESHOLD,
        time_threshold=CLOSEST_NODE_TIME_THRESHOLD
    )
    
    set_rrt_parameters(
        max_nodes=RRT_MAX_NODES, 
        step_size=RRT_STEP_SIZE, 
        search_radius=RRT_SEARCH_RADIUS,
        forward_bias=RRT_FORWARD_BIAS,
        forward_cone_angle=RRT_FORWARD_CONE_ANGLE
    )
    
    initialize_trajectory_integrator(
        max_velocity=LEADER_LINEAR_VEL,
        max_acceleration=MAX_ACCELERATION,
        max_turning_rate=LEADER_ANGULAR_VEL
    )
    
    trajectory_calculator = initialize_trajectory_calculator(max_workers=10)
    
    # Create agents
    print("Creating agents...")
    agent1 = UnicycleModel(
        initial_position=DEFAULT_AGENT_POSITIONS['agent1'],
        walls=environment.get_all_walls(), 
        doors=environment.get_doors()
    )
    
    agent2 = UnicycleModel(
        initial_position=DEFAULT_AGENT_POSITIONS['agent2'],
        walls=environment.get_all_walls(), 
        doors=environment.get_doors()
    )
    
    # Try to load agent states if they exist
    if os.path.exists(AGENT_STATE_FILE):
        try:
            with open(AGENT_STATE_FILE, 'rb') as f:
                agent_state = pickle.load(f)
                agent1.state = agent_state.copy()
                print("Loaded agent 1 state from file")
        except Exception as e:
            print(f"Could not load agent 1 state: {e}")
    
    if os.path.exists(AGENT2_STATE_FILE):
        try:
            with open(AGENT2_STATE_FILE, 'rb') as f:
                agent2_state = pickle.load(f)
                agent2.state = agent2_state.copy()
                print("Loaded agent 2 state from file")
        except Exception as e:
            print(f"Could not load agent 2 state: {e}")
    
    # Load reachability mask
    print("Loading reachability mask...")
    mask_data = load_reachability_mask(REACHABILITY_MASK_NAME)
    if mask_data is not None:
        grid_size = mask_data.get('grid_size', 'unknown')
        world_extent = mask_data.get('world_extent_px', 0)
        cell_size = mask_data.get('cell_size_px', 0)
        print(f"Reachability mask loaded: {grid_size}x{grid_size} grid")
        print(f"Mask covers: ±{world_extent/2:.1f} pixels")
        print(f"Cell size: {cell_size:.3f} px/cell")
    else:
        print("Reachability mask not available - run heatmap.py first")
    
    # Initialize overlay API for reachability processing
    print("Initializing overlay API...")
    overlay_api_instance, overlay_ready = initialize_overlay_api()
    if overlay_ready:
        print("🎯 Overlay API is ready for path processing with reachability analysis")
    else:
        print("⚠️ Overlay API not available - path processing will work without reachability overlay")
    
    # Initialize state variables
    clock = pygame.time.Clock()
    running = True
    
    # Display state
    show_info = DEFAULT_DISPLAY_STATE['show_info']
    show_map_graph = DEFAULT_DISPLAY_STATE['show_map_graph']
    show_rrt_trees = DEFAULT_DISPLAY_STATE['show_rrt_trees']
    show_trajectory = DEFAULT_DISPLAY_STATE['show_trajectory']
    show_reachability_mask = DEFAULT_DISPLAY_STATE['show_reachability_mask']
    
    # Measurement tool state
    show_measurement_tool = False
    measurement_start_pos = None
    measurement_end_pos = None
    measuring = False
    
    # Visibility state
    show_evader_visibility = DEFAULT_VISIBILITY_STATE['show_evader_visibility']
    show_visibility_rays = DEFAULT_VISIBILITY_STATE['show_visibility_rays']
    show_visibility_area = DEFAULT_VISIBILITY_STATE['show_visibility_area']
    visibility_range = DEFAULT_VISIBILITY_STATE['visibility_range']
    evader_analysis = None  # Will hold the unified EvaderAnalysis object
    
    # Timing
    last_position_update = time.time()
    last_travel_time_update = time.time()
    last_evader_analysis_update = time.time()  # Renamed for clarity
    
    # Path visualization
    selected_node = None
    selected_agent_id = None
    path_to_selected = []
    
    # Info panel scrolling
    info_scroll_offset = 0
    max_visible_lines = (WINDOW_HEIGHT - 40) // INFO_PANEL_LINE_SPACING
    
    # Print startup help
    print_startup_help()
    
    # Main simulation loop
    while running:
        current_time = time.time()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                result = handle_key_events(event, show_info, show_map_graph, show_rrt_trees, 
                                         show_reachability_mask, show_evader_visibility, 
                                         mask_data, 
                                         map_graph_loaded, selected_node, selected_agent_id, 
                                         path_to_selected, visibility_range, evader_analysis, 
                                         agent2, environment, show_measurement_tool)
                
                if result.get('quit'):
                    running = False
                else:
                    # Update state variables
                    show_info = result.get('show_info', show_info)
                    show_map_graph = result.get('show_map_graph', show_map_graph)
                    show_rrt_trees = result.get('show_rrt_trees', show_rrt_trees)
                    show_reachability_mask = result.get('show_reachability_mask', show_reachability_mask)
                    show_evader_visibility = result.get('show_evader_visibility', show_evader_visibility)
                    show_measurement_tool = result.get('show_measurement_tool', show_measurement_tool)
                    selected_node = result.get('selected_node', selected_node)
                    selected_agent_id = result.get('selected_agent_id', selected_agent_id)
                    path_to_selected = result.get('path_to_selected', path_to_selected)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_x, mouse_y = event.pos
                    
                    # Only search for nodes in the environment area (not sidebar)
                    if mouse_x < ENVIRONMENT_WIDTH:
                        if show_measurement_tool:
                            # Start measurement
                            measurement_start_pos = (mouse_x, mouse_y)
                            measurement_end_pos = (mouse_x, mouse_y)
                            measuring = True
                            print(f"📏 Starting measurement at ({mouse_x}, {mouse_y})")
                        else:
                            # Try to find RRT nodes near the click position for trajectory visualization
                            for agent_id in ["agent1", "agent2"]:
                                node = find_node_at_position(agent_id, mouse_x, mouse_y, tolerance=15.0)
                                if node:
                                    selected_node = node
                                    selected_agent_id = agent_id
                                    path_to_selected = get_path_to_node(agent_id, node)
                                    print(f"Selected RRT node in {agent_id} tree at ({node.x:.1f}, {node.y:.1f})")
                                    print(f"Path has {len(path_to_selected)} nodes, cost: {node.cost:.2f}")
                                    
                                    # Automatically generate optimized trajectory asynchronously
                                    print(f"Starting async trajectory generation for {agent_id} with {len(path_to_selected)} nodes...")
                                    
                                    def trajectory_callback(result, agent_id):
                                        if result:
                                            print(f"Async trajectory generation completed for {agent_id}!")
                                        else:
                                            print(f"Async trajectory generation failed for {agent_id}")
                                    
                                    trajectory_calculator = get_trajectory_calculator()
                                    trajectory_calculator.generate_trajectory_async(
                                        path_to_selected, agent_id, num_points=TRAJECTORY_NUM_POINTS, callback=trajectory_callback
                                    )
                                    break
                            else:
                                # No RRT node found - clear any existing selection
                                if selected_node:
                                    selected_node = None
                                    selected_agent_id = None
                                    path_to_selected = []
                                    clear_trajectory()
                                    print("No RRT node found at click position - path and trajectory cleared")
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and measuring:  # Left mouse button release
                    mouse_x, mouse_y = event.pos
                    if mouse_x < ENVIRONMENT_WIDTH and measurement_start_pos:
                        measurement_end_pos = (mouse_x, mouse_y)
                        measuring = False
                        
                        # Calculate and display final distance
                        dx = measurement_end_pos[0] - measurement_start_pos[0]
                        dy = measurement_end_pos[1] - measurement_start_pos[1]
                        distance = math.sqrt(dx * dx + dy * dy)
                        world_distance = distance * 0.1  # Approximate world units
                        
                        print(f"📏 Measurement complete:")
                        print(f"   📍 From: ({measurement_start_pos[0]}, {measurement_start_pos[1]})")
                        print(f"   📍 To: ({measurement_end_pos[0]}, {measurement_end_pos[1]})")
                        print(f"   📊 Distance: {distance:.1f} pixels ({world_distance:.1f} world units)")
            
            elif event.type == pygame.MOUSEMOTION:
                if measuring and measurement_start_pos:
                    mouse_x, mouse_y = event.pos
                    if mouse_x < ENVIRONMENT_WIDTH:
                        measurement_end_pos = (mouse_x, mouse_y)
            
            elif event.type == pygame.MOUSEWHEEL:
                # Handle mouse wheel scrolling in the info panel
                if show_info:
                    info_scroll_offset -= event.y * INFO_PANEL_SCROLL_SPEED
        
        # Get current key states for continuous movement
        keys = pygame.key.get_pressed()
        
        # Agent controls
        linear_vel1 = angular_vel1 = 0
        linear_vel2 = angular_vel2 = 0
        
        # Agent 1 control (arrow keys)
        if keys[pygame.K_UP]: linear_vel1 = LEADER_LINEAR_VEL
        if keys[pygame.K_DOWN]: linear_vel1 = -LEADER_LINEAR_VEL
        if keys[pygame.K_LEFT]: angular_vel1 = -LEADER_ANGULAR_VEL
        if keys[pygame.K_RIGHT]: angular_vel1 = LEADER_ANGULAR_VEL
        
        # Agent 2 control (WASD keys)
        if keys[pygame.K_w]: linear_vel2 = LEADER_LINEAR_VEL
        if keys[pygame.K_s]: linear_vel2 = -LEADER_LINEAR_VEL
        if keys[pygame.K_a]: angular_vel2 = -LEADER_ANGULAR_VEL
        if keys[pygame.K_d]: angular_vel2 = LEADER_ANGULAR_VEL
        
        # Update agents
        agent1.set_controls(linear_vel1, angular_vel1)
        agent1.update(dt=0.1, walls=environment.get_all_walls(), doors=environment.get_doors())
        
        agent2.set_controls(linear_vel2, angular_vel2)
        agent2.update(dt=0.1, walls=environment.get_all_walls(), doors=environment.get_doors())
        
        # Update evader analysis system periodically using unified API
        if show_evader_visibility and current_time - last_evader_analysis_update >= VISIBILITY_UPDATE_INTERVAL:
            evader_analysis = calculate_evader_analysis(
                agent_x=agent2.state[0], 
                agent_y=agent2.state[1], 
                agent_theta=agent2.state[2],
                visibility_range=visibility_range,
                walls=environment.get_all_walls(),
                mask_data=mask_data,  # Always pass mask_data when available for projected heatmap
                num_rays=VISIBILITY_NUM_RAYS,
                spatial_index=spatial_index,
                overlay_api=get_overlay_api()  # Pass the overlay API instance
            )
            last_evader_analysis_update = current_time
        
        # Update position evaluator periodically
        if current_time - last_position_update >= POSITION_UPDATE_INTERVAL:
            update_position("agent1", agent1.state[0], agent1.state[1], agent1.state[2], (linear_vel1, angular_vel1))
            update_position("agent2", agent2.state[0], agent2.state[1], agent2.state[2], (linear_vel2, angular_vel2))
            
            # Use spatial index to find nearest map graph nodes to agents (no console output for performance)
            if spatial_index is not None and map_graph_loaded:
                try:
                    # Find nearest map graph node to each agent using O(1) spatial index
                    agent1_nearest = spatial_index.find_nearest_node(agent1.state[0], agent1.state[1])
                    agent2_nearest = spatial_index.find_nearest_node(agent2.state[0], agent2.state[1])
                    
                    # Store this information for use in info panel or other systems (no console spam)
                    if not hasattr(spatial_index, '_debug_info'):
                        spatial_index._debug_info = {}
                    
                    spatial_index._debug_info['agent1_nearest'] = agent1_nearest
                    spatial_index._debug_info['agent2_nearest'] = agent2_nearest
                    
                    # Check for exact coordinate matches (no console output)
                    for agent_name, agent_state in [("Agent 1", agent1.state), ("Agent 2", agent2.state)]:
                        exact_node = spatial_index.get_node_by_coordinates(agent_state[0], agent_state[1])
                        # Store exact match info without printing
                        if exact_node is not None:
                            spatial_index._debug_info[f'{agent_name.lower().replace(" ", "_")}_exact'] = exact_node
                        
                except Exception as e:
                    # Only print errors, not regular operation info
                    print(f"⚠️ Spatial index error: {e}")
            
            last_position_update = current_time
        
        # Update travel time calculations periodically
        if current_time - last_travel_time_update >= TRAVEL_TIME_UPDATE_INTERVAL:
            stats = get_stats()
            if stats.get('rrt_enabled', False):
                agent1_nodes = stats.get('rrt_nodes_agent1', 0)
                agent2_nodes = stats.get('rrt_nodes_agent2', 0)
                
                if agent1_nodes > 0:
                    get_trajectory_calculator().get_travel_times_async("agent1")
                if agent2_nodes > 0:
                    get_trajectory_calculator().get_travel_times_async("agent2")
            
            last_travel_time_update = current_time
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw environment
        environment.draw(screen, font)
        
        # Draw evader visibility system using unified analysis data
        if show_evader_visibility and evader_analysis and evader_analysis.visibility_rays:
            draw_evader_visibility(
                screen, 
                agent2.state[0], agent2.state[1], 
                evader_analysis.visibility_rays,
                show_rays=show_visibility_rays,
                show_visibility_area=show_visibility_area,
                agent_theta=agent2.state[2]
            )
            

            
            # Draw visibility boundary polygon if it exists
            if hasattr(evader_analysis, 'visibility_boundary_polygon') and evader_analysis.visibility_boundary_polygon:
                draw_visibility_boundary_polygon(screen, evader_analysis.visibility_boundary_polygon)
            
            # Draw visibility bounding box (always shown when visibility is enabled)
            if hasattr(evader_analysis, 'visibility_bounding_box') and evader_analysis.visibility_bounding_box:
                bbox = evader_analysis.visibility_bounding_box
                bbox_rect = pygame.Rect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                bbox_color = (255, 255, 0)  # Yellow
                pygame.draw.rect(screen, bbox_color, bbox_rect, 2)
            
            # Highlight map graph nodes within bounding box and visibility circle
            if map_graph_loaded and hasattr(evader_analysis, 'highlighted_nodes') and evader_analysis.highlighted_nodes:
                highlighted_nodes = evader_analysis.highlighted_nodes
                
                # Colors for highlighting nodes
                bbox_node_color = (255, 165, 0)  # Orange for nodes in bounding box
                circle_node_color = (0, 255, 255)  # Cyan for nodes in visibility circle
                
                # Draw nodes in visibility circle (higher priority)
                for node_idx, node_x, node_y in highlighted_nodes.get('circle_nodes', []):
                    pygame.draw.circle(screen, circle_node_color, (int(node_x), int(node_y)), 3, 2)
                
                # Draw nodes only in bounding box
                for node_idx, node_x, node_y in highlighted_nodes.get('bbox_nodes', []):
                    pygame.draw.circle(screen, bbox_node_color, (int(node_x), int(node_y)), 3, 2)
            
            # Draw clipped environment objects to show what's being processed (optional debug visualization)
            if DRAW_CLIPPED_WALLS:
                draw_clipped_environment(screen, evader_analysis)
            
            # Draw breakoff line midpoints if they exist
            if hasattr(evader_analysis, 'breakoff_midpoints') and evader_analysis.breakoff_midpoints:
                draw_breakoff_midpoints(screen, evader_analysis.breakoff_midpoints)
            
            # Draw exploration offset points if they exist
            if hasattr(evader_analysis, 'exploration_offset_points') and evader_analysis.exploration_offset_points:
                draw_exploration_offset_points(screen, evader_analysis.exploration_offset_points)
            
            # Draw highlighted nearest nodes to exploration points if they exist
            if (hasattr(evader_analysis, 'exploration_nearest_nodes') and 
                evader_analysis.exploration_nearest_nodes and map_graph_loaded):
                draw_exploration_nearest_nodes(screen, evader_analysis.exploration_nearest_nodes, map_graph)
            
            # Draw polygon exploration paths if they exist
            if hasattr(evader_analysis, 'polygon_exploration_paths') and evader_analysis.polygon_exploration_paths:
                draw_polygon_exploration_paths(screen, evader_analysis.polygon_exploration_paths)
                
                # Draw path links if they exist
                if hasattr(evader_analysis, 'path_links') and evader_analysis.path_links:
                    from simulation_drawing import draw_path_links
                    draw_path_links(screen, evader_analysis.path_links)
            
            # Draw reachability overlay data if available (from overlay API processing)
            if hasattr(evader_analysis, 'reachability_overlay_data') and evader_analysis.reachability_overlay_data:
                draw_reachability_overlay(screen, evader_analysis.reachability_overlay_data, 
                                        agent2.state[0], agent2.state[1])
            
            # Draw path analysis data (first edges, orientations, etc.)
            if hasattr(evader_analysis, 'path_analysis_data') and evader_analysis.path_analysis_data:
                draw_path_analysis_data(screen, evader_analysis.path_analysis_data)
            
            # Draw sample points where reachability values were evaluated
            if hasattr(evader_analysis, 'sample_points_data') and evader_analysis.sample_points_data:
                draw_sample_points(screen, evader_analysis.sample_points_data)
        
        # Draw map graph if enabled and loaded
        if show_map_graph and map_graph_loaded:
            # Draw the graph edges first
            for edge in map_graph.edges:
                i, j = edge
                start = map_graph.nodes[i]
                end = map_graph.nodes[j]
                pygame.draw.line(screen, MAP_GRAPH_EDGE_COLOR, start, end, 1)
            
            # Draw the graph nodes
            for i, node in enumerate(map_graph.nodes):
                pygame.draw.circle(screen, MAP_GRAPH_NODE_COLOR, node, 4)
        
        # Draw RRT* trees if enabled
        if show_rrt_trees:
            worst_pursuit_nodes = get_worst_pursuit_nodes()
            agent_positions = {
                "agent1": (agent1.state[0], agent1.state[1]),
                "agent2": (agent2.state[0], agent2.state[1])
            }
            draw_rrt_trees(screen, worst_pursuit_nodes, agent_positions)
        
        # Draw optimized trajectory
        if show_trajectory:
            trajectory = get_current_trajectory()
            draw_trajectory(screen, trajectory)
        
        # Draw info panel
        if show_info:
            info_lines = generate_info_lines(
                agent1, agent2, clock, get_stats(), show_map_graph, show_rrt_trees, 
                show_reachability_mask, show_evader_visibility, visibility_range, 
                evader_analysis.visibility_rays if evader_analysis else [], selected_node, selected_agent_id, path_to_selected,
                get_trajectory_info(), map_graph_loaded, get_stats(), 
                get_trajectory_calculator, get_distance, detect_visibility_breakoff_points,
                get_visibility_statistics
            )
            
            # Handle scrolling
            max_scroll = max(0, len(info_lines) - max_visible_lines)
            info_scroll_offset = max(0, min(info_scroll_offset, max_scroll))
            
            # Draw info background
            info_bg = pygame.Surface((300, WINDOW_HEIGHT - 40))
            info_bg.fill((0, 0, 0))
            info_bg.set_alpha(INFO_PANEL_BACKGROUND_ALPHA)
            
            info_x = ENVIRONMENT_WIDTH + INFO_PANEL_MARGIN
            info_y = 20
            screen.blit(info_bg, (info_x, info_y))
            
            # Draw scroll indicator if needed
            if max_scroll > 0:
                scroll_bar_height = max(20, (max_visible_lines / len(info_lines)) * (WINDOW_HEIGHT - 60))
                scroll_bar_y = info_y + 10 + (info_scroll_offset / max_scroll) * (WINDOW_HEIGHT - 80 - scroll_bar_height)
                pygame.draw.rect(screen, (100, 100, 100), (info_x + 280, scroll_bar_y, 8, scroll_bar_height))
            
            # Draw visible info text
            visible_lines = info_lines[info_scroll_offset:info_scroll_offset + max_visible_lines]
            for i, line in enumerate(visible_lines):
                if line:  # Skip empty lines
                    text_surface = font.render(line, True, (255, 255, 255))
                    screen.blit(text_surface, (info_x + INFO_PANEL_MARGIN, info_y + 10 + i * INFO_PANEL_LINE_SPACING))
        
        # Draw reachability mask using unified analysis data
        if show_reachability_mask and evader_analysis and evader_analysis.reachability_grid is not None:
            draw_reachability_grid_overlay(screen, mask_data, agent2.state[0], agent2.state[1], agent2.state[2])
        elif show_reachability_mask and mask_data is not None:
            # Fallback for when evader analysis isn't active but mask display is requested
            draw_reachability_grid_overlay(screen, mask_data, agent2.state[0], agent2.state[1], agent2.state[2])
        
        # Draw measurement tool
        if show_measurement_tool and measurement_start_pos and measurement_end_pos:
            draw_measurement_line(screen, measurement_start_pos, measurement_end_pos, font)
        
        # Draw agents on top of everything
        draw_agent(screen, agent1, AGENT_COLOR, "Agent 1", font)
        draw_agent(screen, agent2, AGENT2_COLOR, "Agent 2", font)
        
        # Update display
        pygame.display.flip()
        clock.tick(TARGET_FPS)
    
    # Save agent states on exit
    try:
        with open(AGENT_STATE_FILE, 'wb') as f:
            pickle.dump(agent1.state, f)
        with open(AGENT2_STATE_FILE, 'wb') as f:
            pickle.dump(agent2.state, f)
        print("Agent states saved on exit")
    except Exception as e:
        print(f"Error saving agent states: {e}")
    
    # Cleanup
    print("Shutting down trajectory calculation threads...")
    get_trajectory_calculator().shutdown()
    
    pygame.quit()
    print("Simulation ended")


if __name__ == "__main__":
    main()
