#!/usr/bin/env python3
"""
Simulation UI Helper
Contains UI text generation and info panel functions for the simple agent simulation.
This helps keep the main simulation file clean and organized.
"""

import math
from simulation_config import *

def generate_info_lines(agent1, agent2, clock, position_evaluator_stats, show_map_graph, 
                       show_rrt_trees, show_reachability_mask, show_evader_visibility, 
                       visibility_range, visibility_data, selected_node, selected_agent_id, 
                       path_to_selected, trajectory_info, map_graph_loaded, stats, 
                       get_trajectory_calculator, get_distance, detect_visibility_breakoff_points,
                       get_visibility_statistics):
    """
    Generate all info panel lines for the simulation.
    
    Returns:
        list: Lines of text to display in the info panel
    """
    info_lines = [
        "Simple Agent Simulation",
        "",
        "Controls:",
        "Arrow Keys: Control Agent 1 (Magenta)",
        "WASD Keys: Control Agent 2 (Cyan)",
        "G: Toggle map graph display",
        "R: Toggle RRT* trees display",
        "M: Toggle reachability grid overlay (Agent 2)",
        "V: Toggle evader visibility (Agent 2 360° rays)",
        "T: Regenerate RRT* trees + auto-map to graph",
        "U: Force update travel times + auto-map to graph",
        "P: Strategic analysis (Agent 1 pursues Agent 2)",
        "I: Toggle this info display",
        "S: Save agent states to file",
        "Mouse Click: Select RRT node (auto-gen traj)",
        "C: Clear selected path & trajectory",
        "X: Clear closest node cache",
        "Z: Force rebuild spatial index",
        "L: Save environment lines to file",
        "ESC: Quit simulation",
        "",
        f"Agent 1 Position: ({agent1.state[0]:.1f}, {agent1.state[1]:.1f})",
        f"Agent 1 Heading: {math.degrees(agent1.state[2]):.1f}°",
        f"Agent 2 Position: ({agent2.state[0]:.1f}, {agent2.state[1]:.1f})",
        f"Agent 2 Heading: {math.degrees(agent2.state[2]):.1f}°",
        "",
        f"FPS: {int(clock.get_fps())}",
        "",
        "Position Evaluator:",
        f"Evaluator Distance: {get_distance('agent1', 'agent2'):.1f}" if get_distance('agent1', 'agent2') else "Evaluator Distance: N/A",
        f"Tracked Agents: {position_evaluator_stats.get('agent_count', 0)}",
        f"Environment Data: {'Yes' if position_evaluator_stats.get('has_environment', False) else 'No'}",
        f"Update Interval: {POSITION_UPDATE_INTERVAL}s",
        "",
        f"Map Graph: {'ON' if show_map_graph else 'OFF'}",
        f"RRT* Trees: {'ON' if show_rrt_trees else 'OFF'}",
        f"Reachability Grid: {'ON' if show_reachability_mask else 'OFF'}" + (" (Agent 2)" if show_reachability_mask else ""),
        f"Evader Visibility: {'ON' if show_evader_visibility else 'OFF'}" + (f" (Range: {visibility_range:.0f}px)" if show_evader_visibility else ""),
    ]
    
    # Add visibility system details if active
    if show_evader_visibility and visibility_data:
        visibility_stats = get_visibility_statistics(visibility_data)
        
        # Get breakoff point statistics
        breakoff_points, breakoff_lines = detect_visibility_breakoff_points(
            visibility_data, 
            min_gap_distance=MIN_GAP_DISTANCE,
            agent_x=agent2.state[0],
            agent_y=agent2.state[1], 
            agent_theta=agent2.state[2]
        )
        
        # Count breakoff points by category
        category_counts = get_breakoff_category_counts(breakoff_points)
        
        info_lines.extend([
            "",
            "Visibility System:",
            f"Total rays: {visibility_stats['total_rays']}",
            f"Clear rays: {visibility_stats['clear_rays']}",
            f"Blocked rays: {visibility_stats['blocked_rays']}",
            f"Visibility %: {visibility_stats['visibility_percentage']:.1f}%",
            f"Avg distance: {visibility_stats['average_visibility_distance']:.1f}px",
            f"Max distance: {visibility_stats['max_visibility_distance']:.1f}px",
            f"Breakoff points: {len(breakoff_points)} total",
            f"  Clockwise near→far (red): {category_counts['clockwise_near_far']}",
            f"  Clockwise far→near (orange-red): {category_counts['clockwise_far_near']}",
            f"  Counter near→far (green): {category_counts['counterclockwise_near_far']}",
            f"  Counter far→near (blue-green): {category_counts['counterclockwise_far_near']}",
            f"Show rays: {'ON' if DEFAULT_VISIBILITY_STATE['show_visibility_rays'] else 'OFF'}",
            f"Show area: {'ON' if DEFAULT_VISIBILITY_STATE['show_visibility_area'] else 'OFF'}",
            f"Update rate: {1/VISIBILITY_UPDATE_INTERVAL:.1f} Hz",
        ])
    
    # Add selected path information
    if selected_node and selected_agent_id:
        info_lines.extend(generate_selected_path_info(selected_node, selected_agent_id, path_to_selected, get_trajectory_calculator))
    else:
        info_lines.extend([
            "",
            "Selected Path: None",
            "(Click on RRT node to auto-generate)",
        ])
    
    # Add trajectory optimization information
    info_lines.extend(generate_trajectory_info(trajectory_info))
    
    # Add RRT* tree information
    if stats.get('rrt_enabled', False):
        info_lines.extend(generate_rrt_info(stats, get_trajectory_calculator))
    
    # Add map graph information
    if map_graph_loaded:
        info_lines.extend(generate_map_graph_info(stats))
    else:
        info_lines.extend([
            "",
            "Map Graph: Not loaded",
            "(Run inspect_environment.py first)"
        ])
    
    return info_lines


def get_breakoff_category_counts(breakoff_points):
    """Count breakoff points by category."""
    category_counts = {
        'clockwise_near_far': 0,
        'clockwise_far_near': 0,
        'counterclockwise_near_far': 0,
        'counterclockwise_far_near': 0,
        'unknown': 0
    }
    
    for point in breakoff_points:
        if len(point) >= 3:
            category = point[2]
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts['unknown'] += 1
    
    return category_counts


def generate_selected_path_info(selected_node, selected_agent_id, path_to_selected, get_trajectory_calculator):
    """Generate info lines for the selected path."""
    info_lines = [
        "",
        "Selected Path:",
        f"Agent: {selected_agent_id}",
        f"Target: ({selected_node.x:.1f}, {selected_node.y:.1f})",
        f"Path Length: {len(path_to_selected)} nodes",
        f"Path Cost: {selected_node.cost:.2f}",
    ]
    
    if len(path_to_selected) > 1:
        path_distance = sum(math.sqrt((path_to_selected[i+1].x - path_to_selected[i].x)**2 + 
                                    (path_to_selected[i+1].y - path_to_selected[i].y)**2) 
                          for i in range(len(path_to_selected)-1))
        info_lines.append(f"Path Distance: {path_distance:.1f}")
    else:
        info_lines.append("Path Distance: 0.0")
    
    # Add travel time for selected node if available
    travel_times_data = get_trajectory_calculator().get_travel_times_async(selected_agent_id)
    if travel_times_data:
        # Find the travel time for the selected node
        selected_travel_time = None
        for node, travel_time, path_len in travel_times_data:
            if abs(node.x - selected_node.x) < 0.1 and abs(node.y - selected_node.y) < 0.1:
                selected_travel_time = travel_time
                break
        
        if selected_travel_time is not None:
            validity_status = "Valid" if selected_travel_time <= TIME_THRESHOLD else "Invalid (too slow)"
            info_lines.append(f"Travel Time: {selected_travel_time:.2f}s ({validity_status})")
        else:
            info_lines.append("Travel Time: Calculating...")
    elif get_trajectory_calculator().is_calculating(selected_agent_id):
        info_lines.append("Travel Time: Calculating in background...")
    else:
        info_lines.append("Travel Time: Not calculated yet")
    
    return info_lines


def generate_trajectory_info(trajectory_info):
    """Generate info lines for trajectory optimization."""
    info_lines = [
        "",
        "Optimized Trajectory:",
        f"Status: {trajectory_info.get('status', 'Unknown')}",
    ]
    
    if trajectory_info.get('status') == 'Active':
        info_lines.extend([
            f"Agent: {trajectory_info.get('agent_id', 'N/A')}",
            f"Travel Time: {trajectory_info.get('total_time', 'N/A')}",
            f"Path Distance: {trajectory_info.get('total_distance', 'N/A')}",
            f"Peak Velocity: {trajectory_info.get('peak_velocity', 'N/A')}",
            f"Avg Velocity: {trajectory_info.get('avg_velocity', 'N/A')}",
            f"Max Constraints:",
            f"  Velocity: {DEFAULT_AGENT_POSITIONS['agent1'][0]:.1f} px/s",  # This should be LEADER_LINEAR_VEL from config
            f"  Turn Rate: {RRT_FORWARD_CONE_ANGLE:.1f} rad/s",  # This should be LEADER_ANGULAR_VEL from config
            f"  Acceleration: {MAX_ACCELERATION:.1f} px/s²",
        ])
    else:
        info_lines.extend([
            "(Click on RRT node for auto-generation)",
            f"Agent Max Velocity: {DEFAULT_AGENT_POSITIONS['agent1'][0]:.1f} px/s",  # This should be LEADER_LINEAR_VEL from config
            f"Agent Max Turn Rate: {RRT_FORWARD_CONE_ANGLE:.1f} rad/s",  # This should be LEADER_ANGULAR_VEL from config
        ])
    
    return info_lines


def generate_rrt_info(stats, get_trajectory_calculator):
    """Generate info lines for RRT* trees."""
    info_lines = [
        "",
        "RRT* Trees:",
        f"Max nodes per tree: {stats.get('rrt_max_nodes', 0)}",
        f"Step size: {stats.get('rrt_step_size', 0):.1f}",
        f"Search radius: {stats.get('rrt_search_radius', 0):.1f}",
        f"Forward bias: {stats.get('rrt_forward_bias', 0):.1%}",
        f"Forward cone: {stats.get('rrt_forward_cone_angle', 0):.0f}°",
        "",
        "Node Visualization Thresholds:",
        f"Distance threshold: {DISTANCE_THRESHOLD:.1f} px",
        f"Time threshold: {TIME_THRESHOLD:.1f}s",
        "(Nodes closer than distance threshold = transparent)",
        "(Nodes slower than time threshold = invalid/transparent)",
    ]
    
    # Add per-agent tree info
    agent1_nodes = stats.get('rrt_nodes_agent1', 0)
    agent2_nodes = stats.get('rrt_nodes_agent2', 0)
    if agent1_nodes > 0:
        info_lines.append(f"Agent 1 tree: {agent1_nodes} nodes")
    if agent2_nodes > 0:
        info_lines.append(f"Agent 2 tree: {agent2_nodes} nodes")
    
    # Add travel time information
    info_lines.extend(generate_travel_time_info(agent1_nodes, agent2_nodes, get_trajectory_calculator))
    
    return info_lines


def generate_travel_time_info(agent1_nodes, agent2_nodes, get_trajectory_calculator):
    """Generate travel time information for both agents."""
    info_lines = [
        "",
        "Longest Travel Times:",
    ]
    
    # Agent 1 travel times
    if agent1_nodes > 0:
        travel_times_1 = get_trajectory_calculator().get_travel_times_async("agent1")
        if travel_times_1:
            info_lines.append("Agent 1:")
            for i, (node, travel_time, path_len) in enumerate(travel_times_1[:2]):  # Top 2
                info_lines.append(f"  #{i+1}: {travel_time:.2f}s ({path_len} nodes)")
                info_lines.append(f"      to ({node.x:.0f}, {node.y:.0f})")
        elif get_trajectory_calculator().is_calculating("agent1"):
            info_lines.append("Agent 1: Calculating in background...")
        else:
            info_lines.append("Agent 1: No data yet")
    
    # Agent 2 travel times
    if agent2_nodes > 0:
        travel_times_2 = get_trajectory_calculator().get_travel_times_async("agent2")
        if travel_times_2:
            info_lines.append("Agent 2:")
            for i, (node, travel_time, path_len) in enumerate(travel_times_2[:2]):  # Top 2
                info_lines.append(f"  #{i+1}: {travel_time:.2f}s ({path_len} nodes)")
                info_lines.append(f"      to ({node.x:.0f}, {node.y:.0f})")
        elif get_trajectory_calculator().is_calculating("agent2"):
            info_lines.append("Agent 2: Calculating in background...")
        else:
            info_lines.append("Agent 2: No data yet")
    
    # Add average travel times
    info_lines.extend(generate_average_travel_times(agent1_nodes, agent2_nodes, get_trajectory_calculator))
    
    return info_lines


def generate_average_travel_times(agent1_nodes, agent2_nodes, get_trajectory_calculator):
    """Generate average travel time statistics."""
    info_lines = [
        "",
        "Average Travel Times (All Nodes):",
    ]
    
    # Agent 1 averages
    if agent1_nodes > 0:
        travel_times_1 = get_trajectory_calculator().get_travel_times_async("agent1")
        if travel_times_1:
            travel_time_values = [travel_time for _, travel_time, _ in travel_times_1]
            total_time = sum(travel_time_values)
            avg_time = total_time / len(travel_time_values)
            min_time = min(travel_time_values)
            max_time = max(travel_time_values)
            
            # Count nodes by time brackets
            fast_nodes = sum(1 for t in travel_time_values if t <= TIME_THRESHOLD)
            slow_nodes = len(travel_time_values) - fast_nodes
            
            info_lines.extend([
                f"Agent 1: {avg_time:.2f}s avg ({len(travel_times_1)} nodes)",
                f"  Range: {min_time:.2f}s - {max_time:.2f}s",
                f"  Fast (≤{TIME_THRESHOLD:.1f}s): {fast_nodes}, Slow (>{TIME_THRESHOLD:.1f}s): {slow_nodes}"
            ])
        elif get_trajectory_calculator().is_calculating("agent1"):
            info_lines.append("Agent 1: Calculating...")
        else:
            info_lines.append("Agent 1: No timing data")
    
    # Agent 2 averages
    if agent2_nodes > 0:
        travel_times_2 = get_trajectory_calculator().get_travel_times_async("agent2")
        if travel_times_2:
            travel_time_values = [travel_time for _, travel_time, _ in travel_times_2]
            total_time = sum(travel_time_values)
            avg_time = total_time / len(travel_time_values)
            min_time = min(travel_time_values)
            max_time = max(travel_time_values)
            
            # Count nodes by time brackets
            fast_nodes = sum(1 for t in travel_time_values if t <= TIME_THRESHOLD)
            slow_nodes = len(travel_time_values) - fast_nodes
            
            info_lines.extend([
                f"Agent 2: {avg_time:.2f}s avg ({len(travel_times_2)} nodes)",
                f"  Range: {min_time:.2f}s - {max_time:.2f}s",
                f"  Fast (≤{TIME_THRESHOLD:.1f}s): {fast_nodes}, Slow (>{TIME_THRESHOLD:.1f}s): {slow_nodes}"
            ])
        elif get_trajectory_calculator().is_calculating("agent2"):
            info_lines.append("Agent 2: Calculating...")
        else:
            info_lines.append("Agent 2: No timing data")
    
    return info_lines


def generate_map_graph_info(stats):
    """Generate map graph information."""
    from position_evaluator import get_closest_node_cache_stats, get_rrt_to_graph_mapping_stats, find_closest_node
    
    cache_stats = get_closest_node_cache_stats()
    info_lines = [
        "",
        f"Map Graph Nodes: {stats.get('map_nodes', 0)}",
        f"Map Graph Edges: {stats.get('map_edges', 0)}",
        "",
        "Closest Node Optimization:",
        f"Method: {cache_stats.get('optimization_method', 'Unknown')}",
        f"Cache Size: {cache_stats.get('cache_size', 0)} agents",
        f"Cache Thresholds:",
        f"  Movement: {cache_stats.get('movement_threshold', 0):.1f} px",
        f"  Time: {cache_stats.get('time_threshold', 0):.1f}s",
        f"SciPy Available: {'Yes' if cache_stats.get('scipy_available', False) else 'No'}",
        f"KD-tree Active: {'Yes' if cache_stats.get('kdtree_available', False) else 'No'}",
    ]
    
    # Add RRT-to-Map-Graph mapping info
    mapping_stats = get_rrt_to_graph_mapping_stats()
    if mapping_stats['total_agents'] > 0:
        info_lines.extend([
            "",
            "RRT to Map Graph Mapping:",
        ])
        
        for agent_id, agent_stats in mapping_stats['mappings_per_agent'].items():
            success_rate = agent_stats['mapping_success_rate'] * 100
            info_lines.append(f"{agent_id}: {agent_stats['mapped_nodes']}/{agent_stats['rrt_nodes']} ({success_rate:.1f}%)")
    else:
        info_lines.extend([
            "",
            "RRT to Map Graph Mapping:",
            "No mappings yet (press T/U to generate)",
        ])
    
    # Add closest node info
    closest_node_agent1 = find_closest_node("agent1")
    closest_node_agent2 = find_closest_node("agent2")
    if closest_node_agent1 is not None:
        info_lines.append(f"Agent 1 → Node {closest_node_agent1}")
    if closest_node_agent2 is not None:
        info_lines.append(f"Agent 2 → Node {closest_node_agent2}")
    
    return info_lines


def print_startup_help():
    """Print startup help text to console."""
    print("Starting simulation...")
    
    for line in CONTROL_HELP_TEXT['basic_controls']:
        print(f"  {line}")
    print("")
    
    for line in CONTROL_HELP_TEXT['reachability_help']:
        print(line)
    print("")
    
    for line in CONTROL_HELP_TEXT['visibility_help']:
        print(line)
    print("")
    
    for line in CONTROL_HELP_TEXT['node_visualization_help']:
        print(line)
