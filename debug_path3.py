#!/usr/bin/env python3
"""
Debug Path 3 specifically with detailed visualization.
"""

import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import polygon_exploration as pe

def debug_path_3():
    """Debug path 3 with detailed visualization and analysis."""
    
    # Load data (same as force_window.py)
    environment_lines = []
    breakoff_lines = []
    agent_x, agent_y = 951.4, 594.8
    visibility_range = 200.0
    
    with open('test_env2.txt', 'r') as f:
        reading_breakoff = False
        for line in f:
            line = line.strip()
            if line.startswith('# BREAKOFF_WALLS'):
                reading_breakoff = True
                continue
            elif line.startswith('#') and reading_breakoff:
                reading_breakoff = False
                continue
            elif line.startswith('LINE'):
                parts = line.split()
                if len(parts) >= 5:
                    x1, y1 = float(parts[1]), float(parts[2])
                    x2, y2 = float(parts[3]), float(parts[4])
                    
                    # Add ALL lines to environment_lines (including breakoff lines)
                    environment_lines.append([(x1, y1), (x2, y2)])
                    
                    if reading_breakoff:
                        # This is a breakoff line - also store it for path generation
                        start_point = (x1, y1)
                        end_point = (x2, y2)
                        gap_size = math.sqrt((x2-x1)**2 + (y2-y1)**2)  # Calculate line length as gap size
                        
                        # Determine category based on distance from agent
                        dist_start = math.sqrt((x1 - agent_x)**2 + (y1 - agent_y)**2)
                        dist_end = math.sqrt((x2 - agent_x)**2 + (y2 - agent_y)**2)
                        
                        if dist_start < dist_end:
                            category = "near_far_transition"
                        else:
                            category = "far_near_transition"
                        
                        breakoff_lines.append((start_point, end_point, gap_size, category))
    
    print(f"Loaded {len(environment_lines)} environment lines and {len(breakoff_lines)} breakoff lines from test_env2.txt")
    
    # Run algorithm using the API - same as force_window.py
    print("Running polygon exploration algorithm...")
    clipped_environment_lines = environment_lines
    test_breakoff_lines = breakoff_lines
    
    print(f"Using ALL {len(test_breakoff_lines)} breakoff lines for polygon exploration")
    for i, (start, end, gap, category) in enumerate(test_breakoff_lines):
        print(f"  Breakoff {i+1}: {start} to {end}, gap={gap:.1f}, category={category}")
    
    paths, exploration_graph = pe.calculate_polygon_exploration_paths(
        test_breakoff_lines, agent_x, agent_y, visibility_range, clipped_environment_lines
    )
    
    print(f"Generated {len(paths)} exploration paths")
    
    # Check if we have at least 3 paths (path 3 = index 2)
    if len(paths) < 3:
        print(f"ERROR: Only {len(paths)} paths found, but we need at least 3 to debug path 3!")
        return
    
    # Extract path 3 data (index 2)
    path3_data = paths[2]  # Path 3 is at index 2
    
    print("\n" + "=" * 80)
    print("PATH 3 DEBUG ANALYSIS")
    print("=" * 80)
    
    # Print detailed path 3 information
    breakoff_info = path3_data.get('breakoff_line', ['Unknown', 'Unknown', 0, 'Unknown'])
    path_points = path3_data.get('path_points', [])
    path_segments = path3_data.get('path_segments', [])
    completed = path3_data.get('completed', False)
    iterations = path3_data.get('iterations', 'Unknown')
    
    print(f"Breakoff Line: {breakoff_info[0]} to {breakoff_info[1]}")
    print(f"Gap Size: {breakoff_info[2]:.2f}")
    print(f"Category: {breakoff_info[3]}")
    print(f"Path Points: {len(path_points)}")
    print(f"Path Segments: {len(path_segments)}")
    print(f"Completed: {completed}")
    print(f"Iterations: {iterations}")
    
    print(f"\nPath 3 Points:")
    for i, point in enumerate(path_points):
        print(f"  Point {i+1}: ({point[0]:.2f}, {point[1]:.2f})")
    
    print(f"\nPath 3 Segments:")
    for i, segment in enumerate(path_segments):
        seg_type = segment.get('type', 'unknown')
        start = segment.get('start', 'unknown')
        end = segment.get('end', 'unknown')
        print(f"  Segment {i+1}: {seg_type} from {start} to {end}")
        
        if seg_type == 'arc' and 'edge_data' in segment:
            edge_data = segment['edge_data']
            if 'start_angle' in edge_data and 'end_angle' in edge_data:
                start_angle_deg = math.degrees(edge_data['start_angle'])
                end_angle_deg = math.degrees(edge_data['end_angle'])
                print(f"    Arc angles: {start_angle_deg:.1f}° to {end_angle_deg:.1f}°")
    
    # Create detailed visualization
    plt.ion()  # Interactive mode
    fig = plt.figure(figsize=(16, 12))
    
    # Create subplot layout: main plot + detail plots
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    ax_breakoff = plt.subplot2grid((3, 3), (0, 2))
    ax_segments = plt.subplot2grid((3, 3), (1, 2))
    ax_info = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    
    # === MAIN PLOT ===
    ax_main.set_aspect('equal')
    ax_main.set_title('Path 3 Detailed Debug View', fontsize=16, fontweight='bold')
    
    # Draw environment lines (dimmed)
    for line in environment_lines:
        x_coords = [line[0][0], line[1][0]]
        y_coords = [line[0][1], line[1][1]]
        ax_main.plot(x_coords, y_coords, 'k-', linewidth=1, alpha=0.3)
    
    # Draw agent and visibility circle
    ax_main.plot(agent_x, agent_y, 'ro', markersize=15, label='Agent', 
                markeredgecolor='darkred', markeredgewidth=2, zorder=10)
    circle = patches.Circle((agent_x, agent_y), visibility_range, 
                           fill=False, color='blue', linestyle='--', alpha=0.8, 
                           linewidth=3, label='Visibility Range')
    ax_main.add_patch(circle)
    
    # Highlight path 3's breakoff line
    if len(breakoff_info) >= 2:
        start_point, end_point = breakoff_info[0], breakoff_info[1]
        if start_point != 'Unknown' and end_point != 'Unknown':
            ax_main.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                        color='lime', linewidth=6, alpha=0.9, label='Path 3 Breakoff Line',
                        zorder=8)
            ax_main.plot([start_point[0]], [start_point[1]], 's', color='lime', 
                        markersize=15, markeredgecolor='darkgreen', markeredgewidth=3,
                        label='Breakoff Start', zorder=9)
            ax_main.plot([end_point[0]], [end_point[1]], 's', color='lime', 
                        markersize=15, markeredgecolor='darkgreen', markeredgewidth=3,
                        label='Breakoff End', zorder=9)
    
    # Draw path 3 with different colors for different segment types
    line_color = 'purple'
    arc_color = 'magenta'
    
    for i, segment in enumerate(path_segments):
        if segment['type'] == 'line':
            start = segment['start']
            end = segment['end']
            ax_main.plot([start[0], end[0]], [start[1], end[1]], 
                        color=line_color, linewidth=4, alpha=0.9, 
                        label='Line Segments' if i == 0 or all(s['type'] != 'line' for s in path_segments[:i]) else '',
                        zorder=7)
            # Add segment number
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax_main.annotate(f'L{i+1}', (mid_x, mid_y), fontsize=10, fontweight='bold',
                            color='white', ha='center', va='center',
                            bbox=dict(boxstyle='circle', facecolor=line_color, alpha=0.8))
            
        elif segment['type'] == 'arc':
            edge_data = segment.get('edge_data', {})
            if 'center' in edge_data and 'radius' in edge_data:
                center = edge_data['center']
                radius = edge_data['radius']
                start_angle = edge_data.get('start_angle', 0)
                end_angle = edge_data.get('end_angle', 0)
                
                # Handle angle wrapping
                if end_angle < start_angle:
                    end_angle += 2 * math.pi
                
                # Create arc with points for smooth curves
                num_points = max(20, int(abs(end_angle - start_angle) * 30))
                angles = [start_angle + j * (end_angle - start_angle) / num_points for j in range(num_points + 1)]
                arc_x = [center[0] + radius * math.cos(a) for a in angles]
                arc_y = [center[1] + radius * math.sin(a) for a in angles]
                
                ax_main.plot(arc_x, arc_y, color=arc_color, linewidth=4, alpha=0.9,
                            label='Arc Segments' if i == 0 or all(s['type'] != 'arc' for s in path_segments[:i]) else '',
                            zorder=7)
                # Add segment number at arc midpoint
                mid_angle = (start_angle + end_angle) / 2
                mid_x = center[0] + (radius + 15) * math.cos(mid_angle)
                mid_y = center[1] + (radius + 15) * math.sin(mid_angle)
                ax_main.annotate(f'A{i+1}', (mid_x, mid_y), fontsize=10, fontweight='bold',
                                color='white', ha='center', va='center',
                                bbox=dict(boxstyle='circle', facecolor=arc_color, alpha=0.8))
    
    # Mark path points with numbers
    for i, point in enumerate(path_points):
        if i == 0:
            # Start point
            ax_main.plot(point[0], point[1], 'o', color='red', markersize=12, 
                        markeredgecolor='darkred', markeredgewidth=2, zorder=10,
                        label='Path Start')
        elif i == len(path_points) - 1:
            # End point
            if completed:
                ax_main.plot(point[0], point[1], 'o', color='blue', markersize=12, 
                            markeredgecolor='darkblue', markeredgewidth=2, zorder=10,
                            label='Path End (Complete)')
            else:
                ax_main.plot(point[0], point[1], 'x', color='orange', markersize=12, 
                            markeredgewidth=3, zorder=10, label='Path End (Incomplete)')
        else:
            # Intermediate points
            ax_main.plot(point[0], point[1], 'o', color='yellow', markersize=8, 
                        markeredgecolor='gold', markeredgewidth=1, zorder=9,
                        label='Intermediate Points' if i == 1 else '')
        
        # Add point number
        ax_main.annotate(f'{i+1}', point, xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax_main.legend(loc='upper right')
    ax_main.grid(True, alpha=0.3)
    
    # Set view to focus on path 3
    if path_points:
        all_x = [p[0] for p in path_points]
        all_y = [p[1] for p in path_points]
        
        # Include breakoff line endpoints
        if len(breakoff_info) >= 2 and breakoff_info[0] != 'Unknown':
            all_x.extend([breakoff_info[0][0], breakoff_info[1][0]])
            all_y.extend([breakoff_info[0][1], breakoff_info[1][1]])
        
        # Include agent position
        all_x.append(agent_x)
        all_y.append(agent_y)
        
        margin = 50
        ax_main.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax_main.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    # === BREAKOFF LINE DETAIL ===
    ax_breakoff.set_title('Breakoff Line Detail', fontsize=12, fontweight='bold')
    ax_breakoff.set_aspect('equal')
    
    if len(breakoff_info) >= 2 and breakoff_info[0] != 'Unknown':
        start_point, end_point = breakoff_info[0], breakoff_info[1]
        
        # Draw visibility circle
        circle_detail = patches.Circle((agent_x, agent_y), visibility_range, 
                                      fill=False, color='blue', linestyle='--', alpha=0.6)
        ax_breakoff.add_patch(circle_detail)
        
        # Draw agent
        ax_breakoff.plot(agent_x, agent_y, 'ro', markersize=8)
        
        # Draw breakoff line
        ax_breakoff.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                        'lime', linewidth=4, marker='s', markersize=8)
        
        # Annotate distances
        dist_start = math.sqrt((start_point[0] - agent_x)**2 + (start_point[1] - agent_y)**2)
        dist_end = math.sqrt((end_point[0] - agent_x)**2 + (end_point[1] - agent_y)**2)
        
        ax_breakoff.annotate(f'Start\nDist: {dist_start:.1f}', start_point, 
                           xytext=(10, 10), textcoords='offset points', fontsize=8)
        ax_breakoff.annotate(f'End\nDist: {dist_end:.1f}', end_point, 
                           xytext=(10, -20), textcoords='offset points', fontsize=8)
        
        # Set view around breakoff line
        margin = 100
        all_x = [start_point[0], end_point[0], agent_x]
        all_y = [start_point[1], end_point[1], agent_y]
        ax_breakoff.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax_breakoff.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    ax_breakoff.grid(True, alpha=0.3)
    
    # === SEGMENTS DETAIL ===
    ax_segments.set_title('Segment Analysis', fontsize=12, fontweight='bold')
    ax_segments.axis('off')
    
    segment_text = f"Path 3 Segments ({len(path_segments)} total):\n\n"
    for i, segment in enumerate(path_segments):
        seg_type = segment.get('type', 'unknown')
        start = segment.get('start', 'unknown')
        end = segment.get('end', 'unknown')
        
        if seg_type == 'line' and start != 'unknown' and end != 'unknown':
            length = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            segment_text += f"{i+1}. LINE: {length:.2f} units\n"
            segment_text += f"   {start[0]:.1f},{start[1]:.1f} → {end[0]:.1f},{end[1]:.1f}\n\n"
        elif seg_type == 'arc':
            edge_data = segment.get('edge_data', {})
            if 'start_angle' in edge_data and 'end_angle' in edge_data:
                start_angle = edge_data['start_angle']
                end_angle = edge_data['end_angle']
                if end_angle < start_angle:
                    end_angle += 2 * math.pi
                arc_length = visibility_range * abs(end_angle - start_angle)
                segment_text += f"{i+1}. ARC: {arc_length:.2f} units\n"
                segment_text += f"   {math.degrees(start_angle):.1f}° → {math.degrees(end_angle):.1f}°\n\n"
            else:
                segment_text += f"{i+1}. ARC: (no angle data)\n\n"
        else:
            segment_text += f"{i+1}. {seg_type.upper()}: (unknown data)\n\n"
    
    ax_segments.text(0.05, 0.95, segment_text, transform=ax_segments.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    # === INFO PANEL ===
    ax_info.set_title('Path 3 Summary', fontsize=12, fontweight='bold')
    ax_info.axis('off')
    
    info_text = f"""
PATH 3 DEBUGGING INFORMATION:

Status: {'✓ COMPLETED' if completed else '✗ INCOMPLETE'}
Iterations: {iterations}
Total Points: {len(path_points)}
Total Segments: {len(path_segments)}

Breakoff Line Info:
  Start: {breakoff_info[0] if breakoff_info[0] != 'Unknown' else 'Unknown'}
  End: {breakoff_info[1] if breakoff_info[1] != 'Unknown' else 'Unknown'}
  Gap Size: {breakoff_info[2]:.2f} units
  Category: {breakoff_info[3]}

Agent Position: ({agent_x}, {agent_y})
Visibility Range: {visibility_range} units

Segment Breakdown:
  Line Segments: {sum(1 for s in path_segments if s.get('type') == 'line')}
  Arc Segments: {sum(1 for s in path_segments if s.get('type') == 'arc')}
  
Path Length: {sum(
    math.sqrt((s['end'][0] - s['start'][0])**2 + (s['end'][1] - s['start'][1])**2) 
    if s.get('type') == 'line' and 'start' in s and 'end' in s
    else (visibility_range * abs(s['edge_data']['end_angle'] - s['edge_data']['start_angle']) 
          if s.get('type') == 'arc' and 'edge_data' in s and 'start_angle' in s['edge_data']
          else 0)
    for s in path_segments
):.2f} units
"""
    
    ax_info.text(0.02, 0.98, info_text, transform=ax_info.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Force window to front
    fig.canvas.manager.window.wm_attributes('-topmost', 1)
    fig.canvas.manager.window.wm_attributes('-topmost', 0)
    
    plt.tight_layout()
    plt.show(block=False)
    
    print("\n" + "=" * 80)
    print("PATH 3 DEBUG WINDOW IS NOW OPEN!")
    print("=" * 80)
    print("The window shows:")
    print("1. Main Plot: Detailed view of Path 3 with segment numbers")
    print("2. Breakoff Detail: Close-up of the breakoff line")
    print("3. Segment Analysis: Text breakdown of each segment")
    print("4. Info Panel: Complete summary of Path 3 data")
    print("")
    print("Legend:")
    print("- Red circle: Path start")
    print("- Blue circle: Completed path end")
    print("- Orange X: Incomplete path end")
    print("- Yellow circles: Intermediate points")
    print("- Purple lines: Line segments (marked L1, L2, etc.)")
    print("- Magenta arcs: Arc segments (marked A1, A2, etc.)")
    print("- Lime line: Breakoff line that generated this path")
    print("")
    print("Use the toolbar to zoom/pan for detailed inspection.")
    print("Press Enter to close the window...")
    print("=" * 80)
    
    input()  # Wait for user input
    plt.close('all')

if __name__ == "__main__":
    debug_path_3()
