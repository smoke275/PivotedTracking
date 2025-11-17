#!/usr/bin/env python3
"""
Reachability Path Viewer - Visualize intersection graphs and polygon exploration paths.
"""

import os
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import polygon_exploration as pe
import polygon_exploration_cpp as pec


class RulerTool:
    """Interactive ruler tool for measuring distances by clicking and dragging."""
    
    def __init__(self, ax):
        self.ax = ax
        self.start_point = None
        self.end_point = None
        self.ruler_line = None
        self.ruler_text = None
        self.start_marker = None
        self.end_marker = None
        self.is_measuring = False
        
        # Connect mouse events
        self.press_cid = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.motion_cid = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.release_cid = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.key_cid = ax.figure.canvas.mpl_connect('key_press_event', self.on_key)
        
        print("🔧 Ruler tool activated:")
        print("   • Left-click and drag to measure distance")
        print("   • Press 'r' to clear ruler")
        print("   • Press 'h' to show/hide help")
    
    def on_press(self, event):
        """Handle mouse press - start measuring."""
        if event.inaxes != self.ax or event.button != 1:  # Only left mouse button
            return
        
        # Clear previous ruler
        self.clear_ruler()
        
        self.start_point = (event.xdata, event.ydata)
        self.is_measuring = True
    
    def on_motion(self, event):
        """Handle mouse motion - update ruler line."""
        if not self.is_measuring or event.inaxes != self.ax:
            return
        
        if self.start_point is None:
            return
        
        self.end_point = (event.xdata, event.ydata)
        self.update_ruler()
    
    def on_release(self, event):
        """Handle mouse release - finish measuring."""
        if not self.is_measuring or event.inaxes != self.ax:
            return
        
        if self.start_point is None:
            return
        
        self.end_point = (event.xdata, event.ydata)
        self.is_measuring = False
        self.update_ruler()
        
        # Calculate and display distance
        distance = self.calculate_distance()
        print(f"📏 Measured distance: {distance:.2f} units")
    
    def on_key(self, event):
        """Handle key press events."""
        if event.key == 'r':
            self.clear_ruler()
            print("🔧 Ruler cleared")
        elif event.key == 'h':
            self.show_help()
    
    def calculate_distance(self):
        """Calculate distance between start and end points."""
        if self.start_point is None or self.end_point is None:
            return 0.0
        
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def update_ruler(self):
        """Update the visual ruler line and text."""
        if self.start_point is None or self.end_point is None:
            return
        
        # Remove old ruler elements
        if self.ruler_line is not None:
            self.ruler_line.remove()
        if self.ruler_text is not None:
            self.ruler_text.remove()
        if self.start_marker is not None:
            self.start_marker.remove()
        if self.end_marker is not None:
            self.end_marker.remove()
        
        # Draw new ruler line
        self.ruler_line, = self.ax.plot(
            [self.start_point[0], self.end_point[0]], 
            [self.start_point[1], self.end_point[1]], 
            'r-', linewidth=3, alpha=0.8, zorder=100
        )
        
        # Add distance text at midpoint
        mid_x = (self.start_point[0] + self.end_point[0]) / 2
        mid_y = (self.start_point[1] + self.end_point[1]) / 2
        distance = self.calculate_distance()
        
        self.ruler_text = self.ax.text(
            mid_x, mid_y, f'{distance:.1f}', 
            fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
            ha='center', va='center', zorder=101
        )
        
        # Draw start and end markers (store references for removal)
        self.start_marker, = self.ax.plot(self.start_point[0], self.start_point[1], 'ro', 
                                         markersize=8, zorder=102, alpha=0.8)
        self.end_marker, = self.ax.plot(self.end_point[0], self.end_point[1], 'ro', 
                                       markersize=8, zorder=102, alpha=0.8)
        
        self.ax.figure.canvas.draw()
    
    def clear_ruler(self):
        """Clear the ruler from the plot."""
        if self.ruler_line is not None:
            self.ruler_line.remove()
            self.ruler_line = None
        if self.ruler_text is not None:
            self.ruler_text.remove()
            self.ruler_text = None
        if self.start_marker is not None:
            self.start_marker.remove()
            self.start_marker = None
        if self.end_marker is not None:
            self.end_marker.remove()
            self.end_marker = None
        
        self.start_point = None
        self.end_point = None
        self.is_measuring = False
        self.ax.figure.canvas.draw()
    
    def show_help(self):
        """Show help information."""
        print("\n🔧 Ruler Tool Help:")
        print("   • Left-click and drag to measure distance")
        print("   • Distance appears as yellow text box")
        print("   • Press 'r' to clear current ruler")
        print("   • Press 'h' to show this help")
        print("   • Each measurement is logged to console\n")
    
    def disconnect(self):
        """Disconnect all event handlers."""
        self.ax.figure.canvas.mpl_disconnect(self.press_cid)
        self.ax.figure.canvas.mpl_disconnect(self.motion_cid)
        self.ax.figure.canvas.mpl_disconnect(self.release_cid)
        self.ax.figure.canvas.mpl_disconnect(self.key_cid)


def load_environment_data(env_filename='test_env2.txt', vis_filename='vis_test2.txt'):
    """Load environment lines, breakoff lines, and agent data from files."""
    environment_lines = []
    breakoff_lines = []
    visibility_polygon = []
    
    # Default values (fallback if vis file is not found)
    agent_x, agent_y = 403.8, 468.1
    agent_orientation = 0.0
    visibility_range = 200.0
    
    # Load environment lines and breakoff lines
    try:
        with open(env_filename, 'r') as f:
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
        
        print(f"✅ Environment: Loaded {len(environment_lines)} lines and {len(breakoff_lines)} breakoff lines from {env_filename}")
    
    except Exception as e:
        print(f"❌ Failed to load environment from {env_filename}: {e}")
        raise
    
    # Load agent data and visibility polygon
    try:
        with open(vis_filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('AGENT_POSITION'):
                    parts = line.split()
                    if len(parts) >= 3:
                        agent_x, agent_y = float(parts[1]), float(parts[2])
                        print(f"✅ Agent position: ({agent_x:.3f}, {agent_y:.3f})")
                
                elif line.startswith('AGENT_ORIENTATION'):
                    parts = line.split()
                    if len(parts) >= 2:
                        agent_orientation = float(parts[1])
                        print(f"✅ Agent orientation: {agent_orientation:.3f} radians ({math.degrees(agent_orientation):.1f}°)")
                
                elif line.startswith('VISIBILITY_RANGE'):
                    parts = line.split()
                    if len(parts) >= 2:
                        visibility_range = float(parts[1])
                        print(f"✅ Visibility range: {visibility_range:.1f} units")
                
                elif line.startswith('VERTEX'):
                    parts = line.split()
                    if len(parts) >= 3:
                        x, y = float(parts[1]), float(parts[2])
                        visibility_polygon.append((x, y))
        
        print(f"✅ Visibility: Loaded {len(visibility_polygon)} polygon vertices from {vis_filename}")
        
        # Now that we have the actual agent position, recalculate breakoff line categories
        if breakoff_lines:
            updated_breakoff_lines = []
            for start_point, end_point, gap_size, _ in breakoff_lines:
                # Recalculate category with correct agent position
                dist_start = math.sqrt((start_point[0] - agent_x)**2 + (start_point[1] - agent_y)**2)
                dist_end = math.sqrt((end_point[0] - agent_x)**2 + (end_point[1] - agent_y)**2)
                
                if dist_start < dist_end:
                    category = "near_far_transition"
                else:
                    category = "far_near_transition"
                
                updated_breakoff_lines.append((start_point, end_point, gap_size, category))
            breakoff_lines = updated_breakoff_lines
            print(f"✅ Updated breakoff line categories with correct agent position")
    
    except Exception as e:
        print(f"⚠️  Could not load visibility data from {vis_filename}: {e}")
        print(f"   Using default agent values: pos=({agent_x:.1f}, {agent_y:.1f}), range={visibility_range:.1f}")
    
    return environment_lines, breakoff_lines, agent_x, agent_y, visibility_range, agent_orientation, visibility_polygon

def show_intersection_graph(environment_lines, agent_x, agent_y, visibility_range, show_environment=True):
    """Show the intersection graph in a separate window.
    
    Args:
        environment_lines: List of environment line segments
        agent_x, agent_y: Agent position
        visibility_range: Visibility radius
        show_environment: If True, show environment lines. If False, show only the graph.
    """
    
    # Create intersection graph
    graph = pe.IntersectionGraph(environment_lines, agent_x, agent_y, visibility_range)
    
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Debug: Print all nodes
    print("\n=== GRAPH NODES ===")
    line_circle_nodes = 0
    line_line_nodes = 0
    unified_nodes = 0
    for node_id, node_data in graph.nodes.items():
        point = node_data['point']
        angle = node_data['angle']
        intersection_type = node_data.get('intersection_type', 'unknown')
        
        if intersection_type == 'line_circle':
            line_circle_nodes += 1
        elif intersection_type == 'line_line':
            line_line_nodes += 1
        elif intersection_type == 'unified':
            unified_nodes += 1
            
        print(f"Node {node_id}: point=({point[0]:.2f}, {point[1]:.2f}), angle={angle:.3f} rad ({math.degrees(angle):.1f}°), type={intersection_type}")
    
    print(f"\nNode Summary: {line_circle_nodes} line-circle nodes, {line_line_nodes} line-line nodes, {unified_nodes} unified nodes")
    
    print(f"\n=== GRAPH EDGES ===")
    line_edges = 0
    arc_edges = 0
    for edge_id, edge_data in graph.edges.items():
        if edge_data['type'] == 'line':
            line_edges += 1
        elif edge_data['type'] == 'arc':
            arc_edges += 1
    
    print(f"Edge Summary: {line_edges} line edges, {arc_edges} arc edges")
    
    # Create figure for graph visualization with navigation toolbar
    fig_graph = plt.figure(figsize=(14, 10))
    ax_graph = fig_graph.add_subplot(111)
    ax_graph.set_aspect('equal')
    
    title = 'Intersection Graph Visualization (Use toolbar to zoom/pan)'
    if not show_environment:
        title += ' - Graph Only Mode'
    ax_graph.set_title(title, fontsize=16)
    
    # Draw environment lines (optional)
    if show_environment:
        for i, line in enumerate(environment_lines):
            x_coords = [line[0][0], line[1][0]]
            y_coords = [line[0][1], line[1][1]]
            ax_graph.plot(x_coords, y_coords, 'k-', linewidth=2, alpha=0.8, 
                         label='Environment' if i == 0 else '')
    
    # Let's also print ALL environment lines to see what you might be referring to
    print(f"\n=== ALL ENVIRONMENT LINES ===")
    for i, line in enumerate(environment_lines):
        line_start, line_end = line
        
        # Calculate distance from agent to this line
        line_vec_x = line_end[0] - line_start[0]
        line_vec_y = line_end[1] - line_start[1]
        line_length_sq = line_vec_x**2 + line_vec_y**2
        
        if line_length_sq == 0:
            dist_to_agent = math.sqrt((line_start[0] - agent_x)**2 + (line_start[1] - agent_y)**2)
        else:
            agent_vec_x = agent_x - line_start[0]
            agent_vec_y = agent_y - line_start[1]
            projection = (agent_vec_x * line_vec_x + agent_vec_y * line_vec_y) / line_length_sq
            projection = max(0, min(1, projection))
            
            closest_x = line_start[0] + projection * line_vec_x
            closest_y = line_start[1] + projection * line_vec_y
            dist_to_agent = math.sqrt((closest_x - agent_x)**2 + (closest_y - agent_y)**2)
        
        # Check if line has intersections
        intersections = pe.line_circle_intersections(
            line_start[0], line_start[1], line_end[0], line_end[1],
            agent_x, agent_y, visibility_range
        )
        
        status = "INTERSECTS" if len(intersections) > 0 else "NO_INTERSECT"
        if 195.0 <= dist_to_agent <= 205.0:  # Lines close to the visibility range
            print(f"Line {i:2d}: {line_start} to {line_end} | dist={dist_to_agent:.1f} | {status}")
        elif len(intersections) > 0:
            print(f"Line {i:2d}: {line_start} to {line_end} | dist={dist_to_agent:.1f} | {status} ({len(intersections)} pts)")
    
    # Draw agent and visibility circle
    ax_graph.plot(agent_x, agent_y, 'ro', markersize=10, label='Agent')
    circle = patches.Circle((agent_x, agent_y), visibility_range, 
                           fill=False, color='blue', linestyle='--', alpha=0.6, linewidth=2, label='Visibility Range')
    ax_graph.add_patch(circle)
    
    # Draw graph nodes with different colors and larger sizes for better visibility
    line_circle_first_id = None
    line_line_first_id = None
    unified_first_id = None
    
    for node_id, node_data in graph.nodes.items():
        point = node_data['point']
        angle = node_data['angle']
        intersection_type = node_data.get('intersection_type', 'unknown')
        
        if intersection_type == 'line_circle':
            # Bright green circles for line-circle intersections (on visibility circle only)
            ax_graph.plot(point[0], point[1], 'o', color='lime', markersize=12, alpha=0.9, 
                         markeredgecolor='darkgreen', markeredgewidth=2,
                         label='Line-Circle Nodes (on visibility circle)' if line_circle_first_id is None else '')
            if line_circle_first_id is None:
                line_circle_first_id = node_id
        elif intersection_type == 'line_line':
            # Bright orange diamonds for line-line intersections (not on visibility circle)
            ax_graph.plot(point[0], point[1], 'D', color='orange', markersize=10, alpha=0.9,
                         markeredgecolor='darkorange', markeredgewidth=2,
                         label='Line-Line Nodes (line intersections)' if line_line_first_id is None else '')
            if line_line_first_id is None:
                line_line_first_id = node_id
        elif intersection_type == 'unified':
            # Purple stars for unified nodes (both line-circle and line-line)
            ax_graph.plot(point[0], point[1], '*', color='purple', markersize=16, alpha=0.9,
                         markeredgecolor='darkmagenta', markeredgewidth=2,
                         label='Unified Nodes (both line-circle & line-line)' if unified_first_id is None else '')
            if unified_first_id is None:
                unified_first_id = node_id
        else:
            # Red squares for unknown type
            ax_graph.plot(point[0], point[1], 's', color='red', markersize=10, alpha=0.9,
                         markeredgecolor='darkred', markeredgewidth=2)
        
        # Add node ID labels with better visibility
        ax_graph.annotate(f'{node_id}', (point[0], point[1]), 
                         xytext=(8, 8), textcoords='offset points', 
                         fontsize=10, alpha=0.9, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # Draw graph edges
    line_edge_count = 0
    arc_edge_count = 0
    
    for edge_id, edge_data in graph.edges.items():
        from_node = graph.nodes[edge_data['from_node']]
        to_node = graph.nodes[edge_data['to_node']]
        from_point = from_node['point']
        to_point = to_node['point']
        
        if edge_data['type'] == 'line':
            # Draw line segment edges with better visibility and distinct patterns
            line_styles = ['-', '--', '-.', ':']
            line_colors = ['red', 'darkred', 'crimson', 'firebrick']
            line_color = line_colors[line_edge_count % len(line_colors)]
            line_style = line_styles[line_edge_count % len(line_styles)]
            
            ax_graph.plot([from_point[0], to_point[0]], [from_point[1], to_point[1]], 
                         color=line_color, linewidth=2, alpha=0.7, linestyle=line_style,
                         label='Line Edges (connecting intersections along environment lines)' if line_edge_count == 0 else '')
            
            # Add edge ID label at midpoint for every 5th edge to reduce clutter
            if line_edge_count % 5 == 0:
                mid_x = (from_point[0] + to_point[0]) / 2
                mid_y = (from_point[1] + to_point[1]) / 2
                ax_graph.annotate(f'L{edge_id}', (mid_x, mid_y), 
                                 xytext=(0, 0), textcoords='offset points', 
                                 fontsize=9, alpha=0.8, color=line_color,
                                 ha='center', va='center', weight='bold',
                                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
            line_edge_count += 1
            
        elif edge_data['type'] == 'arc':
            # Draw arc edges with better visibility and thicker lines
            arc_data = edge_data['data']
            start_angle = arc_data['start_angle']
            end_angle = arc_data['end_angle']
            
            # Distinct colors for arc edges - all solid lines
            arc_colors = ['blue', 'navy', 'royalblue', 'steelblue', 'dodgerblue', 'deepskyblue']
            arc_color = arc_colors[arc_edge_count % len(arc_colors)]
            
            # Handle angle wrapping
            original_end_angle = end_angle
            if end_angle < start_angle:
                end_angle += 2 * math.pi
            
            # Create arc with more points for smoother curves
            angles = [start_angle + i * (end_angle - start_angle) / 30 for i in range(31)]
            arc_x = [agent_x + visibility_range * math.cos(a) for a in angles]
            arc_y = [agent_y + visibility_range * math.sin(a) for a in angles]
            
            # Draw solid arc lines
            ax_graph.plot(arc_x, arc_y, color=arc_color, linewidth=4, alpha=0.8, linestyle='-',
                         label='Arc Edges (connecting intersections along visibility circle)' if arc_edge_count == 0 else '')
            
            # Add edge ID label at arc midpoint with better visibility
            mid_angle = (start_angle + end_angle) / 2
            mid_x = agent_x + (visibility_range + 20) * math.cos(mid_angle)
            mid_y = agent_y + (visibility_range + 20) * math.sin(mid_angle)
            ax_graph.annotate(f'A{edge_id}', (mid_x, mid_y), 
                             xytext=(0, 0), textcoords='offset points', 
                             fontsize=11, alpha=0.9, color=arc_color,
                             ha='center', va='center', weight='bold',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=arc_color))
            arc_edge_count += 1
    
    # Add graph statistics as text
    stats_text = f"Graph Statistics:\n"
    stats_text += f"Nodes: {len(graph.nodes)}\n"
    stats_text += f"Line Edges: {line_edge_count}\n"
    stats_text += f"Arc Edges: {arc_edge_count}\n"
    stats_text += f"Total Edges: {len(graph.edges)}"
    
    ax_graph.text(0.02, 0.98, stats_text, transform=ax_graph.transAxes, 
                 fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax_graph.legend()
    ax_graph.grid(True, alpha=0.3)
    
    # Set axis limits to show the full environment context
    all_x = [node['point'][0] for node in graph.nodes.values()]
    all_y = [node['point'][1] for node in graph.nodes.values()]
    
    # Also include all environment line endpoints
    for line in environment_lines:
        all_x.extend([line[0][0], line[1][0]])
        all_y.extend([line[0][1], line[1][1]])
    
    # Include agent position and visibility circle bounds
    all_x.extend([agent_x - visibility_range, agent_x + visibility_range])
    all_y.extend([agent_y - visibility_range, agent_y + visibility_range])
    
    if all_x and all_y:
        margin = 100  # Larger margin to show more context
        ax_graph.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax_graph.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    # Store references for interactive functionality
    highlighted_edges = []  # Store highlighted edge plot objects
    highlighted_node = None  # Store highlighted node plot object
    highlighted_node_labels = []  # Store highlighted connected node label objects
    selected_node_id = None  # Store currently selected node ID
    
    def find_closest_node(x, y, max_distance=20):
        """Find the closest node to the click coordinates."""
        min_distance = float('inf')
        closest_node_id = None
        
        for node_id, node_data in graph.nodes.items():
            point = node_data['point']
            # Convert to display coordinates
            display_point = ax_graph.transData.transform([point[0], point[1]])
            click_point = ax_graph.transData.transform([x, y])
            
            distance = math.sqrt((display_point[0] - click_point[0])**2 + 
                               (display_point[1] - click_point[1])**2)
            
            if distance < max_distance and distance < min_distance:
                min_distance = distance
                closest_node_id = node_id
                
        return closest_node_id
    
    def highlight_node_edges(node_id):
        """Highlight all edges connected to the given node."""
        nonlocal highlighted_edges, highlighted_node, highlighted_node_labels, selected_node_id
        
        # Clear previous highlights
        for edge_plot in highlighted_edges:
            try:
                edge_plot.remove()
            except:
                pass
        highlighted_edges.clear()
        
        for label_plot in highlighted_node_labels:
            try:
                label_plot.remove()
            except:
                pass
        highlighted_node_labels.clear()
        
        if highlighted_node:
            try:
                highlighted_node.remove()
            except:
                pass
            highlighted_node = None
        
        if node_id is None:
            selected_node_id = None
            fig_graph.canvas.draw()
            return
        
        selected_node_id = node_id
        node_data = graph.nodes[node_id]
        point = node_data['point']
        
        print(f"\n=== CLICKED NODE {node_id} ===")
        print(f"Position: ({point[0]:.2f}, {point[1]:.2f})")
        print(f"Type: {node_data.get('intersection_type', 'unknown')}")
        print(f"Angle: {math.degrees(node_data['angle']):.1f}°")
        
        # Highlight the selected node with a large yellow circle
        highlighted_node = ax_graph.plot(point[0], point[1], 'o', color='yellow', 
                                       markersize=20, alpha=0.7, 
                                       markeredgecolor='orange', markeredgewidth=3)[0]
        
        # Find and highlight all connected edges
        connected_edges = []
        connected_nodes = set()
        for edge_id, edge_data in graph.edges.items():
            if edge_data['from_node'] == node_id or edge_data['to_node'] == node_id:
                connected_edges.append((edge_id, edge_data))
                # Add the other node to connected nodes set
                if edge_data['from_node'] == node_id:
                    connected_nodes.add(edge_data['to_node'])
                else:
                    connected_nodes.add(edge_data['from_node'])
        
        print(f"Connected edges: {len(connected_edges)}")
        print(f"Connected nodes: {sorted(connected_nodes)}")
        
        # Add prominent labels for connected nodes
        for connected_node_id in connected_nodes:
            connected_node = graph.nodes[connected_node_id]
            connected_point = connected_node['point']
            
            # Create a large, prominent label for the connected node
            label_plot = ax_graph.annotate(f'NODE {connected_node_id}', 
                                         (connected_point[0], connected_point[1]), 
                                         xytext=(15, 15), textcoords='offset points', 
                                         fontsize=14, fontweight='bold', color='red',
                                         bbox=dict(boxstyle='round,pad=0.5', 
                                                 facecolor='yellow', alpha=0.9, 
                                                 edgecolor='red', linewidth=2),
                                         arrowprops=dict(arrowstyle='->', 
                                                       connectionstyle='arc3,rad=0.1',
                                                       color='red', lw=2),
                                         zorder=15)
            highlighted_node_labels.append(label_plot)
        
        for edge_id, edge_data in connected_edges:
            from_node = graph.nodes[edge_data['from_node']]
            to_node = graph.nodes[edge_data['to_node']]
            from_point = from_node['point']
            to_point = to_node['point']
            
            print(f"  Edge {edge_id}: {edge_data['type']} from node {edge_data['from_node']} to node {edge_data['to_node']}")
            
            if edge_data['type'] == 'line':
                # Highlight line edge with thick yellow line
                edge_plot = ax_graph.plot([from_point[0], to_point[0]], 
                                        [from_point[1], to_point[1]], 
                                        color='yellow', linewidth=6, alpha=0.8, 
                                        linestyle='-', zorder=10)[0]
                highlighted_edges.append(edge_plot)
                
            elif edge_data['type'] == 'arc':
                # Highlight arc edge with thick yellow arc
                arc_data = edge_data['data']
                start_angle = arc_data['start_angle']
                end_angle = arc_data['end_angle']
                
                # Handle angle wrapping
                if end_angle < start_angle:
                    end_angle += 2 * math.pi
                
                # Create arc with more points for smoother curves
                angles = [start_angle + i * (end_angle - start_angle) / 50 for i in range(51)]
                arc_x = [agent_x + visibility_range * math.cos(a) for a in angles]
                arc_y = [agent_y + visibility_range * math.sin(a) for a in angles]
                
                edge_plot = ax_graph.plot(arc_x, arc_y, color='yellow', 
                                        linewidth=8, alpha=0.8, linestyle='-', 
                                        zorder=10)[0]
                highlighted_edges.append(edge_plot)
        
        fig_graph.canvas.draw()
    
    def on_click(event):
        """Handle mouse click events."""
        if event.inaxes != ax_graph:
            return
        
        if event.button == 1:  # Left click
            # Find the closest node to the click
            closest_node = find_closest_node(event.xdata, event.ydata)
            
            if closest_node is not None:
                if closest_node == selected_node_id:
                    # Clicking the same node again deselects it
                    print(f"\nDeselecting node {closest_node}")
                    highlight_node_edges(None)
                else:
                    # Select new node
                    highlight_node_edges(closest_node)
            else:
                # Click on empty space deselects current node
                if selected_node_id is not None:
                    print(f"\nDeselecting node {selected_node_id}")
                    highlight_node_edges(None)
    
    # Connect the click handler
    fig_graph.canvas.mpl_connect('button_press_event', on_click)
    
    # Update the title to include interaction instructions
    title = 'Intersection Graph Visualization (Click nodes to highlight edges)'
    if not show_environment:
        title += ' - Graph Only Mode'
    ax_graph.set_title(title, fontsize=16)
    
    # Add instruction text
    instruction_text = "INTERACTION:\n• Left-click on any node to highlight its edges & label connected nodes\n• Click same node again to deselect\n• Click empty space to deselect"
    ax_graph.text(0.02, 0.02, instruction_text, transform=ax_graph.transAxes, 
                 fontsize=9, verticalalignment='bottom', 
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Force window to front
    fig_graph.canvas.manager.window.wm_attributes('-topmost', 1)
    fig_graph.canvas.manager.window.wm_attributes('-topmost', 0)
    
    plt.tight_layout()
    plt.show(block=False)
    
    return graph


def show_visualization():
    """Show the visualization in a window that definitely appears."""
    
    # Load data using the enhanced loader function
    print("📂 Loading environment and visibility data...")
    try:
        environment_lines, breakoff_lines, agent_x, agent_y, visibility_range, agent_orientation, visibility_polygon = load_environment_data('test_env2.txt', 'vis_test2.txt')
        print(f"✅ Loaded {len(environment_lines)} environment lines and {len(breakoff_lines)} breakoff lines")
        if (agent_x, agent_y) != (0.0, 0.0):
            print(f"🤖 Agent position: ({agent_x}, {agent_y}), orientation: {math.degrees(agent_orientation):.2f}°")
        if visibility_polygon is not None and len(visibility_polygon) > 0:
            print(f"👁️ Visibility polygon: {len(visibility_polygon)} vertices")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Skip intersection graph visualization - only show paths
    # Run algorithm using the API - let the polygon_exploration module handle all processing
    print("Running polygon exploration algorithm...")
    # Convert environment_lines to the expected format for clipped_environment_lines
    # The API expects already processed/clipped environment lines
    clipped_environment_lines = environment_lines  # Use the loaded environment lines directly
    
    # Use the breakoff lines loaded from the file
    print(f"Loaded {len(breakoff_lines)} breakoff lines from test_env.txt")
    
    # Use ALL breakoff lines for full testing
    test_breakoff_lines = breakoff_lines
    print(f"Using ALL {len(test_breakoff_lines)} breakoff lines for polygon exploration")
    for i, (start, end, gap, category) in enumerate(test_breakoff_lines):
        print(f"  Breakoff {i+1}: {start} to {end}, gap={gap:.1f}, category={category}")
    
    paths, exploration_graph = pe.calculate_polygon_exploration_paths(
        test_breakoff_lines, agent_x, agent_y, visibility_range, clipped_environment_lines
    )
    
    # Use the new unified API to process both path analysis and reachability overlay
    print("🔄 Processing path analysis and reachability overlay using unified API...")
    reachability_data = None
    sample_points_data = []  # Initialize outside try block for global access
    try:
        # Import the reachability mask API and timing module
        from reachability_mask_min_api import ReachabilityMaskAPI
        import time
        
        # Configuration constants for overlay
        CLIP_PIXELS = 32.0
        RESIZE_TARGET = (120, 120)
        UPSCALE_TARGET = (400, 400)
        DOWNSAMPLE_METHOD = 'max_pool'
        # Probability weighting: None for no weighting, or (alpha, beta) for Prelect weighting
        # Example: (0.88, 0.88) for typical Prelect parameters, (0.5, 1.5) for different risk attitudes
        PROBABILITY_WEIGHTING = (0.55, 0.8)  # None to disable, (alpha, beta) to enable
        
        # Create overlay API instance
        overlay_api = ReachabilityMaskAPI(filename_base="unicycle_grid")
        
        if overlay_api.is_loaded():
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
                # STEP 2: Process paths with the configured overlay (per-iteration) - TIMED
                print("🔄 Processing paths with pre-configured overlay...")
                process_start_time = time.perf_counter()
                path_analysis_data, reachability_data, sample_points_data = overlay_api.process_paths_with_reachability(
                    paths=paths,
                    agent_x=agent_x,
                    agent_y=agent_y,
                    agent_orientation=agent_orientation,
                    visibility_range=visibility_range,
                    visibility_polygon=visibility_polygon
                )
                process_end_time = time.perf_counter()
                process_duration = process_end_time - process_start_time
                
                # Store timing data for later display
                timing_data = {
                    'config_duration': config_duration,
                    'process_duration': process_duration,
                    'total_duration': config_duration + process_duration
                }
            else:
                print(f"⚠️ Failed to configure reachability overlay")
                # Fallback: compute path analysis data without reachability overlay
                path_analysis_data = []
                sample_points_data = []
                timing_data = None
        else:
            print(f"⚠️ Failed to load reachability data from unicycle_grid.pkl")
            # Fallback: compute path analysis data without reachability overlay
            path_analysis_data = []
            sample_points_data = []
            timing_data = None
    except Exception as e:
        print(f"⚠️ Could not load reachability overlay: {e}")
        print(f"   Continuing without overlay...")
        # Fallback: compute path analysis data without reachability overlay  
        path_analysis_data = []
        sample_points_data = []
        timing_data = None
    
    # Extract data from API results for visualization
    intersections = []
    
    # Try to get breakoff lines from the API if it provides them
    if hasattr(pe, 'get_last_breakoff_lines'):
        breakoff_lines = pe.get_last_breakoff_lines()
    
    # Get intersections from the API if available, otherwise calculate for visualization only
    if hasattr(pe, 'get_last_intersections'):
        intersections = pe.get_last_intersections()
    else:
        # Fallback: calculate intersections for visualization only
        for line_segment in environment_lines:
            line_start, line_end = line_segment
            line_intersections = pe.line_circle_intersections(
                line_start[0], line_start[1], line_end[0], line_end[1],
                agent_x, agent_y, visibility_range
            )
            intersections.extend(line_intersections)
    
    print(f"Running main visualization with {len(paths)} paths...")
    
    # Print exploration graph information
    if exploration_graph:
        print(f"Exploration graph: {len(exploration_graph.nodes)} nodes, {len(exploration_graph.edges)} edges")
        # Count node types in exploration graph
        line_circle_count = sum(1 for node in exploration_graph.nodes.values() 
                               if node.get('intersection_type') == 'line_circle')
        line_line_count = sum(1 for node in exploration_graph.nodes.values() 
                             if node.get('intersection_type') == 'line_line')
        unified_count = sum(1 for node in exploration_graph.nodes.values() 
                           if node.get('intersection_type') == 'unified')
        print(f"Exploration graph nodes: {line_circle_count} line-circle, {line_line_count} line-line, {unified_count} unified")
    else:
        print("No exploration graph returned")
    
    # Create figure for main visualization with navigation toolbar
    plt.ion()  # Interactive mode
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_title('Polygon Exploration - Algorithm Results (Use toolbar to zoom/pan | Left-click+drag to measure)', fontsize=16)
    
    # Initialize ruler tool
    ruler = RulerTool(ax)
    
    # Draw reachability heatmap overlay as transparent background if available
    if reachability_data is not None:
        reachability_mask, world_bounds = reachability_data
        
        # Extract world coordinate bounds from dictionary
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
        
        print(f"✅ Reachability overlay: {reachability_mask.shape[0]}×{reachability_mask.shape[1]} centered at agent ({agent_x:.1f}, {agent_y:.1f})")
        
        # Calculate the extent for the heatmap (centered on agent)
        extent = [world_x_min, world_x_max, world_y_min, world_y_max]
        
        # Display the reachability heatmap as a transparent overlay
        im = ax.imshow(reachability_mask, 
                      extent=extent,
                      origin='lower',  # Important: origin='lower' for correct orientation
                      alpha=0.6,  # Transparent overlay
                      cmap='hot',  # Heat colormap (red-yellow for high reachability)
                      interpolation='bilinear',  # Smooth interpolation
                      zorder=1,  # Draw behind other elements
                      vmin=0.0,  # Set explicit value range
                      vmax=1.0)
        
        # Add colorbar for reachability values
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Reachability', rotation=270, labelpad=20)
        cbar.ax.tick_params(labelsize=10)
    
    # Draw environment
    for line in environment_lines:
        x_coords = [line[0][0], line[1][0]]
        y_coords = [line[0][1], line[1][1]]
        ax.plot(x_coords, y_coords, 'k-', linewidth=1, alpha=0.7)
    
    # Draw agent and visibility circle with better visibility
    ax.plot(agent_x, agent_y, 'ro', markersize=15, label='Agent', markeredgecolor='darkred', markeredgewidth=2)
    circle = patches.Circle((agent_x, agent_y), visibility_range, 
                           fill=False, color='blue', linestyle='--', alpha=0.8, linewidth=3, label='Visibility Range')
    ax.add_patch(circle)
    
    # Show agent orientation with an arrow
    arrow_length = 30
    arrow_x = agent_x + arrow_length * math.cos(agent_orientation)
    arrow_y = agent_y + arrow_length * math.sin(agent_orientation)
    ax.annotate('', xy=(arrow_x, arrow_y), xytext=(agent_x, agent_y),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    
    # Draw breakoff lines with better visibility and different colors
    # Show all breakoff lines in visualization
    breakoff_colors = ['green', 'darkgreen', 'forestgreen', 'seagreen', 'mediumseagreen', 'springgreen']
    breakoff_markers = ['s', '^', 'o', 'D', 'v', 'p']
    
    for i, (start_point, end_point, gap_size, category) in enumerate(breakoff_lines):
        color = breakoff_colors[i % len(breakoff_colors)]
        linewidth = 4
        alpha = 0.9
        
        marker_start = breakoff_markers[i % len(breakoff_markers)]
        marker_end = breakoff_markers[(i + 1) % len(breakoff_markers)]
        
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
               color=color, linewidth=linewidth, alpha=alpha, 
               label=f'Breakoff Line {i+1} ({category})' if i < 6 else '')
        ax.plot([start_point[0]], [start_point[1]], marker_start, color=color, 
               markersize=10, markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot([end_point[0]], [end_point[1]], marker_end, color=color, 
               markersize=10, markeredgecolor='darkgreen', markeredgewidth=2)
    
    # Draw exploration paths with better visibility and different colors
    path_colors = ['purple', 'magenta', 'indigo', 'darkviolet', 'mediumorchid', 'blueviolet', 'darkmagenta', 'mediumpurple']
    
    for i, path_data in enumerate(paths):
        if 'path_points' in path_data and len(path_data['path_points']) > 1:
            points = path_data['path_points']
            path_segments = path_data.get('path_segments', [])
            path_color = path_colors[i % len(path_colors)]
            
            # Draw path segments properly (lines vs arcs)
            if path_segments:
                # Use path segments to draw the correct geometry
                for seg_idx, segment in enumerate(path_segments):
                    if segment['type'] == 'line':
                        # Draw straight line segment
                        start = segment['start']
                        end = segment['end']
                        ax.plot([start[0], end[0]], [start[1], end[1]], 
                               color=path_color, linewidth=3, alpha=0.9)
                    elif segment['type'] == 'arc':
                        # Draw arc segment along visibility circle
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
                            num_points = max(10, int(abs(end_angle - start_angle) * 30))
                            angles = [start_angle + j * (end_angle - start_angle) / num_points for j in range(num_points + 1)]
                            arc_x = [center[0] + radius * math.cos(a) for a in angles]
                            arc_y = [center[1] + radius * math.sin(a) for a in angles]
                            
                            ax.plot(arc_x, arc_y, color=path_color, linewidth=3, alpha=0.9)
                        else:
                            # Fallback to straight line if arc data is missing
                            start = segment['start']
                            end = segment['end']
                            ax.plot([start[0], end[0]], [start[1], end[1]], 
                                   color=path_color, linewidth=3, alpha=0.9)
                
                # Add label only once per path
                ax.plot([], [], color=path_color, linewidth=3, alpha=0.9,
                       label=f'Path {i+1} ({len(points)} pts, {"✓" if path_data.get("completed") else "✗"})')
            else:
                # Fallback: draw simple lines between points if no segments available
                xs, ys = zip(*points)
                ax.plot(xs, ys, color=path_color, linewidth=3, alpha=0.9,
                       label=f'Path {i+1} ({len(points)} pts, {"✓" if path_data.get("completed") else "✗"})')
            
            # Mark start/end with better visibility
            ax.plot(points[0][0], points[0][1], 's', color='red', markersize=12, 
                   alpha=0.9, markeredgecolor='darkred', markeredgewidth=2,
                   label='Path Starts' if i == 0 else '')
            if path_data.get('completed'):
                ax.plot(points[-1][0], points[-1][1], 's', color='blue', markersize=12, 
                       alpha=0.9, markeredgecolor='darkblue', markeredgewidth=2,
                       label='Completed Paths' if i == 0 else '')
            else:
                ax.plot(points[-1][0], points[-1][1], 'x', color='orange', markersize=12, 
                       alpha=0.9, markeredgewidth=3,
                       label='Incomplete Paths' if i == 0 else '')
    
    # Draw first edges of each path with distinctive styling
    print(f"🎨 Drawing {len(path_analysis_data)} path first edges...")
    first_edge_colors = ['cyan', 'gold', 'lime', 'hotpink', 'turquoise', 'yellow', 'lightgreen', 'orange']
    
    for i, path_info in enumerate(path_analysis_data):
        first_edge = path_info['first_edge']
        if first_edge is None:
            continue
            
        edge_color = first_edge_colors[i % len(first_edge_colors)]
        
        if first_edge['type'] == 'line':
            # Draw first edge as thick highlighted line
            start = first_edge['start']
            end = first_edge['end']
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   color=edge_color, linewidth=6, alpha=0.8, linestyle='-',
                   label=f'First Edge {i+1} (LINE)' if i < 8 else '')
            
            # Add start and end markers for first edge
            ax.plot(start[0], start[1], 'o', color=edge_color, markersize=8, 
                   markeredgecolor='black', markeredgewidth=2, alpha=0.9)
            ax.plot(end[0], end[1], 'o', color=edge_color, markersize=8, 
                   markeredgecolor='black', markeredgewidth=2, alpha=0.9)
                   
        elif first_edge['type'] == 'arc':
            # Draw first edge as thick highlighted arc
            center = first_edge.get('center', (agent_x, agent_y))
            radius = first_edge.get('radius', visibility_range)
            start_angle = first_edge.get('start_angle', 0)
            end_angle = first_edge.get('end_angle', 0)
            
            # Handle angle wrapping
            if end_angle < start_angle:
                end_angle += 2 * math.pi
            
            # Create arc with points for smooth curves
            num_points = max(10, int(abs(end_angle - start_angle) * 30))
            angles = [start_angle + j * (end_angle - start_angle) / num_points for j in range(num_points + 1)]
            arc_x = [center[0] + radius * math.cos(a) for a in angles]
            arc_y = [center[1] + radius * math.sin(a) for a in angles]
            
            ax.plot(arc_x, arc_y, color=edge_color, linewidth=6, alpha=0.8, linestyle='-',
                   label=f'First Edge {i+1} (ARC)' if i < 8 else '')
            
            # Add start and end markers for first edge arc
            start = first_edge['start']
            end = first_edge['end']
            ax.plot(start[0], start[1], 'o', color=edge_color, markersize=8, 
                   markeredgecolor='black', markeredgewidth=2, alpha=0.9)
            ax.plot(end[0], end[1], 'o', color=edge_color, markersize=8, 
                   markeredgecolor='black', markeredgewidth=2, alpha=0.9)
        
        # Add text annotation for first edge identification
        if first_edge['start'] and first_edge['end']:
            mid_x = (first_edge['start'][0] + first_edge['end'][0]) / 2
            mid_y = (first_edge['start'][1] + first_edge['end'][1]) / 2
            ax.annotate(f'1st-{i+1}', (mid_x, mid_y), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=10, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=edge_color, alpha=0.8, edgecolor='black'),
                       ha='center', va='center', zorder=20)
    
    # Draw orientation arrows for each path
    print(f"🧭 Drawing orientation arrows for {len(path_analysis_data)} paths...")
    for i, path_info in enumerate(path_analysis_data):
        first_edge = path_info['first_edge']
        orientation = path_info['orientation']
        target_point = path_info['target_point']
        
        if first_edge is None or orientation is None or target_point is None:
            continue
            
        # Calculate arrow start position (target point - second point of first edge)
        arrow_start_x = target_point[0]
        arrow_start_y = target_point[1]
        
        # Calculate arrow end position using orientation
        arrow_length = 40  # Length of orientation arrow
        arrow_end_x = arrow_start_x + arrow_length * math.cos(orientation)
        arrow_end_y = arrow_start_y + arrow_length * math.sin(orientation)
        
        # Use same color as first edge but make it darker for contrast
        edge_color = first_edge_colors[i % len(first_edge_colors)]
        arrow_color = edge_color
        
        # Draw orientation arrow
        ax.annotate('', xy=(arrow_end_x, arrow_end_y), xytext=(arrow_start_x, arrow_start_y),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=4, alpha=0.9),
                   zorder=25)
        
        # Add small circle at arrow start for clarity
        ax.plot(arrow_start_x, arrow_start_y, 'o', color=arrow_color, markersize=6, 
               markeredgecolor='black', markeredgewidth=1, alpha=0.9, zorder=24)
        
        # Add orientation degree text near arrow end
        orientation_deg = math.degrees(orientation)
        ax.annotate(f'{orientation_deg:.0f}°', (arrow_end_x, arrow_end_y), 
                   xytext=(8, 8), textcoords='offset points', 
                   fontsize=9, fontweight='bold', color=arrow_color,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=arrow_color),
                   ha='center', va='center', zorder=26)
    
    # Draw sample points where reachability values were sampled
    if sample_points_data:
        print(f"📍 Drawing sample points for {len(sample_points_data)} paths...")
        for i, sample_info in enumerate(sample_points_data):
            sample_x, sample_y = sample_info['sample_point']
            target_value = sample_info['target_value']
            
            # Calculate boosted value (50% increase, capped at 1.0) to match processing
            boosted_value = min(1.0, target_value * 1.5)
            
            # Choose color based on boosted value (heat map style)
            if boosted_value > 0.7:
                sample_color = 'red'
            elif boosted_value > 0.4:
                sample_color = 'orange'
            elif boosted_value > 0.1:
                sample_color = 'yellow'
            else:
                sample_color = 'lightblue'
            
            # Draw sample point with distinctive marker
            ax.plot(sample_x, sample_y, 'D', color=sample_color, markersize=12, 
                   markeredgecolor='black', markeredgewidth=2, alpha=0.9, zorder=30)
            
            # Add sample value annotation
            ax.annotate(f'{target_value:.2f}→{boosted_value:.2f}', 
                       (sample_x, sample_y), 
                       xytext=(15, -15), textcoords='offset points', 
                       fontsize=8, fontweight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'),
                       ha='center', va='center', zorder=31)
    
    # Draw polygon paths for each path
    print(f"🔷 Drawing polygon paths for {len(path_analysis_data)} paths...")
    polygon_colors = ['lightcyan', 'lightyellow', 'lightgreen', 'lightpink', 'lightblue', 'lavender', 'wheat', 'mistyrose']
    
    for i, path_info in enumerate(path_analysis_data):
        path_polygon = path_info['path_polygon']
        if path_polygon is None or len(path_polygon) < 3:
            continue  # Need at least 3 points to form a polygon
            
        polygon_color = polygon_colors[i % len(polygon_colors)]
        
        # Extract x and y coordinates
        polygon_x = [point[0] for point in path_polygon]
        polygon_y = [point[1] for point in path_polygon]
        
        # Draw filled polygon with transparency
        ax.fill(polygon_x, polygon_y, color=polygon_color, alpha=0.3, 
               edgecolor='black', linewidth=1, linestyle=':', 
               label=f'Path {i+1} Polygon (Linearized)' if i < 8 else '')
        
        # Draw polygon vertices as small dots
        # Different markers for original vs linearized points
        original_points = paths[i].get('path_points', [])
        
        for j, point in enumerate(path_polygon):
            # Check if this point is close to any original point (within small tolerance)
            is_original = False
            for orig_point in original_points:
                distance = math.sqrt((point[0] - orig_point[0])**2 + (point[1] - orig_point[1])**2)
                if distance < 1e-6:  # Small tolerance for floating point comparison
                    is_original = True
                    break
            
            if is_original:
                # Original path points - larger, darker dots
                ax.plot(point[0], point[1], 'o', color='darkblue', markersize=4, 
                       alpha=0.9, markeredgecolor='black', markeredgewidth=1)
            else:
                # Linearization points - smaller, lighter dots
                ax.plot(point[0], point[1], 'o', color='lightgray', markersize=2, 
                       alpha=0.7, markeredgecolor='gray', markeredgewidth=0.5)
        
        # Add polygon vertex count annotation near the polygon centroid
        if len(path_polygon) > 0:
            # Calculate centroid
            centroid_x = sum(polygon_x) / len(polygon_x)
            centroid_y = sum(polygon_y) / len(polygon_y)
            
            ax.annotate(f'{len(path_polygon)} pts', (centroid_x, centroid_y), 
                       xytext=(0, 0), textcoords='offset points', 
                       fontsize=8, fontweight='bold', color='darkblue',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=polygon_color, alpha=0.8, edgecolor='darkblue'),
                       ha='center', va='center', zorder=15)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Force window to front
    fig.canvas.manager.window.wm_attributes('-topmost', 1)
    fig.canvas.manager.window.wm_attributes('-topmost', 0)
    
    # Set up window close event handler
    def on_close(event):
        """Handle window close event."""
        print("\nWindow closed. Shutting down program...")
        ruler.disconnect()  # Clean up ruler event handlers
        plt.close('all')
        exit(0)
    
    fig.canvas.mpl_connect('close_event', on_close)
    
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)
    
    # Print summary of all paths
    print("\n" + "=" * 80)
    print("POLYGON EXPLORATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Found {len(intersections)} intersection points")
    print(f"Loaded {len(breakoff_lines)} total breakoff lines") 
    print(f"USING ALL {len(test_breakoff_lines)} breakoff lines for polygon exploration")
    print(f"Generated {len(paths)} exploration paths")
    print(f"Analyzed {len(path_analysis_data)} path first edges")
    
    # Display timing information at the end
    if 'timing_data' in locals() and timing_data is not None:
        print("")
        print("REACHABILITY PROCESSING TIMING:")
        print(f"⏱️ Configuration: {timing_data['config_duration']:.4f} seconds")
        print(f"⏱️ Path processing: {timing_data['process_duration']:.4f} seconds")
        print(f"⏱️ Total reachability processing: {timing_data['total_duration']:.4f} seconds")
        print(f"📊 Breakdown: Config={timing_data['config_duration']:.4f}s ({timing_data['config_duration']/timing_data['total_duration']*100:.1f}%), Processing={timing_data['process_duration']:.4f}s ({timing_data['process_duration']/timing_data['total_duration']*100:.1f}%)")
    
    print("")
    
    completed_paths = sum(1 for path in paths if path.get('completed', False))
    incomplete_paths = len(paths) - completed_paths
    
    # Count first edge types
    line_first_edges = sum(1 for info in path_analysis_data if info['first_edge'] and info['first_edge']['type'] == 'line')
    arc_first_edges = sum(1 for info in path_analysis_data if info['first_edge'] and info['first_edge']['type'] == 'arc')
    
    print(f"Path Results:")
    print(f"  ✓ Completed paths: {completed_paths}")
    print(f"  ✗ Incomplete paths: {incomplete_paths}")
    print(f"  📏 First edges - Lines: {line_first_edges}, Arcs: {arc_first_edges}")
    print("")
    
    for i, path_data in enumerate(paths):
        breakoff_info = path_data.get('breakoff_line', ['Unknown', 'Unknown', 0, 'Unknown'])
        category = breakoff_info[3] if len(breakoff_info) > 3 else 'Unknown'
        point_count = len(path_data.get('path_points', []))
        status = "✓ Complete" if path_data.get('completed', False) else "✗ Incomplete"
        iterations = path_data.get('iterations', 'Unknown')
        
        # Add first edge info if available
        first_edge_info = ""
        orientation_info = ""
        if i < len(path_analysis_data):
            if path_analysis_data[i]['first_edge']:
                edge_type = path_analysis_data[i]['first_edge']['type'].upper()
                first_edge_info = f" | 1st edge: {edge_type}"
            
            if path_analysis_data[i]['orientation'] is not None:
                orientation_deg = math.degrees(path_analysis_data[i]['orientation'])
                orientation_info = f" | Orientation: {orientation_deg:.0f}°"
        
        print(f"Path {i+1}: {status} | {point_count} points | {iterations} iterations | Category: {category}{first_edge_info}{orientation_info}")
    
    print("\n" + "=" * 80)
    print("REACHABILITY PATH VISUALIZATION IS NOW VISIBLE!")
    print("Shows ALL exploration paths and algorithm results")
    print("") 
    print("NAVIGATION TIPS:")
    print("- Use the toolbar buttons to zoom and pan")
    print("- Zoom: Click the zoom button (magnifying glass) then drag to select area")
    print("- Pan: Click the pan button (hand) then drag to move around")
    print("- Home: Click home button to reset view")
    print("- Right-click and drag to zoom out")
    print("")
    print("RULER TOOL:")
    print("- Left-click and drag to measure distances")
    print("- Press 'r' to clear ruler measurements")
    print("- Press 'h' for ruler help")
    print("- Measurements are displayed as yellow text boxes")
    print("")
    print("ALGORITHM LEGEND:")
    print("- Green lines: Breakoff lines (multiple, between consecutive intersections)")
    print("- Purple/Magenta lines: Exploration paths (one per breakoff line)")
    print("- Cyan/Gold/Lime thick lines: First edges of each path")
    print("- Colored arrows: Path orientations (45° rotation away from circle center)")
    print("- Light colored polygons: Path polygons (linearized arcs for accurate shape)")
    print("- Dark blue dots: Original path points")
    print("- Light gray dots: Arc linearization points")
    print("- Red squares: Path start points")
    print("- Blue squares: Completed path end points")
    print("- Orange X: Incomplete path end points")
    print("")
    print("If you don't see the window, try Alt+Tab to find it")
    print("Close the window to exit the program...")
    print("=" * 80)
    
    # Keep the program running until the window is closed
    try:
        while plt.get_fignums():  # Check if any figures are still open
            plt.pause(0.1)  # Small pause to prevent busy waiting
    except KeyboardInterrupt:
        print("\nProgram interrupted. Shutting down...")
        plt.close('all')

if __name__ == "__main__":
    show_visualization()
