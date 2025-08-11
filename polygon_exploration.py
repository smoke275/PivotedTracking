#!/usr/bin/env python3
"""
Polygon Exploration Module
Handles polygon breakpoint exploration algorithms for unknown area navigation.

This module implements a graph-based algorithm that:
1. Pre-computes all intersections between environment lines and visibility circle as graph nodes
2. Pre-computes all intersections between environment lines themselves as additional graph nodes
3. Creates edges (line segments and arcs) connecting these intersection points
4. Starts from the breakoff line's far point (which is an intersection node)
5. Traverses the graph using left/right turn preferences until returning to start
6. Builds complete exploration polygons for navigating unknown areas

The intersection graph now includes:
- Line-to-circle intersections (environment lines with visibility circle)
- Line-to-line intersections (environment lines with each other)
- Line segment edges connecting intersection points along the same environment line
- Arc edges connecting intersection points along the visibility circle
"""

import math

# Debug flag - set to True to enable detailed console output
DEBUG_POLYGON_EXPLORATION = False


class IntersectionGraph:
    """
    Graph representation of intersections between environment lines and visibility circle,
    as well as intersections between environment lines themselves.
    
    Nodes represent intersection points, edges represent line segments or arcs connecting them.
    The graph includes:
    - Line-to-circle intersections (environment lines with visibility circle)
    - Line-to-line intersections (environment lines with each other)
    - Line segment edges along environment lines connecting intersection points
    - Arc edges along visibility circle connecting intersection points
    
    Provides methods for graph construction, traversal, and path finding.
    """
    
    def __init__(self, environment_lines, agent_x, agent_y, visibility_range):
        """
        Initialize and build the intersection graph.
        
        Args:
            environment_lines: List of line segments from environment
            agent_x, agent_y: Agent position (center of visibility circle)
            visibility_range: Radius of visibility circle
        """
        self.environment_lines = environment_lines
        self.agent_x = agent_x
        self.agent_y = agent_y
        self.visibility_range = visibility_range
        
        self.nodes = {}  # node_id -> {'point': (x, y), 'angle': angle, 'line': line_segment}
        self.edges = {}  # edge_id -> {'from_node': id, 'to_node': id, 'type': 'line'/'arc', 'data': {}}
        
        self._build_graph()
    
    def _build_graph(self):
        """Build the unified intersection graph by finding all intersections and creating nodes/edges.
        
        Optimized algorithm:
        - O(n) for line-circle intersections
        - O(n²) for line-line intersections (could be optimized with spatial indexing for large n)
        - O(k) for node creation where k = total intersection points
        - O(m) for edge creation where m = total edges
        
        For current environment size (~70 lines), this is very efficient.
        For larger environments (>1000 lines), consider spatial indexing (R-tree, grid).
        """
        edge_id_counter = 0
        node_id_counter = 0
        line_nodes_map = {}  # Maps line index to list of node IDs
        
        # Unified pass: Collect all intersection points and their properties
        all_intersections = {}  # Maps unique keys to intersection info
        
        def get_intersection_key(point, tolerance=1.0):
            """Generate a unique key for an intersection point with tolerance.
            Using spatial hashing for O(1) duplicate detection.
            
            Args:
                point: (x, y) coordinate
                tolerance: Distance tolerance for grouping nearby points (default 1.0)
                          Increased to ensure proper node unification
            """
            # Round to tolerance level to group nearby points
            rounded_x = round(point[0] / tolerance) * tolerance
            rounded_y = round(point[1] / tolerance) * tolerance
            return (rounded_x, rounded_y)
        
        # Phase 1: Line-circle intersections - O(n) where n = number of environment lines
        if DEBUG_POLYGON_EXPLORATION:
            print(f"Computing line-circle intersections for {len(self.environment_lines)} lines...")
        line_circle_count = 0
        
        for line_idx, line_segment in enumerate(self.environment_lines):
            if len(line_segment) != 2:
                continue
            
            line_start, line_end = line_segment
            intersections = line_circle_intersections(
                line_start[0], line_start[1], line_end[0], line_end[1],
                self.agent_x, self.agent_y, self.visibility_range
            )
            
            for intersection_point in intersections:
                key = get_intersection_key(intersection_point)
                line_circle_count += 1
                
                # Find existing intersection within tolerance or create new one
                merged = False
                for existing_key, existing_data in all_intersections.items():
                    existing_point = existing_data['point']
                    distance = math.sqrt(
                        (existing_point[0] - intersection_point[0])**2 + 
                        (existing_point[1] - intersection_point[1])**2
                    )
                    if distance <= 1.0:  # Use same tolerance as get_intersection_key
                        # Merge with existing intersection
                        existing_data['has_line_circle'] = True
                        existing_data['line_circle_lines'].add(line_idx)
                        existing_data['intersecting_lines'].add(line_idx)
                        merged = True
                        break
                
                if not merged:
                    # Create new intersection
                    all_intersections[key] = {
                        'point': intersection_point,
                        'has_line_circle': True,
                        'has_line_line': False,
                        'intersecting_lines': {line_idx},
                        'line_circle_lines': {line_idx}
                    }
        
        if DEBUG_POLYGON_EXPLORATION:
            print(f"Found {line_circle_count} line-circle intersection points")
        
        # Phase 2: Line-line intersections - O(n²) where n = number of environment lines  
        # Note: For very large environments (>1000 lines), consider spatial indexing (R-tree)
        n_lines = len(self.environment_lines)
        total_pairs = n_lines * (n_lines - 1) // 2
        if DEBUG_POLYGON_EXPLORATION:
            print(f"Computing line-line intersections for {total_pairs} line pairs...")
        
        line_line_count = 0
        processed_pairs = 0
        
        for i in range(len(self.environment_lines)):
            for j in range(i + 1, len(self.environment_lines)):
                processed_pairs += 1
                
                # Progress indicator for large environments
                if processed_pairs % 1000 == 0 and DEBUG_POLYGON_EXPLORATION:
                    print(f"  Processed {processed_pairs}/{total_pairs} line pairs...")
                
                line1 = self.environment_lines[i]
                line2 = self.environment_lines[j]
                
                if len(line1) != 2 or len(line2) != 2:
                    continue
                
                # Find intersection between the two lines
                intersection_point = line_line_intersection(
                    line1[0][0], line1[0][1], line1[1][0], line1[1][1],
                    line2[0][0], line2[0][1], line2[1][0], line2[1][1]
                )
                
                if intersection_point:
                    key = get_intersection_key(intersection_point)
                    line_line_count += 1
                    
                    # Find existing intersection within tolerance or create new one
                    merged = False
                    for existing_key, existing_data in all_intersections.items():
                        existing_point = existing_data['point']
                        distance = math.sqrt(
                            (existing_point[0] - intersection_point[0])**2 + 
                            (existing_point[1] - intersection_point[1])**2
                        )
                        if distance <= 1.0:  # Use same tolerance as get_intersection_key
                            # Merge with existing intersection
                            existing_data['has_line_line'] = True
                            existing_data['intersecting_lines'].add(i)
                            existing_data['intersecting_lines'].add(j)
                            merged = True
                            break
                    
                    if not merged:
                        # Create new intersection
                        all_intersections[key] = {
                            'point': intersection_point,
                            'has_line_circle': False,
                            'has_line_line': True,
                            'intersecting_lines': {i, j},
                            'line_circle_lines': set()
                        }
        
        if DEBUG_POLYGON_EXPLORATION:
            print(f"Found {line_line_count} line-line intersection points")
            print(f"Total unique intersection points: {len(all_intersections)}")
            
        # Post-processing: Additional unification pass to catch any remaining duplicates
        if DEBUG_POLYGON_EXPLORATION:
            print("Performing additional unification pass...")
            
        keys_to_merge = []
        tolerance = 1.0
        intersection_items = list(all_intersections.items())
        
        for i in range(len(intersection_items)):
            for j in range(i + 1, len(intersection_items)):
                key1, data1 = intersection_items[i]
                key2, data2 = intersection_items[j]
                
                point1 = data1['point']
                point2 = data2['point']
                
                distance = math.sqrt(
                    (point1[0] - point2[0])**2 + 
                    (point1[1] - point2[1])**2
                )
                
                if distance <= tolerance:
                    # Merge data2 into data1 and mark data2 for removal
                    data1['has_line_circle'] = data1['has_line_circle'] or data2['has_line_circle']
                    data1['has_line_line'] = data1['has_line_line'] or data2['has_line_line']
                    data1['intersecting_lines'].update(data2['intersecting_lines'])
                    data1['line_circle_lines'].update(data2['line_circle_lines'])
                    keys_to_merge.append(key2)
        
        # Remove merged keys
        for key in keys_to_merge:
            if key in all_intersections:
                del all_intersections[key]
        
        if DEBUG_POLYGON_EXPLORATION and keys_to_merge:
            print(f"Unified {len(keys_to_merge)} additional duplicate intersections")
            print(f"Final unique intersection points: {len(all_intersections)}")
        
        # Phase 3: Create unified nodes from all intersection points - O(k) where k = intersection points
        if DEBUG_POLYGON_EXPLORATION:
            print(f"Creating {len(all_intersections)} unified nodes...")
        
        for key, intersection_info in all_intersections.items():
            intersection_point = intersection_info['point']
            angle = math.atan2(intersection_point[1] - self.agent_y, intersection_point[0] - self.agent_x)
            
            # Determine the unified intersection type
            if intersection_info['has_line_circle'] and intersection_info['has_line_line']:
                intersection_type = 'unified'  # Both types at the same point
            elif intersection_info['has_line_circle']:
                intersection_type = 'line_circle'
            else:
                intersection_type = 'line_line'
            
            # Create unified node
            self.nodes[node_id_counter] = {
                'point': intersection_point,
                'angle': angle,
                'intersection_type': intersection_type,
                'has_line_circle': intersection_info['has_line_circle'],
                'has_line_line': intersection_info['has_line_line'],
                'intersecting_lines': list(intersection_info['intersecting_lines']),
                'line_circle_lines': list(intersection_info['line_circle_lines'])
            }
            
            # Add this node to all relevant lines' node lists
            for line_idx in intersection_info['intersecting_lines']:
                if line_idx not in line_nodes_map:
                    line_nodes_map[line_idx] = []
                line_nodes_map[line_idx].append(node_id_counter)
            
            node_id_counter += 1
        
        # Phase 4: Create line segment edges - O(m) where m = total nodes on all lines
        if DEBUG_POLYGON_EXPLORATION:
            print(f"Creating line edges for {len(line_nodes_map)} lines with nodes...")
        line_edge_count = 0
        
        for line_idx, line_nodes in line_nodes_map.items():
            if len(line_nodes) >= 2:
                line_segment = self.environment_lines[line_idx]
                line_start, line_end = line_segment
                
                # Sort nodes along the line - O(k log k) where k = nodes per line
                line_nodes.sort(key=lambda nid: self._distance_along_line(
                    self.nodes[nid]['point'], line_start, line_end
                ))
                
                # Create edges between consecutive nodes - O(k) where k = nodes per line
                for i in range(len(line_nodes) - 1):
                    from_node = line_nodes[i]
                    to_node = line_nodes[i + 1]
                    
                    self.edges[edge_id_counter] = {
                        'from_node': from_node,
                        'to_node': to_node,
                        'type': 'line',
                        'data': {'line_segment': line_segment}
                    }
                    edge_id_counter += 1
                    line_edge_count += 1
        
        if DEBUG_POLYGON_EXPLORATION:
            print(f"Created {line_edge_count} line edges")
        
        # Phase 5: Create arc edges along the visibility circle - O(c log c) where c = circle nodes
        if len(self.nodes) >= 2:
            # Filter nodes to include any that have line-circle intersection capability
            circle_nodes = []
            for node_id, node_data in self.nodes.items():
                # Include nodes that have line_circle capability
                if node_data.get('has_line_circle', False) or node_data.get('intersection_type') == 'line_circle':
                    circle_nodes.append(node_id)
            
            if DEBUG_POLYGON_EXPLORATION:
                print(f"Creating arc edges for {len(circle_nodes)} circle nodes...")
            arc_edge_count = 0
            
            if len(circle_nodes) >= 2:
                # Sort circle nodes by angle around the circle - O(c log c)
                sorted_circle_nodes = sorted(circle_nodes, key=lambda nid: self.nodes[nid]['angle'])
                
                # Create arc edges between consecutive nodes around the circle - O(c)
                for i in range(len(sorted_circle_nodes)):
                    from_node = sorted_circle_nodes[i]
                    to_node = sorted_circle_nodes[(i + 1) % len(sorted_circle_nodes)]  # Wrap around
                    
                    # Skip if from_node and to_node are the same (shouldn't happen with unified approach)
                    if from_node == to_node:
                        continue
                    
                    # Calculate angle difference to avoid zero-length arcs
                    start_angle = self.nodes[from_node]['angle']
                    end_angle = self.nodes[to_node]['angle']
                    
                    # Normalize angle difference and ensure proper end_angle for drawing
                    angle_diff = end_angle - start_angle
                    if angle_diff < 0:
                        angle_diff += 2 * math.pi
                        # Update end_angle to ensure proper arc drawing
                        end_angle = start_angle + angle_diff
                    
                    # Skip very small arcs (less than 0.1 degrees)
                    if angle_diff < math.radians(0.1):
                        continue
                    
                    self.edges[edge_id_counter] = {
                        'from_node': from_node,
                        'to_node': to_node,
                        'type': 'arc',
                        'data': {
                            'center': (self.agent_x, self.agent_y),
                            'radius': self.visibility_range,
                            'start_angle': start_angle,
                            'end_angle': end_angle
                        }
                    }
                    edge_id_counter += 1
                    arc_edge_count += 1
            
            if DEBUG_POLYGON_EXPLORATION:
                print(f"Created {arc_edge_count} arc edges")
        
        if DEBUG_POLYGON_EXPLORATION:
            print(f"Graph construction complete: {len(self.nodes)} nodes, {len(self.edges)} edges")
            print(f"Performance: O(n) line-circle + O(n²) line-line + O(k) nodes + O(m) edges")
            print(f"  where n={len(self.environment_lines)} lines, k={len(self.nodes)} nodes, m={len(self.edges)} edges")
    
    def _distance_along_line(self, point, line_start, line_end):
        """Calculate the distance of a point along a line segment from line_start."""
        line_vec_x = line_end[0] - line_start[0]
        line_vec_y = line_end[1] - line_start[1]
        point_vec_x = point[0] - line_start[0]
        point_vec_y = point[1] - line_start[1]
        
        # Project point vector onto line vector
        line_length_sq = line_vec_x**2 + line_vec_y**2
        if line_length_sq == 0:
            return 0
        
        projection = (point_vec_x * line_vec_x + point_vec_y * line_vec_y) / line_length_sq
        return projection * math.sqrt(line_length_sq)
    
    def _find_duplicate_node(self, point, tolerance=1e-6):
        """Find if a node already exists at this point within tolerance."""
        for node_id, node_data in self.nodes.items():
            existing_point = node_data['point']
            distance = math.sqrt(
                (existing_point[0] - point[0])**2 + 
                (existing_point[1] - point[1])**2
            )
            if distance < tolerance:
                return node_id
        return None
    
    def find_closest_node(self, target_point):
        """Find the node closest to the target point."""
        min_distance = float('inf')
        closest_node_id = None
        
        for node_id, node_data in self.nodes.items():
            node_point = node_data['point']
            distance = math.sqrt(
                (node_point[0] - target_point[0])**2 + 
                (node_point[1] - target_point[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_node_id = node_id
        
        return closest_node_id
    
    def find_edge_toward_point(self, start_node_id, target_point):
        """Find the edge from start_node that leads toward the target point."""
        start_point = self.nodes[start_node_id]['point']
        best_edge_id = None
        best_dot_product = -float('inf')  # We want the most aligned direction
        
        # Vector from start to target
        target_vec_x = target_point[0] - start_point[0]
        target_vec_y = target_point[1] - start_point[1]
        target_length = math.sqrt(target_vec_x**2 + target_vec_y**2)
        
        if target_length == 0:
            return None
        
        target_vec_x /= target_length
        target_vec_y /= target_length
        
        # Check all edges from this node
        for edge_id, edge_data in self.edges.items():
            if edge_data['from_node'] == start_node_id:
                end_node_id = edge_data['to_node']
            elif edge_data['to_node'] == start_node_id:
                end_node_id = edge_data['from_node']
            else:
                continue  # Edge doesn't connect to start node
            
            end_point = self.nodes[end_node_id]['point']
            
            # Vector from start to end of this edge
            edge_vec_x = end_point[0] - start_point[0]
            edge_vec_y = end_point[1] - start_point[1]
            edge_length = math.sqrt(edge_vec_x**2 + edge_vec_y**2)
            
            if edge_length == 0:
                continue
            
            edge_vec_x /= edge_length
            edge_vec_y /= edge_length
            
            # Calculate dot product (cosine of angle)
            dot_product = target_vec_x * edge_vec_x + target_vec_y * edge_vec_y
            
            if dot_product > best_dot_product:
                best_dot_product = dot_product
                best_edge_id = edge_id
        
        return best_edge_id
    
    def find_next_edge_with_turn(self, current_node_id, incoming_edge_id, turn_left):
        """
        Find the next edge from current node using turn preference (left or right).
        
        Args:
            current_node_id: Current node
            incoming_edge_id: Edge we came from
            turn_left: True for left turn, False for right turn
        
        Returns:
            Edge ID of the next edge to follow
        """
        current_point = self.nodes[current_node_id]['point']
        incoming_edge = self.edges[incoming_edge_id]
        
        # Determine the direction we came from
        if incoming_edge['from_node'] == current_node_id:
            prev_node_id = incoming_edge['to_node']
        else:
            prev_node_id = incoming_edge['from_node']
        
        prev_point = self.nodes[prev_node_id]['point']
        
        # Incoming direction vector - depends on edge type
        if incoming_edge['type'] == 'arc':
            # For arc edges, use tangent direction at current node
            arc_data = incoming_edge.get('data', {})
            center = arc_data.get('center', (self.agent_x, self.agent_y))
            
            # Vector from circle center to current point (radial direction)
            radial_x = current_point[0] - center[0]
            radial_y = current_point[1] - center[1]
            radial_length = math.sqrt(radial_x**2 + radial_y**2)
            
            if radial_length == 0:
                return None
            
            # Normalize radial vector
            radial_x /= radial_length
            radial_y /= radial_length
            
            # Tangent direction (perpendicular to radial)
            # Two possible tangent directions: (-radial_y, radial_x) and (radial_y, -radial_x)
            tangent1_x = -radial_y
            tangent1_y = radial_x
            tangent2_x = radial_y
            tangent2_y = -radial_x
            
            # Determine which tangent direction represents the incoming direction
            # by checking which tangent aligns with the chord from prev_point to current_point
            chord_x = current_point[0] - prev_point[0]
            chord_y = current_point[1] - prev_point[1]
            chord_length = math.sqrt(chord_x**2 + chord_y**2)
            
            if chord_length > 0:
                chord_x /= chord_length
                chord_y /= chord_length
                
                # Check which tangent direction is more aligned with chord
                dot1 = tangent1_x * chord_x + tangent1_y * chord_y
                dot2 = tangent2_x * chord_x + tangent2_y * chord_y
                
                if dot1 > dot2:
                    incoming_vec_x = tangent1_x
                    incoming_vec_y = tangent1_y
                else:
                    incoming_vec_x = tangent2_x
                    incoming_vec_y = tangent2_y
            else:
                # Fallback to first tangent direction
                incoming_vec_x = tangent1_x
                incoming_vec_y = tangent1_y
        else:
            # For line edges, use direct chord direction
            incoming_vec_x = current_point[0] - prev_point[0]
            incoming_vec_y = current_point[1] - prev_point[1]
            incoming_length = math.sqrt(incoming_vec_x**2 + incoming_vec_y**2)
            
            if incoming_length == 0:
                return None
            
            incoming_vec_x /= incoming_length
            incoming_vec_y /= incoming_length
        
        # Find all outgoing edges (except the one we came from)
        candidate_edges = []
        
        for edge_id, edge_data in self.edges.items():
            if edge_id == incoming_edge_id:
                continue  # Skip the edge we came from
            
            if edge_data['from_node'] == current_node_id:
                next_node_id = edge_data['to_node']
            elif edge_data['to_node'] == current_node_id:
                next_node_id = edge_data['from_node']
            else:
                continue  # Edge doesn't connect to current node
            
            next_point = self.nodes[next_node_id]['point']
            
            # Calculate outgoing direction vector based on edge type
            if edge_data['type'] == 'arc':
                # For arc edges, use tangent direction at current node
                arc_data = edge_data.get('data', {})
                center = arc_data.get('center', (self.agent_x, self.agent_y))
                
                # Vector from circle center to current point (radial direction)
                radial_x = current_point[0] - center[0]
                radial_y = current_point[1] - center[1]
                radial_length = math.sqrt(radial_x**2 + radial_y**2)
                
                if radial_length == 0:
                    continue
                
                # Normalize radial vector
                radial_x /= radial_length
                radial_y /= radial_length
                
                # Tangent direction (perpendicular to radial)
                # Two possible tangent directions: (-radial_y, radial_x) and (radial_y, -radial_x)
                tangent1_x = -radial_y
                tangent1_y = radial_x
                tangent2_x = radial_y
                tangent2_y = -radial_x
                
                # Determine which tangent direction leads toward next_point
                # Calculate which tangent has better alignment with chord direction
                chord_x = next_point[0] - current_point[0]
                chord_y = next_point[1] - current_point[1]
                chord_length = math.sqrt(chord_x**2 + chord_y**2)
                
                if chord_length > 0:
                    chord_x /= chord_length
                    chord_y /= chord_length
                    
                    # Check which tangent direction is more aligned with chord
                    dot1 = tangent1_x * chord_x + tangent1_y * chord_y
                    dot2 = tangent2_x * chord_x + tangent2_y * chord_y
                    
                    if dot1 > dot2:
                        outgoing_vec_x = tangent1_x
                        outgoing_vec_y = tangent1_y
                    else:
                        outgoing_vec_x = tangent2_x
                        outgoing_vec_y = tangent2_y
                else:
                    # Fallback to first tangent direction
                    outgoing_vec_x = tangent1_x
                    outgoing_vec_y = tangent1_y
            else:
                # For line edges, use direct chord direction
                outgoing_vec_x = next_point[0] - current_point[0]
                outgoing_vec_y = next_point[1] - current_point[1]
                outgoing_length = math.sqrt(outgoing_vec_x**2 + outgoing_vec_y**2)
                
                if outgoing_length == 0:
                    continue
                
                outgoing_vec_x /= outgoing_length
                outgoing_vec_y /= outgoing_length
            
            # Calculate turn angle using cross product
            cross_product = incoming_vec_x * outgoing_vec_y - incoming_vec_y * outgoing_vec_x
            
            # Calculate angle magnitude using dot product
            dot_product = incoming_vec_x * outgoing_vec_x + incoming_vec_y * outgoing_vec_y
            angle = math.atan2(cross_product, dot_product)
            
            candidate_edges.append({
                'edge_id': edge_id,
                'angle': angle,
                'cross_product': cross_product
            })
        
        if not candidate_edges:
            return None
        
        # Choose edge based on turn preference
        if DEBUG_POLYGON_EXPLORATION:
            print(f"      Found {len(candidate_edges)} candidate edges at node {current_node_id}:")
            for i, edge in enumerate(candidate_edges):
                next_node_id = (self.edges[edge['edge_id']]['to_node'] 
                              if self.edges[edge['edge_id']]['from_node'] == current_node_id 
                              else self.edges[edge['edge_id']]['from_node'])
                print(f"        Option {i+1}: Edge {edge['edge_id']} -> Node {next_node_id}, angle={math.degrees(edge['angle']):.1f}°, cross_product={edge['cross_product']:.3f}")
        
        if turn_left:
            # Choose the edge with the most positive cross product (leftmost turn)
            best_edge = max(candidate_edges, key=lambda x: x['angle'])
            if DEBUG_POLYGON_EXPLORATION:
                next_node_id = (self.edges[best_edge['edge_id']]['to_node'] 
                              if self.edges[best_edge['edge_id']]['from_node'] == current_node_id 
                              else self.edges[best_edge['edge_id']]['from_node'])
                print(f"      LEFT turn chosen: Edge {best_edge['edge_id']} -> Node {next_node_id} (cross_product={best_edge['cross_product']:.3f}, angle={math.degrees(best_edge['angle']):.1f}°)")
        else:
            # Choose the edge with the most negative cross product (rightmost turn)
            best_edge = min(candidate_edges, key=lambda x: x['angle'])
            if DEBUG_POLYGON_EXPLORATION:
                next_node_id = (self.edges[best_edge['edge_id']]['to_node'] 
                              if self.edges[best_edge['edge_id']]['from_node'] == current_node_id 
                              else self.edges[best_edge['edge_id']]['from_node'])
                print(f"      RIGHT turn chosen: Edge {best_edge['edge_id']} -> Node {next_node_id} (cross_product={best_edge['cross_product']:.3f}, angle={math.degrees(best_edge['angle']):.1f}°)")
        
        return best_edge['edge_id']
    
    def get_graph_data(self):
        """Return the graph data as a dictionary (for backwards compatibility)."""
        return {'nodes': self.nodes, 'edges': self.edges}


def calculate_polygon_exploration_paths(breakoff_lines, agent_x, agent_y, visibility_range, clipped_environment_lines, max_iterations=50):
    """
    Calculate polygon exploration paths for breakpoints into unknown areas using graph-based approach.
    Algorithm: Pre-compute intersection graph, start from breakoff line's far point (intersection node),
    traverse graph edges using left/right turn preferences until returning to start.
    
    Args:
        breakoff_lines: List of (start_point, end_point, gap_size, category) tuples
        agent_x, agent_y: Agent position (center of visibility circle)
        visibility_range: Radius of visibility circle
        clipped_environment_lines: List of line segments from clipped environment
        max_iterations: Maximum iterations to prevent infinite loops (default 50)
    
    Returns:
        Tuple of (polygon_paths, intersection_graph) where:
        - polygon_paths: List of polygon exploration paths, one for each breakpoint line.
          Each path is a dict with:
          - 'breakoff_line': Original breakoff line data
          - 'path_points': List of (x, y) points forming the exploration polygon
          - 'path_segments': List of path segments with type info
          - 'completed': True if path returned to start, False if terminated early
        - intersection_graph: IntersectionGraph object containing nodes and edges
    """
    # Print what breakoff lines we're receiving
    if DEBUG_POLYGON_EXPLORATION:
        print(f"\n=== calculate_polygon_exploration_paths RECEIVED BREAKOFF LINES ===")
        print(f"Number of breakoff lines: {len(breakoff_lines)}")
        for i, (start_point, end_point, gap_size, category) in enumerate(breakoff_lines):
            print(f"Breakoff Line {i+1}:")
            print(f"  Start point: ({start_point[0]:.2f}, {start_point[1]:.2f})")
            print(f"  End point: ({end_point[0]:.2f}, {end_point[1]:.2f})")
            print(f"  Gap size: {gap_size:.2f}")
            print(f"  Category: {category}")
        print("=" * 70)
    
    if not breakoff_lines:
        return [], None
    
    # Step 1: Build intersection graph ONCE for all breakoff lines
    if DEBUG_POLYGON_EXPLORATION:
        print(f"Building intersection graph for {len(clipped_environment_lines)} environment lines...")
    intersection_graph = IntersectionGraph(clipped_environment_lines, agent_x, agent_y, visibility_range)
    
    if not intersection_graph.nodes:
        if DEBUG_POLYGON_EXPLORATION:
            print("No intersections found in graph - returning empty paths")
        return [], intersection_graph  # Return empty paths but still return the graph
    
    if DEBUG_POLYGON_EXPLORATION:
        print(f"Graph built successfully with {len(intersection_graph.nodes)} nodes and {len(intersection_graph.edges)} edges")
        print(f"Processing {len(breakoff_lines)} breakoff lines using the shared graph...")
    
    polygon_paths = []
    
    for breakoff_idx, (start_point, end_point, gap_size, category) in enumerate(breakoff_lines):      
        if DEBUG_POLYGON_EXPLORATION:
            print(f"\nProcessing breakoff line {breakoff_idx + 1}/{len(breakoff_lines)}: {category}")
        
        # Step 2: Identify starting node from breakoff line
        # Determine which point (start or end) is farther from agent
        dist_start = math.sqrt((start_point[0] - agent_x)**2 + (start_point[1] - agent_y)**2)
        dist_end = math.sqrt((end_point[0] - agent_x)**2 + (end_point[1] - agent_y)**2)
        
        if dist_start > dist_end:
            far_point = start_point
            near_point = end_point
        else:
            far_point = end_point
            near_point = start_point
        
        # Find the node in the graph that corresponds to the far point (intersection point)
        start_node_id = intersection_graph.find_closest_node(far_point)
        if start_node_id is None:
            if DEBUG_POLYGON_EXPLORATION:
                print(f"  No matching node found for far point ({far_point[0]:.2f}, {far_point[1]:.2f}) - skipping")
            continue  # No matching node found
        
        if DEBUG_POLYGON_EXPLORATION:
            print(f"  Starting from node {start_node_id} at ({intersection_graph.nodes[start_node_id]['point'][0]:.2f}, {intersection_graph.nodes[start_node_id]['point'][1]:.2f})")
        
        # Step 3: Find initial direction toward near point
        initial_edge_id = intersection_graph.find_edge_toward_point(start_node_id, near_point)
        if initial_edge_id is None:
            if DEBUG_POLYGON_EXPLORATION:
                print(f"  No suitable initial edge found toward near point ({near_point[0]:.2f}, {near_point[1]:.2f}) - skipping")
            continue  # No suitable initial edge found

        if DEBUG_POLYGON_EXPLORATION:
            print(f"  Found initial edge {initial_edge_id} toward near point")
        
        # Step 4: Traverse the graph using turn preferences
        path_points = []
        path_segments = []
        completed = False
        
        # Determine turn direction from category
        turn_left = 'near_far' in category  # far_near = left turn, near_far = right turn
        if DEBUG_POLYGON_EXPLORATION:
            print(f"  Turn direction: {'LEFT' if turn_left else 'RIGHT'} (based on category: {category})")
        
        # Start graph traversal
        current_node_id = start_node_id
        current_edge_id = initial_edge_id
        visited_edges = set()
        
        path_points.append(intersection_graph.nodes[start_node_id]['point'])
        
        for iteration in range(max_iterations):
            if current_edge_id in visited_edges:
                # Avoid infinite loops by tracking visited edges
                if DEBUG_POLYGON_EXPLORATION:
                    print(f"  Iteration {iteration}: Edge {current_edge_id} already visited - breaking to avoid loop")
                break
            
            visited_edges.add(current_edge_id)
            current_edge = intersection_graph.edges[current_edge_id]
            
            # Move to the other end of the current edge
            if current_edge['from_node'] == current_node_id:
                next_node_id = current_edge['to_node']
            else:
                next_node_id = current_edge['from_node']
            
            next_point = intersection_graph.nodes[next_node_id]['point']
            path_points.append(next_point)

            if DEBUG_POLYGON_EXPLORATION:
                print(f"  Iteration {iteration}: At node {current_node_id} -> moving to node {next_node_id}")
                print(f"    Current point: ({intersection_graph.nodes[current_node_id]['point'][0]:.2f}, {intersection_graph.nodes[current_node_id]['point'][1]:.2f})")
                print(f"    Next point: ({next_point[0]:.2f}, {next_point[1]:.2f})")
                print(f"    Edge type: {current_edge['type']}")
            
            # Add path segment
            path_segments.append({
                'start': intersection_graph.nodes[current_node_id]['point'],
                'end': next_point,
                'type': current_edge['type'],
                'edge_data': current_edge.get('data', {})
            })
            
            # Check if we're back at the start
            if next_node_id == start_node_id:
                completed = True
                if DEBUG_POLYGON_EXPLORATION:
                    print(f"  Iteration {iteration}: Returned to start node {start_node_id} - path completed!")
                break
            
            next_edge_id = intersection_graph.find_next_edge_with_turn(
                next_node_id, current_edge_id, turn_left
            )
            
            
            if next_edge_id is None:
                if DEBUG_POLYGON_EXPLORATION:
                    print(f"  Iteration {iteration}: No next edge found from node {next_node_id} - path incomplete")
                break  # No more edges to follow
            
            current_node_id = next_node_id
            current_edge_id = next_edge_id
        
        if DEBUG_POLYGON_EXPLORATION:
            print(f"  Path complete: {len(path_points)} points, {len(path_segments)} segments, {'COMPLETED' if completed else 'INCOMPLETE'}")
        
        
        polygon_paths.append({
            'breakoff_line': (start_point, end_point, gap_size, category),
            'path_points': path_points,
            'path_segments': path_segments,
            'completed': completed,
            'iterations': iteration + 1 if not completed else iteration + 1
        })
    
    if DEBUG_POLYGON_EXPLORATION:
        print(f"\nGenerated {len(polygon_paths)} exploration paths total")
    
    return polygon_paths, intersection_graph


def find_nearest_intersection(start_point, direction_x, direction_y, environment_lines, agent_x, agent_y, visibility_range):
    """
    Find the nearest intersection of a ray with environment lines or visibility circle.
    
    Args:
        start_point: (x, y) starting point of ray
        direction_x, direction_y: Normalized direction vector
        environment_lines: List of line segments
        agent_x, agent_y: Center of visibility circle
        visibility_range: Radius of visibility circle
    
    Returns:
        Tuple of (intersection_point, intersection_type, intersection_data) or None
        - intersection_point: (x, y) coordinates
        - intersection_type: 'line' or 'circle'
        - intersection_data: Original line segment or circle info
    """
    min_distance = float('inf')
    closest_intersection = None
    
    start_x, start_y = start_point
    
    # Check intersection with all environment lines
    for line_segment in environment_lines:
        if len(line_segment) == 2:  # ((x1, y1), (x2, y2)) format
            line_start, line_end = line_segment
        else:
            continue  # Skip malformed lines
        
        intersection = ray_line_intersection(
            start_x, start_y, direction_x, direction_y,
            line_start[0], line_start[1], line_end[0], line_end[1]
        )
        
        if intersection:
            int_x, int_y = intersection
            distance = math.sqrt((int_x - start_x)**2 + (int_y - start_y)**2)
            
            if distance < min_distance and distance > 1e-6:  # Avoid self-intersection
                min_distance = distance
                closest_intersection = (intersection, 'line', line_segment)
    
    # Check intersection with visibility circle
    circle_intersection = ray_circle_intersection(
        start_x, start_y, direction_x, direction_y,
        agent_x, agent_y, visibility_range
    )
    
    if circle_intersection:
        int_x, int_y = circle_intersection
        distance = math.sqrt((int_x - start_x)**2 + (int_y - start_y)**2)
        
        if distance < min_distance and distance > 1e-6:
            min_distance = distance
            closest_intersection = (circle_intersection, 'circle', {
                'center': (agent_x, agent_y),
                'radius': visibility_range
            })
    
    return closest_intersection


def ray_line_intersection(ray_x, ray_y, ray_dx, ray_dy, line_x1, line_y1, line_x2, line_y2):
    """
    Calculate intersection between a ray and a line segment.
    
    Args:
        ray_x, ray_y: Ray starting point
        ray_dx, ray_dy: Ray direction (normalized)
        line_x1, line_y1, line_x2, line_y2: Line segment endpoints
    
    Returns:
        (x, y) intersection point or None if no intersection
    """
    # Line segment vector
    line_dx = line_x2 - line_x1
    line_dy = line_y2 - line_y1
    
    # Solve: ray_start + t * ray_dir = line_start + s * line_dir
    # ray_x + t * ray_dx = line_x1 + s * line_dx
    # ray_y + t * ray_dy = line_y1 + s * line_dy
    
    denominator = ray_dx * line_dy - ray_dy * line_dx
    
    if abs(denominator) < 1e-10:  # Parallel lines
        return None
    
    # Calculate parameters
    dx = line_x1 - ray_x
    dy = line_y1 - ray_y
    
    t = (dx * line_dy - dy * line_dx) / denominator
    s = (dx * ray_dy - dy * ray_dx) / denominator
    
    # Check if intersection is valid
    if t >= 0 and 0 <= s <= 1:  # Ray forward, within line segment
        int_x = ray_x + t * ray_dx
        int_y = ray_y + t * ray_dy
        return (int_x, int_y)
    
    return None


def line_circle_intersections(line_x1, line_y1, line_x2, line_y2, circle_x, circle_y, radius):
    """
    Find all intersection points between a line segment and a circle.
    
    Args:
        line_x1, line_y1, line_x2, line_y2: Line segment endpoints
        circle_x, circle_y: Circle center
        radius: Circle radius
    
    Returns:
        List of intersection points [(x, y), ...] (can be 0, 1, or 2 points)
    """
    # Line direction vector
    dx = line_x2 - line_x1
    dy = line_y2 - line_y1
    
    # Vector from circle center to line start
    fx = line_x1 - circle_x
    fy = line_y1 - circle_y
    
    # Quadratic equation coefficients: a*t^2 + b*t + c = 0
    a = dx*dx + dy*dy
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - radius*radius
    
    discriminant = b*b - 4*a*c
    
    # Add small tolerance for numerical precision issues
    tolerance = 1e-10
    segment_tolerance = 1e-3  # Larger tolerance for line segment bounds to handle precision issues
    
    if discriminant < -tolerance:
        return []  # No intersection
    
    intersections = []
    
    if abs(discriminant) <= tolerance:
        # One intersection (tangent) - treat near-zero discriminant as tangent
        t = -b / (2*a)
        if -segment_tolerance <= t <= 1 + segment_tolerance:  # Within line segment (with tolerance)
            x = line_x1 + t*dx
            y = line_y1 + t*dy
            intersections.append((x, y))
    else:
        # Two intersections
        sqrt_discriminant = math.sqrt(max(0, discriminant))  # Ensure non-negative
        t1 = (-b - sqrt_discriminant) / (2*a)
        t2 = (-b + sqrt_discriminant) / (2*a)
        
        # Check if intersections are within line segment (with tolerance)
        for t in [t1, t2]:
            if -segment_tolerance <= t <= 1 + segment_tolerance:
                x = line_x1 + t*dx
                y = line_y1 + t*dy
                intersections.append((x, y))
    
    return intersections


def line_line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Find intersection point between two line segments.
    
    Args:
        x1, y1, x2, y2: First line segment endpoints
        x3, y3, x4, y4: Second line segment endpoints
    
    Returns:
        (x, y) intersection point or None if no intersection
    """
    # Direction vectors
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3
    
    # Calculate denominators for parametric equations
    denominator = dx1 * dy2 - dy1 * dx2
    
    if abs(denominator) < 1e-10:  # Lines are parallel
        return None
    
    # Calculate parameters
    dx = x3 - x1
    dy = y3 - y1
    
    t1 = (dx * dy2 - dy * dx2) / denominator
    t2 = (dx * dy1 - dy * dx1) / denominator
    
    # Check if intersection is within both line segments
    tolerance = 1e-6  # Small tolerance for numerical precision
    if (-tolerance <= t1 <= 1 + tolerance) and (-tolerance <= t2 <= 1 + tolerance):
        # Calculate intersection point
        int_x = x1 + t1 * dx1
        int_y = y1 + t1 * dy1
        return (int_x, int_y)
    
    return None


def ray_circle_intersection(ray_x, ray_y, ray_dx, ray_dy, circle_x, circle_y, radius):
    """
    Calculate intersection between a ray and a circle.
    
    Args:
        ray_x, ray_y: Ray starting point
        ray_dx, ray_dy: Ray direction (normalized)
        circle_x, circle_y: Circle center
        radius: Circle radius
    
    Returns:
        (x, y) nearest intersection point or None if no intersection
    """
    # Vector from circle center to ray start
    to_ray_x = ray_x - circle_x
    to_ray_y = ray_y - circle_y
    
    # Quadratic equation coefficients: a*t^2 + b*t + c = 0
    a = ray_dx**2 + ray_dy**2  # Should be 1 for normalized direction
    b = 2 * (to_ray_x * ray_dx + to_ray_y * ray_dy)
    c = to_ray_x**2 + to_ray_y**2 - radius**2
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return None  # No intersection
    
    # Two intersection points (or one if tangent)
    sqrt_discriminant = math.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2*a)
    t2 = (-b + sqrt_discriminant) / (2*a)
    
    # Choose the nearest positive intersection
    candidates = [t for t in [t1, t2] if t > 1e-6]  # Avoid self-intersection
    
    if not candidates:
        return None
    
    t = min(candidates)
    
    # Calculate intersection point
    int_x = ray_x + t * ray_dx
    int_y = ray_y + t * ray_dy
    
    return (int_x, int_y)


def calculate_exploration_turn_direction(current_point, intersection_point, incoming_dir_x, incoming_dir_y,
                                       intersection_type, intersection_data, environment_lines,
                                       agent_x, agent_y, visibility_range, breakoff_category=None, circle_intersections=None):
    """
    Calculate the new direction after hitting an intersection.
    Turn direction depends on distance transition type:
    - Near-to-far transitions (green and orange-red): turn RIGHT
    - Far-to-near transitions (red and blue-green): turn LEFT
    When intersecting a circle, use pre-calculated intersections for binary search.
    
    Args:
        current_point: Previous position
        intersection_point: Current intersection point
        incoming_dir_x, incoming_dir_y: Incoming direction
        intersection_type: 'line' or 'circle'
        intersection_data: Line segment or circle data
        environment_lines: All environment lines
        agent_x, agent_y: Agent position
        visibility_range: Visibility range
        breakoff_category: Category of breakoff point (determines turn direction)
        circle_intersections: Pre-calculated sorted circle intersections
    
    Returns:
        (new_dir_x, new_dir_y) normalized direction or (new_dir_x, new_dir_y, arc_info) for circles
    """
    # Determine turn direction based on distance transition type
    turn_left = True  # Default to left turn
    if breakoff_category:
        if 'near_far' in breakoff_category:
            # Near-to-far transitions turn right
            turn_left = False
        elif 'far_near' in breakoff_category:
            # Far-to-near transitions turn left
            turn_left = True
    
    if intersection_type == 'line':
        # Move along the intersected line
        line_segment = intersection_data
        line_start, line_end = line_segment
        
        # Line direction vector
        line_dx = line_end[0] - line_start[0]
        line_dy = line_end[1] - line_start[1]
        line_length = math.sqrt(line_dx**2 + line_dy**2)
        
        if line_length == 0:
            return None
        
        # Normalize line direction
        line_dx /= line_length
        line_dy /= line_length
        
        # Choose direction based on desired turn direction
        # Cross product of incoming direction with line direction
        cross_product = incoming_dir_x * line_dy - incoming_dir_y * line_dx
        
        if turn_left:
            # Turn left
            if cross_product < 0:
                # Line direction is a left turn
                return (line_dx, line_dy)
            else:
                # Reverse line direction for left turn
                return (-line_dx, -line_dy)
        else:
            # Turn right
            if cross_product > 0:
                # Line direction is a right turn
                return (line_dx, line_dy)
            else:
                # Reverse line direction for right turn
                return (-line_dx, -line_dy)
    
    elif intersection_type == 'circle' and circle_intersections:
        # Use pre-calculated circle intersections with binary search
        circle_center = intersection_data['center']
        circle_radius = intersection_data['radius']
        
        # Calculate angle of current intersection point
        current_angle = math.atan2(
            intersection_point[1] - circle_center[1],
            intersection_point[0] - circle_center[0]
        )
        
        # Normalize current angle to [0, 2π]
        current_angle = current_angle % (2 * math.pi)
        
        # Binary search to find our current position in the sorted intersections
        # Convert all angles to [0, 2π] for comparison
        normalized_intersections = []
        for intersection in circle_intersections:
            norm_angle = intersection['angle'] % (2 * math.pi)
            normalized_intersections.append({
                'angle': norm_angle,
                'original': intersection
            })
        
        # Sort by normalized angle
        normalized_intersections.sort(key=lambda x: x['angle'])
        
        # Find the intersection closest to our current angle
        best_match_idx = 0
        min_angle_diff = float('inf')
        
        for i, intersection in enumerate(normalized_intersections):
            angle_diff = abs(intersection['angle'] - current_angle)
            # Also check wrapped difference
            wrapped_diff = min(angle_diff, 2*math.pi - angle_diff)
            
            if wrapped_diff < min_angle_diff:
                min_angle_diff = wrapped_diff
                best_match_idx = i
        
        # Find next intersection in desired direction
        if turn_left:
            # Counterclockwise - next intersection in array (with wrapping)
            next_idx = (best_match_idx + 1) % len(normalized_intersections)
        else:
            # Clockwise - previous intersection in array (with wrapping)
            next_idx = (best_match_idx - 1) % len(normalized_intersections)
        
        if next_idx < len(normalized_intersections):
            exit_intersection = normalized_intersections[next_idx]['original']
            exit_point = exit_intersection['point']
            
            # Calculate direction toward exit point
            direction_x = exit_point[0] - intersection_point[0]
            direction_y = exit_point[1] - intersection_point[1]
            
            # Normalize direction
            dir_length = math.sqrt(direction_x**2 + direction_y**2)
            if dir_length > 0:
                direction_x /= dir_length
                direction_y /= dir_length
                
                # Return direction with arc information for drawing
                arc_info = {
                    'center': circle_center,
                    'radius': circle_radius,
                    'start_point': intersection_point,
                    'end_point': exit_point
                }
                return (direction_x, direction_y, arc_info)
        
        # Fallback: use tangent direction if no intersections found
        to_intersection_x = intersection_point[0] - circle_center[0]
        to_intersection_y = intersection_point[1] - circle_center[1]
        length = math.sqrt(to_intersection_x**2 + to_intersection_y**2)
        
        if length > 0:
            to_intersection_x /= length
            to_intersection_y /= length
            
            # Tangent direction
            tangent_x = -to_intersection_y
            tangent_y = to_intersection_x
            
            # Choose direction based on turn preference
            if not turn_left:  # Right turn
                tangent_x = -tangent_x
                tangent_y = -tangent_y
            
            # Create a short arc segment in the tangent direction
            arc_distance = min(50.0, circle_radius * 0.5)  # Short arc distance
            end_point = (
                intersection_point[0] + tangent_x * arc_distance,
                intersection_point[1] + tangent_y * arc_distance
            )
            
            arc_info = {
                'center': circle_center,
                'radius': circle_radius,
                'start_point': intersection_point,
                'end_point': end_point
            }
            return (tangent_x, tangent_y, arc_info)
    
    return None
