#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <limits>

namespace py = pybind11;

struct Node {
    int id;
    double x, y;
    double angle; // angle from agent center
    bool has_line_circle;
    bool has_line_line;
    std::vector<int> intersecting_lines;
    std::vector<int> line_circle_lines;
    std::string intersection_type; // unified / line_circle / line_line
};

struct Edge {
    int id;
    int from_node;
    int to_node;
    std::string type; // line / arc
    // For line edge
    std::vector<std::pair<double,double>> line_segment; // 2 points
    // For arc edge
    double center_x{0}, center_y{0}, radius{0}, start_angle{0}, end_angle{0};
};

class IntersectionGraph {
public:
    IntersectionGraph(const std::vector<std::vector<std::pair<double,double>>> &environment_lines,
                      double agent_x, double agent_y, double visibility_range)
        : environment_lines_(environment_lines), agent_x_(agent_x), agent_y_(agent_y), visibility_range_(visibility_range) {
        build_graph();
    }

    const std::unordered_map<int, Node>& nodes() const { return nodes_; }
    const std::unordered_map<int, Edge>& edges() const { return edges_; }

    int find_closest_node(const std::pair<double,double>& target_point) const {
        double min_dist = 1e300; // large sentinel
        int closest = -1;
        for (auto &kv : nodes_) {
            const Node &n = kv.second;
            double dx = n.x - target_point.first;
            double dy = n.y - target_point.second;
            double d = dx*dx + dy*dy;
            if (d < min_dist) { min_dist = d; closest = n.id; }
        }
        return closest;
    }

    int find_edge_toward_point(int start_node_id, const std::pair<double,double>& target_point) const {
        auto it = nodes_.find(start_node_id);
        if (it == nodes_.end()) return -1;
        const Node &start = it->second;
        double target_vec_x = target_point.first - start.x;
        double target_vec_y = target_point.second - start.y;
        double len = std::sqrt(target_vec_x*target_vec_x + target_vec_y*target_vec_y);
        if (len == 0) return -1;
        target_vec_x /= len; target_vec_y /= len;
        double best_dot = -1e18;
        int best_edge = -1;
        for (auto &kv : edges_) {
            const Edge &e = kv.second;
            int end_id = -1;
            if (e.from_node == start_node_id) end_id = e.to_node; else if (e.to_node == start_node_id) end_id = e.from_node; else continue;
            const Node &end_node = nodes_.at(end_id);
            double ex = end_node.x - start.x;
            double ey = end_node.y - start.y;
            double elen = std::sqrt(ex*ex + ey*ey);
            if (elen == 0) continue;
            ex /= elen; ey /= elen;
            double dot = ex * target_vec_x + ey * target_vec_y;
            if (dot > best_dot) { best_dot = dot; best_edge = e.id; }
        }
        return best_edge;
    }

    int find_next_edge_with_turn(int current_node_id, int incoming_edge_id, bool turn_left) const {
        auto it_node = nodes_.find(current_node_id);
        if (it_node == nodes_.end()) return -1;
        auto it_edge = edges_.find(incoming_edge_id);
        if (it_edge == edges_.end()) return -1;
        const Node &current = it_node->second;
        const Edge &incoming = it_edge->second;
        int prev_node_id = (incoming.from_node == current_node_id) ? incoming.to_node : incoming.from_node;
        const Node &prev = nodes_.at(prev_node_id);
        double incoming_vec_x;
        double incoming_vec_y;
        if (incoming.type == "arc") {
            double radial_x = current.x - incoming.center_x;
            double radial_y = current.y - incoming.center_y;
            double radial_len = std::sqrt(radial_x*radial_x + radial_y*radial_y);
            if (radial_len == 0) return -1;
            radial_x /= radial_len; radial_y /= radial_len;
            double t1x = -radial_y, t1y = radial_x;
            double t2x = radial_y, t2y = -radial_x;
            double chord_x = current.x - prev.x; double chord_y = current.y - prev.y; double chord_len = std::sqrt(chord_x*chord_x + chord_y*chord_y);
            if (chord_len > 0) { chord_x /= chord_len; chord_y /= chord_len; }
            double dot1 = t1x * chord_x + t1y * chord_y;
            double dot2 = t2x * chord_x + t2y * chord_y;
            if (dot1 > dot2) { incoming_vec_x = t1x; incoming_vec_y = t1y; } else { incoming_vec_x = t2x; incoming_vec_y = t2y; }
        } else {
            incoming_vec_x = current.x - prev.x; incoming_vec_y = current.y - prev.y; double len = std::sqrt(incoming_vec_x*incoming_vec_x + incoming_vec_y*incoming_vec_y); if (len == 0) return -1; incoming_vec_x /= len; incoming_vec_y /= len;
        }
        struct Cand { int edge_id; double angle; double cross; }; std::vector<Cand> cands; cands.reserve(8);
        for (auto &kv : edges_) {
            const Edge &e = kv.second; if (e.id == incoming_edge_id) continue; int next_node_id; if (e.from_node == current_node_id) next_node_id = e.to_node; else if (e.to_node == current_node_id) next_node_id = e.from_node; else continue; const Node &nextN = nodes_.at(next_node_id);
            double outgoing_vec_x; double outgoing_vec_y;
            if (e.type == "arc") {
                double radial_x = current.x - e.center_x; double radial_y = current.y - e.center_y; double radial_len = std::sqrt(radial_x*radial_x + radial_y*radial_y); if (radial_len == 0) continue; radial_x /= radial_len; radial_y /= radial_len; double t1x = -radial_y, t1y = radial_x; double t2x = radial_y, t2y = -radial_x; double chord_x = nextN.x - current.x; double chord_y = nextN.y - current.y; double chord_len = std::sqrt(chord_x*chord_x + chord_y*chord_y); if (chord_len > 0) { chord_x /= chord_len; chord_y /= chord_len; } double dot1 = t1x*chord_x + t1y*chord_y; double dot2 = t2x*chord_x + t2y*chord_y; if (dot1 > dot2) { outgoing_vec_x = t1x; outgoing_vec_y = t1y; } else { outgoing_vec_x = t2x; outgoing_vec_y = t2y; }
            } else { outgoing_vec_x = nextN.x - current.x; outgoing_vec_y = nextN.y - current.y; double len = std::sqrt(outgoing_vec_x*outgoing_vec_x + outgoing_vec_y*outgoing_vec_y); if (len == 0) continue; outgoing_vec_x /= len; outgoing_vec_y /= len; }
            double cross = incoming_vec_x * outgoing_vec_y - incoming_vec_y * outgoing_vec_x; double dot = incoming_vec_x * outgoing_vec_x + incoming_vec_y * outgoing_vec_y; double angle = ::atan2(cross, dot); cands.push_back({e.id, angle, cross});
        }
        if (cands.empty()) return -1;
        int chosen = cands[0].edge_id; double chosen_angle = cands[0].angle;
        if (turn_left) {
            for (auto &c : cands) if (c.angle > chosen_angle) { chosen_angle = c.angle; chosen = c.edge_id; }
        } else {
            for (auto &c : cands) if (c.angle < chosen_angle) { chosen_angle = c.angle; chosen = c.edge_id; }
        }
        return chosen;
    }

    py::dict to_python_graph() const {
        py::dict nodes_dict;
        for (auto &kv : nodes_) {
            const Node &n = kv.second;
            py::dict d;
            d["point"] = py::make_tuple(n.x, n.y);
            d["angle"] = n.angle;
            d["intersection_type"] = n.intersection_type;
            d["has_line_circle"] = n.has_line_circle;
            d["has_line_line"] = n.has_line_line;
            d["intersecting_lines"] = n.intersecting_lines;
            d["line_circle_lines"] = n.line_circle_lines;
            nodes_dict[py::int_(n.id)] = d;
        }
        py::dict edges_dict;
        for (auto &kv : edges_) {
            const Edge &e = kv.second; py::dict d; d["from_node"] = e.from_node; d["to_node"] = e.to_node; d["type"] = e.type; py::dict edata; if (e.type == "line") { edata["line_segment"] = e.line_segment; } else { edata["center"] = py::make_tuple(e.center_x, e.center_y); edata["radius"] = e.radius; edata["start_angle"] = e.start_angle; edata["end_angle"] = e.end_angle; } d["data"] = edata; edges_dict[py::int_(e.id)] = d; }
        py::dict g; g["nodes"] = nodes_dict; g["edges"] = edges_dict; return g;
    }

private:
    // Geometry helpers
    static std::vector<std::pair<double,double>> line_circle_intersections(double x1,double y1,double x2,double y2,double cx,double cy,double r) {
        double dx = x2 - x1; double dy = y2 - y1; double fx = x1 - cx; double fy = y1 - cy; double a = dx*dx + dy*dy; double b = 2*(fx*dx + fy*dy); double c = fx*fx + fy*fy - r*r; double disc = b*b - 4*a*c; double tol = 1e-10; double seg_tol = 1e-3; std::vector<std::pair<double,double>> res; if (disc < -tol) return res; if (std::abs(disc) <= tol) { double t = -b/(2*a); if (-seg_tol <= t && t <= 1+seg_tol) { res.emplace_back(x1 + t*dx, y1 + t*dy); } } else { double sd = std::sqrt(std::max(0.0, disc)); double t1 = (-b - sd)/(2*a); double t2 = (-b + sd)/(2*a); if (-seg_tol <= t1 && t1 <= 1+seg_tol) res.emplace_back(x1 + t1*dx, y1 + t1*dy); if (-seg_tol <= t2 && t2 <= 1+seg_tol) res.emplace_back(x1 + t2*dx, y1 + t2*dy); }
        return res;
    }
    static std::pair<bool,std::pair<double,double>> line_line_intersection(double x1,double y1,double x2,double y2,double x3,double y3,double x4,double y4) {
        double dx1 = x2 - x1; double dy1 = y2 - y1; double dx2 = x4 - x3; double dy2 = y4 - y3; double denom = dx1*dy2 - dy1*dx2; if (std::abs(denom) < 1e-10) return {false,{0,0}}; double dx = x3 - x1; double dy = y3 - y1; double t1 = (dx * dy2 - dy * dx2)/denom; double t2 = (dx * dy1 - dy * dx1)/denom; double tol = 1e-6; if (-tol <= t1 && t1 <= 1+tol && -tol <= t2 && t2 <= 1+tol) { return {true,{x1 + t1*dx1, y1 + t1*dy1}}; } return {false,{0,0}}; }

    double distance_along_line(const std::pair<double,double>& p, const std::pair<double,double>& a, const std::pair<double,double>& b) const {
        double vx = b.first - a.first; double vy = b.second - a.second; double wx = p.first - a.first; double wy = p.second - a.second; double len2 = vx*vx + vy*vy; if (len2 == 0) return 0; double proj = (wx*vx + wy*vy)/len2; return proj * std::sqrt(len2);
    }

    void build_graph() {
        int edge_id_counter = 0; int node_id_counter = 0; std::unordered_map<int,std::vector<int>> line_nodes_map; struct IntersectionInfo { std::pair<double,double> point; bool has_line_circle=false; bool has_line_line=false; std::unordered_set<int> intersecting_lines; std::unordered_set<int> line_circle_lines; };
        std::vector<IntersectionInfo> intersections; // we will unify manually

        // Line-circle
        for (int i=0;i<(int)environment_lines_.size();++i) {
            const auto &seg = environment_lines_[i]; if (seg.size()!=2) continue; auto p1 = seg[0]; auto p2 = seg[1]; auto pts = line_circle_intersections(p1.first,p1.second,p2.first,p2.second,agent_x_,agent_y_,visibility_range_);
            for (auto &pt: pts) { bool merged=false; for (auto &info: intersections) { double dx = info.point.first - pt.first; double dy = info.point.second - pt.second; if (std::hypot(dx,dy) <= 1.0) { info.has_line_circle=true; info.intersecting_lines.insert(i); info.line_circle_lines.insert(i); merged=true; break; } } if (!merged) { IntersectionInfo inf; inf.point=pt; inf.has_line_circle=true; inf.intersecting_lines.insert(i); inf.line_circle_lines.insert(i); intersections.push_back(std::move(inf)); } }
        }
        // Line-line
        int n = (int)environment_lines_.size();
        for (int i=0;i<n;++i) {
            for (int j=i+1;j<n;++j) {
                const auto &l1=environment_lines_[i]; const auto &l2=environment_lines_[j]; if (l1.size()!=2 || l2.size()!=2) continue; auto p1=l1[0]; auto p2=l1[1]; auto p3=l2[0]; auto p4=l2[1]; auto inter = line_line_intersection(p1.first,p1.second,p2.first,p2.second,p3.first,p3.second,p4.first,p4.second); if (inter.first) { auto pt = inter.second; bool merged=false; for (auto &info: intersections) { double dx=info.point.first-pt.first; double dy=info.point.second-pt.second; if (std::hypot(dx,dy)<=1.0) { info.has_line_line=true; info.intersecting_lines.insert(i); info.intersecting_lines.insert(j); merged=true; break; } } if (!merged) { IntersectionInfo inf; inf.point=pt; inf.has_line_line=true; inf.intersecting_lines.insert(i); inf.intersecting_lines.insert(j); intersections.push_back(std::move(inf)); } }
            }
        }
        // Unify pass O(k^2) for small k
        for (size_t i=0;i<intersections.size();++i){ for(size_t j=i+1;j<intersections.size();++j){ double dx=intersections[i].point.first - intersections[j].point.first; double dy=intersections[i].point.second - intersections[j].point.second; if (std::hypot(dx,dy)<=1.0){ intersections[i].has_line_circle |= intersections[j].has_line_circle; intersections[i].has_line_line |= intersections[j].has_line_line; intersections[i].intersecting_lines.insert(intersections[j].intersecting_lines.begin(),intersections[j].intersecting_lines.end()); intersections[i].line_circle_lines.insert(intersections[j].line_circle_lines.begin(),intersections[j].line_circle_lines.end()); intersections.erase(intersections.begin()+j); --j; }}}
        // Create nodes
        for (auto &info: intersections) {
            Node node; node.id = node_id_counter; node.x = info.point.first; node.y = info.point.second; node.angle = ::atan2(node.y - agent_y_, node.x - agent_x_); node.has_line_circle = info.has_line_circle; node.has_line_line = info.has_line_line; node.intersection_type = (info.has_line_circle && info.has_line_line) ? "unified" : (info.has_line_circle ? "line_circle" : "line_line"); node.intersecting_lines.assign(info.intersecting_lines.begin(), info.intersecting_lines.end()); node.line_circle_lines.assign(info.line_circle_lines.begin(), info.line_circle_lines.end()); nodes_[node.id] = node; for (int li : node.intersecting_lines) line_nodes_map[li].push_back(node.id); ++node_id_counter; }
        // Line edges
        for (auto &kv : line_nodes_map) {
            auto line_idx = kv.first; auto &node_ids = kv.second; if (node_ids.size()<2) continue; const auto &seg = environment_lines_[line_idx]; auto a = seg[0]; auto b = seg[1]; std::sort(node_ids.begin(), node_ids.end(), [&](int lhs,int rhs){ return distance_along_line({nodes_[lhs].x,nodes_[lhs].y}, a, b) < distance_along_line({nodes_[rhs].x,nodes_[rhs].y}, a, b); }); for (size_t i=0;i+1<node_ids.size();++i){ Edge e; e.id = edge_id_counter++; e.from_node = node_ids[i]; e.to_node = node_ids[i+1]; e.type = "line"; e.line_segment = {seg[0], seg[1]}; edges_[e.id]=e; }
        }
        // Arc edges
        std::vector<int> circle_nodes; circle_nodes.reserve(nodes_.size()); for (auto &kvn : nodes_) { const Node &n = kvn.second; if (n.has_line_circle || n.intersection_type == "line_circle") circle_nodes.push_back(n.id); }
        if (circle_nodes.size()>=2) {
            std::sort(circle_nodes.begin(), circle_nodes.end(), [&](int a,int b){ return nodes_[a].angle < nodes_[b].angle; });
            for (size_t i=0;i<circle_nodes.size();++i){ int from = circle_nodes[i]; int to = circle_nodes[(i+1)%circle_nodes.size()]; if (from==to) continue; double start_angle = nodes_[from].angle; double end_angle = nodes_[to].angle; double diff = end_angle - start_angle; if (diff < 0) { diff += 2*M_PI; end_angle = start_angle + diff; } if (diff < (M_PI/1800.0)) continue; Edge e; e.id = edge_id_counter++; e.from_node=from; e.to_node=to; e.type="arc"; e.center_x=agent_x_; e.center_y=agent_y_; e.radius=visibility_range_; e.start_angle=start_angle; e.end_angle=end_angle; edges_[e.id]=e; }
        }
    }

    std::vector<std::vector<std::pair<double,double>>> environment_lines_;
    double agent_x_, agent_y_, visibility_range_;
    std::unordered_map<int, Node> nodes_;
    std::unordered_map<int, Edge> edges_;
};

// Main function analogous to Python calculate_polygon_exploration_paths
py::tuple calculate_polygon_exploration_paths_cpp(
    const std::vector<std::tuple<std::pair<double,double>, std::pair<double,double>, double, std::string>> &breakoff_lines,
    double agent_x,
    double agent_y,
    double visibility_range,
    const std::vector<std::vector<std::pair<double,double>>> &clipped_environment_lines,
    int max_iterations = 50
) {
    IntersectionGraph graph(clipped_environment_lines, agent_x, agent_y, visibility_range);
    py::list polygon_paths;

    for (size_t idx=0; idx<breakoff_lines.size(); ++idx) {
        const auto &[start_point, end_point, gap_size, category] = breakoff_lines[idx];
        double dist_start = std::sqrt((start_point.first - agent_x)*(start_point.first - agent_x) + (start_point.second - agent_y)*(start_point.second - agent_y));
        double dist_end   = std::sqrt((end_point.first - agent_x)*(end_point.first - agent_x) + (end_point.second - agent_y)*(end_point.second - agent_y));
        std::pair<double,double> far_point = dist_start > dist_end ? start_point : end_point;
        std::pair<double,double> near_point = dist_start > dist_end ? end_point : start_point;
        int start_node_id = graph.find_closest_node(far_point);
        if (start_node_id < 0) continue;
        int initial_edge_id = graph.find_edge_toward_point(start_node_id, near_point);
        if (initial_edge_id < 0) continue;
        bool turn_left = category.find("near_far") != std::string::npos; // replicate Python logic
        py::list path_points; py::list path_segments; bool completed=false; std::unordered_set<int> visited_edges; int current_node_id = start_node_id; int current_edge_id = initial_edge_id; path_points.append(py::make_tuple(graph.nodes().at(start_node_id).x, graph.nodes().at(start_node_id).y)); int iteration=0;
        for (iteration=0; iteration<max_iterations; ++iteration) {
            if (visited_edges.count(current_edge_id)) break; visited_edges.insert(current_edge_id); const Edge &edge = graph.edges().at(current_edge_id); int next_node_id = (edge.from_node == current_node_id) ? edge.to_node : edge.from_node; const Node &next_node = graph.nodes().at(next_node_id); path_points.append(py::make_tuple(next_node.x, next_node.y)); // segment
            py::dict seg; seg["start"] = py::make_tuple(graph.nodes().at(current_node_id).x, graph.nodes().at(current_node_id).y); seg["end"] = py::make_tuple(next_node.x, next_node.y); seg["type"] = edge.type; py::dict edge_data; if (edge.type == "arc") { edge_data["center"] = py::make_tuple(edge.center_x, edge.center_y); edge_data["radius"] = edge.radius; edge_data["start_angle"] = edge.start_angle; edge_data["end_angle"] = edge.end_angle; } else { edge_data["line_segment"] = edge.line_segment; } seg["edge_data"] = edge_data; path_segments.append(seg); if (next_node_id == start_node_id) { completed=true; break; } int next_edge_id = graph.find_next_edge_with_turn(next_node_id, current_edge_id, turn_left); if (next_edge_id < 0) break; current_node_id = next_node_id; current_edge_id = next_edge_id; }
        py::dict path_dict; path_dict["breakoff_line"] = py::make_tuple(py::make_tuple(start_point.first,start_point.second), py::make_tuple(end_point.first,end_point.second), gap_size, category); path_dict["path_points"] = path_points; path_dict["path_segments"] = path_segments; path_dict["completed"] = completed; path_dict["iterations"] = iteration + 1; polygon_paths.append(path_dict);
    }

    py::dict graph_dict = graph.to_python_graph();
    return py::make_tuple(polygon_paths, graph_dict);
}

PYBIND11_MODULE(polygon_exploration_cpp, m) {
    m.doc() = "C++ accelerated polygon exploration graph builder and path calculator";
    m.def("calculate_polygon_exploration_paths_cpp", &calculate_polygon_exploration_paths_cpp,
          py::arg("breakoff_lines"), py::arg("agent_x"), py::arg("agent_y"), py::arg("visibility_range"), py::arg("clipped_environment_lines"), py::arg("max_iterations")=50,
          "Calculate polygon exploration paths (C++ accelerated)");
}
