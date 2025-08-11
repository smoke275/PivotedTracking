#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <tuple>

namespace py = pybind11;

struct Point {
    double x, y;
    Point(double x = 0, double y = 0) : x(x), y(y) {}
};

struct Rectangle {
    double x, y, width, height;
    Rectangle(double x, double y, double w, double h) : x(x), y(y), width(w), height(h) {}
};

struct LineSegment {
    Point p1, p2;
    LineSegment(const Point& p1, const Point& p2) : p1(p1), p2(p2) {}
};

class FastVisibilityCalculator {
private:
    std::vector<LineSegment> wall_segments;
    std::vector<Rectangle> door_rects;
    
    // Fast line intersection
    std::pair<bool, Point> line_intersection(
        double x1, double y1, double x2, double y2,
        double x3, double y3, double x4, double y4) const {
        
        const double denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if (std::abs(denom) < 1e-10) return {false, Point()};
        
        const double t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
        const double u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;
        
        if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
            return {true, Point(x1 + t * (x2 - x1), y1 + t * (y2 - y1))};
        }
        
        return {false, Point()};
    }
    
    // Check if point is inside any door
    bool is_point_in_door(double x, double y) const {
        for (const auto& door : door_rects) {
            if (x >= door.x - 2 && x <= door.x + door.width + 2 &&
                y >= door.y - 2 && y <= door.y + door.height + 2) {
                return true;
            }
        }
        return false;
    }
    
    // Cast single ray
    Point cast_single_ray(double start_x, double start_y, double angle, double max_distance) const {
        const double end_x = start_x + std::cos(angle) * max_distance;
        const double end_y = start_y + std::sin(angle) * max_distance;
        
        Point closest_point(end_x, end_y);
        double closest_dist_sq = max_distance * max_distance;
        
        // Test against all wall segments
        for (const auto& segment : wall_segments) {
            auto result = line_intersection(
                start_x, start_y, end_x, end_y,
                segment.p1.x, segment.p1.y, segment.p2.x, segment.p2.y
            );
            
            if (result.first && !is_point_in_door(result.second.x, result.second.y)) {
                const double dx = result.second.x - start_x;
                const double dy = result.second.y - start_y;
                const double dist_sq = dx * dx + dy * dy;
                
                if (dist_sq < closest_dist_sq) {
                    closest_dist_sq = dist_sq;
                    closest_point = result.second;
                }
            }
        }
        
        return closest_point;
    }
    
public:
    void set_walls(const std::vector<std::tuple<double, double, double, double>>& walls) {
        wall_segments.clear();
        wall_segments.reserve(walls.size() * 4);
        
        for (const auto& wall_tuple : walls) {
            double x = std::get<0>(wall_tuple);
            double y = std::get<1>(wall_tuple);
            double w = std::get<2>(wall_tuple);
            double h = std::get<3>(wall_tuple);
            
            // Add 4 segments for each wall rectangle
            wall_segments.emplace_back(Point(x, y), Point(x + w, y));           // Top
            wall_segments.emplace_back(Point(x, y), Point(x, y + h));           // Left
            wall_segments.emplace_back(Point(x + w, y), Point(x + w, y + h));   // Right
            wall_segments.emplace_back(Point(x, y + h), Point(x + w, y + h));   // Bottom
        }
    }
    
    void set_doors(const std::vector<std::tuple<double, double, double, double>>& doors) {
        door_rects.clear();
        door_rects.reserve(doors.size());
        
        for (const auto& door_tuple : doors) {
            double x = std::get<0>(door_tuple);
            double y = std::get<1>(door_tuple);
            double w = std::get<2>(door_tuple);
            double h = std::get<3>(door_tuple);
            door_rects.emplace_back(x, y, w, h);
        }
    }
    
    // Main visibility calculation
    std::vector<std::tuple<double, double, double, double, bool>> calculate_visibility(
        double agent_x, double agent_y, double visibility_range, int num_rays = 100) {
        
        std::vector<std::tuple<double, double, double, double, bool>> results;
        results.reserve(num_rays);
        
        const double angle_step = 2.0 * M_PI / num_rays;
        const double tolerance = 1.0;
        
        for (int i = 0; i < num_rays; ++i) {
            const double angle = i * angle_step;
            const Point endpoint = cast_single_ray(agent_x, agent_y, angle, visibility_range);
            
            const double dx = endpoint.x - agent_x;
            const double dy = endpoint.y - agent_y;
            const double distance = std::sqrt(dx * dx + dy * dy);
            const bool blocked = distance < (visibility_range - tolerance);
            
            results.emplace_back(angle, endpoint.x, endpoint.y, distance, blocked);
        }
        
        return results;
    }
};

// Convenience function matching original Python API
std::vector<std::tuple<double, double, double, double, bool>> calculate_fast_visibility(
    double agent_x, double agent_y, double visibility_range,
    const std::vector<std::tuple<double, double, double, double>>& walls,
    const std::vector<std::tuple<double, double, double, double>>& doors,
    int num_rays = 100) {
    
    FastVisibilityCalculator calculator;
    calculator.set_walls(walls);
    calculator.set_doors(doors);
    return calculator.calculate_visibility(agent_x, agent_y, visibility_range, num_rays);
}

PYBIND11_MODULE(fast_visibility, m) {
    m.doc() = "Fast C++ visibility calculation for agent tracking";
    
    py::class_<FastVisibilityCalculator>(m, "FastVisibilityCalculator")
        .def(py::init<>())
        .def("set_walls", &FastVisibilityCalculator::set_walls,
             "Set wall rectangles as list of (x, y, width, height) tuples")
        .def("set_doors", &FastVisibilityCalculator::set_doors,
             "Set door rectangles as list of (x, y, width, height) tuples")
        .def("calculate_visibility", &FastVisibilityCalculator::calculate_visibility,
             "Calculate 360-degree visibility from agent position",
             py::arg("agent_x"), py::arg("agent_y"), py::arg("visibility_range"), py::arg("num_rays") = 100);
    
    m.def("calculate_fast_visibility", &calculate_fast_visibility,
        "Fast visibility calculation matching original Python API",
        py::arg("agent_x"), py::arg("agent_y"), py::arg("visibility_range"),
        py::arg("walls"), py::arg("doors"), py::arg("num_rays") = 100);
}
