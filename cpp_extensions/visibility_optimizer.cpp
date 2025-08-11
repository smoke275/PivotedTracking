#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <tuple>
#ifndef __AVX__
#define __AVX__
#endif
#include <immintrin.h>  // For SIMD operations

using namespace std;

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

struct RayResult {
    double angle;
    Point endpoint;
    double distance;
    bool blocked;
    
    RayResult(double a, const Point& ep, double d, bool b) 
        : angle(a), endpoint(ep), distance(d), blocked(b) {}
};

class VisibilityOptimizer {
private:
    std::vector<LineSegment> wall_segments;
    std::vector<Rectangle> door_rects;
    
    // Precomputed trigonometric values for faster ray casting
    std::vector<double> cos_table, sin_table;
    
    void precompute_trig_tables(int num_rays) {
        cos_table.resize(num_rays);
        sin_table.resize(num_rays);
        
        const double angle_step = 2.0 * M_PI / num_rays;
        for (int i = 0; i < num_rays; ++i) {
            double angle = i * angle_step;
            cos_table[i] = cos(angle);
            sin_table[i] = sin(angle);
        }
    }
    
    // Fast line intersection using optimized math
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
    
    // Check if point is inside any door (optimized)
    bool is_point_in_door(double x, double y) const {
        for (const auto& door : door_rects) {
            // Slightly expanded door rect for tolerance
            if (x >= door.x - 2 && x <= door.x + door.width + 2 &&
                y >= door.y - 2 && y <= door.y + door.height + 2) {
                return true;
            }
        }
        return false;
    }
    
    // Optimized single ray casting
    Point cast_single_ray(double start_x, double start_y, double cos_angle, double sin_angle, double max_distance) const {
        const double end_x = start_x + cos_angle * max_distance;
        const double end_y = start_y + sin_angle * max_distance;
        
        Point closest_point(end_x, end_y);
        double closest_dist_sq = max_distance * max_distance;
        
        // Test against all wall segments
        for (const auto& segment : wall_segments) {
            auto [has_intersection, intersection] = line_intersection(
                start_x, start_y, end_x, end_y,
                segment.p1.x, segment.p1.y, segment.p2.x, segment.p2.y
            );
            
            if (has_intersection && !is_point_in_door(intersection.x, intersection.y)) {
                const double dx = intersection.x - start_x;
                const double dy = intersection.y - start_y;
                const double dist_sq = dx * dx + dy * dy;
                
                if (dist_sq < closest_dist_sq) {
                    closest_dist_sq = dist_sq;
                    closest_point = intersection;
                }
            }
        }
        
        return closest_point;
    }
    
public:
    void set_walls(const std::vector<std::tuple<double, double, double, double>>& walls) {
        wall_segments.clear();
        wall_segments.reserve(walls.size() * 4);  // 4 segments per rectangle
        
        for (const auto& [x, y, w, h] : walls) {
            Rectangle wall(x, y, w, h);
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
        
        for (const auto& [x, y, w, h] : doors) {
            door_rects.emplace_back(x, y, w, h);
        }
    }
    
    // Main optimized visibility calculation
    std::vector<std::tuple<double, double, double, double, bool>> calculate_visibility(
        double agent_x, double agent_y, double visibility_range, int num_rays = 100) {
        
        precompute_trig_tables(num_rays);
        
        std::vector<std::tuple<double, double, double, double, bool>> results;
        results.reserve(num_rays);
        
        const double tolerance = 1.0;  // Distance tolerance for blocked detection
        
        // Process rays in batches for better cache performance
        constexpr int batch_size = 8;
        
        for (int i = 0; i < num_rays; i += batch_size) {
            const int batch_end = std::min(i + batch_size, num_rays);
            
            // Process batch
            for (int j = i; j < batch_end; ++j) {
                const double angle = j * (2.0 * M_PI) / num_rays;
                const Point endpoint = cast_single_ray(agent_x, agent_y, cos_table[j], sin_table[j], visibility_range);
                
                // Calculate distance using optimized math
                const double dx = endpoint.x - agent_x;
                const double dy = endpoint.y - agent_y;
                const double distance = std::sqrt(dx * dx + dy * dy);
                const bool blocked = distance < (visibility_range - tolerance);
                
                results.emplace_back(angle, endpoint.x, endpoint.y, distance, blocked);
            }
        }
        
        return results;
    }
    
    // SIMD-optimized batch distance calculation (for future enhancement)
    void calculate_distances_simd(const std::vector<Point>& points, double agent_x, double agent_y, std::vector<double>& distances) const {
        distances.resize(points.size());
        
        // Process 4 points at a time using AVX
        size_t simd_end = (points.size() / 4) * 4;
        
        const __m256d agent_x_vec = _mm256_set1_pd(agent_x);
        const __m256d agent_y_vec = _mm256_set1_pd(agent_y);
        
        for (size_t i = 0; i < simd_end; i += 4) {
            // Load 4 x coordinates
            __m256d x_vec = _mm256_set_pd(points[i+3].x, points[i+2].x, points[i+1].x, points[i].x);
            __m256d y_vec = _mm256_set_pd(points[i+3].y, points[i+2].y, points[i+1].y, points[i].y);
            
            // Calculate dx and dy
            __m256d dx = _mm256_sub_pd(x_vec, agent_x_vec);
            __m256d dy = _mm256_sub_pd(y_vec, agent_y_vec);
            
            // Calculate distance squared
            __m256d dx_sq = _mm256_mul_pd(dx, dx);
            __m256d dy_sq = _mm256_mul_pd(dy, dy);
            __m256d dist_sq = _mm256_add_pd(dx_sq, dy_sq);
            
            // Calculate square root
            __m256d dist = _mm256_sqrt_pd(dist_sq);
            
            // Store results
            double dist_array[4];
            _mm256_store_pd(dist_array, dist);
            
            for (int j = 0; j < 4; ++j) {
                distances[i + j] = dist_array[j];
            }
        }
        
        // Handle remaining points
        for (size_t i = simd_end; i < points.size(); ++i) {
            const double dx = points[i].x - agent_x;
            const double dy = points[i].y - agent_y;
            distances[i] = std::sqrt(dx * dx + dy * dy);
        }
    }
};

// Pybind11 module definition
PYBIND11_MODULE(visibility_optimizer, m) {
    m.doc() = "Fast C++ visibility calculation for agent tracking";
    
    py::class_<VisibilityOptimizer>(m, "VisibilityOptimizer")
        .def(py::init<>())
        .def("set_walls", &VisibilityOptimizer::set_walls,
             "Set wall rectangles as list of (x, y, width, height) tuples")
        .def("set_doors", &VisibilityOptimizer::set_doors,
             "Set door rectangles as list of (x, y, width, height) tuples")
        .def("calculate_visibility", &VisibilityOptimizer::calculate_visibility,
             "Calculate 360-degree visibility from agent position",
             py::arg("agent_x"), py::arg("agent_y"), py::arg("visibility_range"), py::arg("num_rays") = 100);
    
    // Convenience function that matches the original Python API
    m.def("calculate_fast_visibility", 
        [](double agent_x, double agent_y, double visibility_range,
           const std::vector<std::tuple<double, double, double, double>>& walls,
           const std::vector<std::tuple<double, double, double, double>>& doors,
           int num_rays) {
            
            VisibilityOptimizer optimizer;
            optimizer.set_walls(walls);
            optimizer.set_doors(doors);
            return optimizer.calculate_visibility(agent_x, agent_y, visibility_range, num_rays);
        },
        "Fast visibility calculation matching original Python API",
        py::arg("agent_x"), py::arg("agent_y"), py::arg("visibility_range"),
        py::arg("walls"), py::arg("doors"), py::arg("num_rays") = 100);
}
