#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>

namespace py = pybind11;

/**
 * Fast C++ implementation of polygon rasterization using scanline algorithm
 * Replaces _rasterize_polygon_scanline() from Python
 */
py::array_t<bool> rasterize_polygon_scanline(
    py::array_t<float> vertices_array,
    int height,
    int width
) {
    auto vertices_buf = vertices_array.request();
    
    if (vertices_buf.ndim != 2 || vertices_buf.shape[1] != 2) {
        throw std::runtime_error("Vertices array must be Nx2");
    }
    
    int num_vertices = vertices_buf.shape[0];
    if (num_vertices < 3) {
        // Return empty mask for invalid polygons
        auto result = py::array_t<bool>(height * width);
        auto result_buf = result.request();
        std::fill((bool*)result_buf.ptr, (bool*)result_buf.ptr + height * width, false);
        result.resize({height, width});
        return result;
    }
    
    // Extract vertices as float array
    float* vertices = static_cast<float*>(vertices_buf.ptr);
    
    // Convert to integer coordinates
    std::vector<std::pair<int, int>> int_vertices;
    for (int i = 0; i < num_vertices; i++) {
        int x = static_cast<int>(std::round(vertices[i * 2]));
        int y = static_cast<int>(std::round(vertices[i * 2 + 1]));
        int_vertices.emplace_back(x, y);
    }
    
    // Ensure polygon is closed
    if (int_vertices[0] != int_vertices[num_vertices - 1]) {
        int_vertices.push_back(int_vertices[0]);
    }
    
    // Build edge table
    struct Edge {
        int y_min, y_max, x_start;
        float dx_dy;
    };
    
    std::vector<Edge> edges;
    for (size_t i = 0; i < int_vertices.size() - 1; i++) {
        int x1 = int_vertices[i].first;
        int y1 = int_vertices[i].second;
        int x2 = int_vertices[i + 1].first;
        int y2 = int_vertices[i + 1].second;
        
        // Skip horizontal edges
        if (y1 == y2) continue;
        
        // Ensure y1 < y2
        if (y1 > y2) {
            std::swap(x1, x2);
            std::swap(y1, y2);
        }
        
        int dy = y2 - y1;
        if (dy != 0) {
            float dx_dy = static_cast<float>(x2 - x1) / static_cast<float>(dy);
            edges.push_back({y1, y2, x1, dx_dy});
        }
    }
    
    // Create output mask
    auto result = py::array_t<bool>(height * width);
    auto result_buf = result.request();
    bool* mask = static_cast<bool*>(result_buf.ptr);
    std::fill(mask, mask + height * width, false);
    
    if (edges.empty()) {
        result.resize({height, width});
        return result;
    }
    
    // Find scanline bounds
    int y_min = height, y_max = -1;
    for (const auto& edge : edges) {
        y_min = std::min(y_min, edge.y_min);
        y_max = std::max(y_max, edge.y_max);
    }
    y_min = std::max(0, y_min);
    y_max = std::min(height, y_max + 1);
    
    // Process each scanline
    std::vector<float> intersections;
    intersections.reserve(edges.size());
    
    for (int y = y_min; y < y_max; y++) {
        intersections.clear();
        
        // Find intersections for this scanline
        for (const auto& edge : edges) {
            if (edge.y_min <= y && y < edge.y_max) {
                float x_intersect = edge.x_start + (y - edge.y_min) * edge.dx_dy;
                intersections.push_back(x_intersect);
            }
        }
        
        if (intersections.size() < 2) continue;
        
        // Sort intersections
        std::sort(intersections.begin(), intersections.end());
        
        // Fill between pairs of intersections
        for (size_t i = 0; i + 1 < intersections.size(); i += 2) {
            int x_start = std::max(0, static_cast<int>(std::round(intersections[i])));
            int x_end = std::min(width, static_cast<int>(std::round(intersections[i + 1])) + 1);
            
            if (x_start < x_end) {
                for (int x = x_start; x < x_end; x++) {
                    mask[y * width + x] = true;
                }
            }
        }
    }
    
    result.resize({height, width});
    return result;
}

/**
 * Fast C++ implementation of Prelect probability weighting
 * Replaces _apply_prelect_probability_weighting() from Python
 */
py::array_t<float> apply_prelect_weighting(
    py::array_t<float> grid,
    float alpha,
    float beta
) {
    auto buf = grid.request();
    int size = buf.size;
    
    auto result = py::array_t<float>(size);
    auto result_buf = result.request();
    
    float* input = static_cast<float*>(buf.ptr);
    float* output = static_cast<float*>(result_buf.ptr);
    
    const float epsilon = 1e-10f;
    const float one_minus_epsilon = 1.0f - epsilon;
    
    for (int i = 0; i < size; i++) {
        // Clamp input to valid range [epsilon, 1-epsilon]
        float p = std::max(epsilon, std::min(one_minus_epsilon, input[i]));
        
        // Apply Prelect transformation: p^α / (p^α + (1-p)^β)
        float p_alpha = std::pow(p, alpha);
        float one_minus_p_beta = std::pow(1.0f - p, beta);
        
        output[i] = p_alpha / (p_alpha + one_minus_p_beta);
    }
    
    result.resize(buf.shape);
    return result;
}

/**
 * Fast C++ implementation of grid statistics calculation
 * Replaces _calculate_grid_statistics() from Python
 */
py::dict calculate_grid_statistics(py::array_t<float> grid, int grid_size) {
    auto buf = grid.request();
    float* data = static_cast<float*>(buf.ptr);
    int total_cells = buf.size;
    
    // Calculate basic statistics
    float min_value = std::numeric_limits<float>::max();
    float max_value = std::numeric_limits<float>::lowest();
    int reachable_cells = 0;
    
    for (int i = 0; i < total_cells; i++) {
        float val = data[i];
        min_value = std::min(min_value, val);
        max_value = std::max(max_value, val);
        if (val > 0.0f) {
            reachable_cells++;
        }
    }
    
    int unreachable_cells = total_cells - reachable_cells;
    float reachable_percentage = 100.0f * reachable_cells / total_cells;
    
    // Calculate statistics for reachable cells only
    float mean_reachable = 0.0f;
    float std_reachable = 0.0f;
    float min_reachable = 0.0f;
    float max_reachable = 0.0f;
    
    if (reachable_cells > 0) {
        // Calculate mean
        float sum = 0.0f;
        min_reachable = std::numeric_limits<float>::max();
        max_reachable = std::numeric_limits<float>::lowest();
        
        for (int i = 0; i < total_cells; i++) {
            float val = data[i];
            if (val > 0.0f) {
                sum += val;
                min_reachable = std::min(min_reachable, val);
                max_reachable = std::max(max_reachable, val);
            }
        }
        mean_reachable = sum / reachable_cells;
        
        // Calculate standard deviation
        float var_sum = 0.0f;
        for (int i = 0; i < total_cells; i++) {
            float val = data[i];
            if (val > 0.0f) {
                float diff = val - mean_reachable;
                var_sum += diff * diff;
            }
        }
        std_reachable = std::sqrt(var_sum / reachable_cells);
    }
    
    // Find maximum location
    int max_idx = 0;
    for (int i = 1; i < total_cells; i++) {
        if (data[i] > data[max_idx]) {
            max_idx = i;
        }
    }
    
    // Convert linear index to 2D coordinates
    int max_row = max_idx / grid_size;
    int max_col = max_idx % grid_size;
    
    // Build result dictionary
    py::dict result;
    result["grid_size"] = grid_size;
    result["total_cells"] = total_cells;
    result["reachable_cells"] = reachable_cells;
    result["unreachable_cells"] = unreachable_cells;
    result["reachable_percentage"] = reachable_percentage;
    result["min_value"] = min_value;
    result["max_value"] = max_value;
    result["mean_reachable"] = mean_reachable;
    result["std_reachable"] = std_reachable;
    result["min_reachable"] = min_reachable;
    result["max_reachable"] = max_reachable;
    result["max_location_grid"] = py::make_tuple(max_row, max_col);
    
    return result;
}

/**
 * Fast C++ implementation of memory format conversion
 * Replaces _ensure_float32_array() from Python
 */
py::array_t<float> ensure_float32_contiguous(py::array_t<float> input) {
    auto buf = input.request();
    
    // Check if already float32 and contiguous
    if (input.dtype().is(py::dtype::of<float>()) && 
        (buf.strides[buf.ndim-1] == sizeof(float))) {
        
        // Check if C-contiguous
        bool is_contiguous = true;
        ssize_t expected_stride = sizeof(float);
        for (int i = buf.ndim - 1; i >= 0; i--) {
            if (buf.strides[i] != expected_stride) {
                is_contiguous = false;
                break;
            }
            expected_stride *= buf.shape[i];
        }
        
        if (is_contiguous) {
            return input; // Already in correct format
        }
    }
    
    // Create new contiguous array
    auto result = py::array_t<float>(buf.size);
    auto result_buf = result.request();
    
    float* input_data = static_cast<float*>(buf.ptr);
    float* output_data = static_cast<float*>(result_buf.ptr);
    
    // Copy data in C-contiguous order
    if (buf.ndim == 2) {
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int input_idx = i * buf.strides[0] / sizeof(float) + j * buf.strides[1] / sizeof(float);
                int output_idx = i * cols + j;
                output_data[output_idx] = input_data[input_idx];
            }
        }
    } else {
        // Handle 1D and other dimensions generically
        std::copy(input_data, input_data + buf.size, output_data);
    }
    
    result.resize(buf.shape);
    return result;
}

/**
 * Optimized coordinate transformation
 */
py::tuple grid_to_world_fast(int row, int col, int center_idx, float cell_size) {
    float world_x = (col - center_idx) * cell_size;
    float world_y = (center_idx - row) * cell_size;  // Flip Y axis
    return py::make_tuple(world_x, world_y);
}

/**
 * Element-wise maximum operation for two arrays
 * Replaces np.maximum() calls in apply_mask_to_canvas()
 */
py::array_t<float> element_wise_maximum(
    py::array_t<float> canvas,
    py::array_t<float> mask
) {
    auto canvas_buf = canvas.request();
    auto mask_buf = mask.request();
    
    if (canvas_buf.size != mask_buf.size) {
        throw std::runtime_error("Arrays must have the same size");
    }
    
    auto result = py::array_t<float>(canvas_buf.size);
    auto result_buf = result.request();
    
    float* canvas_data = static_cast<float*>(canvas_buf.ptr);
    float* mask_data = static_cast<float*>(mask_buf.ptr);
    float* result_data = static_cast<float*>(result_buf.ptr);
    
    // Element-wise maximum
    for (py::ssize_t i = 0; i < canvas_buf.size; ++i) {
        result_data[i] = std::max(canvas_data[i], mask_data[i]);
    }
    
    result.resize(canvas_buf.shape);
    return result;
}

PYBIND11_MODULE(fast_reachability, m) {
    
    m.def("rasterize_polygon_scanline", &rasterize_polygon_scanline,
          "Fast polygon rasterization using scanline algorithm",
          py::arg("vertices"), py::arg("height"), py::arg("width"));
    
    m.def("apply_prelect_weighting", &apply_prelect_weighting,
          "Fast Prelect probability weighting transformation",
          py::arg("grid"), py::arg("alpha"), py::arg("beta"));
    
    m.def("calculate_grid_statistics", &calculate_grid_statistics,
          "Fast grid statistics calculation",
          py::arg("grid"), py::arg("grid_size"));
    
    m.def("ensure_float32_contiguous", &ensure_float32_contiguous,
          "Ensure array is float32 and C-contiguous",
          py::arg("input"));
    
    m.def("grid_to_world_fast", &grid_to_world_fast,
          "Fast coordinate transformation",
          py::arg("row"), py::arg("col"), py::arg("center_idx"), py::arg("cell_size"));
    
    m.def("element_wise_maximum", &element_wise_maximum,
          "Fast element-wise maximum operation",
          py::arg("a"), py::arg("b"));
}
