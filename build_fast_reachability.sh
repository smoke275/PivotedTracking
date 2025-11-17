#!/bin/bash

# Build script for fast_reachability C++ extension
echo "🔧 Building fast_reachability C++ extension..."

# Check if pybind11 is installed
python -c "import pybind11" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ pybind11 not found. Installing..."
    pip install pybind11
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/
rm -rf fast_reachability.cpython-*.so
rm -rf *.egg-info

# Build the extension
echo "🔨 Building C++ extension..."
python setup.py build_ext --inplace

# Check if build was successful
if [ -f fast_reachability.cpython-*.so ]; then
    echo "✅ Build successful! Extension created:"
    ls -la fast_reachability.cpython-*.so
    echo ""
    echo "🧪 Testing the extension..."
    python -c "
import fast_reachability
import numpy as np
print('✅ Import successful!')

# Test polygon rasterization
vertices = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)
mask = fast_reachability.rasterize_polygon_scanline(vertices, 20, 20)
print(f'✅ Polygon rasterization: {mask.shape}, filled pixels: {np.sum(mask)}')

# Test Prelect weighting
grid = np.random.rand(100, 100).astype(np.float32)
weighted = fast_reachability.apply_prelect_weighting(grid, 0.8, 0.8)
print(f'✅ Prelect weighting: input [{grid.min():.3f}, {grid.max():.3f}] -> output [{weighted.min():.3f}, {weighted.max():.3f}]')

# Test statistics
stats = fast_reachability.calculate_grid_statistics(grid, 100)
print(f'✅ Statistics: {stats[\"reachable_cells\"]} reachable cells, mean = {stats[\"mean_reachable\"]:.3f}')

print('🎉 All tests passed!')
"
else
    echo "❌ Build failed!"
    exit 1
fi
