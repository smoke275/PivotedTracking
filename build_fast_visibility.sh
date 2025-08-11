#!/bin/bash
# Build script for fast visibility C++ extension

echo "🔨 Building Fast Visibility C++ Extension..."
echo "============================================="

# Check if required packages are installed
echo "📦 Checking dependencies..."

python3 -c "import pybind11" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ pybind11 not found. Installing..."
    pip install pybind11
fi

python3 -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ numpy not found. Installing..."
    pip install numpy
fi

# Build the extension
echo "🏗️  Building C++ extension..."
python3 setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "🧪 Running benchmark..."
    python3 fast_visibility_calculator.py
    echo ""
    echo "🎉 Fast visibility extension is ready!"
    echo "💡 The system will automatically use C++ optimization when available."
else
    echo "❌ Build failed!"
    echo "🐍 The system will fall back to Python implementation."
    echo ""
    echo "Troubleshooting:"
    echo "- Make sure you have a C++ compiler installed (gcc/clang)"
    echo "- Install required packages: pip install pybind11 numpy"
    echo "- Check that Python development headers are installed"
fi
