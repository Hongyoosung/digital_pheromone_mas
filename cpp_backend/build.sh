#!/bin/bash

# Build script for C++ accelerators module
# This script builds the C++ extension and installs it to src/core/

set -e  # Exit on error

echo "======================================"
echo "Building C++ Accelerators Module"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found!"
    echo "Please run this script from the cpp_backend directory"
    exit 1
fi

# Check for required dependencies
echo ""
echo "Checking dependencies..."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found!"
    exit 1
fi

echo "  ✓ Python found: $(python3 --version)"

# Check for pybind11
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "  ✗ pybind11 not found"
    echo "    Installing pybind11..."
    pip install pybind11
else
    echo "  ✓ pybind11 found"
fi

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not found!"
    echo "Please install cmake: sudo apt-get install cmake"
    exit 1
fi

echo "  ✓ CMake found: $(cmake --version | head -n1)"

# Check for C++ compiler
if ! command -v g++ &> /dev/null; then
    echo "Error: g++ not found!"
    echo "Please install g++: sudo apt-get install build-essential"
    exit 1
fi

echo "  ✓ g++ found: $(g++ --version | head -n1)"

# Check for AVX2 support
if grep -q avx2 /proc/cpuinfo; then
    echo "  ✓ AVX2 support detected"
else
    echo "  ⚠ AVX2 not detected - SIMD optimizations will be limited"
fi

echo ""
echo "Creating build directory..."
mkdir -p build
cd build

echo ""
echo "Running CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo ""
echo "Compiling..."
make -j$(nproc)

echo ""
echo "Installing to src/core/..."
make install

echo ""
echo "======================================"
echo "Build completed successfully!"
echo "======================================"

# Verify installation
cd ../..
if python3 -c "import sys; sys.path.insert(0, 'src/core'); import cpp_accelerators; print('✓ Module can be imported successfully')" 2>/dev/null; then
    echo ""
    python3 -c "import sys; sys.path.insert(0, 'src/core'); import cpp_accelerators; print('C++ Accelerators version:', cpp_accelerators.version())"
    python3 -c "import sys; sys.path.insert(0, 'src/core'); import cpp_accelerators; print('Hardware concurrency:', cpp_accelerators.get_hardware_concurrency(), 'threads')"
    echo ""
    echo "Ready to use! Import with:"
    echo "  from src.core.cpp_accelerators import MessageCodecWrapper"
else
    echo ""
    echo "⚠ Warning: Module built but import failed"
    echo "You may need to check your PYTHONPATH"
fi

echo ""
