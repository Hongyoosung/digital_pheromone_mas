#!/bin/bash

# Performance Profiling Example Script
# =====================================
# This script demonstrates the complete workflow for profiling and optimization

set -e  # Exit on error

echo "================================================================"
echo "Digital Pheromone MAS - Performance Profiling Workflow"
echo "================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Create output directories
echo -e "${YELLOW}Step 1: Creating output directories...${NC}"
mkdir -p profiling_results
mkdir -p benchmark_results
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 2: Run performance profiler
echo -e "${YELLOW}Step 2: Running performance profiler (quick mode)...${NC}"
echo "This will profile communication, field operations, and spatial queries"
python performance_profiler.py \
    --config config/experiment_config.yaml \
    --output profiling_results/ \
    --quick

echo -e "${GREEN}✓ Profiling complete${NC}"
echo "  Results saved to: profiling_results/"
echo "  - performance_analysis.png (visualization)"
echo "  - PERFORMANCE_REPORT.md (detailed report)"
echo "  - bottleneck_analysis.json (JSON data)"
echo ""

# Step 3: Run baseline benchmark
echo -e "${YELLOW}Step 3: Running baseline benchmark...${NC}"
echo "This establishes Python baseline for comparison"
python benchmark_comparison.py \
    --mode baseline \
    --output benchmark_results/baseline_results.json

echo -e "${GREEN}✓ Baseline benchmark complete${NC}"
echo "  Results saved to: benchmark_results/baseline_results.json"
echo ""

# Step 4: Display summary
echo "================================================================"
echo -e "${GREEN}Profiling Workflow Complete!${NC}"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Review profiling_results/PERFORMANCE_REPORT.md"
echo "  2. Review profiling_results/performance_analysis.png"
echo "  3. Implement C++ optimizations following CPP_IMPLEMENTATION_GUIDE.md"
echo "  4. Run: python benchmark_comparison.py --mode cpp"
echo "  5. Compare: python benchmark_comparison.py --mode compare \\"
echo "       --baseline benchmark_results/baseline_results.json \\"
echo "       --cpp benchmark_results/cpp_results.json"
echo ""
echo "Expected improvements:"
echo "  - Communication: 4-10x speedup"
echo "  - Field operations: 2-5x speedup"
echo "  - Spatial queries: 3-10x speedup"
echo "  - Overall simulation: 2-5x speedup"
echo ""
echo "================================================================"
