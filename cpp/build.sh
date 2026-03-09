#!/bin/bash

# Build script for C++ multithreaded backtest optimizer
# Usage: ./cpp/build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Building C++ Grid Search Optimizer..."
echo "======================================="

# Detect OS
OS_TYPE=$(uname -s)
if [[ "$OS_TYPE" == "Darwin" ]]; then
    # macOS
    echo "Detected macOS - using Clang"
    COMPILER="clang++"
    SHARED_FLAG="-dynamiclib"
    OUTPUT_FILE="$SCRIPT_DIR/backtest_optimizer.dylib"
else
    # Linux
    echo "Detected Linux - using g++"
    COMPILER="g++"
    SHARED_FLAG="-shared"
    OUTPUT_FILE="$SCRIPT_DIR/backtest_optimizer.so"
fi

# Check if compiler exists
if ! command -v $COMPILER &> /dev/null; then
    echo "Error: $COMPILER not found. Please install C++ compiler."
    exit 1
fi

echo "Compiler: $COMPILER"
echo "Output: $OUTPUT_FILE"

# Compile
$COMPILER \
    -O3 \
    -std=c++17 \
    -fPIC \
    $SHARED_FLAG \
    -pthread \
    -o "$OUTPUT_FILE" \
    "$SCRIPT_DIR/backtest_optimizer.cpp"

echo "✓ Build successful!"
echo "Library created: $OUTPUT_FILE"
