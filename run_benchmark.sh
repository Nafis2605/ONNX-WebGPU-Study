#!/bin/bash

# Simple wrapper script for running ONNX benchmarks
# Edit the variables below to match your setup

# Configuration
BENCHMARK_URL="http://localhost:5173"
MODELS_JSON="./public/models/models.json"
BACKEND="wasm"  # Options: wasm, webgl, webgpu, webnn
WARMUP=1
MEASURE=100
OUTPUT_FORMAT="csv"  # Options: csv, excel

echo "=========================================="
echo "ONNX Runtime Web Benchmark Runner"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  URL: $BENCHMARK_URL"
echo "  Models: $MODELS_JSON"
echo "  Backend: $BACKEND"
echo "  Warmup: $WARMUP runs"
echo "  Measure: $MEASURE runs"
echo "  Format: $OUTPUT_FORMAT"
echo ""
echo "=========================================="
echo ""

# Check if Python script exists
if [ ! -f "run_benchmark_cli.py" ]; then
    echo "ERROR: run_benchmark_cli.py not found!"
    echo "Make sure you're in the correct directory."
    exit 1
fi

# Check if models.json exists
if [ ! -f "$MODELS_JSON" ]; then
    echo "ERROR: $MODELS_JSON not found!"
    echo "Please update MODELS_JSON path in this script."
    exit 1
fi

# Run the benchmark
python3 run_benchmark_cli.py \
    --url "$BENCHMARK_URL" \
    --models "$MODELS_JSON" \
    --backend "$BACKEND" \
    --warmup "$WARMUP" \
    --measure "$MEASURE" \
    --format "$OUTPUT_FORMAT"

echo ""
echo "=========================================="
echo "Benchmark complete!"
echo "Check the benchmark_results/ directory for output"
echo "=========================================="
