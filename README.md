# ONNX Runtime Web Benchmark - Complete Guide

A comprehensive benchmarking tool for ONNX Runtime Web models with detailed performance metrics including kernel execution time, memory usage, and end-to-end latency across different execution providers (WASM, WebGL, WebGPU, WebNN).

## 🚀 Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
# Python packages
pip install -r requirements.txt

# Browser for testing
playwright install chromium
```

### 2. Start the Web Server

```bash
npm run dev
# Server runs at http://localhost:5173 (or next available port)
```

### 3. Run Benchmark

```bash
# Activate Python environment (if using venv)
source .venv/bin/activate

# Run benchmark
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend webgpu \
    --warmup 1 \
    --measure 100
```

Results are saved to `benchmark_results/` directory as CSV or Excel files.

---

## 📋 Features

- **Multiple Backends**: WASM, WebGL, WebGPU, WebNN
- **10 Performance Metrics**:
  1. Kernel Execution Time
  2. Kernel Launch Latency
  3. Operator Fusion Rate
  4. Per-Operator Latency
  5. Kernel Compilation Time
  6. Effective Memory Bandwidth (GB/s)
  7. Synchronization Overhead
  8. Peak Memory Usage (MB)
  9. Time to First Output
  10. End-to-End Latency

- **Automatic Results Export**: CSV or Excel format
- **Browser Automation**: Playwright-based with cache clearing
- **Headless & Headed Modes**: Run visible browser or automated headless
- **Model Filtering**: Run specific models by keyword
- **Progress Monitoring**: Real-time status updates

---

## 🔧 Setup

### Prerequisites
- Node.js 16+
- Python 3.8+
- Chrome/Chromium browser

### Full Installation

```bash
# Install Node dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Build WASM resources
npm run prepare-wasm
```

---

## 📊 Usage

### Command Line Options

```bash
python run_benchmark.py \
    --url http://localhost:5173              # Dev server URL
    --models ./public/models/models.json     # Models config file
    --backend webgpu                         # Execution provider: wasm, webgl, webgpu, webnn
    --warmup 1                               # Warmup runs per model
    --measure 100                            # Measurement runs per model
    --filter bertsquad mobilenet             # Filter models by keyword (optional)
    --output-dir benchmark_results           # Output directory (default)
    --format csv                             # Output format: csv or excel
    --headless                               # Run without visible browser
```

### Examples

**Benchmark all models with WASM backend:**
```bash
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend wasm \
    --warmup 2 \
    --measure 50
```

**Benchmark only MobileNet with WebGPU:**
```bash
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend webgpu \
    --filter mobilenet \
    --warmup 1 \
    --measure 100
```

**Run in headless mode with Excel output:**
```bash
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend webgl \
    --format excel \
    --headless
```

---

## 📁 Project Structure

```
.
├── src/
│   └── main.js                    # Main benchmark application
├── public/
│   ├── models/                    # ONNX model files
│   │   ├── models.json            # Model configuration
│   │   └── *.onnx                 # Model files
│   └── ort-wasm/                  # ONNX Runtime WASM binaries
├── run_benchmark.py         # Main benchmark automation script
├── requirements.txt               # Python dependencies
├── index.html                     # Web UI
├── vite.config.js                 # Vite configuration
└── package.json                   # Node dependencies
```

---

## 📈 Output Files

Results are saved in `benchmark_results/` with timestamp:
```
benchmark_results/benchmark_metrics_20251229_143022.csv
```

### CSV Columns
- `timestamp`: Benchmark start time
- `model`: Model name
- `backend`: Execution provider used
- `warmup_runs`: Number of warmup iterations
- `measure_runs`: Number of measurement iterations
- `avg_warmup_ms`: Average warmup latency
- `avg_inference_ms`: Average inference latency
- `kernel_execution_time_ms`: Kernel computation time
- `kernel_launch_latency_ms`: Time to launch kernels
- `operator_fusion_rate%`: Percentage of fused operators
- `per_operator_latency_ms`: Average per-operator time
- `kernel_compilation_time_ms`: Graph compilation time
- `effective_memory_bandwidth_gbps`: Memory throughput
- `synchronization_overhead_ms`: GPU sync wait time
- `peak_memory_usage_mb`: Maximum memory used
- `time_to_first_output_ms`: Latency to first result
- `end_to_end_latency_ms`: Total wall-clock time
- `status`: SUCCESS or FAILED
- `notes`: Error messages or additional info

---

## ⚠️ Known Issues

### ONNX Runtime Web int64 Support
**Issue**: BERT models require int64 inputs but ONNX Runtime Web v1.23.2 has a bug where int64 tensors are created as float32.

**Error**:
```
ERROR_CODE: 2
ERROR_MESSAGE: Unexpected input data type. Actual: (tensor(float)), expected: (tensor(int64))
```

**Status**: BERT models (bertsquad-12) are currently not supported with WebGPU backend due to this limitation.

**Workaround**: Use WASM backend instead:
```bash
python run_benchmark.py \
    --url http://localhost:5173 \
    --backend wasm \
    --filter bertsquad
```

**Potential Solutions**:
1. Upgrade ONNX Runtime Web: `npm install onnxruntime-web@latest`
2. Use WASM backend instead of WebGPU
3. Check https://github.com/microsoft/onnxruntime/issues for updates

---

## 🐛 Troubleshooting

### Browser Won't Open
```bash
# Install/update Playwright browsers
playwright install chromium --force
```

### Cannot Connect to Server
```bash
# Verify server is running
curl http://localhost:5173

# Check if port is in use
lsof -i :5173

# Try a different port
npm run dev -- --port 3000
```

### Model Files Not Found
- Verify files exist in `public/models/`
- Check `public/models/models.json` has correct paths
- Ensure model files are readable: `ls -la public/models/*.onnx`

### Benchmark Hangs
- Increase Python timeout (edit `run_benchmark.py`)
- Run single model to isolate issue: `--filter mobilenet`
- Check browser console: run without `--headless`

### Memory Issues
- Reduce `--measure` value (e.g., 10 instead of 100)
- Increase system memory or close other applications
- Run one model at a time with fresh browser context

---

## 📚 Development

### Adding New Models

1. Place ONNX file in `public/models/`
2. Update `public/models/models.json`:

```json
{
  "name": "my-model",
  "path": "/models/my-model.onnx",
  "inputs": [
    { "name": "input", "type": "float32", "dims": [1, 224, 224, 3] }
  ]
}
```

### Testing Locally

```bash
# Start dev server
npm run dev

# Open in browser
open http://localhost:5173

# Run benchmark programmatically
python run_benchmark.py --url http://localhost:5173
```

### Modifying Metrics

Edit metrics calculation in `src/main.js`:
- `PerformanceProfiler` class: Metrics collection
- `calculateMetrics()`: Aggregation and statistics
- `addResultRow()`: Result display

---

## 📝 File Guide

**Active & Important**:
- `run_benchmark.py` - Main benchmark script (recommended, always use this)
- `requirements.txt` - Python dependencies
- `src/main.js` - Web application core
- `public/models/models.json` - Model configuration
- `index.html` - Web UI

**Can be removed** (redundant or outdated):
- `run_benchmark.py` - Superseded by run_benchmark.py
- `run_benchmark_cli.py` - Older CLI version
- `run_benchmark.sh`, `run_benchmark.bat` - Shell wrappers (legacy)
- `run_benchmark (1).py` - Duplicate copy
- `requirements (1).txt` - Duplicate copy
- `QUICKSTART.md` - Merged into README.md
- `README_BENCHMARK.md` - Merged into README.md
- `INT64_ISSUE.md` - Documented in Known Issues
- `test-int64.js`, `debug-tensor-api.js` - Development test files
- `automation/METRICS_GUIDE.md`, `automation/COMPLETE_METRICS_GUIDE.md` - Reference documentation

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review console output and error messages
3. Check browser console (F12) for JavaScript errors
4. Review ONNX Runtime documentation: https://onnxruntime.ai/

---

## 📄 License

See LICENSE file for details.

---

**Last Updated**: December 29, 2025  
**ONNX Runtime Web Version**: 1.23.2  
**Node Version**: 16+ recommended  
**Python Version**: 3.8+ required