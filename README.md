# ONNX Runtime Web Benchmark

Benchmarking system for ONNX Runtime Web with support for WASM, WebGL, and WebGPU backends.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
npm install

# Start development server
npm run dev
# Server runs at http://localhost:5173/
```

## Running Benchmarks

### Browser Testing
1. Open http://localhost:5173/ in your browser
2. Select a model from the dropdown
3. Choose a backend (WASM, WebGL, or WebGPU)
4. Click "Run Benchmark"
5. Results display in the table

### Automated Benchmarking

```bash
# WASM backend
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend wasm \
    --filter gpt2 \
    --warmup 1 \
    --measure 2

# WebGL backend
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend webgl \
    --filter gpt2 \
    --warmup 1 \
    --measure 2

# WebGPU backend
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend webgpu \
    --filter gpt2 \
    --warmup 1 \
    --measure 2
```

## Benchmark Parameters

- `--url`: Web server URL (default: http://localhost:5173)
- `--models`: Path to models.json file
- `--backend`: Backend to test (wasm, webgl, webgpu)
- `--filter`: Model name filter (e.g., gpt2)
- `--warmup`: Number of warmup runs (default: 1)
- `--measure`: Number of measurement runs (default: 10)

## Output

Results are saved to `benchmark_results/benchmark_metrics_TIMESTAMP.csv`

CSV columns include:
- Model name, backend, warmup/measure runs
- Inference timing metrics (avg warmup, avg inference)
- Kernel metrics (execution time, launch latency, fusion rate, etc.)
- GPU metrics (vendor, memory usage, core utilization)
- Top 5 kernels with execution times

## Files

- `src/main.js` - Frontend benchmark application
- `run_benchmark.py` - Python automation script for CLI benchmarking
- `index.html` - Web interface
- `public/models/` - ONNX model files
- `benchmark_results/` - Benchmark output CSV files


**Solution:** Load JSEP-enabled WASM bundle (`ort-wasm-simd-threaded.jsep.wasm`)

**Code Fix (src/main.js):**
```javascript
// Load JSEP-enabled WASM for GPU backend support
const mjsUrl = `${wasmBasePath}ort-wasm-simd-threaded.jsep.mjs`;
const wasmUrl = `${wasmBasePath}ort-wasm-simd-threaded.jsep.wasm`;
```

**Key Change:**
- ❌ Old: `ort-wasm-simd-threaded.wasm` → No GPU support
- ✅ New: `ort-wasm-simd-threaded.jsep.wasm` → Full GPU support

This enables WebGPU and WebGL execution providers to initialize correctly.

---

## 🔍 Backend Detection System

### What It Solves
**Problem:** How do I know if WebGPU/WebGL are actually being used or if it's falling back to WASM?

**Solution:** Detects actual backend via performance analysis (warmup spike, variance, latency patterns)

### Features
✅ **Test Button** - 5-second backend verification without full benchmark  
✅ **Performance Analysis** - Detects backend from execution patterns  
✅ **Fallback Prevention** - No silent WASM fallback (now explicit errors)  
✅ **Comprehensive Logging** - All backend decisions visible in status console  
✅ **CSV Export** - Detection results included in benchmark output  
✅ **GPU Metrics** - Real-time GPU utilization, memory, power monitoring

### How Detection Works

Each backend has unique execution characteristics:

| Metric | WebGPU | WebGL | WASM |
|--------|--------|-------|------|
| **Warmup Spike** | 10-100x slower | 2-5x slower | No spike |
| **Variance** | Very low (<15%) | Low (15-30%) | High (20-50%) |
| **Speed** | 1-50ms | 10-100ms | 10-200ms |
| **Signature** | Big spike + consistent | Medium spike + stable | No spike + variable |

**Detection Process:**
1. Checks browser GPU support (navigator.gpu, WebGL context)
2. Loads small test model
3. Runs warmup inference (detects GPU shader compilation)
4. Runs 5 steady-state inferences (measures variance)
5. Analyzes patterns: warmup/avg ratio, variance, latency
6. Compares with requested backend
7. Reports result with confidence level

### Using the Test Button

**Step 1:** Select backend from dropdown (WebGPU, WebGL, WASM)  
**Step 2:** Click **🧪 Test Backend Detection** button  
**Step 3:** Wait ~5 seconds for analysis  
**Step 4:** Check status console for results

**Expected Output - Working Backend:**
```
Requested Backend: webgpu
Performance-Detected Backend: webgpu
Detection Confidence: high

Performance Metrics:
  warmupMs: 145.67
  avgTimeMs: 12.34
  varianceMs: 0.90

✓ Backend match is good! Likely using webgpu as expected.
```

**Expected Output - Fallback Detected:**
```
Requested Backend: webgpu
Performance-Detected Backend: wasm
Detection Confidence: high

Performance Metrics:
  warmupMs: 12.34
  avgTimeMs: 8.89
  varianceMs: 1.22

⚠ WARNING: Requested webgpu but got wasm!
  This indicates a fallback to wasm.
```

### Expected Performance Hierarchy
```
WebGPU (fastest)    ~8ms
   ↓ 2-3x difference
WebGL (medium)      ~15ms
   ↓ 5-10x difference  
WASM (slowest)      ~85ms
```

### Browser Support
| Browser | WebGPU | WebGL | WASM | Test Button |
|---------|--------|-------|------|-------------|
| Chrome 113+ | ✓ | ✓ | ✓ | ✓ |
| Edge 113+ | ✓ | ✓ | ✓ | ✓ |
| Firefox | ✗ | ✓ | ✓ | ✓ |
| Safari | ✗ | ✓ | ✓ | ✓ |

### Troubleshooting Backend Detection

| Issue | Solution |
|-------|----------|
| "WebGPU not available" | Use Chrome/Edge 113+, or update browser |
| "Detection uncertain" | Try with larger model (ResNet50, BERT) - GPU benefit shows on complex models only |
| "Same metrics all backends" | Check detection confidence - if HIGH + warning = fallback detected |
| Button doesn't work | Refresh page, check browser console (F12) |
| "No GPU acceleration" | Check browser settings, update GPU drivers, try different browser |

---

## Web Benchmark Interface

Open **http://localhost:5173/** for interactive testing:

**Controls:**
- Backend selector: WebGPU, WebGL, WASM, WebNN
- Model multi-select: Hold Ctrl/Cmd to select multiple
- Warmup runs & measure runs
- Run/Stop buttons
- **Test Backend Detection** button (NEW!)

**Output:**
- Real-time status console
- Results table with 10 detailed metrics:
  - Kernel Execution Time, Kernel Launch Latency
  - Operator Fusion Rate, Per-Operator Latency
  - Kernel Compilation Time, Memory Bandwidth
  - Synchronization Overhead, Peak Memory Usage
  - Time to First Output, End-to-End Latency

---

## Python Benchmark (run_benchmark.py)

Run benchmarks from command line:

```bash
python run_benchmark.py \
    --url http://localhost:5173 \
    --backend webgpu \
    --warmup 1 \
    --measure 100
```

**Options:**
- `--url` - Web server URL (default: http://localhost:5173)
- `--backend` - webgpu, webgl, wasm, webnn (default: wasm)
- `--warmup` - Warmup runs (default: 1)
- `--measure` - Measurement runs (default: 100)
- `--models` - Models JSON path (default: ./public/models/models.json)
- `--filter` - Filter models by keyword (e.g., `--filter mobilenet`)
- `--headless` - Run browser in headless mode
- `--executable-path` - Path to Chrome/Chromium executable
- `--output-dir` - Output directory for results (default: benchmark_results)

### Platform-Specific Examples

#### macOS (Chrome Dev)
```bash
# Basic benchmark with Chrome Dev
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend webgpu \
    --warmup 1 \
    --measure 10 \
    --executable-path "/Applications/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev"

# Filter specific model (e.g., gpt2)
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend webgpu \
    --warmup 1 \
    --measure 10 \
    --filter gpt2 \
    --executable-path "/Applications/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev"

# With headless mode
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend webgpu \
    --filter mobilenet \
    --headless \
    --executable-path "/Applications/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev"
```

#### Windows (Chrome Dev)
```bash
# Basic benchmark with Chrome Dev
python run_benchmark.py ^
    --url http://localhost:5173 ^
    --models .\public\models\models.json ^
    --backend webgpu ^
    --warmup 1 ^
    --measure 10 ^
    --executable-path "C:\Program Files\Google\Chrome Dev\Application\chrome.exe"

# Filter specific model (e.g., gpt2)
python run_benchmark.py ^
    --url http://localhost:5173 ^
    --models .\public\models\models.json ^
    --backend webgpu ^
    --warmup 1 ^
    --measure 10 ^
    --filter gpt2 ^
    --executable-path "C:\Program Files\Google\Chrome Dev\Application\chrome.exe"

# With headless mode
python run_benchmark.py ^
    --url http://localhost:5173 ^
    --models .\public\models\models.json ^
    --backend webgpu ^
    --filter mobilenet ^
    --headless ^
    --executable-path "C:\Program Files\Google\Chrome Dev\Application\chrome.exe"
```

**Note:** On Windows, use `^` for line continuation instead of `\`

### Common Example Commands (All Platforms)

```bash
# Benchmark only MobileNet with WebGPU
python run_benchmark.py --url http://localhost:5173 --models ./public/models/models.json --backend webgpu --filter mobilenet

# Run with headless browser
python run_benchmark.py --url http://localhost:5173 --models ./public/models/models.json --backend wasm --headless

# Custom warmup and measurement runs
python run_benchmark.py --url http://localhost:5173 --models ./public/models/models.json --backend webgl --warmup 2 --measure 50

# Benchmark multiple models
python run_benchmark.py --url http://localhost:5173 --models ./public/models/models.json --backend webgpu --warmup 1 --measure 5
```

---

## Output Files

Results saved in `benchmark_results/` directory:

### benchmark_results.csv
Aggregated metrics per model run:
- `timestamp` - Run start time
- `model` - Model name
- `backend` - Execution provider (webgpu/webgl/wasm)
- `avg_warmup_ms` - Average warmup latency
- `avg_inference_ms` - Average inference latency
- All 10 detailed performance metrics
- Backend detection info: actual_backend, detection_confidence

### gpu_utilization_intervals.csv
Per-second GPU metrics during inference:
- `timestamp_sec` - Seconds into inference (1.05, 2.08, 3.12, etc.)
- `gpu_utilization_percent` - GPU utilization at that second
- `gpu_memory_utilization_percent` - GPU memory utilization
- `gpu_power_draw_mw` - GPU power draw in milliwatts (mW)
- `memory_mb` - System memory usage

---

## Analyzing Results

### Python Analysis
```python
import pandas as pd

# Load summary results
df = pd.read_csv('benchmark_results.csv')

# Performance by backend
print(df.groupby('backend')[['avg_inference_ms', 'performance_detected_backend']].mean())

# Load interval samples  
df_intervals = pd.read_csv('gpu_utilization_intervals.csv')

# Peak GPU utilization per model
print(df_intervals.groupby('model')['gpu_utilization_percent'].max())

# Utilization over time for specific model
model_data = df_intervals[df_intervals['model'] == 'mobilenetv2-7']
print(model_data[['timestamp_sec', 'gpu_utilization_percent']])
```

---

## System Requirements

### macOS (Apple Silicon/Metal)
- Python 3.8+
- `system_profiler` (built-in)
- Dependencies: psutil, pandas, playwright

### Linux/Windows (NVIDIA GPU)
- Python 3.8+
- `nvidia-smi` (NVIDIA drivers)
- Dependencies: psutil, pandas, playwright

### All Platforms
- Node.js 16+
- Chrome/Chromium browser
- Web server running at target URL

---

## Project Structure

```
.
├── src/
│   └── main.js                   # Web UI with backend detection
├── public/
│   ├── models/                   # ONNX models
│   │   ├── models.json
│   │   └── *.onnx
│   └── ort-wasm/                 # ONNX Runtime WASM
├── benchmark_results/            # Output CSV files
├── run_benchmark.py              # Main benchmark script
├── gpu_metrics.py                # GPU metrics collector
├── index.html                    # Web UI entry point
├── package.json                  # Node dependencies
├── vite.config.js                # Vite config
└── README.md                     # This file
```

---

## Performance Metrics Collected

**Per-run metrics:**
- Warmup latency (ms)
- Inference latency (ms)
- Kernel execution time
- Kernel launch latency
- Operator fusion rate (%)
- Per-operator latency
- Kernel compilation time
- Memory bandwidth (GB/s)
- Synchronization overhead
- Peak memory usage (MB)

**Backend detection metrics (NEW!):**
- Actual backend detected
- Detection confidence level
- Warmup time (ms)
- Average inference time (ms)
- Variance between runs (ms)

**1-second interval samples:**
- GPU utilization (%)
- GPU memory utilization (%)
- GPU power draw (mW)
- System memory (MB)

---

## Important Notes

- GPU metrics are optional; benchmarking continues if GPU collection fails
- Real power draw measurement requires direct system access (powermetrics/nvidia-smi)
- Browser WebGPU workloads typically show lower power than native applications
- For production power analysis, use native ONNX Runtime Python (non-browser)
- Backend detection is most reliable on complex models (ResNet50, BERT, ViT)
- Results are appended to CSV files (multiple runs accumulate data)

---

## Code Changes for Backend Detection

**src/main.js:**
- `checkBackendAvailability()` - Detect WebGPU/WebGL browser support
- `detectBackendViaPerformance()` - Analyze execution patterns for backend detection
- `executionProvidersForBackend()` - Modified to prevent silent fallbacks
- `testBackendBtn.onclick` - Test button handler
- Enhanced logging throughout session creation

**index.html:**
- Added blue "🧪 Test Backend Detection" button
- Added CSS styling for new button

**Total addition:** ~420 lines of backend detection logic + enhanced logging

---

## Troubleshooting

### Benchmark Hangs
- Ensure web server is running: `npm run dev`
- Check URL: `curl http://localhost:5173`

### Browser Not Found
- Install playwright browsers: `playwright install chromium`

### Models Not Found
- Verify: `./public/models/models.json` exists
- Check: `./public/models/` directory

### GPU Metrics Missing
- GPU collection is optional; benchmarking continues
- Check browser console (F12) for errors
- Some metrics may show "N/A" on unsupported hardware

---

## Support & Questions

For backend detection issues:
1. Open browser DevTools (F12)
2. Go to Console tab
3. Run backend test again
4. Check console output for error messages
5. Verify browser version supports your target backend

For general questions:
- Check status console output during benchmark
- Review CSV results for detection info
- Validate browser supports target backend

---

**Happy benchmarking!** 🚀
