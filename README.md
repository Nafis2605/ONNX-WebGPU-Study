# ONNX Runtime Web Benchmark with GPU Metrics

A comprehensive benchmarking system for ONNX Runtime Web models with GPU performance metrics (utilization, memory, power draw) collected at 1-second intervals across all platforms.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Setup (macOS Only - For Real Power Draw)
```bash
echo '%admin ALL=(ALL) NOPASSWD: /usr/bin/powermetrics' | sudo tee -a /etc/sudoers.d/powermetrics
```

### 3. Start Web Server
```bash
npm run dev
# Server will be available at http://localhost:5173
```

### 4. Run Benchmark
```bash
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend webgpu \
    --warmup 1 \
    --measure 100
```

---

## Running the Benchmark

### Basic Command
```bash
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend webgpu
```

### Available Options
- `--url` - Web server URL (default: http://localhost:5173)
- `--models` - Path to models.json (default: ./public/models/models.json)
- `--backend` - Execution provider: `wasm`, `webgl`, `webgpu`, `webnn` (default: webgpu)
- `--warmup` - Warmup runs per model (default: 1)
- `--measure` - Measurement runs per model (default: 100)
- `--filter` - Filter models by keyword (e.g., `--filter mobilenet` runs only matching models)
- `--headless` - Run browser in headless mode
- `--output-dir` - Output directory for results (default: benchmark_results)

### Example Commands

**Benchmark only MobileNet:**
```bash
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend webgpu \
    --filter mobilenet
```

**Run with headless browser:**
```bash
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend wasm \
    --headless
```

**Custom WASM backend:**
```bash
python run_benchmark.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --backend wasm \
    --warmup 2 \
    --measure 50
```

---

## Output Files

Results are saved in `benchmark_results/` directory:

### benchmark_results.csv
**Aggregated metrics per model run**

Key columns:
- `timestamp` - Benchmark start time (ISO format)
- `model` - Model name
- `backend` - Execution provider used
- `avg_inference_ms` - Average inference latency
- `gpu_vendor` - GPU vendor (Apple, NVIDIA, AMD, Intel, or N/A)
- `gpu_name` - GPU model name
- `gpu_utilization_percent` - Average GPU utilization during inference
- `gpu_memory_utilization_percent` - Average GPU memory utilization
- `gpu_power_draw_mw` - Average GPU power draw in milliwatts (mW)

**Note:** `gpu_power_draw_mw` records real values from powermetrics/nvidia-smi when available (e.g., 10.50 mW instead of 0.01 W).

### gpu_utilization_intervals.csv
**Per-second GPU metrics samples**

Key columns:
- `timestamp` - Run start time
- `model` - Model name
- `backend` - Execution provider
- `timestamp_sec` - Seconds into inference (1.05, 2.08, 3.12, etc.)
- `gpu_utilization_percent` - GPU utilization at that second
- `gpu_memory_utilization_percent` - GPU memory utilization at that second
- `gpu_power_draw_mw` - GPU power draw at that second in milliwatts (mW)
- `memory_mb` - System memory usage at that second

---

## Analyzing Results

### View Results (Bash)
```bash
# See first few rows
head -5 benchmark_results.csv

# Count runs
wc -l benchmark_results.csv

# View interval samples
head -10 gpu_utilization_intervals.csv
```

### Analyze in Python
```python
import pandas as pd

# Load summary results
df = pd.read_csv('benchmark_results.csv')

# GPU metrics by model
print(df[['model', 'avg_inference_ms', 'gpu_utilization_percent', 'gpu_power_draw_watts']])

# Average GPU utilization by backend
print(df.groupby('backend')['gpu_utilization_percent'].mean())

# Load interval samples
df_intervals = pd.read_csv('gpu_utilization_intervals.csv')

# Peak power draw per model
print(df_intervals.groupby('model')['gpu_power_draw_watts'].max())

# Utilization over time for a specific model
model_data = df_intervals[df_intervals['model'] == 'mobilnetv2-7']
print(model_data[['timestamp_sec', 'gpu_utilization_percent']])
```

---

## System Requirements

### macOS (Apple Silicon/Metal)
- Python 3.8+
- `system_profiler` (built-in)
- `powermetrics` (optional, for real power draw - requires passwordless sudo)
- Dependencies: psutil, pandas, playwright

### Linux/Windows (NVIDIA GPU)
- Python 3.8+
- `nvidia-smi` (NVIDIA drivers)
- Dependencies: psutil, pandas, playwright

### All Platforms
- Node.js 16+
- Chrome/Chromium browser
- Web server running on target URL

---

## GPU Metrics Behavior

### GPU Utilization & Memory
- **macOS:** System memory utilization (proxy, since GPU memory is unified)
- **Linux/Windows (NVIDIA):** Direct GPU metrics from nvidia-smi

### GPU Power Draw

- **macOS:** Via `powermetrics` utility (direct) or GPU utilization estimation (fallback)
  - Direct: Requires passwordless sudo setup (see Quick Start, step 2)
  - Fallback: Estimates power from 85%+ utilization (typical browser workload: 12-18W)
  - Real values when powermetrics available; estimates otherwise

- **Linux/Windows (NVIDIA):** Via `nvidia-smi` (direct) or GPU utilization estimation (fallback)
  - Direct: Reports actual power when hardware supports it
  - Fallback: Estimates power from utilization + memory (typical 250-350W at 85%)
  - Real values when supported; estimates otherwise

**Browser workloads rarely expose true GPU power.** Power draw values are useful for relative comparisons (model A vs model B) but should not be treated as absolute measurements for power budget planning.

---

## Troubleshooting

### Power Draw Shows "N/A"

**Power metrics may show "N/A" or 0.0 watts:**

**macOS:**
- Verify passwordless sudo setup for direct measurement:
  ```bash
  sudo -n powermetrics -s gpu_power -n 1
  ```
  - If prompted for password: Run setup again (see Quick Start, step 2)
  - If command not found: Using fallback estimation instead (still reasonable)
  - System will automatically fall back to estimation if unavailable

**Linux/Windows:**
- Verify nvidia-smi is available:
  ```bash
  nvidia-smi
  ```
  - Should show GPU info
  - If not found: Using fallback estimation instead (still reasonable)
  - Older GPUs may not support power monitoring, system will estimate

**Either way, benchmarking continues with estimated power if direct measurement fails.**

### Benchmark Hangs
- Ensure web server is running: `npm run dev`
- Check URL: `curl http://localhost:5173`
- Try headless mode: add `--headless` flag

### Browser Not Found
- Reinstall playwright browsers:
  ```bash
  playwright install chromium
  ```

### Models Not Found
- Verify models file exists: `./public/models/models.json`
- Check models directory: `./public/models/`

---

## Project Structure

```
.
├── src/                          # Web UI source
│   └── main.js
├── public/
│   ├── models/                   # ONNX models
│   │   ├── models.json
│   │   └── *.onnx
│   └── ort-wasm/                 # ONNX Runtime WASM
├── benchmark_results/            # Output directory
│   └── benchmark_metrics_*.csv
├── run_benchmark.py              # Main benchmark script
├── gpu_metrics.py                # GPU metrics collector
├── requirements.txt              # Python dependencies
├── package.json                  # Node dependencies
├── vite.config.js                # Vite config
└── index.html                    # Web UI entry point
```

---

## Performance Metrics Collected

Per-run metrics:
- Average warmup latency (ms)
- Average inference latency (ms)
- Kernel execution time (ms)
- Kernel launch latency (ms)
- Operator fusion rate (%)
- Per-operator latency (ms)
- Kernel compilation time (ms)
- Effective memory bandwidth (GB/s)
- Synchronization overhead (ms)
- Peak memory usage (MB)
- Time to first output (ms)
- End-to-end latency (ms)
- Top 5 kernels by execution time
- GPU vendor & name
- GPU utilization (%)
- GPU memory utilization (%)
- GPU power draw (W)

1-second interval samples:
- GPU utilization (%)
- GPU memory utilization (%)
- GPU power draw (W)
- System memory (MB)

---

## Important: GPU Power Draw on Browser Workloads

**Real power draw collection in milliwatts (mW) for precision:**

The system now uses real powermetrics/nvidia-smi readings with intelligent fallback estimation, recorded in **milliwatts (mW)** with 2 decimal precision (e.g., 10.50 mW instead of 0.01 W):

- **macOS:** 
  - Uses direct `powermetrics` (if passwordless sudo configured)
  - Falls back to utilization-based estimation if powermetrics unavailable
  - Example values: 10.50 mW (idle) to 20000 mW (peak browser workload)
  - **Key finding:** Browser WebGPU shows 10-100 mW even at 85%+ GPU utilization
  
- **Linux/Windows (NVIDIA):**
  - Uses direct `nvidia-smi power.draw` if hardware supports it
  - Falls back to utilization-based estimation if unsupported
  - Example values: 50000 mW (idle) to 350000 mW (peak workload)
  - **Key finding:** Like macOS, browsers don't trigger peak GPU power states

**Why browser power draw is low:**
- Browser GPU execution is fundamentally different from native CUDA/Metal
- WebGPU/WebGL have additional overhead and don't fully utilize power-efficient GPU modes
- System profilers may not capture browser GPU power accurately
- This is a platform limitation, not a code issue

**For accurate power metrics, use:**
- External USB power meters (most accurate for system power)
- Xcode Instruments Metal GPU Profiler (macOS)
- NVIDIA GPU profiling tools with native CUDA
- Native ONNX Runtime Python benchmarks

---

- All GPU metrics are optional and gracefully degrade to "N/A" if collection fails
- Benchmarking always completes successfully, even if GPU metrics are unavailable
- Results are appended to CSV files (multiple runs accumulate data)
- Use `--filter` to run specific models and reduce benchmark time
- Power draw data may be unreliable on older hardware or drivers