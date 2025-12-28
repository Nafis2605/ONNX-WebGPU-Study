# Quick Start Guide - ONNX Benchmark Automation

## 🚀 5-Minute Setup

### Step 1: Install Dependencies

```bash
# Install Python packages
pip install playwright pandas openpyxl

# Install browser
playwright install chromium
```

### Step 2: Start Your Web Server

```bash
# In your project directory, start the dev server
npm run dev
# Note the URL (usually http://localhost:5173)
```

### Step 3: Run the Benchmark

**Option A: Use the Simple Script (Recommended for beginners)**

Edit configuration in `run_benchmark.py` or `run_benchmark.sh`:
```bash
# Linux/Mac
chmod +x run_benchmark.sh
./run_benchmark.sh

# Windows
run_benchmark.bat
```

**Option B: Use CLI (More flexible)**

```bash
python run_benchmark_cli.py \
    --url http://localhost:5173 \
    --models ./public/models/models.json \
    --warmup 1 \
    --measure 100
```

### Step 4: Get Results

Results will be saved in `benchmark_results/` directory as CSV or Excel.

---

## 📋 Common Use Cases

### Run with Different Backend

```bash
# Test WebGPU
python run_benchmark_cli.py --url http://localhost:5173 \
    --models ./public/models/models.json --backend webgpu

# Test WebGL
python run_benchmark_cli.py --url http://localhost:5173 \
    --models ./public/models/models.json --backend webgl
```

### Run Only Specific Models

```bash
# Only MobileNet models
python run_benchmark_cli.py --url http://localhost:5173 \
    --models ./public/models/models.json --filter mobilenet

# Multiple filters
python run_benchmark_cli.py --url http://localhost:5173 \
    --models ./public/models/models.json --filter mobilenet resnet
```

### Run in Headless Mode (No Browser Window)

```bash
python run_benchmark_cli.py --url http://localhost:5173 \
    --models ./public/models/models.json --headless
```

### Save as Excel

```bash
python run_benchmark_cli.py --url http://localhost:5173 \
    --models ./public/models/models.json --format excel
```

---

## 🔧 Troubleshooting

**Problem: "Failed to load model"**
- ✅ Make sure your web server is running
- ✅ Check that model files exist in `/public/models/`
- ✅ Verify the models.json path is correct

**Problem: "Benchmark timed out"**
- ✅ Increase timeout in `run_benchmark_cli.py` (line ~120)
- ✅ Try with fewer measure runs: `--measure 10`
- ✅ Check browser console for errors (run without `--headless`)

**Problem: Browser doesn't open**
- ✅ Run: `playwright install chromium`
- ✅ Make sure you have Chrome/Chromium installed

**Problem: Results show "ERROR"**
- ✅ Check the notes column in the CSV for details
- ✅ Run without `--headless` to see what's happening
- ✅ Make sure the web page loads correctly in a normal browser first

---

## 📊 Understanding the Results

The output CSV/Excel contains these columns:

- **timestamp**: When the test ran
- **model**: Model name
- **backend**: Which execution provider was used
- **avg_warmup_ms**: Average warmup time (milliseconds)
- **avg_inference_ms**: Average inference time (milliseconds)
- **status**: SUCCESS or FAILED
- **notes**: Additional info or error messages

### Example Output

```
model              backend  avg_warmup_ms  avg_inference_ms  status
mobilenetv2-1.0    wasm     45.231         23.456           SUCCESS
resnet50-v1        wasm     78.923         45.678           SUCCESS
squeezenet1.1      wasm     12.345         8.901            SUCCESS
```

---

## 🎯 Tips for Best Results

1. **Close other applications** to avoid interference
2. **Run multiple times** and average the results
3. **Use warmup runs** to let the JIT compiler optimize (we use 1 by default)
4. **Test different backends** to find the fastest for your models
5. **Check CPU/GPU usage** during benchmarks
6. **Use headless mode** for automated CI/CD pipelines

---

## 📝 Advanced: Batch Testing

Create a script to test multiple configurations:

```bash
#!/bin/bash

# Test all backends
for backend in wasm webgl webgpu; do
    echo "Testing backend: $backend"
    python run_benchmark_cli.py \
        --url http://localhost:5173 \
        --models ./public/models/models.json \
        --backend $backend \
        --warmup 1 \
        --measure 100 \
        --filename "results_${backend}.csv"
done

echo "All tests complete!"
```

---

## 🆘 Getting Help

1. Check README_BENCHMARK.md for detailed documentation
2. Run with `--help` to see all options:
   ```bash
   python run_benchmark_cli.py --help
   ```
3. Run without `--headless` to see what's happening in the browser
4. Check the browser console (F12) for JavaScript errors

---

## ✅ Checklist Before Running

- [ ] Web server is running (check URL in browser)
- [ ] models.json exists and is accessible
- [ ] Model files (.onnx) are in the correct directory
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Playwright installed (`playwright install chromium`)
- [ ] Enough disk space for results files

---

Happy benchmarking! 🎉
