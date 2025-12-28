# ONNX Runtime Web Benchmark Automation

This script automates running benchmarks for ONNX models in the browser, one at a time, with cache clearing between each run.

## Features

- Runs each model individually with a fresh browser context
- Clears browser cache between each model
- Configurable warmup and measurement runs
- Automatic result collection and CSV/Excel export
- Progress monitoring and error handling
- Headless or headed browser mode

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Playwright Browsers

```bash
playwright install chromium
```

### 3. Start Your Web Server

Make sure your benchmark web application is running:

```bash
npm run dev
# or
vite
# or whatever command starts your server
```

Note the URL (typically `http://localhost:5173` or `http://localhost:3000`)

## Usage

### Basic Usage

Edit the configuration in `run_benchmark.py`:

```python
# Configuration
BENCHMARK_URL = "http://localhost:5173"  # Your dev server URL
MODELS_CONFIG = "./public/models/models.json"  # Path to models.json
BACKEND = "wasm"  # Options: wasm, webgl, webgpu, webnn
WARMUP_RUNS = 1
MEASURE_RUNS = 100
OUTPUT_FORMAT = "csv"  # Options: csv, excel
```

Then run:

```bash
python run_benchmark.py
```

### Command Line Version

For more flexibility, you can create a CLI version. Here's a quick example:

```bash
# Run with custom settings
python run_benchmark.py --url http://localhost:5173 \
                        --models ./public/models/models.json \
                        --backend wasm \
                        --warmup 1 \
                        --measure 100 \
                        --format csv
```

### Headless Mode

To run without opening a visible browser window, edit `run_benchmark.py`:

```python
browser = await p.chromium.launch(
    headless=True,  # Change to True
    args=['--disable-cache', '--disk-cache-size=0']
)
```

## Output

The script generates a timestamped CSV or Excel file in the `benchmark_results/` directory:

```
benchmark_results/
  benchmark_results_20240115_143022.csv
```

### Output Columns

- `timestamp`: When the benchmark was run
- `model`: Model name
- `backend`: Execution provider (wasm, webgl, etc.)
- `warmup_runs`: Number of warmup runs
- `measure_runs`: Number of measurement runs
- `avg_warmup_ms`: Average warmup time in milliseconds
- `avg_inference_ms`: Average inference time in milliseconds
- `status`: SUCCESS or FAILED
- `notes`: Additional notes or error messages
- `session_create_ms`: Time to create the inference session

## Troubleshooting

### Browser doesn't open
- Make sure Playwright is installed: `playwright install chromium`
- Try running with `headless=False` to see what's happening

### Cannot connect to URL
- Verify your dev server is running
- Check the URL in the configuration
- Try accessing the URL in your regular browser first

### Models not found
- Verify the path to `models.json` is correct
- Make sure the models are actually available in your `/public/models` directory

### Benchmark hangs
- Check the browser console (visible when `headless=False`)
- Increase timeout values in the script
- Check the status logs for error messages

### Results show "ERROR" or "FAILED"
- Check the `notes` column in the output for error details
- Verify the model files are accessible
- Check browser console for JavaScript errors

## Advanced Configuration

### Testing with Different Backends

Run the script multiple times with different backends:

```python
for backend in ['wasm', 'webgl', 'webgpu']:
    runner = BenchmarkRunner(BENCHMARK_URL, MODELS_CONFIG)
    await runner.run_all_models(backend=backend, warmup=1, measure=100)
    runner.save_results(format='csv')
```

### Selecting Specific Models

Modify the `run_all_models` method to filter models:

```python
models = await self.load_models_list()
# Only run on specific models
models = [m for m in models if 'mobilenet' in m.lower()]
```

### Custom Result Processing

Access the results DataFrame before saving:

```python
df = pd.DataFrame(runner.results)

# Calculate statistics
df['throughput'] = 1000 / df['avg_inference_ms'].astype(float)

# Filter successful runs
successful = df[df['status'] == 'SUCCESS']

# Save with custom processing
df.to_csv('custom_results.csv', index=False)
```

## Example Output

```
================================================================================
SUMMARY
================================================================================
                    timestamp              model backend  warmup_runs  avg_warmup_ms  measure_runs  avg_inference_ms    status                notes
 2024-01-15T14:30:22.123456  mobilenetv2-1.0    wasm            1         45.231           100            23.456   SUCCESS  session_create_ms=234.5
 2024-01-15T14:32:15.789012      resnet50-v1    wasm            1         78.923           100            45.678   SUCCESS  session_create_ms=456.7
 2024-01-15T14:34:08.456789    squeezenet1.1    wasm            1         12.345           100             8.901   SUCCESS  session_create_ms=123.4
================================================================================
```

## Notes

- Each model runs in a completely fresh browser context
- Browser cache is cleared between runs to ensure fair comparison
- The script waits for each benchmark to fully complete before moving to the next
- All console logs from the browser are captured for debugging
- Results are saved with a timestamp to prevent overwriting previous runs
