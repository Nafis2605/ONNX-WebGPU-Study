# ONNX Runtime Web Benchmark Study

Automated, measurement-only benchmarking of ONNX Runtime Web on the **WASM**, **WebGPU** and
**WebGL** backends, on Windows with an NVIDIA GPU and **Google Chrome Dev**.

Design rules the code follows:

- **No fabricated values.** Every number is a direct measurement (timers, GPU timestamp and timer
  queries, graphics-API call arguments, ORT's own trace, NVML, OS process counters). A value that
  can't be measured is left empty, never estimated.
- **Strict backends.** Each session has exactly one execution provider. WebGPU sessions refuse to
  place any node on the CPU (`session.disable_cpu_ep_fallback`). A run only counts as `ok` when
  API counters prove the backend did the work: WebGPU submits/dispatches, WebGL draws, and no
  GPU calls at all for WASM.
- **Timing and instrumentation are separate.** Latency comes from uninstrumented runs. Counters,
  kernel timing and memory accounting run in their own browser launches (`profile` mode).
- **Cold, isolated launches.** A new Chrome Dev process with an empty profile and no shader disk
  cache for every (trial, model, backend).
- **Raw data first.** The browser returns per-run records; statistics are computed offline by
  `analysis/*.py`.

## Setup (Windows)

```bash
winget install Python.Python.3.12
winget install Google.Chrome.Dev
npm install
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Chrome Dev is auto-detected (`CHROME_DEV_PATH` overrides it). Models (`*.onnx`, gitignored) go in
`public/models/` and are registered in `public/models/models.json`.

Benchmarks run against the production build, served with cross-origin isolation (5 µs timer
resolution) and the model files served in place:

```bash
npm run bench:serve
```

(`npm run dev` still works for manual use. Don't benchmark against it: editing a source file
live-reloads the page and kills a running experiment.)

## Running

Whole study, all stages plus analysis:

```bash
.venv\Scripts\python run_study.py
```

Quick smoke run of every stage:

```bash
.venv\Scripts\python run_study.py --quick
```

Re-run only some configs × models inside an existing study, merged into each stage's newest result (same (trial, model, config) experiments replaced; output in `<stage>/<timestamp>_merged/`):

```bash
.venv\Scripts\python run_study.py --output-dir benchmark_results\study_full --supplement --config webgl --models resnet50-v2 deeplabv3p
```

Diagnostic stages (not in the default study): where WebGL's inference time goes, and the raw readback cost per API path. `webgl_readback` needs the unminified diagnostic build, so V8 profiles show ORT's function names:

```bash
npx vite build --mode diag
```

```bash
npx vite preview --mode diag
```

```bash
.venv\Scripts\python run_study.py --output-dir benchmark_results\study_full --stages readback_micro webgl_readback
```

- **`--gl-timing` (profile mode, WebGL):** adds phases that time every WebGL call, splitting `readPixels` into GPU drain and transfer.
- **`--cpu-profile`:** saves a V8 sampling profile of extra runs (`logs/<tag>.cpuprofile`).
- **`--trace`:** saves a Chrome trace with GPU-process categories (`logs/<tag>.trace.json.gz`).
- **`run_readback_micro.py`:** times GPU→JS readback without ORT, for WebGL `readPixels` (RGBA, RED, PBO), WebGPU `mapAsync`, and ORT's RGBA-unpack expression vs a strided loop.
- **`analysis/webgl_readback.py <study>`:** summarizes both stages.

A single stage:

```bash
.venv\Scripts\python run_benchmark.py --mode latency --config wasm-8t webgpu webgl --filter resnet50-v2 --warmup 10 --measure 200 --trials 3
```

Configs (`--config`, defined in `run_benchmark.py` `CONFIGS`); the CSVs' `backend` column holds the config label:

| Config | Meaning |
|---|---|
| `wasm-1t`, `wasm-4t`, `wasm-8t` | WASM EP with 1 / 4 / 8 threads |
| `webgl` | WebGL EP |
| `webgpu` | JSEP WebGPU EP, CPU I/O |
| `webgpu-native` | Native C++ WebGPU EP (`onnxruntime-web/webgpu`) |
| `webgpu-gpuio` | JSEP with GPU-resident inputs/outputs |
| `webgpu-capture` | GPU-resident I/O plus graph capture |

`--outputs primary` fetches only the primary output; `--exclude model:config` skips pairs; `--cool-to-c` waits for the GPU to cool before each launch.

Figures for the paper (PNG + PDF, plus `findings.json`):

```bash
.venv\Scripts\python analysis\make_figures.py benchmark_results\study_full --out study_report\figures
```

| `--mode` | Study group | What it does |
|---|---|---|
| `latency` | 1, 2 | Closed-loop runs at the base shape: fetch, session create, first/second inference, steady-state distribution |
| `batch` | 2 | Per-sample latency across the model's batch sizes (`shape_sweep`) |
| `shapes` | 1 | Seeded replay of input shapes (models with `shape_sweep`): first-seen vs revisited vs steady |
| `profile` | 5, 6, 7 | Instrumented pass: per-run API counters, per-kernel GPU/CPU time, data movement, memory |
| `correctness` | 10 | Outputs vs native ONNX Runtime (CPU EP) on the identical input bytes |
| `responsiveness` | 4 | Frame timing and click latency with and without inference running |
| `llm` | 3 | Greedy token generation timing (models with an `llm` spec) |
| `load` | 8 | Open-loop Poisson arrivals at swept rates, bounded queue, deadlines |

Useful flags: `--trials N` (independent browser launches), `--allow-cpu-nodes` (let WebGPU place
unsupported nodes on the CPU and record how many), `--wasm-threads N`, `--pacing yield|back_to_back`,
`--no-telemetry`, `--headless` (headed is the default and recommended for GPU runs).

Analysis (each writes CSVs next to the data it reads):

| Script | Output |
|---|---|
| `analysis/summarize.py` | Latency p50/p90/p95/p99, IQR, CoV, closed-loop throughput, bootstrap CIs |
| `analysis/init_reuse.py` | Fetch / session-create / first-run overhead; first-seen vs revisited shapes |
| `analysis/gpu_telemetry.py` | Energy per inference, power, utilization, clocks, throttle reasons, device memory |
| `analysis/profile_summary.py` | CPU-side vs GPU time, dispatch groups, bytes moved, memory, top op types |
| `analysis/responsiveness.py` | Frame-interval distribution, missed-deadline fraction, click input delay |
| `analysis/llm.py` | Time to first token, inter-token gaps, tokens/s |
| `analysis/load.py` | Goodput, deadline-miss / reject rates, max sustainable tested rate |
| `analysis/predict.py` | Leave-one-model-out latency prediction, backend selection regret, WebGL feasibility rule |
| `calculate_flops.py` | MACs / FLOPs per model (Conv, MatMul, Gemm) → `flops_table.csv` |

## Output (`benchmark_results/<timestamp>/`)

- `env.json`: host, CPU, RAM, NVIDIA GPU/driver/VBIOS, Chrome version and flags, ORT version, WebGPU adapter, WebGL renderer, git commit, all arguments
- `experiments.jsonl`: one raw record per experiment (the source of truth)
- `sessions.csv`: status, failure stage/category/message, load and session times, verification counters, node placement
- `runs.csv`: one row per inference (profile mode adds per-run counters)
- `kernels.csv` (profile): per-kernel records (WebGPU timestamp queries, WebGL timer queries, ORT trace nodes)
- `gpu_samples.csv`, `gpu_util_samples.csv`, `gpu_pcie_samples.csv`, `gpu_marks.csv`: NVML telemetry
- `chrome_memory_samples.csv`: private/RSS memory of this launch's browser, renderer and GPU processes
- `correctness.csv`, `frames.csv`, `clicks.csv`, `interactions.csv`, `loaf.csv`, `load_requests.csv`, `load_frames.csv`: mode-specific data
- `logs/`: browser console per experiment

## What is measured, and what isn't

| Group | Measured | Caveats / not feasible |
|---|---|---|
| 1 Init & reuse | Model fetch (localhost), session creation, first and second inference; ORT session-init breakdown (`model_loading_array`, `session_initialization`) in profile mode; first-seen vs revisited shape latency | Shape study only for models with symbolic dims (ResNet-50 and DeepLab batch). BERT and GPT-2 are exported with a fixed sequence length of 128. |
| 2 Inference | Per-run E2E latency (outputs read back to the CPU), p50/p95/p99, closed-loop throughput | p99 is flagged unless ≥100 runs |
| 3 LLM | TTFT, inter-token gaps, tokens/s | The GPT-2 export has no KV-cache inputs, so every token is a full 128-token forward pass. Prefill and decode can't be separated; results are labelled no-KV-cache. |
| 4 Responsiveness | rAF frame intervals, missed-deadline fraction (display refresh measured in the baseline), per-click input delay and time to next frame, Event Timing entries, Long Animation Frames | Event Timing only reports events ≥16 ms and rounds them to 8 ms, so the page's own click measurement is the complete stream |
| 5 Execution | WebGPU: per-kernel GPU time (timestamp queries), CPU-side node time (ORT trace), dispatches, submits, dispatches per submit, compute passes. WebGL: per-draw GPU time (`EXT_disjoint_timer_query_webgl2`) and draws. WASM: per-op CPU time (ORT trace) | Chrome exposes only `timestamp-query`, not timestamps inside passes. ORT therefore ends a compute pass after every dispatch in the GPU-timing phase, so that phase's wall times aren't latency numbers. ORT's dispatch batch size (16) is hardcoded, so no submission-size sweep. |
| 6 Data movement | WebGPU `writeBuffer` bytes, `mapAsync(READ)` bytes and resolve latency, buffer copies; WebGL texture upload and `readPixels` bytes; CPU↔GPU boundaries (ORT `Memcpy` nodes) | Readback latency includes waiting for queued GPU work |
| 7 Memory | Logical GPU memory (live WebGPU buffer bytes, WebGL texture bytes: after session create, peak, final); page memory (`measureUserAgentSpecificMemory`); Chrome process private bytes; NVML device memory; failure rate | Windows (WDDM) doesn't report per-process GPU memory, so NVML memory is device-wide and compared with an idle baseline |
| 8 Application | Goodput, deadline-miss, reject and unfinished rates, max sustainable tested rate, frame misses under load | Max rate is the highest *tested* rate; nothing is interpolated |
| 9 Prediction | Leave-one-model-out prediction error, selection violations and regret, calibration time; static WebGL feasibility rule validated against measured outcomes | Few models per backend; the WebGL rule was derived on these 12 models |
| 10 Correctness | Max/mean abs error, RMSE, max relative error, cosine vs native ORT; top-1/top-5 agreement for declared logits outputs | Agreement with the reference, not task accuracy (no labelled datasets) |
| NVIDIA | Energy via the NVML energy counter (~96 ms resolution), power, SM/memory utilization (driver samples, ~200 ms), clocks, temperature, P-state, throttle reasons, PCIe TX/RX | Single-run energy isn't resolvable. NVML utilization is device-wide (includes the desktop compositor). |

## Findings about the runtime (ORT Web 1.24.0-dev, Chrome Dev 156, RTX 3060)

- **The default `onnxruntime-web` entry doesn't register WebGL.** A `["webgl", "wasm"]` session silently ran on WASM, so earlier "WebGL" results were WASM. `src/ort-setup.js` now imports `onnxruntime-web/all`.
- **WebGL rejects models with symbolic input dims** (`expected shape '[,3,224,224]'`). The graph parser leaves `dim_param` undefined, but the check only accepts 0. This affects ResNet-50 and DeepLab. For WebGL only, they load a copy with the input dims pinned: `public/patch/pin_input_dims.py <model>` writes `<file>_b1.onnx` (same graph and weights, checked bit-identical against the original on native ORT), and the `webgl_path` field in `models.json` selects it. `sessions.csv` records the file used in `model_path`.
- **WebGL can't load int64 values outside the int32 range** (e.g. `INT64_MAX` Slice bounds in ViT and MobileBERT), and lacks `ConstantOfShape`, `LayerNormalization`, `HardSwish`, among others.
- **Starting ORT's WebGL profiler breaks inference** (`reading 'inputTypes'`), so WebGL draws are timed with the page's own timer queries.
- **WebGPU runs all 12 models in strict mode** (0 CPU-placed nodes, 0 memcpy boundaries).

## NVIDIA Nsight (not automated yet)

Nsight Systems and Nsight Graphics require a download from NVIDIA's developer site (account
login), and only Nsight Compute is in winget. Nsight Compute profiles CUDA kernels, not
Chrome's D3D12/D3D11 work. After installing Nsight Systems, a system-wide GPU trace of one
experiment would look like the following. This command is untested here and must be verified
after installation:

```bash
nsys profile --trace=dx12,wddm --gpu-metrics-devices=0 -o nsight\resnet50_webgpu .venv\Scripts\python run_benchmark.py --mode latency --filter resnet50-v2 --config webgpu --trials 1
```

## Repository layout

```
src/            browser side: main.js (UI + window.__bench), bench.js (experiment core),
                ort-setup.js, feeds.js, instrument.js, ortlog.js, profile.js,
                responsiveness.js, llm.js, load.js, timing.js
benchkit/       Python side: Chrome launching, env capture, NVML telemetry, correctness, results
analysis/       offline statistics per study group
run_benchmark.py  one mode, many (trial, model, backend) experiments
run_study.py      every mode + analysis
public/models/  models.json registry (+ local .onnx files)
public/test/    cpu_node_probe.onnx: negative test for strict WebGPU mode
public/patch/   int64 → int32 input patch (*_int32.onnx); input-dim pinning for WebGL (*_b1.onnx)
```
