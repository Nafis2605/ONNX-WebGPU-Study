import * as ort from "onnxruntime-web";

/* ================= WASM CONFIG ================= */

// REQUIRED
const wasmBasePath = `${import.meta.env.BASE_URL}ort-wasm/`;

// IMPORTANT: disable threaded WASM (avoids .jsep.mjs import)
ort.env.wasm.numThreads = 1;

async function configureWasmPaths() {
  const mjsUrl = `${wasmBasePath}ort-wasm-simd-threaded.mjs`;
  const wasmUrl = `${wasmBasePath}ort-wasm-simd-threaded.wasm`;

  const res = await fetch(mjsUrl, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to load ORT WASM module (${res.status})`);
  }

  const mjsSource = await res.text();
  const mjsBlob = new Blob([mjsSource], { type: "text/javascript" });
  const mjsBlobUrl = URL.createObjectURL(mjsBlob);

  ort.env.wasm.wasmPaths = {
    mjs: mjsBlobUrl,
    wasm: wasmUrl
  };
}

// SIMD is safe
ort.env.wasm.simd = true;

// Enable profiling
ort.env.logLevel = 'verbose';

// WebGL (optional; may still be unavailable)
ort.env.webgl = {
  contextId: "webgl2",
  pack: true
};

// WebGPU (optional)
ort.env.webgpu = {
  deviceType: "gpu"
};


/* ============================================================
   UI ELEMENTS
   ============================================================ */

const backendSelect = document.getElementById("backendSelect");
const modelsSelect = document.getElementById("modelsSelect");
const warmupRunsEl = document.getElementById("warmupRuns");
const measureRunsEl = document.getElementById("measureRuns");
const runBtn = document.getElementById("runBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const resultsBody = document.getElementById("resultsBody");

let stopRequested = false;

/* ============================================================
   MEMORY PROFILING UTILITIES
   ============================================================ */

class MemoryProfiler {
  constructor() {
    this.baseline = 0;
    this.peakUsage = 0;
    this.currentUsage = 0;
    this.samples = [];
    this.monitoringInterval = null;
  }

  async getMemoryUsage() {
    if (performance.memory) {
      // Chrome-specific API
      return {
        used: performance.memory.usedJSHeapSize,
        total: performance.memory.totalJSHeapSize,
        limit: performance.memory.jsHeapSizeLimit
      };
    } else if (typeof window.gc === 'function') {
      // Try to get memory info after GC
      window.gc();
      return { used: 0, total: 0, limit: 0 };
    } else {
      // Estimate based on object count (very rough)
      return { used: 0, total: 0, limit: 0 };
    }
  }

  async setBaseline() {
    const mem = await this.getMemoryUsage();
    this.baseline = mem.used;
    this.peakUsage = mem.used;
    this.currentUsage = mem.used;
    this.samples = [];
  }

  startMonitoring(intervalMs = 50) {
    this.stopMonitoring();
    this.monitoringInterval = setInterval(async () => {
      const mem = await this.getMemoryUsage();
      this.currentUsage = mem.used;
      this.samples.push(mem.used);
      if (mem.used > this.peakUsage) {
        this.peakUsage = mem.used;
      }
    }, intervalMs);
  }

  stopMonitoring() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
  }

  getPeakMemoryMB() {
    return ((this.peakUsage - this.baseline) / (1024 * 1024)).toFixed(2);
  }

  getAverageMemoryMB() {
    if (this.samples.length === 0) return 0;
    const avg = this.samples.reduce((a, b) => a + b, 0) / this.samples.length;
    return ((avg - this.baseline) / (1024 * 1024)).toFixed(2);
  }

  reset() {
    this.stopMonitoring();
    this.baseline = 0;
    this.peakUsage = 0;
    this.currentUsage = 0;
    this.samples = [];
  }
}

/* ============================================================
   COMPREHENSIVE PERFORMANCE PROFILER
   ============================================================ */

class PerformanceProfiler {
  constructor() {
    this.metrics = {
      // Original 5 metrics
      kernelExecutionTimes: [],
      kernelLaunchLatencies: [],
      operatorLatencies: {},
      compilationTime: 0,
      sessionCreateTime: 0,
      firstRunTime: 0,
      subsequentRunTimes: [],
      
      // New 5 metrics
      memoryBandwidthSamples: [],
      synchronizationOverheads: [],
      peakMemoryUsage: 0,
      timeToFirstOutput: 0,
      endToEndLatencies: []
    };
    this.performanceObserver = null;
    this.memoryProfiler = new MemoryProfiler();
  }

  reset() {
    this.metrics = {
      kernelExecutionTimes: [],
      kernelLaunchLatencies: [],
      operatorLatencies: {},
      compilationTime: 0,
      sessionCreateTime: 0,
      firstRunTime: 0,
      subsequentRunTimes: [],
      memoryBandwidthSamples: [],
      synchronizationOverheads: [],
      peakMemoryUsage: 0,
      timeToFirstOutput: 0,
      endToEndLatencies: []
    };
    this.memoryProfiler.reset();
  }

  startProfiling() {
    if (typeof PerformanceObserver !== 'undefined') {
      try {
        this.performanceObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.name.includes('ort') || entry.name.includes('kernel')) {
              this.metrics.kernelExecutionTimes.push(entry.duration);
            }
          }
        });
        this.performanceObserver.observe({ entryTypes: ['measure'] });
      } catch (e) {
        console.log('Performance Observer not available:', e);
      }
    }
  }

  stopProfiling() {
    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
      this.performanceObserver = null;
    }
  }

  analyzeOperatorFusion(session) {
    try {
      const inputNames = session.inputNames || [];
      const outputNames = session.outputNames || [];
      const estimatedOperators = Math.max(inputNames.length, outputNames.length, 1);
      
      return {
        estimatedOperators,
        inputNodes: inputNames.length,
        outputNodes: outputNames.length
      };
    } catch (e) {
      return {
        estimatedOperators: 0,
        inputNodes: 0,
        outputNodes: 0
      };
    }
  }

  // Calculate effective memory bandwidth
  calculateMemoryBandwidth(dataTransferred, timeMs) {
    // Bandwidth in GB/s = (bytes / time_seconds) / (1024^3)
    if (timeMs === 0) return 0;
    const timeSeconds = timeMs / 1000;
    const bandwidthGBps = (dataTransferred / (1024 * 1024 * 1024)) / timeSeconds;
    return bandwidthGBps;
  }

  // Estimate data transfer size based on tensors
  estimateDataSize(feeds) {
    let totalBytes = 0;
    for (const key in feeds) {
      const tensor = feeds[key];
      if (tensor && tensor.data) {
        totalBytes += tensor.data.byteLength || tensor.data.length * 4; // Assume float32
      }
    }
    return totalBytes;
  }

  // Profile individual run with comprehensive timing
  async profileRun(session, feeds, runIndex) {
    const runMetrics = {
      totalTime: 0,
      kernelTime: 0,
      overhead: 0,
      launchLatency: 0,
      syncOverhead: 0,
      timeToFirstOutput: 0,
      endToEndLatency: 0,
      memoryBandwidth: 0
    };

    // Start memory monitoring
    await this.memoryProfiler.setBaseline();
    this.memoryProfiler.startMonitoring(25); // Sample every 25ms

    // End-to-end timing (includes all overhead)
    const e2eStart = performance.now();

    // Measure launch latency (time to initiate execution)
    const launchStart = performance.now();
    const runPromise = session.run(feeds);
    const launchEnd = performance.now();
    runMetrics.launchLatency = launchEnd - launchStart;

    // Measure time to first output (actual execution start)
    const execStart = performance.now();
    
    // Wait for completion
    const outputs = await runPromise;
    
    const execEnd = performance.now();
    const e2eEnd = performance.now();

    // Stop memory monitoring
    this.memoryProfiler.stopMonitoring();

    // Calculate timings
    runMetrics.timeToFirstOutput = execEnd - execStart; // Time from exec start to output
    runMetrics.endToEndLatency = e2eEnd - e2eStart; // Total wall-clock time
    runMetrics.totalTime = execEnd - execStart;
    runMetrics.kernelTime = Math.max(0, runMetrics.totalTime - runMetrics.launchLatency);

    // Estimate synchronization overhead (time waiting for GPU/async operations)
    // This is the difference between end-to-end and actual execution
    runMetrics.syncOverhead = Math.max(0, runMetrics.endToEndLatency - runMetrics.totalTime);

    // Calculate memory bandwidth
    const dataSize = this.estimateDataSize(feeds);
    runMetrics.memoryBandwidth = this.calculateMemoryBandwidth(dataSize, runMetrics.totalTime);

    // Store metrics
    if (runIndex === 0) {
      this.metrics.timeToFirstOutput = runMetrics.timeToFirstOutput;
    }
    
    this.metrics.endToEndLatencies.push(runMetrics.endToEndLatency);
    this.metrics.synchronizationOverheads.push(runMetrics.syncOverhead);
    this.metrics.memoryBandwidthSamples.push(runMetrics.memoryBandwidth);
    
    // Update peak memory
    const peakMB = parseFloat(this.memoryProfiler.getPeakMemoryMB());
    if (peakMB > this.metrics.peakMemoryUsage) {
      this.metrics.peakMemoryUsage = peakMB;
    }

    return runMetrics;
  }

  calculateMetrics() {
    // Original 5 metrics
    const kernelExecTime = this.metrics.kernelExecutionTimes.length > 0
      ? avg(this.metrics.kernelExecutionTimes)
      : (this.metrics.subsequentRunTimes.length > 0 ? avg(this.metrics.subsequentRunTimes) : 0);

    const kernelLaunchLatency = this.metrics.kernelLaunchLatencies.length > 0
      ? avg(this.metrics.kernelLaunchLatencies)
      : 0;

    const fusionRate = this.metrics.firstRunTime > 0 && this.metrics.subsequentRunTimes.length > 0
      ? Math.min(100, Math.max(0, (1 - avg(this.metrics.subsequentRunTimes) / this.metrics.firstRunTime) * 100))
      : 0;

    const estimatedOps = Object.keys(this.metrics.operatorLatencies).length || 1;
    const perOperatorLatency = kernelExecTime / estimatedOps;

    const compilationTime = this.metrics.firstRunTime > 0 && this.metrics.subsequentRunTimes.length > 0
      ? Math.max(0, this.metrics.firstRunTime - avg(this.metrics.subsequentRunTimes))
      : this.metrics.compilationTime;

    // New 5 metrics
    const effectiveMemoryBandwidth = this.metrics.memoryBandwidthSamples.length > 0
      ? avg(this.metrics.memoryBandwidthSamples)
      : 0;

    const synchronizationOverhead = this.metrics.synchronizationOverheads.length > 0
      ? avg(this.metrics.synchronizationOverheads)
      : 0;

    const peakMemoryUsage = this.metrics.peakMemoryUsage;

    const timeToFirstOutput = this.metrics.timeToFirstOutput;

    const endToEndLatency = this.metrics.endToEndLatencies.length > 0
      ? avg(this.metrics.endToEndLatencies)
      : 0;

    return {
      // Original 5
      kernelExecutionTime: kernelExecTime.toFixed(3),
      kernelLaunchLatency: kernelLaunchLatency.toFixed(3),
      operatorFusionRate: fusionRate.toFixed(1),
      perOperatorLatency: perOperatorLatency.toFixed(3),
      kernelCompilationTime: compilationTime.toFixed(3),
      
      // New 5
      effectiveMemoryBandwidth: effectiveMemoryBandwidth.toFixed(3),
      synchronizationOverhead: synchronizationOverhead.toFixed(3),
      peakMemoryUsage: peakMemoryUsage.toFixed(2),
      timeToFirstOutput: timeToFirstOutput.toFixed(3),
      endToEndLatency: endToEndLatency.toFixed(3)
    };
  }
}

/* ============================================================
   UTILS
   ============================================================ */

function logStatus(msg) {
  statusEl.textContent += msg + "\n";
  console.log(msg);
}

function clearStatus() {
  statusEl.textContent = "";
}

function clearResults() {
  resultsBody.innerHTML = "";
}

function addResultRow(result) {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td>${result.model}</td>
    <td>${result.backend}</td>
    <td>${result.warmupRuns}</td>
    <td>${result.avgWarmup}</td>
    <td>${result.measureRuns}</td>
    <td>${result.avgInference}</td>
    <td>${result.kernelExecutionTime}</td>
    <td>${result.kernelLaunchLatency}</td>
    <td>${result.operatorFusionRate}%</td>
    <td>${result.perOperatorLatency}</td>
    <td>${result.kernelCompilationTime}</td>
    <td>${result.effectiveMemoryBandwidth}</td>
    <td>${result.synchronizationOverhead}</td>
    <td>${result.peakMemoryUsage}</td>
    <td>${result.timeToFirstOutput}</td>
    <td>${result.endToEndLatency}</td>
    <td style="font-size: 11px;">${result.notes}</td>
  `;
  resultsBody.appendChild(tr);
}

function avg(arr) {
  if (!arr.length) return NaN;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function numel(dims) {
  return dims.reduce((a, b) => a * b, 1);
}

/* ============================================================
   EXECUTION PROVIDER SELECTION (SAFE FALLBACK)
   ============================================================ */

function executionProvidersForBackend(backend) {
  switch (backend) {
    case "wasm":
      return ["wasm"];
    case "webgl":
      return ["webgl", "wasm"];
    case "webgpu":
      return ["webgpu", "wasm"];
    case "webnn":
      return ["webnn", "wasm"];
    default:
      return ["wasm"];
  }
}

/* ============================================================
   MODEL / INPUT HANDLING
   ============================================================ */

async function loadModelsConfig() {
  const res = await fetch("/models/models.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load models.json (${res.status})`);
  return await res.json();
}

function populateModelsSelect(models) {
  modelsSelect.innerHTML = "";
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = m.name;
    modelsSelect.appendChild(opt);
  }
}

function getSelectedModelNames() {
  return Array.from(modelsSelect.selectedOptions).map(o => o.value);
}

function makeRandomTypedArray(type, size) {
  if (type === "float32") {
    const a = new Float32Array(size);
    for (let i = 0; i < size; i++) a[i] = Math.random() * 2 - 1;
    return a;
  }
  if (type === "int32") {
    const a = new Int32Array(size);
    for (let i = 0; i < size; i++) a[i] = (Math.random() * 1000) | 0;
    return a;
  }
  if (type === "uint8") {
    const a = new Uint8Array(size);
    for (let i = 0; i < size; i++) a[i] = Math.random() * 255;
    return a;
  }
  if (type === "int64") {
    // For BERT models, use reasonable token IDs (0-30000 range)
    // CRITICAL: Must use BigInt64Array for int64 tensors
    try {
      const a = new BigInt64Array(size);
      for (let i = 0; i < size; i++) {
        // Ensure we create proper BigInt values (min 0, max 30000)
        const randomValue = Math.floor(Math.random() * 30000);
        a[i] = BigInt(randomValue);
      }
      // Verify the array was created correctly
      if (!(a instanceof BigInt64Array)) {
        console.error(`CRITICAL ERROR: Failed to create BigInt64Array! Got ${a.constructor.name}`);
        throw new Error(`Failed to create BigInt64Array for int64 type`);
      }
      console.log(`Created BigInt64Array: size=${size}, sample=${a[0]}, type=${typeof a[0]}, instanceof=${a instanceof BigInt64Array}`);
      
      // ADDITIONAL VERIFICATION: Check array buffer
      console.log(`BigInt64Array details:`);
      console.log(`  - buffer.byteLength: ${a.buffer.byteLength}`);
      console.log(`  - length: ${a.length}`);
      console.log(`  - BYTES_PER_ELEMENT: ${BigInt64Array.BYTES_PER_ELEMENT}`);
      console.log(`  - constructor name: ${a.constructor.name}`);
      
      return a;
    } catch (e) {
      console.error(`ERROR creating BigInt64Array: ${e.message}`);
      throw e;
    }
  }
  if (type === "uint32") {
    const a = new Uint32Array(size);
    for (let i = 0; i < size; i++) a[i] = Math.floor(Math.random() * 1000);
    return a;
  }
  if (type === "float16") {
    // Float16 not directly supported, use Float32Array
    const a = new Float32Array(size);
    for (let i = 0; i < size; i++) a[i] = Math.random() * 2 - 1;
    return a;
  }
  if (type === "bool") {
    const a = new Uint8Array(size);
    for (let i = 0; i < size; i++) a[i] = Math.random() > 0.5 ? 1 : 0;
    return a;
  }
  
  // Default to float32 for unknown types
  console.warn(`Unsupported input type: ${type}, using float32 as fallback`);
  const a = new Float32Array(size);
  for (let i = 0; i < size; i++) a[i] = Math.random() * 2 - 1;
  return a;
}

function makeFeedsFromModelConfig(modelCfg) {
  const feeds = {};
  for (const inp of modelCfg.inputs) {
    logStatus(`Creating feed for input: "${inp.name}", type: "${inp.type}", dims: [${inp.dims.join(', ')}]`);
    try {
      const data = makeRandomTypedArray(inp.type, numel(inp.dims));
      logStatus(`  Data: ${data.constructor.name}, length: ${data.length}`);
      
      // CRITICAL: For int64, ensure BigInt64Array is used
      if (inp.type === 'int64') {
        if (!(data instanceof BigInt64Array)) {
          throw new Error(`int64 type requires BigInt64Array, but got ${data.constructor.name}`);
        }
        logStatus(`  ✓ BigInt64Array validated`);
      }
      
      // Create the tensor
      let tensor;
      try {
        tensor = new ort.Tensor(inp.type, data, inp.dims);
        logStatus(`  Created with explicit type: ${tensor.type}`);
      } catch (e1) {
        logStatus(`  Explicit type failed: ${e1.message}`);
        if (inp.type === 'int64' && data instanceof BigInt64Array) {
          logStatus(`  Trying inferred type...`);
          tensor = new ort.Tensor(data, inp.dims);
          logStatus(`  Created with inferred type: ${tensor.type}`);
        } else {
          throw e1;
        }
      }
      
      // Verify
      if (tensor.type !== inp.type) {
        logStatus(`  ✗ TYPE MISMATCH: Expected '${inp.type}', got '${tensor.type}'`);
        throw new Error(`Type mismatch for ${inp.name}: expected ${inp.type}, got ${tensor.type}`);
      }
      
      feeds[inp.name] = tensor;
    } catch (e) {
      logStatus(`  ✗ FAILED: ${e.message}`);
      throw e;
    }
  }
  return feeds;
}

function normalizeDims(dims) {
  if (!Array.isArray(dims) || dims.length === 0) return [1];
  return dims.map(d => (typeof d === "number" && d > 0 ? d : 1));
}

function makeFeedsFromSession(session, fallbackCfg) {
  try {
    const inputNames = session.inputNames || [];
    
    if (!inputNames.length) {
      logStatus("No input names found, using config fallback");
      return makeFeedsFromModelConfig(fallbackCfg);
    }

    const feeds = {};
    
    for (const name of inputNames) {
      let type = "float32";
      let dims = [1, 3, 224, 224];
      let configUsed = false;
      
      // PRIORITY 1: Check config first (most reliable)
      if (fallbackCfg && fallbackCfg.inputs) {
        const configInput = fallbackCfg.inputs.find(inp => inp.name === name);
        if (configInput) {
          type = configInput.type || type;
          dims = configInput.dims || dims;
          configUsed = true;
          logStatus(`✓ Using config for "${name}": ${type} [${dims.join(", ")}]`);
        }
      }
      
      // PRIORITY 2: Try session metadata (if config not found)
      if (!configUsed) {
        try {
          if (session.inputMetadata && session.inputMetadata[name]) {
            const meta = session.inputMetadata[name];
            
            // Extract type - handle different metadata formats
            if (meta.type) {
              type = meta.type;
            } else if (meta.elementType) {
              type = meta.elementType;
            }
            
            // WORKAROUND: If model requires int64 but ONNX Runtime Web doesn't support it,
            // fall back to int32 which should work for token IDs
            if (type === 'int64') {
              logStatus(`⚠️  Model specifies int64 for "${name}", but int64 support in ONNX Runtime Web is broken`);
              logStatus(`    Falling back to int32 for compatibility`);
              type = 'int32';
            }
            
            // Extract dimensions
            if (meta.dimensions) {
              dims = normalizeDims(meta.dimensions);
            } else if (meta.dims) {
              dims = normalizeDims(meta.dims);
            } else if (meta.shape) {
              dims = normalizeDims(meta.shape);
            }
            
            logStatus(`Using metadata for "${name}": ${type} [${dims.join(", ")}]`);
          }
        } catch (e) {
          logStatus(`Could not read metadata for ${name}: ${e.message}`);
        }
      }
      
      logStatus(`Creating input tensor "${name}": ${type} [${dims.join(", ")}]`);
      const data = makeRandomTypedArray(type, numel(dims));
      
      // Verify the data type is correct
      const arrayType = data.constructor.name;
      logStatus(`  → Created ${arrayType} with ${data.length} elements`);
      
      // Create tensor with explicit type
      try {
        // CRITICAL: For int64, validate BigInt64Array before tensor creation
        if (type === 'int64') {
          if (!(data instanceof BigInt64Array)) {
            throw new Error(`int64 type requires BigInt64Array, but got ${arrayType}`);
          }
          logStatus(`  → Validated int64: BigInt64Array with ${data.length} elements`);
        }
        
        feeds[name] = new ort.Tensor(type, data, dims);
        
        // Double-check the tensor type
        logStatus(`  → Tensor created with type: ${feeds[name].type}`);
        
        // CRITICAL: Verify int64 tensors were created correctly
        if (type === 'int64' && feeds[name].type !== 'int64') {
          throw new Error(`Type verification failed: requested int64 but got ${feeds[name].type}`);
        }
      } catch (tensorError) {
        logStatus(`ERROR creating tensor for ${name}: ${tensorError.message}`);
        throw tensorError;
      }
    }
    
    return feeds;
  } catch (e) {
    logStatus(`Error creating feeds from session: ${e.message}, using config fallback`);
    return makeFeedsFromModelConfig(fallbackCfg);
  }
}

const USE_STREAMING_DOWNLOAD = false;

async function fetchModelArrayBuffer(url, onProgress) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to load model (${res.status})`);
  }

  const contentLength = res.headers.get("content-length");
  const total = contentLength ? Number(contentLength) : 0;

  if (!USE_STREAMING_DOWNLOAD || !res.body || typeof res.body.getReader !== "function") {
    const buffer = await res.arrayBuffer();
    if (onProgress) onProgress(total || buffer.byteLength, total || buffer.byteLength);
    return buffer;
  }

  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (onProgress) onProgress(received, total);
  }

  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  return buffer.buffer;
}

/* ============================================================
   BENCHMARK CORE WITH COMPREHENSIVE PROFILING
   ============================================================ */

async function timeRuns(session, feeds, runs, label, profiler, isWarmup = false) {
  const times = [];
  
  for (let i = 0; i < runs; i++) {
    if (stopRequested) break;
    
    try {
      const runMetrics = await profiler.profileRun(session, feeds, i);
      times.push(runMetrics.totalTime);
      
      // Store detailed metrics
      if (!isWarmup) {
        if (i === 0) {
          profiler.metrics.firstRunTime = runMetrics.totalTime;
        } else {
          profiler.metrics.subsequentRunTimes.push(runMetrics.totalTime);
        }
        
        profiler.metrics.kernelExecutionTimes.push(runMetrics.kernelTime);
        profiler.metrics.kernelLaunchLatencies.push(runMetrics.launchLatency);
      }
      
      if (i === 0 || (i + 1) % Math.max(1, Math.floor(runs / 5)) === 0) {
        logStatus(`${label}: ${i + 1}/${runs} (${runMetrics.totalTime.toFixed(2)} ms, e2e: ${runMetrics.endToEndLatency.toFixed(2)} ms)`);
      }
    } catch (e) {
      logStatus(`ERROR in ${label} run ${i + 1}: ${e.message}`);
      throw e;
    }
  }
  return times;
}

async function runOneModel({ modelCfg, backend, warmupRuns, measureRuns }) {
  const executionProviders = executionProvidersForBackend(backend);
  const notes = [];
  const profiler = new PerformanceProfiler();

  let session;
  try {
    logStatus(`Loading model: ${modelCfg.path}`);
    let lastLog = 0;
    const modelBuffer = await fetchModelArrayBuffer(
      modelCfg.path,
      (received, total) => {
        const now = performance.now();
        if (now - lastLog < 250) return;
        lastLog = now;
        if (total) {
          const pct = ((received / total) * 100).toFixed(1);
          logStatus(`Model download: ${pct}% (${received}/${total} bytes)`);
        } else {
          logStatus(`Model download: ${received} bytes`);
        }
      }
    );
    logStatus(`Model download complete: ${modelBuffer.byteLength} bytes`);

    const t0 = performance.now();
    logStatus("Creating inference session...");
    
    const createPromise = ort.InferenceSession.create(modelBuffer, {
      executionProviders,
      enableProfiling: true,
      graphOptimizationLevel: 'all'
    });
    
    const timeoutMs = 30000;
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`Session creation timed out after ${timeoutMs} ms`)), timeoutMs)
    );
    session = await Promise.race([createPromise, timeoutPromise]);
    
    const sessionCreateTime = performance.now() - t0;
    profiler.metrics.sessionCreateTime = sessionCreateTime;
    notes.push(`session_create_ms=${sessionCreateTime.toFixed(1)}`);
    
    logStatus("Session created.");
    
    // Analyze operator fusion potential
    const fusionInfo = profiler.analyzeOperatorFusion(session);
    logStatus(`Estimated operators: ${fusionInfo.estimatedOperators}`);
    
    if (session.inputNames?.length) {
      logStatus(`Session inputs: ${session.inputNames.join(", ")}`);
    }
    if (session.outputNames?.length) {
      logStatus(`Session outputs: ${session.outputNames.join(", ")}`);
    }
  } catch (e) {
    const errMsg = `session_create_failed: ${e.message}`;
    logStatus(`ERROR: ${errMsg}`);
    return { 
      ok: false, 
      notes: errMsg,
      kernelExecutionTime: 'N/A',
      kernelLaunchLatency: 'N/A',
      operatorFusionRate: 'N/A',
      perOperatorLatency: 'N/A',
      kernelCompilationTime: 'N/A',
      effectiveMemoryBandwidth: 'N/A',
      synchronizationOverhead: 'N/A',
      peakMemoryUsage: 'N/A',
      timeToFirstOutput: 'N/A',
      endToEndLatency: 'N/A'
    };
  }

  let feeds;
  try {
    logStatus("Preparing input tensors...");
    logStatus(`Model config has ${modelCfg.inputs ? modelCfg.inputs.length : 0} inputs defined`);
    if (modelCfg.inputs) {
      for (const inp of modelCfg.inputs) {
        logStatus(`  Config input: "${inp.name}" type=${inp.type} dims=[${inp.dims.join(", ")}]`);
      }
    }
    
    logStatus(`Session has ${session.inputNames ? session.inputNames.length : 0} inputs`);
    if (session.inputNames) {
      for (const name of session.inputNames) {
        logStatus(`  Session input: "${name}"`);
      }
    }
    
    feeds = makeFeedsFromSession(session, modelCfg);
    
    // Log what we created
    logStatus("Input tensors created:");
    for (const name in feeds) {
      const tensor = feeds[name];
      logStatus(`  ${name}: type=${tensor.type}, data type=${tensor.data.constructor.name}, dims=[${tensor.dims.join(", ")}], elements=${tensor.data.length}`);
    }
    
    logStatus("Inputs ready.");
  } catch (e) {
    const errMsg = `feed_error: ${e.message}`;
    logStatus(`ERROR: ${errMsg}`);
    logStatus(`Stack: ${e.stack}`);
    return { 
      ok: false, 
      notes: errMsg,
      kernelExecutionTime: 'N/A',
      kernelLaunchLatency: 'N/A',
      operatorFusionRate: 'N/A',
      perOperatorLatency: 'N/A',
      kernelCompilationTime: 'N/A',
      effectiveMemoryBandwidth: 'N/A',
      synchronizationOverhead: 'N/A',
      peakMemoryUsage: 'N/A',
      timeToFirstOutput: 'N/A',
      endToEndLatency: 'N/A'
    };
  }

  // Start profiling
  profiler.startProfiling();

  let warmTimes = [];
  try {
    if (warmupRuns > 0) {
      logStatus("Warmup starting...");
      warmTimes = await timeRuns(session, feeds, warmupRuns, "warmup", profiler, true);
      logStatus("Warmup complete.");
    }
  } catch (e) {
    profiler.stopProfiling();
    const errMsg = `warmup_failed: ${e.message}`;
    logStatus(`ERROR: ${errMsg}`);
    return { 
      ok: false, 
      notes: errMsg,
      kernelExecutionTime: 'N/A',
      kernelLaunchLatency: 'N/A',
      operatorFusionRate: 'N/A',
      perOperatorLatency: 'N/A',
      kernelCompilationTime: 'N/A',
      effectiveMemoryBandwidth: 'N/A',
      synchronizationOverhead: 'N/A',
      peakMemoryUsage: 'N/A',
      timeToFirstOutput: 'N/A',
      endToEndLatency: 'N/A'
    };
  }

  let infTimes = [];
  try {
    logStatus("Measure starting...");
    infTimes = await timeRuns(session, feeds, measureRuns, "inference", profiler, false);
    logStatus("Measure complete.");
  } catch (e) {
    profiler.stopProfiling();
    const errMsg = `inference_failed: ${e.message}`;
    logStatus(`ERROR: ${errMsg}`);
    return { 
      ok: false, 
      notes: errMsg,
      kernelExecutionTime: 'N/A',
      kernelLaunchLatency: 'N/A',
      operatorFusionRate: 'N/A',
      perOperatorLatency: 'N/A',
      kernelCompilationTime: 'N/A',
      effectiveMemoryBandwidth: 'N/A',
      synchronizationOverhead: 'N/A',
      peakMemoryUsage: 'N/A',
      timeToFirstOutput: 'N/A',
      endToEndLatency: 'N/A'
    };
  }

  // Stop profiling and calculate metrics
  profiler.stopProfiling();
  const detailedMetrics = profiler.calculateMetrics();

  logStatus(`\nDetailed Metrics (First 5):`);
  logStatus(`  Kernel Execution Time: ${detailedMetrics.kernelExecutionTime} ms`);
  logStatus(`  Kernel Launch Latency: ${detailedMetrics.kernelLaunchLatency} ms`);
  logStatus(`  Operator Fusion Rate: ${detailedMetrics.operatorFusionRate}%`);
  logStatus(`  Per-Operator Latency: ${detailedMetrics.perOperatorLatency} ms`);
  logStatus(`  Kernel Compilation Time: ${detailedMetrics.kernelCompilationTime} ms`);
  
  logStatus(`\nDetailed Metrics (Next 5):`);
  logStatus(`  Effective Memory Bandwidth: ${detailedMetrics.effectiveMemoryBandwidth} GB/s`);
  logStatus(`  Synchronization Overhead: ${detailedMetrics.synchronizationOverhead} ms`);
  logStatus(`  Peak Memory Usage: ${detailedMetrics.peakMemoryUsage} MB`);
  logStatus(`  Time to First Output: ${detailedMetrics.timeToFirstOutput} ms`);
  logStatus(`  End-to-End Latency: ${detailedMetrics.endToEndLatency} ms`);

  return {
    ok: true,
    avgWarmupMs: warmTimes.length ? avg(warmTimes) : 0,
    avgInferenceMs: infTimes.length ? avg(infTimes) : NaN,
    actualWarmupRuns: warmTimes.length,
    actualMeasureRuns: infTimes.length,
    notes: notes.join(" | "),
    ...detailedMetrics
  };
}

/* ============================================================
   MAIN
   ============================================================ */

async function main() {
  await configureWasmPaths();

  clearStatus();
  clearResults();

  logStatus(`UserAgent: ${navigator.userAgent}`);

  if (ort.env.versions) {
    logStatus(`ONNX Runtime version: ${ort.env.versions.common || 'unknown'}`);
  }

  // TEST int64 tensor creation early
  logStatus(`\n=== TESTING int64 TENSOR SUPPORT ===`);
  logStatus(`ONNX Runtime version: ${ort.env.versions?.common || 'unknown'}`);
  
  try {
    const testBigIntArray = new BigInt64Array([BigInt(1), BigInt(2), BigInt(3)]);
    logStatus(`✓ Created BigInt64Array: ${testBigIntArray.constructor.name}`);
    
    const testTensor = new ort.Tensor('int64', testBigIntArray, [3]);
    logStatus(`Created tensor with type='int64': ${testTensor.type}`);
    logStatus(`  Tensor.data type: ${testTensor.data.constructor.name}`);
    
    if (testTensor.type !== 'int64') {
      logStatus(`\n⚠️  CRITICAL ISSUE DETECTED ⚠️`);
      logStatus(`int64 tensor creation FAILED - tensor was created as '${testTensor.type}' instead of 'int64'`);
      logStatus(`This is a known issue with ONNX Runtime Web ${ort.env.versions?.common || 'this version'}`);
      logStatus(`BERT models (like bertsquad-12) CANNOT run because they require int64 inputs.`);
      logStatus(`\nNeed to upgrade ONNX Runtime Web or find a workaround.`);
    } else {
      logStatus(`✓ int64 tensor creation successful`);
    }
  } catch (e) {
    logStatus(`✗ Exception during int64 test: ${e.message}`);
  }
  logStatus(`=== END TEST ===\n`);

  if (typeof ort.env.wasm !== 'undefined') {
    logStatus(`WASM threads: ${ort.env.wasm.numThreads}`);
    logStatus(`WASM SIMD: ${ort.env.wasm.simd}`);
  }

  // Check for memory profiling support
  if (performance.memory) {
    logStatus(`Memory profiling: Available (Chrome)`);
  } else {
    logStatus(`Memory profiling: Limited (estimates only)`);
  }

  const cfg = await loadModelsConfig();
  populateModelsSelect(cfg.models ?? []);

  runBtn.onclick = async () => {
    stopRequested = false;
    runBtn.disabled = true;
    stopBtn.disabled = false;

    clearStatus();
    clearResults();

    const backend = backendSelect.value;
    const warmupRuns = Number(warmupRunsEl.value);
    const measureRuns = Number(measureRunsEl.value);
    const selected = getSelectedModelNames();

    if (!selected.length) {
      logStatus("Select at least one model.");
      runBtn.disabled = false;
      stopBtn.disabled = true;
      return;
    }

    logStatus(`Backend: ${backend}`);
    logStatus(`Warmup: ${warmupRuns}, Measure: ${measureRuns}`);

    for (const name of selected) {
      if (stopRequested) break;
      const modelCfg = cfg.models.find(m => m.name === name);
      logStatus(`\n${"=".repeat(60)}`);
      logStatus(`Running: ${name}`);
      logStatus(`${"=".repeat(60)}`);

      const r = await runOneModel({
        modelCfg,
        backend,
        warmupRuns,
        measureRuns
      });

      if (!r.ok) {
        logStatus(`FAILED: ${r.notes}`);
        addResultRow({
          model: name,
          backend,
          warmupRuns: 0,
          avgWarmup: 'N/A',
          measureRuns: 0,
          avgInference: 'N/A',
          kernelExecutionTime: r.kernelExecutionTime,
          kernelLaunchLatency: r.kernelLaunchLatency,
          operatorFusionRate: r.operatorFusionRate,
          perOperatorLatency: r.perOperatorLatency,
          kernelCompilationTime: r.kernelCompilationTime,
          effectiveMemoryBandwidth: r.effectiveMemoryBandwidth,
          synchronizationOverhead: r.synchronizationOverhead,
          peakMemoryUsage: r.peakMemoryUsage,
          timeToFirstOutput: r.timeToFirstOutput,
          endToEndLatency: r.endToEndLatency,
          notes: r.notes
        });
      } else {
        const msg = `SUCCESS: warmup=${r.avgWarmupMs.toFixed(3)} ms, inference=${r.avgInferenceMs.toFixed(3)} ms`;
        logStatus(msg);
        addResultRow({
          model: name,
          backend,
          warmupRuns: r.actualWarmupRuns,
          avgWarmup: r.avgWarmupMs.toFixed(3),
          measureRuns: r.actualMeasureRuns,
          avgInference: r.avgInferenceMs.toFixed(3),
          kernelExecutionTime: r.kernelExecutionTime,
          kernelLaunchLatency: r.kernelLaunchLatency,
          operatorFusionRate: r.operatorFusionRate,
          perOperatorLatency: r.perOperatorLatency,
          kernelCompilationTime: r.kernelCompilationTime,
          effectiveMemoryBandwidth: r.effectiveMemoryBandwidth,
          synchronizationOverhead: r.synchronizationOverhead,
          peakMemoryUsage: r.peakMemoryUsage,
          timeToFirstOutput: r.timeToFirstOutput,
          endToEndLatency: r.endToEndLatency,
          notes: r.notes
        });
      }
    }

    runBtn.disabled = false;
    stopBtn.disabled = true;
    logStatus("\n" + "=".repeat(60));
    logStatus("All benchmarks complete.");
  };

  stopBtn.onclick = () => {
    stopRequested = true;
    logStatus("Stop requested...");
  };
}

main().catch(e => {
  logStatus(`Fatal error: ${e.message}`);
  console.error(e);
});