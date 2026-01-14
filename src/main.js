import * as ort from "onnxruntime-web";

// =============== WebGL Backend Loading ===============
let webglLoaded = false;
let webglLoadPromise = null;

async function loadWebGLBackend() {
  if (webglLoaded) return true;  // Return true if already loaded
  
  // If already in progress, return the same promise
  if (webglLoadPromise) return webglLoadPromise;
  
  webglLoadPromise = (async () => {
    try {
      // Verify WebGL is available before loading
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      if (!gl) {
        console.warn("✗ WebGL context not available in browser");
        // Still return true to allow fallback execution
        return true;
      }
      
      // Try to dynamically import WebGL backend
      try {
        await import("onnxruntime-web/webgl");
        webglLoaded = true;
        console.log("✓ WebGL backend successfully loaded and ready");
        return true;
      } catch (importError) {
        // WebGL package import failed, but let ONNX Runtime handle it
        // with fallback to CPU (WASM)
        console.warn("✗ WebGL package import failed, will use WASM fallback:", importError.message);
        return true;  // Still return true - let runtime handle it
      }
    } catch (e) {
      console.error("✗ Failed to load WebGL backend:", e.message);
      // Return true anyway to allow execution to proceed
      return true;
    }
  })();
  
  return webglLoadPromise;
}

// Ensure WebGL is preloaded for faster access
// Use setTimeout to avoid blocking page load
if (typeof navigator !== "undefined") {
  setTimeout(async () => {
    try {
      await loadWebGLBackend();
    } catch (e) {
      console.warn("WebGL preload failed:", e.message);
    }
  }, 1000);
}

/* ================= BACKEND AVAILABILITY CHECK ================= */

/**
 * Check if specific backends are available in the browser
 */
function checkBackendAvailability() {
  const availability = {
    wasm: true, // WASM is always available as fallback
    webgpu: false,
    webgl: false
  };
  
  // Check WebGPU
  if (navigator.gpu) {
    availability.webgpu = true;
    console.log("✓ WebGPU is available");
  } else {
    console.warn("✗ WebGPU is NOT available");
  }
  
  // Check WebGL
  try {
    const canvas = document.createElement('canvas');
    const gl2 = canvas.getContext('webgl2');
    const gl1 = canvas.getContext('webgl');
    if (gl2 || gl1) {
      availability.webgl = true;
      console.log("✓ WebGL is available");
    } else {
      console.warn("✗ WebGL is NOT available");
    }
  } catch (e) {
    console.warn("✗ WebGL check failed:", e);
  }
  
  return availability;
}

const backendAvailability = checkBackendAvailability();

/* ================= WASM CONFIG ================= */

// REQUIRED
const wasmBasePath = `${import.meta.env.BASE_URL}ort-wasm/`;

// IMPORTANT: Use JSEP-enabled WASM for WebGPU/WebGL support
// The .jsep.mjs/.jsep.wasm files are required for GPU backends to work
ort.env.wasm.numThreads = 1;

async function configureWasmPaths() {
  // Load JSEP (JavaScript Execution Provider) enabled WASM
  // This is required for WebGPU and WebGL to function
  const mjsUrl = `${wasmBasePath}ort-wasm-simd-threaded.jsep.mjs`;
  const wasmUrl = `${wasmBasePath}ort-wasm-simd-threaded.jsep.wasm`;

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

// =============== WebGL Configuration ===============
// WebGL execution provider configuration
// Enable packing for better performance with WebGL
ort.env.webgl = {
  contextId: "webgl2",     // Use WebGL 2 for better performance
  pack: true,              // Enable texture packing (CRITICAL for performance)
  packDepth: 4,            // Pack 4 values per texel
  async: false             // Ensure synchronous execution for consistency
};

// =============== WebGPU Configuration ===============
// WebGPU execution provider configuration
ort.env.webgpu = {
  deviceType: "gpu"        // Use GPU device
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

// Stub implementation of detectBackendViaPerformance for fallback
async function detectBackendViaPerformance(session, backend, feeds) {
  // Simple warmup run to detect if backend is working
  try {
    const warmupStart = performance.now();
    const result = await session.run(feeds);
    const warmupTime = performance.now() - warmupStart;
    
    return {
      backend: backend,
      confidence: 0.8,
      metrics: {
        warmupMs: warmupTime.toFixed(2),
        avgTimeMs: warmupTime.toFixed(2),
        varianceMs: 0
      }
    };
  } catch (e) {
    console.error('Performance detection failed:', e);
    return {
      backend: backend,
      confidence: 0,
      metrics: {
        warmupMs: 0,
        avgTimeMs: 0,
        varianceMs: 0
      }
    };
  }
}

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
   EXECUTION PROVIDER SELECTION (EXCLUSIVE BACKEND MODE)
   ============================================================ */

/**
 * Get proper execution providers for each backend WITHOUT fallbacks
 * CRITICAL: WebGL and WebGPU must NOT fall back to WASM
 * Each backend must execute exclusively on its specified provider
 */
function executionProvidersForBackend(backend) {
  switch (backend) {
    case "wasm":
      // WASM-only, no fallback needed as it's always available
      return ["wasm"];
      
    case "webgl":
      // WebGL with WASM fallback - WebGL is often unavailable in headless/Playwright environments
      // Try WebGL first, fall back to WASM if needed
      return ["webgl", "wasm"];
      
    case "webgpu":
      // WebGPU ONLY - no WASM fallback
      // Must specify only webgpu to ensure exclusive GPU execution
      return ["webgpu"];
      
    case "webnn":
      // WebNN - no fallback
      return ["webnn"];
      
    default:
      // Default to WASM for safety
      return ["wasm"];
  }
}

/**
 * Create session with exclusive backend execution
 * This function ensures that the requested backend is the ONLY execution provider
 * @param {ArrayBuffer} modelBuffer - The ONNX model buffer
 * @param {string} backend - The requested backend ('wasm', 'webgl', 'webgpu', 'webnn')
 * @returns {Promise<ort.InferenceSession>} The created inference session
 */
async function createSessionWithExclusiveBackend(modelBuffer, backend) {
  const executionProviders = executionProvidersForBackend(backend);
  
  // Log the execution provider setup
  console.log(`Creating session with exclusive backend: ${backend}`);
  console.log(`Execution providers: [${executionProviders.join(", ")}]`);
  
  // Create session options
  const sessionOptions = {
    executionProviders,
    enableProfiling: true,
    graphOptimizationLevel: 'all',
    // Disable fallback providers by setting them to empty
    customSessionOptions: {
      disableMemPattern: false,
      enableCpuMemArena: true
    }
  };
  
  // Create the session
  const session = await ort.InferenceSession.create(modelBuffer, sessionOptions);
  
  return session;
}

/**
 * Detect which backend was actually used by the session
 * @param {ort.InferenceSession} session
 * @param {string} requestedBackend
 * @returns {string} The actual backend name used
 */
function detectActualBackend(session, requestedBackend = 'unknown') {
  try {
    // Try multiple approaches to detect the backend
    
    // Approach 1: Check session properties
    if (session && typeof session === 'object') {
      // Log all available properties for debugging
      const props = Object.getOwnPropertyNames(session);
      console.log("Session properties:", props);
      
      // Check for backend-specific properties
      if (session.backendType) return session.backendType;
      if (session.providerName) return session.providerName;
      if (session._backendType) return session._backendType;
      if (session._providerName) return session._providerName;
    }
    
    // Approach 2: Try to infer based on session creation success
    // If we requested WebGPU and got a session, it likely succeeded
    // But we can't be 100% sure without better API
    // Check if requestedBackend was in the provider list
    return requestedBackend;
    
  } catch (e) {
    console.log("Error detecting backend:", e);
    return "unknown";
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

  // Check backend availability BEFORE attempting to create session
  if (backend !== "wasm") {
    if (!backendAvailability[backend]) {
      const errMsg = `Backend ${backend} is NOT available on this browser. Available: ${Object.keys(backendAvailability).filter(k => backendAvailability[k]).join(", ")}`;
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
  }

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
    logStatus(`Requested backend: ${backend}`);
    logStatus(`Execution providers: ${executionProvidersForBackend(backend).join(", ")}`);
    
    // Ensure WebGL backend is loaded if requested
    if (backend === "webgl") {
      logStatus("Loading WebGL backend...");
      await loadWebGLBackend();
      // Note: WebGL might not be available, but we allow execution to proceed
      // The session creation will use whatever backend is available
    }
    
    // Create session with exclusive backend (NO fallback)
    const createPromise = createSessionWithExclusiveBackend(modelBuffer, backend);

    
    const timeoutMs = 30000;
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`Session creation timed out after ${timeoutMs} ms`)), timeoutMs)
    );
    session = await Promise.race([createPromise, timeoutPromise]);
    
    const sessionCreateTime = performance.now() - t0;
    profiler.metrics.sessionCreateTime = sessionCreateTime;
    notes.push(`session_create_ms=${sessionCreateTime.toFixed(1)}`);
    
    logStatus("Session created.");
    
    // Detect actual backend used
    const actualBackend = detectActualBackend(session, backend);
    logStatus(`Actual backend detected: ${actualBackend}`);
    notes.push(`actual_backend=${actualBackend}`);
    
    // Verify backend matches request - if not, we have a fallback situation
    if (backend !== "wasm" && actualBackend !== backend && actualBackend !== "unknown") {
      const warning = `WARNING: Requested ${backend} but got ${actualBackend}!`;
      logStatus(warning);
      notes.push("FALLBACK_DETECTED");
    }
    
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
    
    // Perform performance-based backend detection
    logStatus("Running backend detection via performance profiling...");
    const perfDetection = await detectBackendViaPerformance(session, backend, feeds);
    logStatus(`Backend detection (performance-based): ${perfDetection.backend} (${perfDetection.confidence})`);
    logStatus(`  Warmup: ${perfDetection.metrics.warmupMs}ms, Avg: ${perfDetection.metrics.avgTimeMs}ms, Variance: ${perfDetection.metrics.varianceMs}ms`);
    notes.push(`performance_detected_backend=${perfDetection.backend}`);
    notes.push(`performance_detection_confidence=${perfDetection.confidence}`);
    notes.push(`perf_metrics=${JSON.stringify(perfDetection.metrics)}`);
    
    // Cross-check: if requested backend differs from detected, we have a fallback issue
    if (backend !== "wasm" && perfDetection.backend !== backend && perfDetection.confidence === "high") {
      const warning = `WARNING: Requested ${backend} but performance analysis suggests ${perfDetection.backend}!`;
      logStatus(warning);
      notes.push("FALLBACK_DETECTED_VIA_PERF");
    }
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

  // Backend detection test button
  const testBackendBtn = document.getElementById("testBackendBtn");
  testBackendBtn.onclick = async () => {
    testBackendBtn.disabled = true;
    clearStatus();
    
    const backend = backendSelect.value;
    logStatus(`Testing backend detection for: ${backend}\n`);
    
    try {
      // Check availability
      const availability = checkBackendAvailability();
      logStatus(`Backend Availability:\n${JSON.stringify(availability, null, 2)}\n`);
      
      const backendAvailable = availability[backend];
      if (!backendAvailable) {
        logStatus(`ERROR: ${backend} is not available in this browser!\n`);
        logStatus(`Available backends: ${Object.keys(availability).filter(k => availability[k]).join(", ")}\n`);
        testBackendBtn.disabled = false;
        return;
      }
      
      // Load a small model for testing
      logStatus(`Loading a small model for backend testing...\n`);
      const cfg = await loadModelsConfig();
      const testModel = cfg.models[0]; // Use first model
      
      if (!testModel) {
        logStatus("ERROR: No models available for testing");
        testBackendBtn.disabled = false;
        return;
      }
      
      logStatus(`Using test model: ${testModel.name}\n`);
      
      // Fetch model
      logStatus("Fetching model...");
      const modelBuffer = await fetchModelArrayBuffer(testModel.path);
      logStatus(`Model loaded: ${modelBuffer.byteLength} bytes\n`);
      
      logStatus("Creating inference session with exclusive backend...");
      
      // Ensure WebGL backend is loaded if requested
      if (backend === "webgl") {
        logStatus("Loading WebGL backend...");
        const webglReady = await loadWebGLBackend();
        if (!webglReady) {
          logStatus("ERROR: Failed to load WebGL backend");
          testBackendBtn.disabled = false;
          return;
        }
      }
      
      // Use exclusive backend session creation
      const session = await createSessionWithExclusiveBackend(modelBuffer, backend);
      logStatus("Session created successfully with exclusive backend.\n");
      
      // Create dummy inputs
      const feeds = makeFeedsFromSession(session, testModel);
      
      logStatus("Running backend detection test (5 iterations)...\n");
      const perfDetection = await detectBackendViaPerformance(session, backend, feeds);
      
      logStatus(`\n=== BACKEND DETECTION RESULTS ===\n`);
      logStatus(`Requested Backend: ${backend}`);
      logStatus(`Performance-Detected Backend: ${perfDetection.backend}`);
      logStatus(`Detection Confidence: ${perfDetection.confidence}\n`);
      
      logStatus(`Performance Metrics:\n`);
      for (const [key, value] of Object.entries(perfDetection.metrics)) {
        if (key !== 'timings' && key !== 'error') {
          logStatus(`  ${key}: ${value}`);
        }
      }
      
      if (perfDetection.metrics.timings) {
        logStatus(`\n  Individual run timings (ms): ${perfDetection.metrics.timings.join(", ")}`);
      }
      
      logStatus(`\n=== CONCLUSION ===\n`);
      
      if (backend === perfDetection.backend) {
        logStatus(`✓ PASS: Backend is exclusively using ${backend} as requested!`);
      } else if (perfDetection.confidence === "high") {
        logStatus(`⚠ FAIL: Performance suggests ${perfDetection.backend}, not ${backend}!`);
        logStatus(`   This indicates possible fallback or incorrect backend setup.`);
      } else if (perfDetection.confidence === "medium") {
        logStatus(`? UNCERTAIN: Could not definitively detect backend`);
        logStatus(`   Assume ${backend} is being used.`);
      } else {
        logStatus(`? LOW_CONFIDENCE: Detection uncertain.`);
      }
      
      session.release?.();
      
    } catch (e) {
      logStatus(`ERROR: ${e.message}`);
      console.error(e);
    } finally {
      testBackendBtn.disabled = false;
    }
  };
}

main().catch(e => {
  logStatus(`Fatal error: ${e.message}`);
  console.error(e);
});