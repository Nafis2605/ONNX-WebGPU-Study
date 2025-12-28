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

function addResult(row) {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td>${row.model}</td>
    <td>${row.backend}</td>
    <td>${row.warmupRuns}</td>
    <td>${row.avgWarmup}</td>
    <td>${row.measureRuns}</td>
    <td>${row.avgInference}</td>
    <td>${row.notes}</td>
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
    // Note: ONNX Runtime Web doesn't support true int64, use int32
    const a = new BigInt64Array(size);
    for (let i = 0; i < size; i++) a[i] = BigInt(Math.floor(Math.random() * 1000));
    return a;
  }
  throw new Error(`Unsupported input type: ${type}`);
}

function makeFeedsFromModelConfig(modelCfg) {
  const feeds = {};
  for (const inp of modelCfg.inputs) {
    const data = makeRandomTypedArray(inp.type, numel(inp.dims));
    feeds[inp.name] = new ort.Tensor(inp.type, data, inp.dims);
  }
  return feeds;
}

function normalizeDims(dims) {
  if (!Array.isArray(dims) || dims.length === 0) return [1];
  return dims.map(d => (typeof d === "number" && d > 0 ? d : 1));
}

function makeFeedsFromSession(session, fallbackCfg) {
  try {
    // Try to get input information from the session
    const inputNames = session.inputNames || [];
    
    if (!inputNames.length) {
      logStatus("No input names found, using config fallback");
      return makeFeedsFromModelConfig(fallbackCfg);
    }

    const feeds = {};
    
    for (const name of inputNames) {
      let type = "float32";
      let dims = [1, 3, 224, 224]; // Common default for image models
      
      // Try different ways to get metadata
      try {
        if (session.inputMetadata && session.inputMetadata[name]) {
          const meta = session.inputMetadata[name];
          type = meta.type || type;
          dims = normalizeDims(meta.dimensions || meta.dims || dims);
        }
      } catch (e) {
        logStatus(`Could not read metadata for ${name}, using defaults: ${e.message}`);
      }
      
      logStatus(`Creating input "${name}": ${type} [${dims.join(", ")}]`);
      const data = makeRandomTypedArray(type, numel(dims));
      feeds[name] = new ort.Tensor(type, data, dims);
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
   BENCHMARK CORE
   ============================================================ */

async function timeRuns(session, feeds, runs, label) {
  const times = [];
  for (let i = 0; i < runs; i++) {
    if (stopRequested) break;
    
    try {
      const t0 = performance.now();
      
      // Use a proper timeout wrapper
      const timeoutMs = 30000; // Increased to 30 seconds
      const runPromise = session.run(feeds);
      
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(
          () => reject(new Error(`${label} run ${i + 1} timed out after ${timeoutMs} ms`)),
          timeoutMs
        )
      );
      
      await Promise.race([runPromise, timeoutPromise]);
      
      const t1 = performance.now();
      times.push(t1 - t0);
      
      // Log progress
      if (i === 0 || (i + 1) % Math.max(1, Math.floor(runs / 5)) === 0) {
        logStatus(`${label}: ${i + 1}/${runs} (${(t1 - t0).toFixed(2)} ms)`);
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
      executionProviders
    });
    const timeoutMs = 30000; // Increased to 30 seconds
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`Session creation timed out after ${timeoutMs} ms`)), timeoutMs)
    );
    session = await Promise.race([createPromise, timeoutPromise]);
    notes.push(`session_create_ms=${(performance.now() - t0).toFixed(1)}`);
    logStatus("Session created.");
    
    if (session.inputNames?.length) {
      logStatus(`Session inputs: ${session.inputNames.join(", ")}`);
    }
    if (session.outputNames?.length) {
      logStatus(`Session outputs: ${session.outputNames.join(", ")}`);
    }
  } catch (e) {
    const errMsg = `session_create_failed: ${e.message}`;
    logStatus(`ERROR: ${errMsg}`);
    return { ok: false, notes: errMsg };
  }

  let feeds;
  try {
    logStatus("Preparing input tensors...");
    feeds = makeFeedsFromSession(session, modelCfg);
    logStatus("Inputs ready.");
  } catch (e) {
    const errMsg = `feed_error: ${e.message}`;
    logStatus(`ERROR: ${errMsg}`);
    return { ok: false, notes: errMsg };
  }

  let warmTimes = [];
  try {
    if (warmupRuns > 0) {
      logStatus("Warmup starting...");
      warmTimes = await timeRuns(session, feeds, warmupRuns, "warmup");
      logStatus("Warmup complete.");
    }
  } catch (e) {
    const errMsg = `warmup_failed: ${e.message}`;
    logStatus(`ERROR: ${errMsg}`);
    return { ok: false, notes: errMsg };
  }

  let infTimes = [];
  try {
    logStatus("Measure starting...");
    infTimes = await timeRuns(session, feeds, measureRuns, "inference");
    logStatus("Measure complete.");
  } catch (e) {
    const errMsg = `inference_failed: ${e.message}`;
    logStatus(`ERROR: ${errMsg}`);
    return { ok: false, notes: errMsg };
  }

  return {
    ok: true,
    avgWarmupMs: warmTimes.length ? avg(warmTimes) : 0,
    avgInferenceMs: infTimes.length ? avg(infTimes) : NaN,
    actualWarmupRuns: warmTimes.length,
    actualMeasureRuns: infTimes.length,
    notes: notes.join(" | ")
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

  if (typeof ort.env.wasm !== 'undefined') {
    logStatus(`WASM threads: ${ort.env.wasm.numThreads}`);
    logStatus(`WASM SIMD: ${ort.env.wasm.simd}`);
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
        addResult({
          model: name,
          backend,
          warmupRuns: 0,
          avgWarmup: 'N/A',
          measureRuns: 0,
          avgInference: 'N/A',
          notes: r.notes
        });
      } else {
        const msg = `SUCCESS: warmup=${r.avgWarmupMs.toFixed(3)} ms, inference=${r.avgInferenceMs.toFixed(3)} ms`;
        logStatus(msg);
        addResult({
          model: name,
          backend,
          warmupRuns: r.actualWarmupRuns,
          avgWarmup: r.avgWarmupMs.toFixed(3),
          measureRuns: r.actualMeasureRuns,
          avgInference: r.avgInferenceMs.toFixed(3),
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