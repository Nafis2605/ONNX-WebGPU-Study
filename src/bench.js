// Benchmark core. Every value reported here is a direct measurement; anything that cannot be
// measured is left out or reported as null with a reason, never estimated.
import { startCapture, stopCapture, parsePlacement } from "./ortlog.js";
import { ort, ortBundle, configureOrt, sessionOptions } from "./ort-setup.js";
import { installCounters, uninstallCounters, resetCounters, readCounters, readMemory, verifyBackend } from "./instrument.js";
import { makeRunner, runnerRuns } from "./runner.js";
import { profileMode } from "./profile.js";
import { responsivenessMode } from "./responsiveness.js";
import { llmMode } from "./llm.js";
import { loadMode } from "./load.js";

const SCHEMA_VERSION = 2;

let log = msg => console.log(msg);
export function setLogger(fn) { log = fn; }

/** Browser / GPU environment facts for env.json. */
export async function probeEnv() {
  const env = {
    user_agent: navigator.userAgent,
    cross_origin_isolated: self.crossOriginIsolated,
    hardware_concurrency: navigator.hardwareConcurrency,
    time_origin_epoch_ms: performance.timeOrigin,
    timer_granularity_ms: timerGranularity(),
    webgpu_adapter: null,
    webgl: null,
  };
  if (navigator.gpu) {
    const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (a) {
      env.webgpu_adapter = {
        vendor: a.info.vendor, architecture: a.info.architecture,
        device: a.info.device, description: a.info.description,
        features: [...a.features].sort(),
      };
    }
  }
  const gl = document.createElement("canvas").getContext("webgl2");
  if (gl) {
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    env.webgl = {
      version: gl.getParameter(gl.VERSION),
      renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
      vendor: dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
      timer_query: !!gl.getExtension("EXT_disjoint_timer_query_webgl2"),
    };
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
  return env;
}

/** Smallest positive step of performance.now() seen in a tight loop (5 µs when isolated). */
function timerGranularity() {
  let min = Infinity;
  let last = performance.now();
  for (let i = 0; i < 200000; i++) {
    const t = performance.now();
    if (t > last && t - last < min) min = t - last;
    last = t;
  }
  return min;
}

async function fetchModel(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
  return res.arrayBuffer();
}

function withTimeout(promise, ms, what) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${what} timed out after ${ms} ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/** Label a failure from the raw ORT error text. The raw message is always kept alongside. */
function classifyError(stage, message) {
  if (/disable_cpu_ep_fallback|fallback to CPU|assigned to the default CPU EP/i.test(message)) return "cpu_nodes_blocked";
  if (/\b(int64|int8|uint8|float16|bool|string)\b is not supported/i.test(message)) return "dtype_unsupported_on_ep";
  if (/cannot resolve operator|not implemented|unsupported/i.test(message)) return "op_unsupported_on_ep";
  // ORT WebGL parses symbolic input dims (dim_param) as undefined but only accepts 0 as "any".
  if (/expected shape '\[(,|[^']*,,|[^']*,\])/.test(message)) {
    return "webgl_symbolic_dim_rejected";
  }
  if (/backend not found|no available backend/i.test(message)) return "backend_unavailable";
  if (/timed out/i.test(message)) return "timeout";
  if (/out of memory|OOM|allocation failed|Array buffer allocation/i.test(message)) return "out_of_memory";
  return `${stage}_error`;
}

/** Create a throwaway session with verbose ORT logging and parse where each node was placed. */
async function inspectPlacement(model, backend, cfg) {
  const buffer = await fetchModel(model.path);
  startCapture();
  let lines;
  let s = null;
  try {
    s = await ort.InferenceSession.create(buffer, sessionOptions(backend, {
      allowCpuNodes: cfg.allowCpuNodes, logSeverityLevel: 0, gpuIo: cfg.gpuIo, graphCapture: cfg.graphCapture,
    }));
  } finally {
    lines = stopCapture();
    try { await s?.release(); } catch { /* inspection session only */ }
  }
  return parsePlacement(lines);
}

/** Input/output metadata (names, types, symbolic shapes); the WebGL backend only exposes names. */
function sessionMetadata(session, kind) {
  try {
    return session[`${kind}Metadata`];
  } catch {
    return session[`${kind}Names`].map(name => ({ name }));
  }
}

/**
 * Page memory (JS heap + WASM memory + DOM) from performance.measureUserAgentSpecificMemory().
 * Needs cross-origin isolation; Chrome runs it at the next GC unless launched with
 * --enable-blink-features=ForceEagerMeasureMemory (run_benchmark.py adds that in profile mode).
 */
async function pageMemory() {
  if (!self.crossOriginIsolated || !performance.measureUserAgentSpecificMemory) {
    return { bytes: null, reason: "measureUserAgentSpecificMemory unavailable (needs cross-origin isolation)" };
  }
  try {
    const m = await withTimeout(performance.measureUserAgentSpecificMemory(), 30000, "memory measurement");
    return { bytes: m.bytes };
  } catch (e) {
    return { bytes: null, reason: String(e?.message ?? e) };
  }
}

/** Seeded Fisher-Yates shuffle. */
function shuffled(values, rand) {
  const a = [...values];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/**
 * Shape order for the reuse study: `rounds` seeded permutations of the sweep values,
 * arranged so consecutive rounds never join on the same value (every block boundary is a real
 * shape switch).
 */
export function shapeSequence(values, rounds, seed) {
  let s = seed >>> 0;
  const rand = () => ((s = (Math.imul(s ^ (s >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0) / 4294967296);
  const seq = [];
  for (let r = 0; r < rounds; r++) {
    let perm = shuffled(values, rand);
    if (seq.length && perm[0] === seq[seq.length - 1] && perm.length > 1) perm = [...perm.slice(1), perm[0]];
    seq.push(...perm);
  }
  return seq;
}

const MODES = {
  /** Groups 1-2: closed-loop latency at the model's base shape. */
  async latency(session, model, cfg, result) {
    const runner = await makeRunner(session, model, cfg);
    await runnerRuns(runner, "warmup", cfg.warmup, result.runs);
    await runnerRuns(runner, "measure", cfg.measure, result.runs);
    return runner;
  },

  /**
   * Group 1 reuse: after warming up at the base shape, replay a seeded sequence of input shapes.
   * Each shape runs `repeats` times in a row. Every call is labelled with whether its shape was
   * seen before in this session and its position in the block.
   */
  async shapes(session, model, cfg, result) {
    const sweep = model.shape_sweep;
    const base = await makeRunner(session, model, cfg);
    await runnerRuns(base, "warmup", cfg.warmup, result.runs);

    const baseDims = model.inputs.find(i => i.name === sweep.input).dims;
    const seen = new Set([baseDims[sweep.axis]]);
    const sequence = shapeSequence(sweep.values, cfg.shapeRounds ?? 3, cfg.seed);
    result.config.shape_sequence = sequence;
    result.config.shape_repeats = cfg.shapeRepeats ?? 3;

    let i = 0;
    for (const [block, value] of sequence.entries()) {
      const dims = [...baseDims];
      dims[sweep.axis] = value;
      const runner = await makeRunner(session, model, cfg, { [sweep.input]: dims });
      const firstSeen = !seen.has(value);
      seen.add(value);
      for (let pos = 0; pos < result.config.shape_repeats; pos++) {
        result.runs.push({
          phase: "shape", i: i++, block, block_pos: pos,
          shape_value: value, shape: dims.join("x"),
          first_seen: firstSeen && pos === 0,
          ...(await runner.run()),
        });
      }
      runner.dispose();
    }
    return base;
  },

  /**
   * Batch scaling: for every value of the model's shape_sweep (ascending), warm up at that shape
   * and then measure cfg.measure runs. Per-sample latency = dur_ms / batch.
   */
  async batch(session, model, cfg, result) {
    const sweep = model.shape_sweep;
    const baseDims = model.inputs.find(i => i.name === sweep.input).dims;
    const values = [...(cfg.batchValues ?? sweep.batch_values ?? sweep.values)].sort((a, b) => a - b);
    result.config.batch_values = values;
    let base = null;
    for (const value of values) {
      const dims = [...baseDims];
      dims[sweep.axis] = value;
      const runner = await makeRunner(session, model, cfg, { [sweep.input]: dims });
      await runnerRuns(runner, "batch_warmup", cfg.warmup, result.runs, { batch: value });
      await runnerRuns(runner, "batch", cfg.measure, result.runs, { batch: value });
      if (value === baseDims[sweep.axis]) base = runner;
      else runner.dispose();
    }
    return base ?? makeRunner(session, model, cfg);
  },

  /** Groups 5-7: instrumented pass (see profile.js). */
  profile: profileMode,

  /** Main-thread availability while inference runs (see responsiveness.js). */
  responsiveness: responsivenessMode,

  /** Group 3: token generation timing, no-KV-cache greedy decoding (see llm.js). */
  llm: llmMode,

  /** Group 8: open-loop Poisson load with deadlines and a bounded queue (see load.js). */
  load: loadMode,

  /**
   * Group 10: return the exact inputs and outputs of one inference (after warmup) as raw bytes,
   * so run_benchmark.py can compare them against native ONNX Runtime on the same inputs.
   */
  async correctness(session, model, cfg, result) {
    const runner = await makeRunner(session, model, cfg);
    await runnerRuns(runner, "warmup", cfg.warmup, result.runs);
    const outputs = await runner.outputs();
    result.io = {
      inputs: Object.entries(runner.feeds).map(([name, t]) => encodeTensor(name, t)),
      outputs: Object.entries(outputs).map(([name, t]) => encodeTensor(name, t)),
    };
    return runner;
  },
};

function encodeTensor(name, tensor) {
  const bytes = new Uint8Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return { name, type: tensor.type, dims: tensor.dims, b64: btoa(binary) };
}

/**
 * Run one (model, backend) experiment in this page.
 * cfg: {model: <models.json entry>, backend, warmup, measure, seed, allowCpuNodes, sessionTimeoutMs}
 */
export async function runExperiment(cfg) {
  const { model, backend } = cfg;
  cfg = { seed: 1, ...cfg };
  const result = {
    schema: SCHEMA_VERSION,
    model: model.name,
    backend,
    mode: cfg.mode ?? "latency",
    config_label: cfg.configLabel ?? backend,
    config: {
      warmup: cfg.warmup, measure: cfg.measure, seed: cfg.seed,
      allow_cpu_nodes: !!cfg.allowCpuNodes,
      graph_optimization_level: "all",
      ep_impl: cfg.epImpl ?? "jsep",
      wasm_threads: cfg.wasmThreads ?? 1,
      gpu_io: !!cfg.gpuIo,
      graph_capture: !!cfg.graphCapture,
      outputs: cfg.outputs ?? "all",
    },
    ort: null,
    status: "failed",
    failure: null,
    model_bytes: null,
    fetch_ms: null,
    session_create_ms: null,
    session_inputs: null,
    session_outputs: null,
    feeds: null,
    runs: [],
    verification: null,
  };

  await configureOrt({ epImpl: cfg.epImpl, wasmThreads: cfg.wasmThreads, webglPack: cfg.webglPack });
  result.ort = {
    bundle: ortBundle, versions: ort.env.versions,
    wasm_threads: ort.env.wasm.numThreads, simd: ort.env.wasm.simd,
    webgl_pack: ortBundle === "jsep" ? ort.env.webgl.pack ?? null : null,
  };
  const profiling = result.mode === "profile";
  if (profiling) {
    // The WebGL profiler logs through ORT's JS logger, whose level is fixed when the WebGL
    // backend initialises (first WebGL session in this page).
    if (backend === "webgl") ort.env.logLevel = "verbose";
    // Instrument from before session creation so buffers/textures holding weights are counted.
    installCounters();
  }

  let stage = "fetch";
  let session = null;
  try {
    if (profiling) result.page_memory = { before_fetch: await pageMemory() };
    log(`[${model.name}/${backend}] fetching ${model.path}`);
    const f0 = performance.now();
    const buffer = await fetchModel(model.path);
    result.fetch_ms = performance.now() - f0;
    result.model_bytes = buffer.byteLength;

    stage = "session_create";
    log(`[${model.name}/${backend}] creating session`);
    const s0 = performance.now();
    session = await withTimeout(
      ort.InferenceSession.create(buffer, sessionOptions(backend, {
        allowCpuNodes: cfg.allowCpuNodes, enableProfiling: profiling,
        gpuIo: cfg.gpuIo, graphCapture: cfg.graphCapture,
      })),
      cfg.sessionTimeoutMs ?? 600000, "session creation");
    result.session_create_ms = performance.now() - s0;
    if (profiling) result.page_memory.after_session_create = await pageMemory();
    result.session_inputs = sessionMetadata(session, "input");
    result.session_outputs = sessionMetadata(session, "output");

    stage = "run";
    log(`[${model.name}/${result.config_label}] mode ${result.mode}: warmup ${cfg.warmup}, measure ${cfg.measure}`);
    const runner = await MODES[result.mode](session, model, cfg, result);
    result.feeds = runner.spec;
    if (profiling) {
      result.page_memory.after_runs = await pageMemory();
      result.gpu_memory_final = readMemory();
    }

    // Verification pass: one extra run with API counters, after (not inside) the timed runs.
    stage = "verify";
    if (!profiling) installCounters();
    try {
      resetCounters();
      await runner.run();
      const counters = readCounters();
      result.verification = { counters, ...verifyBackend(backend, counters) };
    } finally {
      uninstallCounters();
    }
    if (backend === "webgpu" && ort.env.webgpu.adapter?.info) {
      const i = ort.env.webgpu.adapter.info;
      result.verification.ort_adapter = { vendor: i.vendor, architecture: i.architecture, description: i.description };
    }

    // Node placement comes from a separate, verbose-logging session so the timed session's
    // creation time isn't inflated by logging.
    if (backend === "webgpu" && cfg.placement !== false) {
      stage = "placement";
      await session.release();
      session = null;
      runner.dispose();
      result.placement = await inspectPlacement(model, backend, cfg);
    }

    const cpuNodes = result.placement?.cpu_nodes ?? 0;
    if (!result.verification.verified) result.status = "backend_not_verified";
    else if (cpuNodes > 0) result.status = "ok_with_cpu_nodes";
    // Strict mode already guarantees no CPU nodes (session creation fails otherwise); in allow
    // mode an unparsed placement report means CPU placement can't be ruled out.
    else if (cfg.allowCpuNodes && result.placement && !result.placement.found) result.status = "placement_unknown";
    else result.status = "ok";
    log(`[${model.name}/${backend}] ${result.status}: ${result.verification.detail}` +
        (result.placement ? `; placement ${JSON.stringify(result.placement.nodes_per_ep)}` : ""));
  } catch (e) {
    const message = String(e?.message ?? e);
    result.failure = { stage, category: classifyError(stage, message), message };
    log(`[${model.name}/${backend}] FAILED at ${stage}: ${message}`);
  } finally {
    uninstallCounters();
    try { await session?.release(); } catch { /* the session is discarded either way */ }
  }
  return result;
}
