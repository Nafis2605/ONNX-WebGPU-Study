// Profile mode (groups 5-7): an instrumented pass, run in its own browser launch so it never
// shares a session with the latency (timing) passes.
//
// Phases after warmup:
//   profile_counters  graphics-API counters per run (dispatches, submits, execution groups,
//                     upload/readback bytes, mapAsync latency, logical GPU memory)
//   profile_gpu       per-kernel GPU time:
//                       WebGPU  ORT timestamp queries (env.webgpu.profiling). Chrome exposes only
//                               "timestamp-query", so ORT ends a compute pass after every dispatch
//                               in this phase; its wall times aren't comparable to latency runs.
//                       WebGL   EXT_disjoint_timer_query_webgl2 around every draw (one draw per
//                               ORT program run), attributed to ORT nodes via the verbose log
//                       WASM    none (no GPU); per-op CPU time comes from the ORT trace instead
// WebGL with cfg.glTiming, between the two (where the time goes on the CPU side):
//   profile_gltime    CPU time inside every WebGL call, by method; the rest of the run is JS
//   profile_readsplit readPixels split into GPU drain (1x1 read first) and payload transfer
//   profile_cpuprof   (cfg.cpuProfile) the same runs inside console.profile(), which the driver
//                     collects over CDP as a V8 sampling profile of the page's main thread
// For WASM and WebGPU, ORT's C++ profiler traces every run of the session (per-node CPU time,
// model_run duration, session-initialization breakdown).
import { ort } from "./ort-setup.js";
import { makeRunner, runnerRuns } from "./runner.js";
import {
  resetCounters, readCounters, readGroups, readMemory, resetPeak, startDrawTimers, collectDrawTimers,
  startGlTiming, readGlTiming, stopGlTiming, startReadSplit, readReadSplit, stopReadSplit,
} from "./instrument.js";
import { startCapture, stopCapture, parseOrtTrace, webglOpAt } from "./ortlog.js";

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function instrumentedRuns(runner, phase, count, runs) {
  const done = [];
  for (let i = 0; i < count; i++) {
    resetCounters();
    const rec = { phase, i, ...(await runner.run()) };
    rec.counters = readCounters();
    rec.groups = readGroups();
    rec.memory = readMemory();
    runs.push(rec);
    done.push(rec);
  }
  return done;
}

/** Split a flat, in-order list of kernel records into runs using the per-run counts. */
function splitByCounts(items, counts) {
  const total = counts.reduce((a, b) => a + b, 0);
  if (total !== items.length) return null;
  const out = [];
  let k = 0;
  for (const n of counts) {
    out.push(items.slice(k, k + n));
    k += n;
  }
  return out;
}

async function webgpuKernelPhase(runner, count, runs, profile) {
  const kernels = [];
  ort.env.webgpu.profiling = { mode: "default", ondata: d => kernels.push(d) };
  let recs;
  try {
    recs = await instrumentedRuns(runner, "profile_gpu", count, runs);
    // Timestamp results arrive through mapAsync callbacks; let the queue drain first.
    await ort.env.webgpu.device?.queue.onSubmittedWorkDone();
    await sleep(100);
  } finally {
    ort.env.webgpu.profiling = { mode: "off" };
  }
  const perRun = splitByCounts(kernels, recs.map(r => r.counters.webgpu_dispatches));
  profile.kernel_assignment = perRun ? "split_by_dispatch_counts" :
    `unassigned: ${kernels.length} kernel records vs ${recs.reduce((a, r) => a + r.counters.webgpu_dispatches, 0)} dispatches`;
  recs.forEach((rec, r) => {
    const ks = perRun ? perRun[r] : [];
    if (ks.length) {
      rec.gpu_kernels = ks.length;
      rec.gpu_busy_ms = ks.reduce((a, k) => a + (k.endTime - k.startTime), 0) / 1e6;
      rec.gpu_span_ms = (ks[ks.length - 1].endTime - ks[0].startTime) / 1e6;
    }
    for (const k of ks) {
      profile.kernels.push({
        source: "webgpu_timestamp", phase: rec.phase, run: rec.i,
        op: k.kernelType, node: k.kernelName, program: k.programName,
        start_us: k.startTime / 1e3, dur_us: (k.endTime - k.startTime) / 1e3,
        input_shapes: JSON.stringify(k.inputsMetadata.map(m => m.dims)),
      });
    }
  });
}

async function webglKernelPhase(runner, count, runs, profile) {
  const perRun = [];
  const recs = [];
  startCapture();
  let lines;
  try {
    for (let i = 0; i < count; i++) {
      startDrawTimers();
      const [rec] = await instrumentedRuns(runner, "profile_gpu", 1, runs);
      rec.i = i;
      recs.push(rec);
      perRun.push(await collectDrawTimers());
    }
  } finally {
    lines = stopCapture();
  }
  const opAt = webglOpAt(lines);
  profile.kernel_assignment = "per_draw_timer_query";
  recs.forEach((rec, r) => {
    const draws = perRun[r];
    const valid = draws.filter(d => d.gpu_ms !== null);
    rec.gpu_kernels = draws.length;
    rec.gpu_timer_invalid = draws.length - valid.length;
    rec.gpu_busy_ms = valid.length === draws.length ? valid.reduce((a, d) => a + d.gpu_ms, 0) : null;
    for (const d of draws) {
      profile.kernels.push({
        source: "webgl_timer_query", phase: rec.phase, run: rec.i,
        node: opAt[d.logPos] ?? null, program: d.program, dur_us: d.gpu_ms === null ? null : d.gpu_ms * 1e3,
        disjoint: d.disjoint,
      });
    }
  });
}

// WebGL method -> category for the per-run CPU split.
const GL_CATEGORY = {
  readPixels: "readpixels",
  drawArrays: "draw", drawElements: "draw", drawArraysInstanced: "draw", drawElementsInstanced: "draw",
  texImage2D: "upload", texSubImage2D: "upload", texImage3D: "upload", texSubImage3D: "upload",
  getError: "sync", finish: "sync", flush: "sync", fenceSync: "sync", clientWaitSync: "sync",
  getSyncParameter: "sync", getQueryParameter: "sync", getParameter: "sync",
};

function glCategory(name) {
  if (GL_CATEGORY[name]) return GL_CATEGORY[name];
  if (/^(bind|uniform|useProgram|vertexAttrib|enable|disable|viewport|scissor|active|framebuffer|pixelStore|texParameter)/.test(name)) return "state";
  return "other";
}

async function webglCpuPhases(runner, cfg, count, runs, profile) {
  const methods = {};
  startGlTiming();
  try {
    readGlTiming();
    for (let i = 0; i < count; i++) {
      const rec = { phase: "profile_gltime", i, ...(await runner.run()) };
      const t = readGlTiming();
      const cat = { readpixels: 0, draw: 0, upload: 0, sync: 0, state: 0, other: 0 };
      let total = 0, calls = 0;
      for (const [name, { calls: n, ms }] of Object.entries(t)) {
        cat[glCategory(name)] += ms;
        total += ms;
        calls += n;
        const m = methods[name] ?? (methods[name] = { calls: 0, ms: 0 });
        m.calls += n;
        m.ms += ms;
      }
      rec.gl_calls = calls;
      rec.gl_ms_total = total;
      for (const [k, v] of Object.entries(cat)) rec[`gl_${k}_ms`] = v;
      rec.js_ms = rec.dur_ms - total;
      runs.push(rec);
    }
  } finally {
    stopGlTiming();
  }
  profile.gl_methods = Object.entries(methods).sort((a, b) => b[1].ms - a[1].ms).slice(0, 15)
    .map(([name, m]) => ({ name, calls_per_run: m.calls / count, ms_per_run: m.ms / count }));

  startReadSplit();
  try {
    readReadSplit();
    for (let i = 0; i < count; i++) {
      const rec = { phase: "profile_readsplit", i, ...(await runner.run()) };
      const s = readReadSplit();
      rec.readpixels_calls = s.calls;
      rec.readpixels_split_calls = s.split_calls;
      rec.readpixels_drain_ms = s.drain_ms;
      rec.readpixels_transfer_ms = s.transfer_ms;
      rec.readpixels_unsplit_ms = s.unsplit_ms;
      runs.push(rec);
    }
  } finally {
    stopReadSplit();
  }

  if (cfg.cpuProfile) {
    // console.profile() starts V8's sampling profiler; the driver has the CDP Profiler domain
    // enabled and receives the finished profile (Profiler.consoleProfileFinished).
    performance.mark("cpuprof_start");
    console.profile("cpuprof");
    try {
      await runnerRuns(runner, "profile_cpuprof", count, runs);
    } finally {
      console.profileEnd("cpuprof");
      performance.measure("cpuprof", "cpuprof_start");
    }
  }
}

/** Attach ORT C++ trace events (WASM / WebGPU sessions) to the runs they occurred in. */
function attachOrtTrace(events, runs, warmup, profile, graphCapture = false) {
  profile.session_events = events.filter(e => e.cat === "Session" && e.name !== "model_run" && e.name !== "SequentialExecutor::Execute")
    .map(e => ({ name: e.name, dur_us: e.dur }));
  let modelRuns = events.filter(e => e.name === "model_run").sort((a, b) => a.ts - b.ts);
  const nodes = events.filter(e => e.cat === "Node");
  const traced = runs.filter(r => r.phase.startsWith("profile"));
  const expected = warmup + traced.length;
  profile.ort_trace_assignment = "by_model_run_interval";
  // With graph capture, ORT's first run() executes the graph once normally and once to record it,
  // which shows up as exactly one extra model_run event at the start.
  if (graphCapture && modelRuns.length === expected + 1) {
    modelRuns = modelRuns.slice(1);
    profile.ort_trace_assignment = "by_model_run_interval (capture run dropped)";
  }
  // The session's runs in order: warmup, then the profiled runs.
  if (modelRuns.length !== expected) {
    profile.ort_trace_assignment = `unassigned: ${modelRuns.length} model_run events vs ${expected} runs`;
    return;
  }
  traced.forEach((rec, idx) => {
    const mr = modelRuns[warmup + idx];
    const inRun = nodes.filter(n => n.ts >= mr.ts && n.ts + n.dur <= mr.ts + mr.dur);
    rec.ort_model_run_ms = mr.dur / 1e3;
    rec.ort_nodes = inRun.length;
    rec.ort_node_time_ms = inRun.reduce((a, n) => a + n.dur, 0) / 1e3;
    for (const n of inRun) {
      profile.kernels.push({
        source: "ort_trace", phase: rec.phase, run: rec.i,
        op: n.args?.op_name, node: n.name.replace(/_kernel_time$/, ""), provider: n.args?.provider,
        start_us: n.ts, dur_us: n.dur,
        input_shapes: JSON.stringify((n.args?.input_type_shape ?? []).map(o => Object.values(o)[0])),
      });
    }
  });
}

export async function profileMode(session, model, cfg, result) {
  const backend = result.backend;
  const runner = await makeRunner(session, model, cfg);
  const profile = { kernels: [], memory: {} };
  result.profile = profile;
  const count = cfg.measure;

  profile.memory.after_session_create = readMemory();
  await runnerRuns(runner, "warmup", cfg.warmup, result.runs);
  profile.memory.after_warmup = readMemory();

  resetPeak();
  // WebGL runs with verbose logging in profile mode; swallow it here so console output doesn't
  // inflate the counters phase.
  if (backend === "webgl") startCapture();
  try {
    await instrumentedRuns(runner, "profile_counters", count, result.runs);
  } finally {
    if (backend === "webgl") stopCapture();
  }
  profile.memory.after_counter_runs = readMemory();

  // The JSEP WebGPU EP exposes per-kernel GPU timestamps (env.webgpu.profiling); the native EP
  // doesn't implement that hook, so its kernel phase is skipped rather than approximated.
  if (backend === "webgpu" && (cfg.epImpl ?? "jsep") === "jsep") await webgpuKernelPhase(runner, count, result.runs, profile);
  if (backend === "webgl" && cfg.glTiming) {
    // Verbose WebGL logging stays swallowed, as in the counters phase.
    startCapture();
    try {
      await webglCpuPhases(runner, cfg, count, result.runs, profile);
    } finally {
      stopCapture();
    }
  }
  if (backend === "webgl") await webglKernelPhase(runner, count, result.runs, profile);

  if (backend === "wasm" || backend === "webgpu") {
    startCapture();
    let lines;
    try {
      session.endProfiling();
      await sleep(50);
    } finally {
      lines = stopCapture();
    }
    attachOrtTrace(parseOrtTrace(lines), result.runs, cfg.warmup, profile, !!cfg.graphCapture);
  }
  return runner;
}
