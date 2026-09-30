// Capture of ONNX Runtime's own output. ORT's WASM core writes logs through console.error and
// profiler traces (endProfiling) through console.log; the WebGL backend's profiler logs through
// console.log.
//
// This module must be imported before the first InferenceSession is created: Emscripten binds
// console.log / console.error when the WASM module initialises, so the wrappers have to be in
// place first. Outside a capture window they only forward, and at the default "warning" log
// level ORT prints nothing during inference.

let captured = null;

for (const method of ["log", "error"]) {
  const original = console[method];
  console[method] = function (...args) {
    if (captured) {
      captured.push(args.join(" "));
      return undefined;
    }
    return original.apply(this, args);
  };
}

export function startCapture() {
  captured = [];
}

/** Number of lines captured so far in the current window (0 when not capturing). */
export function captureLength() {
  return captured ? captured.length : 0;
}

export function stopCapture() {
  const lines = captured ?? [];
  captured = null;
  return lines;
}

/**
 * Parse ORT's "Node placements" report (session_state.cc) and memcpy insertions
 * (transformer_memcpy.cc) from verbose session-creation logs.
 */
export function parsePlacement(lines) {
  const nodesPerEp = {};
  const opsPerEp = {};
  const memcpy = { MemcpyToHost: 0, MemcpyFromHost: 0 };
  const memcpyTensors = [];
  let currentEp = null;
  for (const line of lines) {
    // "All nodes placed on [EP]" when a single EP owns the graph, "Node(s) placed on [EP]" per EP otherwise.
    const placed = line.match(/(?:All nodes|Node\(s\)) placed on \[(\w+)\]\. Number of nodes: (\d+)/);
    if (placed) {
      currentEp = placed[1];
      nodesPerEp[currentEp] = Number(placed[2]);
      opsPerEp[currentEp] = {};
      continue;
    }
    const node = line.match(/VerifyEachNodeIsAssignedToAnEp\]\s{3,}(\S+) \(/);
    if (node && currentEp) {
      opsPerEp[currentEp][node[1]] = (opsPerEp[currentEp][node[1]] ?? 0) + 1;
      continue;
    }
    const copy = line.match(/Add (MemcpyToHost|MemcpyFromHost) (before|after) (\S+) for (\w+)/);
    if (copy) {
      memcpy[copy[1]]++;
      memcpyTensors.push({ kind: copy[1], position: copy[2], tensor: copy[3], ep: copy[4] });
    }
  }
  return {
    found: Object.keys(nodesPerEp).length > 0,
    nodes_per_ep: nodesPerEp,
    // Op types per EP are only listed for the EPs ORT prints individually (CPU when mixed).
    ops_per_ep: opsPerEp,
    cpu_nodes: nodesPerEp.CPUExecutionProvider ?? 0,
    memcpy_nodes: memcpy,
    memcpy_tensors: memcpyTensors,
  };
}

/** Parse the Chrome-trace JSON array that ORT's profiler prints on endProfiling(). */
export function parseOrtTrace(lines) {
  const events = [];
  for (const line of lines) {
    const t = line.trim().replace(/,$/, "");
    if (!t.startsWith("{") || !t.includes('"ph"')) continue;
    try {
      events.push(JSON.parse(t));
    } catch { /* not a trace event */ }
  }
  return events;
}

/**
 * For each captured-log position, the ORT WebGL node that was running: the most recent
 * "ExecPlan ... Running op:<node>" line before it (verbose log level only).
 */
export function webglOpAt(lines) {
  const opAt = new Array(lines.length + 1);
  let current = null;
  for (let i = 0; i < lines.length; i++) {
    opAt[i] = current;
    const m = lines[i].match(/Running op:(\S+)/);
    if (m) current = m[1];
  }
  opAt[lines.length] = current;
  return opAt;
}
