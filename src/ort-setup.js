// ONNX Runtime Web setup.
//
// Two bundles can be loaded (one per page, chosen by the experiment config):
//   "jsep"    onnxruntime-web/all     WASM + WebGL + WebGPU through JSEP (the JS WebGPU EP).
//             This is the only entry that registers WebGL alongside the others; the default
//             "onnxruntime-web" entry does NOT register WebGL, so a ["webgl", ...] session
//             created through it silently resolves to another EP.
//   "native"  onnxruntime-web/webgpu  WASM + the native (C++) WebGPU EP, asyncify build.
// `ort` is a live binding: it's null until loadOrt() has run.

export let ort = null;
export let ortBundle = null;

const wasmBasePath = `${import.meta.env.BASE_URL}ort-wasm/`;
const WASM_FILES = {
  jsep: "ort-wasm-simd-threaded.jsep",
  native: "ort-wasm-simd-threaded.asyncify",
};

async function loadOrt(impl) {
  if (ort) {
    if (ortBundle !== impl) throw new Error(`ORT bundle "${ortBundle}" already loaded; this page can't switch to "${impl}"`);
    return;
  }
  ort = impl === "native" ? await import("onnxruntime-web/webgpu") : await import("onnxruntime-web/all");
  ortBundle = impl;
}

/**
 * Load and configure the runtime once per page, before the first session is created.
 * @param {{epImpl?: "jsep"|"native", wasmThreads?: number, webglPack?: boolean}} opts
 */
export async function configureOrt(opts = {}) {
  const impl = opts.epImpl ?? "jsep";
  if (ortBundle === impl) return ort;
  await loadOrt(impl);

  // The glue .mjs is loaded through a Blob URL because Vite refuses to import JavaScript from
  // /public during development.
  const base = `${wasmBasePath}${WASM_FILES[impl]}`;
  const res = await fetch(`${base}.mjs`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ORT WASM module (${res.status})`);
  const mjsBlobUrl = URL.createObjectURL(new Blob([await res.text()], { type: "text/javascript" }));
  ort.env.wasm.wasmPaths = { mjs: mjsBlobUrl, wasm: `${base}.wasm` };

  // More than one thread needs SharedArrayBuffer, i.e. cross-origin isolation (COOP/COEP).
  ort.env.wasm.numThreads = opts.wasmThreads ?? 1;
  ort.env.wasm.simd = true;
  ort.env.wasm.proxy = false;
  ort.env.logLevel = "warning";

  ort.env.webgpu.powerPreference = "high-performance";

  if (impl === "jsep") {
    ort.env.webgl.contextId = "webgl2";
    if (typeof opts.webglPack === "boolean") ort.env.webgl.pack = opts.webglPack;
  }
  return ort;
}

/** Execution providers per backend. Exactly one EP each: no list-level fallback. */
export function executionProviders(backend) {
  switch (backend) {
    case "wasm":
    case "webgpu":
    case "webgl":
      return [backend];
    default:
      throw new Error(`Unsupported backend: ${backend}`);
  }
}

/**
 * Session options for a strict, single-backend session.
 * For WebGPU, `session.disable_cpu_ep_fallback` makes session creation fail if ORT would
 * assign any node to the CPU EP, unless `allowCpuNodes` is set.
 *   gpuIo         WebGPU outputs stay on the GPU (preferredOutputLocation "gpu-buffer")
 *   graphCapture  WebGPU graph capture/replay (requires gpuIo and static shapes)
 */
export function sessionOptions(backend, {
  allowCpuNodes = false, enableProfiling = false, logSeverityLevel, gpuIo = false, graphCapture = false,
} = {}) {
  const opts = {
    executionProviders: executionProviders(backend),
    graphOptimizationLevel: "all",
    // WASM / WebGPU: ORT's C++ profiler for the whole session. The WebGL backend's own profiler
    // breaks inference once started in this ORT build ("reading 'inputTypes'"), so WebGL draws
    // are timed by instrument.js instead.
    enableProfiling: enableProfiling && backend !== "webgl",
  };
  if (backend === "webgpu" && !allowCpuNodes) {
    opts.extra = { session: { disable_cpu_ep_fallback: "1" } };
  }
  if (backend === "webgpu" && gpuIo) opts.preferredOutputLocation = "gpu-buffer";
  if (backend === "webgpu" && graphCapture) opts.enableGraphCapture = true;
  if (logSeverityLevel !== undefined) opts.logSeverityLevel = logSeverityLevel;
  return opts;
}
