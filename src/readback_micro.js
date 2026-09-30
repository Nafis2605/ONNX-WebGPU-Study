// Readback microbenchmark, independent of ORT: how long the browser takes to move N bytes of
// fp32 data from GPU memory into a JS ArrayBuffer, per API path.
//
//   webgl_sync_rgba   readPixels(RGBA, FLOAT) of an RGBA32F framebuffer, after a 1x1 read has
//                     drained the GPU (so the timed call is the transfer). ORT's WebGL path.
//   webgl_sync_red    the same from an R32F framebuffer read as (RED, FLOAT), if the
//                     implementation reports that as its read format (else not run)
//   webgl_pbo_rgba    readPixels into a PIXEL_PACK_BUFFER (issue), fence + poll (wait), then
//                     getBufferSubData (copy)
//   webgpu_map        copyBufferToBuffer into a MAP_READ buffer, queue drained, then
//                     mapAsync (map) + getMappedRange().slice() (copy)
//
// Every value is a performance.now() interval; bytes are the bytes each call returns.

const MiB = 1 << 20;
const WIDTH = 4096;

// Yield to the event loop without setTimeout's clamping (used while polling a fence).
function yieldTask() {
  return new Promise(resolve => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => resolve();
    ch.port2.postMessage(null);
  });
}

function makeTarget(gl, internalFormat, format, texels) {
  const h = Math.ceil(texels / WIDTH);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texStorage2D(gl.TEXTURE_2D, 1, internalFormat, WIDTH, h);
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  const complete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  const readFormat = gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_FORMAT);
  const readType = gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE);
  return { tex, fb, h, complete, readFormat, readType, format,
           dispose() { gl.deleteFramebuffer(fb); gl.deleteTexture(tex); } };
}

function fillAndDrain(gl, h) {
  // GPU work that writes the whole target, then a 1x1 read that waits for it.
  gl.viewport(0, 0, WIDTH, h);
  gl.clearColor(Math.random(), 0.25, 0.5, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  const t0 = performance.now();
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, new Float32Array(4));
  return performance.now() - t0;
}

async function webglPaths(sizes, reps, warmup, out, env) {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2");
  if (!gl) { env.webgl_error = "no webgl2 context"; return; }
  if (!gl.getExtension("EXT_color_buffer_float")) { env.webgl_error = "EXT_color_buffer_float unavailable"; return; }
  env.webgl_max_texture_size = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  const dbg = gl.getExtension("WEBGL_debug_renderer_info");
  env.webgl_renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);

  for (const mib of sizes) {
    const bytes = mib * MiB;
    const dst = new Float32Array(bytes / 4);

    // RGBA32F, synchronous readPixels.
    const rgba = makeTarget(gl, gl.RGBA32F, gl.RGBA, bytes / 16);
    if (!rgba.complete) { env[`rgba_incomplete_${mib}`] = true; rgba.dispose(); continue; }
    for (let r = 0; r < warmup + reps; r++) {
      const drain = fillAndDrain(gl, rgba.h);
      const t0 = performance.now();
      gl.readPixels(0, 0, WIDTH, bytes / 16 / WIDTH, gl.RGBA, gl.FLOAT, dst);
      const t1 = performance.now();
      if (r >= warmup) out.push({ path: "webgl_sync_rgba", bytes, rep: r - warmup, total_ms: t1 - t0, drain_ms: drain });
    }

    // Unpacking the RGBA readback to one value per texel. "decode_ort_filter" is ORT WebGL's
    // RGBAFloatDataEncoder.decode for 1-channel data, verbatim (a per-element callback);
    // "decode_strided_loop" produces the same array with a plain loop. Any GC these trigger
    // synchronously is inside the timed interval. `bytes` is the RGBA input size.
    let sink = 0;
    for (let r = 0; r < warmup + reps; r++) {
      const t0 = performance.now();
      const filtered = dst.filter((_value, index) => index % 4 === 0);
      const t1 = performance.now();
      sink += filtered[r % filtered.length];
      if (r >= warmup) out.push({ path: "decode_ort_filter", bytes, rep: r - warmup, total_ms: t1 - t0 });
    }
    for (let r = 0; r < warmup + reps; r++) {
      const t0 = performance.now();
      const n = dst.length / 4;
      const strided = new Float32Array(n);
      for (let i = 0, j = 0; j < n; i += 4, j++) strided[j] = dst[i];
      const t1 = performance.now();
      sink += strided[r % n];
      if (r >= warmup) out.push({ path: "decode_strided_loop", bytes, rep: r - warmup, total_ms: t1 - t0 });
    }
    env.sink = (env.sink ?? 0) + sink;   // keeps the results observable

    // RGBA32F through a PIXEL_PACK_BUFFER.
    const pbo = gl.createBuffer();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, bytes, gl.STREAM_READ);
    for (let r = 0; r < warmup + reps; r++) {
      // The drain read goes to client memory, which needs the PBO unbound.
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      const drain = fillAndDrain(gl, rgba.h);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
      const t0 = performance.now();
      gl.readPixels(0, 0, WIDTH, bytes / 16 / WIDTH, gl.RGBA, gl.FLOAT, 0);
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const t1 = performance.now();
      let status;
      while ((status = gl.clientWaitSync(sync, 0, 0)) === gl.TIMEOUT_EXPIRED) await yieldTask();
      const t2 = performance.now();
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, dst);
      const t3 = performance.now();
      gl.deleteSync(sync);
      if (r >= warmup) out.push({ path: "webgl_pbo_rgba", bytes, rep: r - warmup, total_ms: t3 - t0, drain_ms: drain,
                                  issue_ms: t1 - t0, wait_ms: t2 - t1, copy_ms: t3 - t2, wait_status: status });
    }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    gl.deleteBuffer(pbo);
    rgba.dispose();

    // R32F read as (RED, FLOAT), only if that is the implementation's read format.
    const red = makeTarget(gl, gl.R32F, gl.RED, bytes / 4);
    env.red_float_read = red.complete && red.readFormat === gl.RED && red.readType === gl.FLOAT;
    if (env.red_float_read) {
      for (let r = 0; r < warmup + reps; r++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, red.fb);
        gl.viewport(0, 0, WIDTH, red.h);
        gl.clearColor(Math.random(), 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        const d0 = performance.now();
        gl.readPixels(0, 0, 1, 1, gl.RED, gl.FLOAT, new Float32Array(1));
        const t0 = performance.now();
        gl.readPixels(0, 0, WIDTH, bytes / 4 / WIDTH, gl.RED, gl.FLOAT, dst);
        const t1 = performance.now();
        if (r >= warmup) out.push({ path: "webgl_sync_red", bytes, rep: r - warmup, total_ms: t1 - t0, drain_ms: t0 - d0 });
      }
    }
    red.dispose();
    // Any GL error invalidates this size's WebGL records; the analysis drops them.
    const err = gl.getError();
    if (err !== gl.NO_ERROR) env[`gl_error_${mib}mib`] = err;
  }
  const ext = gl.getExtension("WEBGL_lose_context");
  if (ext) ext.loseContext();
}

async function webgpuPath(sizes, reps, warmup, out, env) {
  if (!navigator.gpu) { env.webgpu_error = "navigator.gpu unavailable"; return; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) { env.webgpu_error = "no adapter"; return; }
  const device = await adapter.requestDevice({
    requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize },
  });
  for (const mib of sizes) {
    const bytes = mib * MiB;
    const src = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(src, 0, new Float32Array(bytes / 4).fill(0.5));
    const dstBuf = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    await device.queue.onSubmittedWorkDone();
    for (let r = 0; r < warmup + reps; r++) {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, dstBuf, 0, bytes);
      const t0 = performance.now();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      const t1 = performance.now();
      await dstBuf.mapAsync(GPUMapMode.READ);
      const t2 = performance.now();
      const copy = dstBuf.getMappedRange().slice(0);
      const t3 = performance.now();
      dstBuf.unmap();
      if (r >= warmup && copy.byteLength === bytes) {
        out.push({ path: "webgpu_map", bytes, rep: r - warmup, total_ms: t3 - t1, drain_ms: t1 - t0,
                   wait_ms: t2 - t1, copy_ms: t3 - t2 });
      }
    }
    src.destroy();
    dstBuf.destroy();
  }
  device.destroy();
}

/** cfg: {sizesMiB, reps, warmup}. Returns {env, records}. */
export async function readbackMicro(cfg = {}) {
  const sizes = cfg.sizesMiB ?? [1, 4, 16, 20, 80];
  const reps = cfg.reps ?? 20;
  const warmup = cfg.warmup ?? 2;
  const records = [];
  const env = { sizes_mib: sizes, reps, warmup, width: WIDTH };
  await webglPaths(sizes, reps, warmup, records, env);
  await webgpuPath(sizes, reps, warmup, records, env);
  return { env, records };
}
