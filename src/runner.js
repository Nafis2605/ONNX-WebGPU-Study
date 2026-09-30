// One timed inference, the same way in every mode.
//
// I/O variants (from the experiment config):
//   CPU I/O (default)  inputs are CPU tensors (uploaded by ORT on every run) and requested outputs
//                      are read back to the CPU inside session.run().
//   GPU I/O (gpuIo)    WebGPU only: inputs are uploaded to GPU buffers ONCE, outside timing, and
//                      outputs stay on the GPU (preferredOutputLocation "gpu-buffer"). session.run()
//                      then returns after submission, so each run also downloads the primary output
//                      (getData) to make the result usable. dur_ms covers both; submit_ms and
//                      readback_ms split them.
//   outputs "primary"  fetch only the model's primary output (models.json logits_outputs[0], or
//                      the first graph output) instead of every output.
//
// dur_ms is always "request start to primary result available on the CPU", so it compares across
// configs.
import { ort } from "./ort-setup.js";
import { makeFeeds } from "./feeds.js";

export function primaryOutput(session, model) {
  return model.logits_outputs?.[0] ?? session.outputNames[0];
}

function uploadToGpu(feeds) {
  const device = ort.env.webgpu.device;
  if (!device) throw new Error("GPU I/O needs ort.env.webgpu.device (create the WebGPU session first)");
  const gpuFeeds = {};
  const buffers = [];
  for (const [name, t] of Object.entries(feeds)) {
    const bytes = t.data.byteLength;
    const buffer = device.createBuffer({
      size: Math.ceil(bytes / 16) * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, t.data.buffer, t.data.byteOffset, bytes);
    gpuFeeds[name] = ort.Tensor.fromGpuBuffer(buffer, { dataType: t.type, dims: t.dims });
    buffers.push(buffer);
  }
  return { gpuFeeds, buffers };
}

/**
 * Build a runner for (session, model, config), optionally with overridden input dims.
 * runner.run() -> {t_start_epoch_ms, dur_ms, [submit_ms, readback_ms]}
 * runner.outputs() -> {name: CPU tensor} for correctness checks (not timed)
 */
export async function makeRunner(session, model, cfg, dimsOverride = null) {
  const { feeds, spec } = makeFeeds(session, model, cfg.seed, dimsOverride);
  const primary = primaryOutput(session, model);
  const fetches = cfg.outputs === "primary" ? [primary] : undefined;

  if (!cfg.gpuIo) {
    return {
      feeds, spec, primary,
      async run() {
        const t0 = performance.now();
        await session.run(feeds, fetches);
        const t1 = performance.now();
        return { t_start_epoch_ms: performance.timeOrigin + t0, dur_ms: t1 - t0 };
      },
      async outputs() {
        return session.run(feeds, fetches);
      },
      dispose() {},
    };
  }

  const { gpuFeeds, buffers } = uploadToGpu(feeds);
  // With graph capture ORT owns and reuses the output buffers of the captured graph.
  const disposeOutputs = !cfg.graphCapture;
  return {
    feeds, spec, primary,
    async run() {
      const t0 = performance.now();
      const out = await session.run(gpuFeeds, fetches);
      const t1 = performance.now();
      await out[primary].getData();
      const t2 = performance.now();
      if (disposeOutputs) for (const o of Object.values(out)) o.dispose();
      return { t_start_epoch_ms: performance.timeOrigin + t0, dur_ms: t2 - t0, submit_ms: t1 - t0, readback_ms: t2 - t1 };
    },
    async outputs() {
      const out = await session.run(gpuFeeds, fetches);
      const cpu = {};
      for (const [name, t] of Object.entries(out)) {
        cpu[name] = new ort.Tensor(t.type, await t.getData(), t.dims);
        if (disposeOutputs) t.dispose();
      }
      return cpu;
    },
    dispose() {
      for (const b of buffers) b.destroy();
    },
  };
}

export async function runnerRuns(runner, phase, count, records, extra = {}) {
  for (let i = 0; i < count; i++) {
    records.push({ phase, i, ...extra, ...(await runner.run()) });
  }
}
