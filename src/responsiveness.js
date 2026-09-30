// Group 4: does inference interfere with the rest of the page?
//
// The study measures deep-learning cost, not rendering, so the question here is how much of the
// main thread inference takes away from the page. Declared page workload (identical for every
// model/config):
//   - a button whose click handler updates a text node (so every interaction produces a paint);
//   - optionally (cfg.animate) a requestAnimationFrame loop redrawing a 480x270 2D canvas.
// run_benchmark.py clicks the button with real input events (CDP) at a fixed interval during
// both phases:
//   baseline   animation + clicks, no inference (cfg.baselineSeconds)
//   inference  animation + clicks + repeated session.run() (cfg.loadSeconds, at least one run)
//
// Pacing between inferences (cfg.pacing):
//   "yield"         yield to the browser (scheduler.yield) after every run: how a well-behaved app
//                   runs repeated inference on the main thread
//   "back_to_back"  await the next run immediately. For WASM and WebGL, ORT resolves run() without
//                   an intervening task, so this starves rendering and input for the whole phase
//
// Recorded (all browser measurements):
//   frames        every rAF timestamp -> frame intervals
//   interactions  Event Timing entries (pointerdown/pointerup/click): startTime, processingStart,
//                 processingEnd, duration. Chrome only reports events lasting >= 16 ms and rounds
//                 `duration` to 8 ms, so fast interactions are missing from this stream.
//   clicks        every click, measured in the page's own pointerdown listener: event timestamp,
//                 handler start (input delay) and the start of the next animation frame after the
//                 handler, taken when that frame's rAF callbacks run (a lower bound on time to the
//                 next paint). Complete and unthresholded.
//   loaf          Long Animation Frame entries (frames > 50 ms): duration, blockingDuration
import { yieldToBrowser } from "./timing.js";
import { makeRunner } from "./runner.js";

const sleep = ms => new Promise(r => setTimeout(r, ms));

export function startAnimation(canvas) {
  const ctx = canvas.getContext("2d");
  const frames = [];
  let running = true;
  const draw = ts => {
    if (!running) return;
    frames.push(ts);
    ctx.fillStyle = "#101820";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < 60; i++) {
      const x = (ts / 8 + i * 37) % canvas.width;
      const y = (i * 53 + Math.sin(ts / 300 + i) * 40 + canvas.height) % canvas.height;
      ctx.fillStyle = `hsl(${(i * 6 + ts / 20) % 360} 70% 60%)`;
      ctx.fillRect(x, y, 18, 18);
    }
    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
  return { frames, stop: () => { running = false; } };
}

export async function responsivenessMode(session, model, cfg, result) {
  const runner = await makeRunner(session, model, cfg);
  const canvas = document.getElementById("animCanvas");
  const button = document.getElementById("interactBtn");
  const counter = document.getElementById("interactCount");
  let clicks = 0;
  const clickLog = [];
  button.onclick = () => { counter.textContent = String(++clicks); };
  button.onpointerdown = e => {
    const rec = { event_ts: e.timeStamp, handler_start: performance.now(), next_frame: null };
    clickLog.push(rec);
    // performance.now() rather than the rAF timestamp: the latter is the frame's begin time,
    // which after a long task can predate the input event.
    requestAnimationFrame(() => { rec.next_frame = performance.now(); });
  };

  const interactions = [];
  const loaf = [];
  const eventObserver = new PerformanceObserver(list => {
    for (const e of list.getEntries()) {
      interactions.push({
        type: e.name, interaction_id: e.interactionId ?? null,
        start: e.startTime, processing_start: e.processingStart, processing_end: e.processingEnd, duration: e.duration,
      });
    }
  });
  eventObserver.observe({ type: "event", durationThreshold: 16, buffered: false });
  let loafObserver = null;
  if (PerformanceObserver.supportedEntryTypes.includes("long-animation-frame")) {
    loafObserver = new PerformanceObserver(list => {
      for (const e of list.getEntries()) loaf.push({ start: e.startTime, duration: e.duration, blocking_duration: e.blockingDuration });
    });
    loafObserver.observe({ type: "long-animation-frame", buffered: false });
  }

  // Warmup first so the inference phase measures steady-state interference, not compilation.
  for (let i = 0; i < cfg.warmup; i++) await runner.run();

  const anim = cfg.animate ? startAnimation(canvas) : { frames: [], stop() {} };
  const phases = {};
  window.__bench.interactionTarget = true;   // tells run_benchmark.py to start clicking

  phases.baseline = { start: performance.now() };
  await sleep((cfg.baselineSeconds ?? 5) * 1000);
  phases.baseline.end = performance.now();

  phases.inference = { start: performance.now() };
  const loadMs = (cfg.loadSeconds ?? 10) * 1000;
  const pacing = cfg.pacing ?? "yield";
  let yieldMechanism = null;
  let i = 0;
  do {
    result.runs.push({ phase: "load", i: i++, ...(await runner.run()) });
    if (pacing === "yield") yieldMechanism = await yieldToBrowser();
  } while (performance.now() - phases.inference.start < loadMs);
  phases.inference.end = performance.now();

  window.__bench.interactionTarget = false;
  await sleep(300);  // let the last event-timing / LoAF entries arrive
  anim.stop();
  eventObserver.disconnect();
  loafObserver?.disconnect();

  result.responsiveness = {
    workload: cfg.animate ? "rAF 2D canvas 480x270, 60 rects/frame; click handler updates one text node"
                          : "click handler updates one text node (no animation)",
    pacing, yield_mechanism: yieldMechanism,
    time_origin_epoch_ms: performance.timeOrigin,
    phases,
    frames: anim.frames,
    interactions,
    clicks: clickLog,
    loaf,
    loaf_supported: loafObserver !== null,
    clicks_handled: clicks,
  };
  return runner;
}
