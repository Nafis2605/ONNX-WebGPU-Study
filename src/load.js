// Group 8: open-loop load. Requests arrive by a seeded Poisson process at each rate in
// cfg.rates (requests/s) for cfg.loadSeconds, independent of how fast they complete. One session
// serves them FIFO, one at a time (ORT Web runs one inference per session at a time).
//
// Latency is measured from the SCHEDULED arrival time, so time a request spends waiting while
// the main thread is busy counts against it (no coordinated omission). A request is rejected when
// the queue already holds cfg.maxQueue requests at the moment its arrival is processed. After
// the arrival window closes, the queue drains for up to cfg.drainSeconds; what's left counts as
// unfinished.
//
// Optional: cfg.withAnimation runs the responsiveness page workload (rAF canvas) so frame timing
// under each load level is recorded too.
import { yieldToBrowser } from "./timing.js";
import { makeRunner } from "./runner.js";

const sleep = ms => new Promise(r => setTimeout(r, ms));

function expSampler(seed) {
  let a = seed >>> 0;
  const rand = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return rate => -Math.log(1 - rand()) / rate;
}

async function runRate(runner, rate, cfg, seed, requests) {
  const windowMs = cfg.loadSeconds * 1000;
  const next = expSampler(seed);
  const schedule = [];
  for (let t = next(rate) * 1000; t < windowMs; t += next(rate) * 1000) schedule.push(t);

  const t0 = performance.now();
  const queue = [];
  let arrivalsDone = false;
  let wake = null;
  const rows = schedule.map((offset, id) => ({ rate, id, arrival: t0 + offset }));

  // Arrival generator: processes every arrival that's due, in order, whenever it gets the main thread.
  const arrivals = (async () => {
    for (const req of rows) {
      const wait = req.arrival - performance.now();
      if (wait > 0) await sleep(wait);
      req.enqueued = performance.now();
      req.queue_len_at_arrival = queue.length;
      if (queue.length >= cfg.maxQueue) {
        req.rejected = true;
      } else {
        queue.push(req);
        wake?.();
      }
    }
    arrivalsDone = true;
    wake?.();
  })();

  const drainDeadline = () => t0 + windowMs + cfg.drainSeconds * 1000;
  while (true) {
    if (!queue.length) {
      if (arrivalsDone) break;
      await new Promise(r => { wake = r; });
      wake = null;
      continue;
    }
    if (performance.now() > drainDeadline()) break;
    const req = queue.shift();
    req.start = performance.now();
    await runner.run();
    req.end = performance.now();
    // Let arrivals, input and rendering run between requests (see responsiveness.js pacing).
    if (cfg.pacing !== "back_to_back") await yieldToBrowser();
  }
  await arrivals;
  for (const req of rows) {
    requests.push({
      rate, id: req.id,
      arrival_ms: req.arrival - t0,
      queue_len_at_arrival: req.queue_len_at_arrival,
      // How late the harness noticed the arrival (timer wake-up / busy main thread); included in latency.
      dispatch_lag_ms: req.enqueued - req.arrival,
      rejected: !!req.rejected,
      unfinished: !req.rejected && req.end === undefined,
      queue_wait_ms: req.start !== undefined ? req.start - req.arrival : null,
      service_ms: req.end !== undefined ? req.end - req.start : null,
      latency_ms: req.end !== undefined ? req.end - req.arrival : null,
      t_arrival_epoch_ms: performance.timeOrigin + req.arrival,
    });
  }
  return { rate, window_start_epoch_ms: performance.timeOrigin + t0, arrivals: rows.length };
}

export async function loadMode(session, model, cfg, result) {
  const runner = await makeRunner(session, model, cfg);
  for (let i = 0; i < cfg.warmup; i++) await runner.run();

  let anim = null;
  if (cfg.withAnimation) {
    const { startAnimation } = await import("./responsiveness.js");
    anim = startAnimation(document.getElementById("animCanvas"));
  }
  const requests = [];
  const windows = [];
  for (const [k, rate] of (cfg.rates ?? []).entries()) {
    const w = await runRate(runner, rate, cfg, cfg.seed * 1000 + k, requests);
    if (anim) w.frames = anim.frames.splice(0);
    windows.push(w);
    await sleep(500);
  }
  anim?.stop();
  result.load = {
    rates: cfg.rates, load_seconds: cfg.loadSeconds, max_queue: cfg.maxQueue, drain_seconds: cfg.drainSeconds,
    pacing: cfg.pacing ?? "yield",
    with_animation: !!cfg.withAnimation, time_origin_epoch_ms: performance.timeOrigin, windows, requests,
  };
  return runner;
}
