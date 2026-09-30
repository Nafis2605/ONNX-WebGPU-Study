// Timed inference: nothing but performance.now() around the awaited session.run().

export async function timedRun(session, feeds) {
  const t0 = performance.now();
  await session.run(feeds);
  const t1 = performance.now();
  return { t_start_epoch_ms: performance.timeOrigin + t0, dur_ms: t1 - t0 };
}

export async function timedRuns(session, feeds, phase, count, records) {
  for (let i = 0; i < count; i++) {
    records.push({ phase, i, ...(await timedRun(session, feeds)) });
  }
}

/**
 * Give the main thread back to the browser (input, rendering) between inferences.
 * scheduler.yield() where available (Chrome 129+), otherwise a MessageChannel task (no 4 ms
 * timer clamping). Returns the mechanism used, so results record it.
 */
export async function yieldToBrowser() {
  if (globalThis.scheduler?.yield) {
    await scheduler.yield();
    return "scheduler.yield";
  }
  await new Promise(r => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => r();
    ch.port2.postMessage(null);
  });
  return "message_channel";
}
