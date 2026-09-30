// Page entry point: manual UI plus the `window.__bench` API that run_benchmark.py drives.
import { runExperiment, probeEnv, setLogger } from "./bench.js";
import * as ortSetup from "./ort-setup.js";
import { readbackMicro } from "./readback_micro.js";

const $ = id => document.getElementById(id);
const statusEl = $("status");
const resultsBody = $("resultsBody");

function logStatus(msg) {
  statusEl.textContent += msg + "\n";
  console.log(msg);
}
setLogger(logStatus);

async function loadModelsConfig() {
  const res = await fetch("/models/models.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load models.json (${res.status})`);
  return (await res.json()).models ?? [];
}

// Display-only convenience; the paper's statistics are computed offline from the raw runs.
function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function addResultRow(r) {
  const measure = r.runs.filter(x => x.phase === "measure").map(x => x.dur_ms).sort((a, b) => a - b);
  const first = r.runs[0]?.dur_ms;
  const fmt = v => (v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(2));
  const tr = document.createElement("tr");
  const cells = [
    r.model, r.backend, r.status, r.verification?.detail ?? r.failure?.message ?? "",
    fmt(r.fetch_ms), fmt(r.session_create_ms), fmt(first), String(measure.length),
    fmt(quantile(measure, 0.5)), fmt(quantile(measure, 0.95)),
  ];
  for (const c of cells) {
    const td = document.createElement("td");
    td.textContent = c;
    tr.appendChild(td);
  }
  resultsBody.appendChild(tr);
}

async function main() {
  const models = await loadModelsConfig();
  const select = $("modelsSelect");
  for (const m of models) select.add(new Option(m.name, m.name));

  let stopRequested = false;

  $("runBtn").onclick = async () => {
    const names = Array.from(select.selectedOptions).map(o => o.value);
    if (!names.length) { logStatus("Select at least one model."); return; }
    stopRequested = false;
    $("runBtn").disabled = true;
    $("stopBtn").disabled = false;
    statusEl.textContent = "";
    resultsBody.innerHTML = "";
    for (const name of names) {
      if (stopRequested) break;
      addResultRow(await runExperiment({
        model: models.find(m => m.name === name),
        backend: $("backendSelect").value,
        warmup: Number($("warmupRuns").value),
        measure: Number($("measureRuns").value),
        seed: 1,
      }));
    }
    $("runBtn").disabled = false;
    $("stopBtn").disabled = true;
  };
  $("stopBtn").onclick = () => { stopRequested = true; logStatus("Stopping after the current model..."); };

  // `ort` and `configureOrt` are exposed for diagnostics scripts only.
  window.__bench = {
    ready: true, probeEnv, runExperiment, models, configureOrt: ortSetup.configureOrt, readbackMicro,
    get ort() { return ortSetup.ort; },
  };
}

main().catch(e => {
  logStatus(`Fatal error: ${e.message}`);
  window.__bench = { ready: false, error: String(e.message ?? e) };
});
