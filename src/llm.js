// Group 3: token generation timing for decoder models (models.json entries with an "llm" spec).
//
// The available GPT-2 export has a fixed 128-token window and no past-key/value inputs, so this is
// greedy decoding WITHOUT a KV cache: every generated token is a full forward pass over the
// window, with the prompt left-aligned and padding masked out (attention_mask = 0). Prefill and
// decode therefore run the same computation; the first step is reported as time-to-first-token.
// Prompt tokens are seeded random ids (no tokenizer): the timing is what's measured, not the text.
//
// Per generation g and step s: dur_ms = session.run (logits output only) + argmax over the
// vocabulary for the last prompt/generated position, i.e. the real per-token latency of this loop.
import { ort } from "./ort-setup.js";
import { makeRunner } from "./runner.js";

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function argmaxRow(logits, row, vocab) {
  let best = 0;
  let bestVal = -Infinity;
  const base = row * vocab;
  for (let v = 0; v < vocab; v++) {
    const x = logits[base + v];
    if (x > bestVal) { bestVal = x; best = v; }
  }
  return best;
}

async function generate(session, spec, prompt, genTokens, seqLen, record) {
  const ids = new Int32Array(seqLen);
  const mask = new Float32Array(seqLen);
  ids.set(prompt);
  mask.fill(1, 0, prompt.length);
  let pos = prompt.length;
  const tokens = [];
  for (let s = 0; s < genTokens && pos < seqLen; s++) {
    const feeds = {
      [spec.input_ids]: new ort.Tensor("int32", ids, [1, seqLen]),
      [spec.attention_mask]: new ort.Tensor("float32", mask, [1, seqLen]),
    };
    const t0 = performance.now();
    const out = await session.run(feeds, [spec.logits]);
    const logits = out[spec.logits];
    const next = argmaxRow(logits.data, pos - 1, logits.dims[2]);
    const t1 = performance.now();
    record?.({ step: s, t_start_epoch_ms: performance.timeOrigin + t0, dur_ms: t1 - t0 });
    ids[pos] = next;
    mask[pos] = 1;
    tokens.push(next);
    pos++;
  }
  return tokens;
}

export async function llmMode(session, model, cfg, result) {
  const spec = model.llm;
  if (!spec) throw new Error(`model ${model.name} has no "llm" spec in models.json`);
  if (cfg.gpuIo) throw new Error("llm mode uses CPU I/O only (the argmax needs the logits on the CPU every step)");
  const seqLen = model.inputs.find(i => i.name === spec.input_ids).dims[1];
  const promptLen = Math.min(cfg.promptTokens ?? 32, seqLen - 1);
  const genTokens = Math.min(cfg.genTokens ?? 32, seqLen - promptLen);
  const rand = mulberry32(cfg.seed);
  const prompt = Array.from({ length: promptLen }, () => Math.floor(rand() * 1000));
  result.llm = {
    kv_cache: false, window: seqLen, prompt_tokens: promptLen, gen_tokens: genTokens,
    prompt: "seeded random token ids (no tokenizer)", fetched_outputs: [spec.logits],
    generations: [],
  };

  for (let w = 0; w < cfg.warmup; w++) await generate(session, spec, prompt, genTokens, seqLen);
  for (let g = 0; g < cfg.measure; g++) {
    const tokens = await generate(session, spec, prompt, genTokens, seqLen,
      rec => result.runs.push({ phase: "gen", i: result.runs.length, generation: g, ...rec }));
    result.llm.generations.push({ generation: g, tokens });
  }
  return makeRunner(session, model, { ...cfg, gpuIo: false });
}
