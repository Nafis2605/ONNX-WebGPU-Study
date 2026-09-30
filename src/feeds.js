// Deterministic input tensors built from the models.json input spec.
import { ort } from "./ort-setup.js";

/** mulberry32: small, fast, seedable PRNG returning floats in [0, 1). */
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

const numel = dims => dims.reduce((a, b) => a * b, 1);

/**
 * Fill rules:
 *   default float  -> uniform [-1, 1)
 *   default int    -> uniform integers in [0, 1000)  (valid token ids for BERT/GPT-2 vocabularies)
 *   fill: "ones"   -> all ones (e.g. attention masks)
 *   fill: "zeros"  -> all zeros
 *   fill: {int_range: [lo, hi]} -> uniform integers in [lo, hi)
 */
function makeData(type, size, fill, rand) {
  const intRange = fill && typeof fill === "object" && fill.int_range ? fill.int_range : [0, 1000];
  const constant = fill === "ones" ? 1 : fill === "zeros" ? 0 : null;
  const randInt = () => intRange[0] + Math.floor(rand() * (intRange[1] - intRange[0]));

  switch (type) {
    case "float32": {
      const a = new Float32Array(size);
      for (let i = 0; i < size; i++) a[i] = constant ?? rand() * 2 - 1;
      return a;
    }
    case "int32": {
      const a = new Int32Array(size);
      for (let i = 0; i < size; i++) a[i] = constant ?? randInt();
      return a;
    }
    case "int64": {
      const a = new BigInt64Array(size);
      for (let i = 0; i < size; i++) a[i] = BigInt(constant ?? randInt());
      return a;
    }
    case "uint8":
    case "bool": {
      const a = new Uint8Array(size);
      const hi = type === "bool" ? 2 : 256;
      for (let i = 0; i < size; i++) a[i] = constant ?? Math.floor(rand() * hi);
      return a;
    }
    default:
      throw new Error(`Input type '${type}' is not supported by the feed generator`);
  }
}

/**
 * Build feeds for `modelCfg.inputs`, checking them against the session's declared inputs.
 * @returns {{feeds: Record<string, ort.Tensor>, spec: Array<{name, type, dims}>}}
 */
export function makeFeeds(session, modelCfg, seed, dimsOverride = null) {
  const declared = session.inputNames ?? [];
  const cfgInputs = modelCfg.inputs ?? [];
  const missing = declared.filter(n => !cfgInputs.some(i => i.name === n));
  if (missing.length) {
    throw new Error(`models.json has no spec for session input(s): ${missing.join(", ")}`);
  }

  const rand = mulberry32(seed);
  const feeds = {};
  const spec = [];
  for (const name of declared) {
    const inp = cfgInputs.find(i => i.name === name);
    const dims = dimsOverride?.[name] ?? inp.dims;
    feeds[name] = new ort.Tensor(inp.type, makeData(inp.type, numel(dims), inp.fill, rand), dims);
    spec.push({ name, type: inp.type, dims });
  }
  return { feeds, spec };
}
