import { createReadStream, statSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";
import { defineConfig } from "vite";

// Cross-origin isolation (COOP + COEP) gives the page a 5 µs performance.now() resolution
// (100 µs otherwise), SharedArrayBuffer for multi-threaded WASM, and
// performance.measureUserAgentSpecificMemory().
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

const publicDir = resolve("public");
const mimeTypes = {
  ".json": "application/json",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
};

// Benchmarks run against the production build (`npm run bench:serve`): no HMR client and no
// on-demand transforms in the measured page, and editing sources can't reload a running
// experiment. public/ (≈3 GB of models) is served in place instead of being copied into dist/.
function servePublicInPreview() {
  return {
    name: "serve-public-in-preview",
    configurePreviewServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = normalize(join(publicDir, decodeURIComponent(req.url.split("?")[0])));
        if (!path.startsWith(publicDir)) return next();
        let stat;
        try {
          stat = statSync(path);
        } catch {
          return next();
        }
        if (!stat.isFile()) return next();
        res.setHeader("Content-Type", mimeTypes[extname(path)] ?? "application/octet-stream");
        res.setHeader("Content-Length", stat.size);
        res.setHeader("Cache-Control", "no-store");
        for (const [k, v] of Object.entries(isolationHeaders)) res.setHeader(k, v);
        createReadStream(path).pipe(res);
      });
    },
  };
}

// `vite build --mode diag` (served by `vite preview --mode diag` on :4174): a diagnostic build for
// CPU profiling only. Nothing is minified and onnxruntime-web/all resolves to ORT's unminified
// ESM build, so V8 profiles show real function names. Benchmarks use the default build.
export default defineConfig(({ mode }) => ({
  plugins: [servePublicInPreview()],
  resolve: mode === "diag"
    ? { alias: [{ find: /^onnxruntime-web\/all$/, replacement: resolve("node_modules/onnxruntime-web/dist/ort.all.mjs") }] }
    : {},
  build: {
    copyPublicDir: false,
    ...(mode === "diag" ? { minify: false, outDir: "dist-diag" } : {}),
  },
  server: {
    host: true,
    port: 5173,
    headers: isolationHeaders,
  },
  preview: {
    port: mode === "diag" ? 4174 : 4173,
    strictPort: true,
    headers: isolationHeaders,
  },
}));
