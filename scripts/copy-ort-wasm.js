import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");

const srcDir = path.join(rootDir, "node_modules/onnxruntime-web/dist");
const destDir = path.join(rootDir, "public/ort-wasm");

const filesToCopy = [
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm"
];

async function main() {
  await fs.mkdir(destDir, { recursive: true });

  const missing = await Promise.all(
    filesToCopy.map(async file => {
      const src = path.join(srcDir, file);
      try {
        await fs.access(src);
        return null;
      } catch {
        return file;
      }
    })
  );

  const missingFiltered = missing.filter(Boolean);
  if (missingFiltered.length) {
    throw new Error(
      `The following ORT WASM assets are missing under ${srcDir}: ${missingFiltered.join(", ")}`
    );
  }

  for (const file of filesToCopy) {
    const src = path.join(srcDir, file);
    const dest = path.join(destDir, file);
    await fs.copyFile(src, dest);
    console.log(`Copied ${file}`);
  }

  console.log(`\nCopied ${filesToCopy.length} ORT WASM assets into ${destDir}`);
}

main().catch(err => {
  console.error(err.message);
  process.exitCode = 1;
});
