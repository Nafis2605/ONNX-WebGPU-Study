"""
Readback microbenchmark driver: GPU -> JS copy time per API path and payload size, without ORT
(src/readback_micro.js). One fresh Chrome Dev process per trial.

Output (<output-dir>/<timestamp>/): env.json, readback_micro.csv, experiments.jsonl

Usage:
    npm run bench:serve
    python run_readback_micro.py --trials 3 --reps 20
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from playwright.async_api import async_playwright

from benchkit.chrome import CHROME_FLAGS, chrome_page, find_chrome_dev
from benchkit.envinfo import host_info
from benchkit.results import ResultWriter


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:4173/")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--sizes-mib", type=int, nargs="+", default=[1, 4, 16, 20, 80],
                   help="payload sizes in MiB (20 and 80 MiB = DeepLabV3+'s output as fp32 and as RGBA fp32)")
    p.add_argument("--cooldown-s", type=float, default=5.0)
    p.add_argument("--executable-path", default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--output-dir", default="benchmark_results/readback_micro")
    return p.parse_args()


async def main():
    args = parse_args()
    chrome = args.executable_path or find_chrome_dev()
    if not chrome:
        sys.exit("Google Chrome Dev not found (set CHROME_DEV_PATH or pass --executable-path).")
    out = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = ResultWriter(out)
    writer.write_json("env.json", {"host": host_info(), "chrome_flags": CHROME_FLAGS, "args": vars(args)})
    print(f"Output: {out}")
    rows = []
    async with async_playwright() as pw:
        for trial in range(1, args.trials + 1):
            async with chrome_page(pw, chrome, args.url, args.headless, log_path=out / "logs" / f"t{trial}.log") as (page, version):
                res = await page.evaluate("cfg => window.__bench.readbackMicro(cfg)",
                                          {"sizesMiB": args.sizes_mib, "reps": args.reps, "warmup": args.warmup})
            res.update({"trial": trial, "browser_version": version})
            writer.append(res)
            rows += [{"trial": trial, **r} for r in res["records"]]
            errs = {k: v for k, v in res["env"].items() if "error" in k or "incomplete" in k}
            print(f"trial {trial}: {len(res['records'])} records, red_float_read={res['env'].get('red_float_read')} {errs or ''}")
            await asyncio.sleep(args.cooldown_s)
    pd.DataFrame(rows).to_csv(out / "readback_micro.csv", index=False)
    print(f"Done -> {out / 'readback_micro.csv'}")


if __name__ == "__main__":
    asyncio.run(main())
