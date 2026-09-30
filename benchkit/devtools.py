"""
DevTools-protocol collectors for diagnostic runs (profile mode, WebGL CPU split).

  cpu_profile  V8 sampling profile of the page's main thread. The page brackets the runs it wants
               with console.profile()/profileEnd(); with the Profiler domain enabled, Chrome sends
               the finished profile as Profiler.consoleProfileFinished. Saved as <prefix>.cpuprofile.
  trace        Chrome trace (all processes, GPU-side categories) for the whole experiment, read
               back as a stream after it ends. Saved as <prefix>.trace.json.gz. The page marks its
               phases with performance.mark/measure (blink.user_timing).

Collectors never raise into the experiment: failures are recorded in .status.
"""
import asyncio
import base64
import gzip
import json
from pathlib import Path

TRACE_CATEGORIES = ["gpu", "gpu.service", "disabled-by-default-gpu.service", "disabled-by-default-gpu.decoder", "gpu.angle",
                    "blink.user_timing", "toplevel", "__metadata"]


class DiagCollector:
    def __init__(self, prefix: Path, cpu_profile=False, trace=False, sampling_us=100):
        self.prefix = Path(prefix)
        self.cpu_profile, self.trace, self.sampling_us = cpu_profile, trace, sampling_us
        self.status = {}
        self._profiles = []
        self._trace_done = None

    async def setup(self, cdp):
        if self.cpu_profile:
            try:
                await cdp.send("Profiler.enable")
                await cdp.send("Profiler.setSamplingInterval", {"interval": self.sampling_us})
                cdp.on("Profiler.consoleProfileFinished", lambda ev: self._profiles.append(ev))
                self.status["cpu_profile"] = "enabled"
            except Exception as e:
                self.status["cpu_profile"] = f"failed: {type(e).__name__}: {e}"
        if self.trace:
            try:
                loop = asyncio.get_running_loop()
                self._trace_done = loop.create_future()
                cdp.on("Tracing.tracingComplete",
                       lambda ev: self._trace_done.done() or self._trace_done.set_result(ev))
                await cdp.send("Tracing.start", {
                    "transferMode": "ReturnAsStream", "streamFormat": "json",
                    "traceConfig": {"recordMode": "recordAsMuchAsPossible",
                                    "includedCategories": TRACE_CATEGORIES},
                })
                self.status["trace"] = "started"
            except Exception as e:
                self.status["trace"] = f"failed: {type(e).__name__}: {e}"
                self._trace_done = None

        async def teardown():
            await self._finish(cdp)
        return teardown

    async def _finish(self, cdp):
        if self.cpu_profile and self.status.get("cpu_profile") == "enabled":
            if self._profiles:
                path = self.prefix.with_suffix(".cpuprofile")
                # One profile per console.profile() pair; keep them all, in order.
                path.write_text(json.dumps([{"title": p.get("title"), "profile": p["profile"]} for p in self._profiles]),
                                encoding="utf-8")
                self.status["cpu_profile"] = f"saved {len(self._profiles)} profile(s) to {path.name}"
            else:
                self.status["cpu_profile"] = "no profile received"
        if self._trace_done is not None:
            try:
                await cdp.send("Tracing.end")
                ev = await asyncio.wait_for(self._trace_done, timeout=90)
                handle = ev["stream"]
                path = self.prefix.with_suffix(".trace.json.gz")
                with gzip.open(path, "wb") as f:
                    while True:
                        chunk = await cdp.send("IO.read", {"handle": handle, "size": 4 << 20})
                        data = chunk.get("data", "")
                        f.write(base64.b64decode(data) if chunk.get("base64Encoded") else data.encode("utf-8"))
                        if chunk.get("eof"):
                            break
                await cdp.send("IO.close", {"handle": handle})
                self.status["trace"] = f"saved to {path.name}"
            except Exception as e:
                self.status["trace"] = f"failed at end: {type(e).__name__}: {e}"
