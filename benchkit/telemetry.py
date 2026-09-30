"""NVIDIA GPU telemetry via NVML, sampled in background threads during each experiment.

What each stream measures (and its real resolution on this system, RTX 3060 / WDDM):
    samples       every `interval_s` (default 50 ms): power (the driver refreshes it about every
                  390 ms), cumulative energy counter (refreshes about every 96 ms), SM/memory/graphics
                  clocks, device memory used, temperature, P-state, clock-event (throttle) reasons.
    util          the driver's own timestamped utilization samples (GPU = fraction of time a kernel
                  was executing; memory = memory-controller busy fraction), about 200 ms apart,
                  fetched with nvmlDeviceGetSamples so no sample is missed or duplicated.
    pcie          PCIe TX/RX throughput (KB/s). Each NVML query blocks for about 20-30 ms while the
                  driver measures, so this runs in its own thread.

Per-process GPU memory isn't available under WDDM, so memory is device-wide (compare with the
idle baseline). All timestamps are Unix epoch milliseconds, the same clock as the browser's
performance.timeOrigin + performance.now().
"""
import threading
import time

import psutil
import pynvml as nv

CLOCK_EVENT_REASONS = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}


def decode_clock_reasons(mask):
    return "|".join(name for bit, name in CLOCK_EVENT_REASONS.items() if mask & bit) or "none"


class NvmlTelemetry:
    def __init__(self, interval_s=0.05, pcie=True, device_index=0):
        nv.nvmlInit()
        self.h = nv.nvmlDeviceGetHandleByIndex(device_index)
        self.interval_s = interval_s
        self.pcie_enabled = pcie
        self._stop = threading.Event()
        self._threads = []
        self.marks = []

    def _read_clock_reasons(self):
        fn = getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None) or nv.nvmlDeviceGetCurrentClocksThrottleReasons
        return fn(self.h)

    def _sample_loop(self):
        h = self.h
        last_util_ts = {nv.NVML_GPU_UTILIZATION_SAMPLES: 0, nv.NVML_MEMORY_UTILIZATION_SAMPLES: 0}
        next_util_poll = 0.0
        while not self._stop.is_set():
            t = time.time() * 1000.0
            try:
                self.samples.append({
                    "t_epoch_ms": t,
                    "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
                    "energy_mj": nv.nvmlDeviceGetTotalEnergyConsumption(h),
                    "sm_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                    "mem_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                    "gr_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_GRAPHICS),
                    "mem_used_mb": nv.nvmlDeviceGetMemoryInfo(h).used / 2**20,
                    "temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                    "pstate": nv.nvmlDeviceGetPerformanceState(h),
                    "clock_reasons": self._read_clock_reasons(),
                })
            except nv.NVMLError as e:
                self.errors.append(f"{t:.0f} sample: {e}")
            if time.monotonic() >= next_util_poll:
                self._poll_util(last_util_ts)
                next_util_poll = time.monotonic() + 1.0
            self._stop.wait(self.interval_s)
        self._poll_util(last_util_ts)

    def _poll_util(self, last_ts):
        for kind, name in ((nv.NVML_GPU_UTILIZATION_SAMPLES, "gpu"), (nv.NVML_MEMORY_UTILIZATION_SAMPLES, "mem")):
            try:
                _, samples = nv.nvmlDeviceGetSamples(self.h, kind, last_ts[kind])
            except nv.NVMLError as e:
                self.errors.append(f"util {name}: {e}")
                continue
            for s in samples:
                if s.timeStamp > last_ts[kind]:
                    self.util.append({"t_epoch_ms": s.timeStamp / 1000.0, "kind": name, "percent": s.sampleValue.uiVal})
            if samples:
                last_ts[kind] = max(last_ts[kind], max(s.timeStamp for s in samples))

    def _pcie_loop(self):
        while not self._stop.is_set():
            try:
                t0 = time.time() * 1000.0
                tx = nv.nvmlDeviceGetPcieThroughput(self.h, nv.NVML_PCIE_UTIL_TX_BYTES)
                rx = nv.nvmlDeviceGetPcieThroughput(self.h, nv.NVML_PCIE_UTIL_RX_BYTES)
                self.pcie.append({"t_epoch_ms": t0, "tx_kbps": tx, "rx_kbps": rx})
            except nv.NVMLError as e:
                self.errors.append(f"pcie: {e}")
                self._stop.wait(1.0)

    def watch_chrome(self, user_data_dir):
        """Sample memory of the Chrome processes launched with this (unique) --user-data-dir."""
        self._chrome_udd = user_data_dir

    def _chrome_loop(self, interval_s=0.1, refresh_s=2.0):
        """Per-process memory of the watched Chrome instance, summed by process type.

        private_mb is the process's private committed memory (Windows "private bytes"),
        rss_mb its working set, cpu_user_s / cpu_system_s its cumulative CPU time (summed over the
        processes of that type), so CPU-seconds spent in a window = difference across the window.
        Also samples system-wide CPU utilization (all processes) to expose background load. The GPU process holds Chrome's side of GPU allocations;
        WDDM exposes no per-process GPU memory, so device memory comes from NVML instead.
        """
        procs, next_refresh = [], 0.0
        while not self._stop.is_set():
            udd = getattr(self, "_chrome_udd", None)
            if udd and time.monotonic() >= next_refresh:
                procs = []
                for p in psutil.process_iter(["name", "cmdline"]):
                    try:
                        cmd = p.info["cmdline"] or []
                        if p.info["name"] and p.info["name"].lower().startswith("chrome") and any(udd in a for a in cmd):
                            kind = next((a.split("=", 1)[1] for a in cmd if a.startswith("--type=")), "browser")
                            procs.append((p, kind))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                next_refresh = time.monotonic() + refresh_s
            if procs:
                t = time.time() * 1000.0
                by_kind = {}
                for p, kind in procs:
                    try:
                        m = p.memory_info()
                        ct = p.cpu_times()
                        agg = by_kind.setdefault(kind, {"rss_mb": 0.0, "private_mb": 0.0, "cpu_user_s": 0.0,
                                                        "cpu_system_s": 0.0, "processes": 0})
                        agg["rss_mb"] += m.rss / 2**20
                        agg["private_mb"] += getattr(m, "private", 0) / 2**20
                        agg["cpu_user_s"] += ct.user
                        agg["cpu_system_s"] += ct.system
                        agg["processes"] += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                for kind, agg in by_kind.items():
                    self.chrome.append({"t_epoch_ms": t, "process_type": kind, **agg})
            self.system_cpu.append({"t_epoch_ms": time.time() * 1000.0, "cpu_percent": psutil.cpu_percent(interval=None)})
            self._stop.wait(interval_s)

    def start(self):
        self.samples, self.util, self.pcie, self.errors, self.marks, self.chrome = [], [], [], [], [], []
        self.system_cpu = []
        psutil.cpu_percent(interval=None)  # prime the counter
        self._chrome_udd = None
        self._stop.clear()
        self._threads = [threading.Thread(target=self._sample_loop, daemon=True),
                         threading.Thread(target=self._chrome_loop, daemon=True)]
        if self.pcie_enabled:
            self._threads.append(threading.Thread(target=self._pcie_loop, daemon=True))
        for th in self._threads:
            th.start()

    def mark(self, name):
        """Record a named instant (e.g. baseline end, browser launch) on the telemetry clock."""
        self.marks.append({"t_epoch_ms": time.time() * 1000.0, "mark": name})

    def stop(self):
        self._stop.set()
        for th in self._threads:
            th.join(timeout=5)
        return {"samples": self.samples, "util": self.util, "pcie": self.pcie,
                "chrome": self.chrome, "system_cpu": self.system_cpu, "marks": self.marks, "errors": self.errors}
