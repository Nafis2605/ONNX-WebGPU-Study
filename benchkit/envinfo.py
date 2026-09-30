"""Host environment capture for env.json."""
import platform
import subprocess
import sys
from importlib import metadata

import psutil


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def nvidia_info():
    """Static NVIDIA GPU facts via NVML, or {'error': ...} if NVML is unavailable."""
    try:
        import pynvml as nv
        nv.nvmlInit()
        h = nv.nvmlDeviceGetHandleByIndex(0)

        def safe(fn, *a):
            try:
                return fn(*a)
            except nv.NVMLError as e:
                return f"unavailable: {e}"

        info = {
            "name": safe(nv.nvmlDeviceGetName, h),
            "driver_version": safe(nv.nvmlSystemGetDriverVersion),
            "vbios_version": safe(nv.nvmlDeviceGetVbiosVersion, h),
            "memory_total_mb": safe(lambda: nv.nvmlDeviceGetMemoryInfo(h).total / 2**20),
            "power_limit_w": safe(lambda: nv.nvmlDeviceGetPowerManagementLimit(h) / 1000),
            "max_sm_clock_mhz": safe(nv.nvmlDeviceGetMaxClockInfo, h, nv.NVML_CLOCK_SM),
            "max_mem_clock_mhz": safe(nv.nvmlDeviceGetMaxClockInfo, h, nv.NVML_CLOCK_MEM),
            "pcie_max_gen": safe(nv.nvmlDeviceGetMaxPcieLinkGeneration, h),
            "pcie_max_width": safe(nv.nvmlDeviceGetMaxPcieLinkWidth, h),
            "driver_model": safe(lambda: "WDDM" if nv.nvmlDeviceGetDriverModel(h)[0] == nv.NVML_DRIVER_WDDM else "TCC"),
        }
        nv.nvmlShutdown()
        return info
    except Exception as e:
        return {"error": str(e)}


def _powershell(cmd):
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return None


def host_info():
    packages = {}
    for name in ("playwright", "pandas", "numpy", "psutil", "nvidia-ml-py", "onnx", "onnxruntime"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "os": platform.platform(),
        "cpu": (_powershell("(Get-CimInstance Win32_Processor).Name") if platform.system() == "Windows" else None)
               or platform.processor(),
        "cpu_max_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else None,
        "power_plan": _powershell("powercfg /getactivescheme") if platform.system() == "Windows" else None,
        "on_battery": (psutil.sensors_battery().power_plugged is False) if psutil.sensors_battery() else False,
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "python": sys.version,
        "packages": packages,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "nvidia": nvidia_info(),
    }
