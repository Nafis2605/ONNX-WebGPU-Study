"""Launch a fresh Google Chrome Dev process per experiment and drive the benchmark page."""
import asyncio
import os
import platform
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

# Flags applied to every launch. Each one is recorded in env.json.
CHROME_FLAGS = [
    "--no-first-run",
    "--no-default-browser-check",
    # WebGPU: unquantized timestamp queries and full adapter info (vendor/description).
    "--enable-webgpu-developer-features",
    # Cold GPU state: no shader/pipeline cache survives between launches.
    "--disable-gpu-shader-disk-cache",
    # Keep timers and rendering at full rate even if the window is occluded.
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--disable-features=CalculateNativeWinOcclusion",
]


def find_chrome_dev():
    """Return the Google Chrome Dev executable path, or None if it is not installed.

    Checks CHROME_DEV_PATH first, then the standard install locations.
    """
    candidates = [os.environ.get("CHROME_DEV_PATH")]
    if platform.system() == "Windows":
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LOCALAPPDATA")):
            if base:
                candidates.append(os.path.join(base, "Google", "Chrome Dev", "Application", "chrome.exe"))
    elif platform.system() == "Darwin":
        candidates.append("/Applications/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev")
    for path in candidates:
        if path and Path(path).is_file():
            return path
    return None


@asynccontextmanager
async def chrome_page(playwright, executable_path, url, headless, extra_flags=(), log_path=None, on_launch=None,
                      cdp_setup=None):
    """Yield (page, browser_version) for a brand-new Chrome process with an empty profile.

    on_launch(user_data_dir) is called right after the browser starts (the directory name is
    unique per launch, so it identifies this launch's Chrome processes).
    cdp_setup(cdp) (async) runs on the page's CDP session before navigation; it may return an
    async teardown, which runs after the caller's block and before the browser closes.
    """
    user_data_dir = tempfile.mkdtemp(prefix="ortbench-profile-")
    log_file = open(log_path, "w", encoding="utf-8") if log_path else None
    context = None
    teardown = None
    try:
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir,
            executable_path=executable_path,
            headless=headless,
            args=CHROME_FLAGS + list(extra_flags),
            viewport={"width": 1280, "height": 900},
        )
        if on_launch:
            on_launch(user_data_dir)
        page = context.pages[0] if context.pages else await context.new_page()
        if log_file:
            page.on("console", lambda m: log_file.write(f"[console.{m.type}] {m.text}\n"))
            page.on("pageerror", lambda e: log_file.write(f"[pageerror] {e}\n"))
            page.on("crash", lambda _: log_file.write("[crash] renderer crashed\n"))
        cdp = await context.new_cdp_session(page)
        version = (await cdp.send("Browser.getVersion"))["product"]
        if cdp_setup:
            teardown = await cdp_setup(cdp)
        await page.goto(url, wait_until="load", timeout=60_000)
        await page.wait_for_function("() => window.__bench !== undefined", timeout=60_000)
        if not await page.evaluate("() => window.__bench.ready"):
            raise RuntimeError(f"benchmark page failed to initialise: {await page.evaluate('() => window.__bench.error')}")
        yield page, version
    finally:
        if teardown is not None:
            try:
                await asyncio.wait_for(teardown(), timeout=120)
            except Exception as e:
                if log_file:
                    log_file.write(f"[cdp teardown failed] {type(e).__name__}: {e}\n")
        if context is not None:
            try:
                await asyncio.wait_for(context.close(), timeout=30)
            except Exception:
                pass
        if log_file:
            log_file.close()
        shutil.rmtree(user_data_dir, ignore_errors=True)
