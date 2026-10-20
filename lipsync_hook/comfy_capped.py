#!/usr/bin/env python3
"""Run ComfyUI's main.py under a LIVE per-process CUDA cap that keeps the box's last GiBs free.

    python comfy_capped.py main.py --listen 127.0.0.1 --port 8188 ...    (same argv as main.py)

Why: a DGX Spark (GB10) has ONE unified 121 GiB pool. The GPU driver runs out of it before Linux's OOM killer acts,
and then the whole box freezes (power button). No outside guard can stop that early enough; the only clean stop is
INSIDE the process: torch's allocator refuses an allocation over its cap with a clean OutOfMemoryError, ComfyUI marks
the prompt failed and stays up.

The cap is live: every PERIOD_S the watcher reads MemFree (real free; MemAvailable lies on unified memory) and sets
allowed = reserved_now + (MemFree - FLOOR_GIB). LIPSYNC_CUDA_FLOOR_GIB (default 8; 0 = cap off)."""
import os
import runpy
import sys
import threading
import time

FLOOR_GIB = float(os.environ.get("LIPSYNC_CUDA_FLOOR_GIB", "8") or 0)
PERIOD_S = float(os.environ.get("LIPSYNC_CUDA_CAP_PERIOD_S", "1") or 1)
GIB = float(1 << 30)


def _log(msg):
    sys.stderr.write(f"[cuda-cap {time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def mem_free_gib():
    with open("/proc/meminfo") as fh:
        for ln in fh:
            if ln.startswith("MemFree:"):
                return int(ln.split()[1]) / (1 << 20)
    return 0.0


def allowed_bytes(reserved, free_gib, floor_gib, total):
    return max(min(reserved + (free_gib - floor_gib) * GIB, total), GIB)


def watcher():
    # main.py configures the allocator before importing torch; wait for its torch, never import our own first
    while "torch" not in sys.modules:
        time.sleep(0.5)
    torch = sys.modules["torch"]
    while not (hasattr(torch, "cuda") and torch.cuda.is_initialized()):
        time.sleep(0.5)
    total = float(torch.cuda.get_device_properties(0).total_memory)
    _log(f"live cap ON: allowed = reserved + (MemFree - {FLOOR_GIB:g} GiB); pool {total / GIB:.0f} GiB; every {PERIOD_S:g} s")
    last_frac, last_floor_log, at_floor_since = None, 0.0, None
    while True:
        try:
            free = mem_free_gib()
            reserved = float(torch.cuda.memory_reserved())
            allowed = allowed_bytes(reserved, free, FLOOR_GIB, total)
            frac = allowed / total
            if last_frac is None or abs(frac - last_frac) > 0.002:
                torch.cuda.set_per_process_memory_fraction(frac)
                last_frac = frac
            now = time.time()
            if allowed <= reserved + GIB:
                at_floor_since = at_floor_since or now
                if now - last_floor_log > 30:
                    _log(f"AT THE FLOOR: MemFree {free:.1f} GiB, torch reserved {reserved / GIB:.1f} GiB, allowed "
                         f"{allowed / GIB:.1f} GiB; a new allocation now fails cleanly instead of freezing the box")
                    last_floor_log = now
            elif at_floor_since:
                _log(f"off the floor after {now - at_floor_since:.0f} s: MemFree {free:.1f} GiB")
                at_floor_since = None
        except Exception as e:
            _log(f"watcher error (ignored): {e!r}")
            time.sleep(5)
        time.sleep(PERIOD_S)


def main():
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
        sys.exit("usage: comfy_capped.py main.py [main.py args...]")
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    if FLOOR_GIB > 0:
        threading.Thread(target=watcher, name="cuda-cap", daemon=True).start()
    else:
        _log("cap OFF (LIPSYNC_CUDA_FLOOR_GIB=0)")
    sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
