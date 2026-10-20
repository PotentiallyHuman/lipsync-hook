"""Start, stop and watch the ComfyUI server the proven way (unified-memory box, 22B video model).

- Fresh server per render: an aged ComfyUI process gets ~10x slower on the refine pass after about 90 minutes
  (measured), and an idle server keeps ~50-60 GiB of GPU pages. Stop it after each render.
- Memory gate before launch: drop the page cache, require MemFree >= min_free_gib_to_launch (real free, not
  MemAvailable: on unified memory GPU-resident weights are counted as page cache).
- The server runs through comfy_capped.py (live CUDA cap) with --highvram --disable-pinned-memory
  --disable-async-offload. With the BAKED checkpoint the 46 GiB of weights sit on the GPU once.
- Cache warden thread during a render: when MemFree < free_min_gib and the cache is large, drop the cache. On this
  kind of box the page cache is NOT evicted for a GPU allocation; without the warden a render stalls at
  "Model Initializing" or OOMs on the last refine step.

Needs passwordless sudo for `sync; echo 3 > /proc/sys/vm/drop_caches` (see setup/INSTALL.md)."""
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _url(cfg, path):
    return f"http://127.0.0.1:{cfg['comfy']['port']}{path}"


def http(cfg, path, data=None, timeout=60):
    req = urllib.request.Request(_url(cfg, path), data=(json.dumps(data).encode() if data is not None else None),
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def is_up(cfg):
    try:
        http(cfg, "/queue", timeout=5)
        return True
    except Exception:
        return False


def queue_busy(cfg):
    try:
        d = http(cfg, "/queue", timeout=5)
        return bool(d.get("queue_running")) or bool(d.get("queue_pending"))
    except Exception:
        return False


def meminfo():
    d = {}
    for ln in open("/proc/meminfo"):
        k, v = ln.split(":", 1)
        d[k] = int(v.split()[0])
    return d


def drop_caches():
    subprocess.run(["sudo", "-n", "sync"], timeout=180, capture_output=True)
    r = subprocess.run(["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"], input=b"3\n", timeout=180, capture_output=True)
    return r.returncode == 0


def real_free_gib(drop=False):
    if drop:
        drop_caches()
    return meminfo()["MemFree"] // 1048576


def memory_gate(cfg, log=print):
    need = int(cfg["comfy"]["min_free_gib_to_launch"])
    free = real_free_gib(drop=True)
    log(f"[server] real free {free} GiB after a cache drop (need {need})")
    if free < need:
        raise RuntimeError(f"real free {free} GiB < {need} GiB: not launching a 22B model into a crowded pool. "
                           f"Unload other models (ollama, other servers) first.")
    return free


def launch_argv(cfg):
    c = cfg["comfy"]
    envs = []
    if c.get("hf_home"):
        envs.append(f"HF_HOME={c['hf_home']}")
    envs.append(f"LIPSYNC_CUDA_FLOOR_GIB={int(c.get('cuda_floor_gib', 8))}")
    words = [c["python"], str(HERE / "comfy_capped.py"), "main.py", "--listen", "127.0.0.1", "--port", str(c["port"]),
             "--highvram", "--disable-pinned-memory", "--disable-async-offload"]
    if c.get("run_with_sudo"):
        words = ["sudo", "-n", "env", *envs] + words
    else:
        words = ["env", *envs] + words
    if int(c.get("memory_max_gib", 0) or 0) > 0 and shutil.which("systemd-run"):
        words = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={int(c['memory_max_gib'])}G",
                 "-p", "MemorySwapMax=0"] + words
    return words


def start(cfg, log=print, timeout=400):
    if is_up(cfg):
        log("[server] already up")
        return True
    memory_gate(cfg, log)
    argv = launch_argv(cfg)
    log(f"[server] launching: {' '.join(argv)}")
    lf = open(cfg["comfy"]["log"], "a")
    subprocess.Popen(argv, cwd=cfg["comfy"]["dir"], stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_up(cfg):
            time.sleep(3)
            log(f"[server] READY on :{cfg['comfy']['port']} after {time.time() - t0:.0f} s")
            return True
        time.sleep(3)
    log(f"[server] TIMEOUT; see {cfg['comfy']['log']}")
    return False


def our_server_pids(cfg):
    """PIDs of ComfyUI servers on our port whose command line runs OUR comfy_capped.py: never another user's process."""
    pattern = f"main.py --listen 127.0.0.1 --port {cfg['comfy']['port']}"
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    pids = []
    for pid in out:
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if str(HERE / "comfy_capped.py") in cmd:
            pids.append(int(pid))
    return pids


def stop(cfg, log=print):
    pids = our_server_pids(cfg)
    for pid in pids:
        if cfg["comfy"].get("run_with_sudo"):
            subprocess.run(["sudo", "-n", "kill", "-9", str(pid)], check=False)
        else:
            subprocess.run(["kill", "-9", str(pid)], check=False)
    if not pids:
        log("[server] no server of ours to stop")
    for _ in range(30):
        if not is_up(cfg):
            break
        time.sleep(1)
    log(f"[server] :{cfg['comfy']['port']} {'is free' if not is_up(cfg) else 'STILL LISTENING'}")
    return not is_up(cfg)


class CacheWarden(threading.Thread):
    """Keep MemFree above free_min_gib while a render runs by dropping the page cache (the loaded weights' cache
    is dead weight once the model sits on the GPU)."""

    def __init__(self, free_min_gib=30, period_s=5, cache_min_gib=24, log=print):
        super().__init__(daemon=True, name="cache-warden")
        self.free_min, self.period, self.cache_min, self.log = free_min_gib, period_s, cache_min_gib, log
        self.stop_flag = threading.Event()
        self.drops = 0

    def run(self):
        while not self.stop_flag.is_set():
            m = meminfo()
            free = m["MemFree"] // 1048576
            cache = (m["Cached"] + m["Buffers"]) // 1048576
            if free < self.free_min and cache > self.cache_min:
                ok = drop_caches()
                after = meminfo()["MemFree"] // 1048576
                self.drops += 1
                self.log(f"[warden] free {free} GiB cache {cache} GiB -> {'dropped, free now ' + str(after) + ' GiB' if ok else 'drop FAILED (sudo -n)'}")
                if not ok:
                    time.sleep(300)
            self.stop_flag.wait(self.period)

    def stop(self):
        self.stop_flag.set()


if __name__ == "__main__":
    import sys
    from .config import load  # noqa
    cfg = load()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        print(f"up={is_up(cfg)} busy={queue_busy(cfg)} free={real_free_gib()} GiB")
    elif cmd == "start":
        start(cfg)
    elif cmd == "stop":
        stop(cfg)
