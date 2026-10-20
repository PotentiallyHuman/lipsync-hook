#!/usr/bin/env python3
"""ONE COMMAND: song in, finished lip-sync hook clip out (plus the exact number the uploader needs).

    bin/make_hook.py --song SONG.wav [--lyric "first words of the chorus"] [--from 0] [--frame PATH] [--out out/<stamp>]
                     [--prompt "..."] [--config config.json] [--dry]

Steps (each one verified before the next):
  1. hook window   lipsync_hook.hook_window   -> hook_audio.wav + hook.json (bar-aligned, loops, 8-20 s)
  2. start frame   lipsync_hook.frame_gen     -> start_frame.png (Codex image_gen from the references, face-gated)
                   or --frame PATH (your own picture, still face-gated and cropped)
  3. render        lipsync_hook.render        -> clip.mp4 / clip_silent.mp4 (one continuous shot, audio-locked)
  4. check         lipsync_hook.check_clip    -> checks.json (cfr, no jumps, no dups, identity, one face)
  5. deliver       DELIVERY.json: clip paths + "start the library sound at <cut_start> s under clip time 0"
Uploading and posting are NOT part of this tool: do that yourself, on your own account."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from lipsync_hook.config import load, tools_python  # noqa: E402


def say(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def run_tools(cfg, module, args):
    r = subprocess.run([tools_python(cfg), "-m", module, *map(str, args)], cwd=str(HERE), text=True, capture_output=True,
                       env=dict(os.environ, PYTHONPATH=str(HERE)))
    sys.stdout.write("".join(l + "\n" for l in r.stdout.splitlines() if l.startswith("[") or l.startswith("{")))
    if r.returncode != 0:
        raise SystemExit(f"{module} failed:\n{r.stderr[-1500:]}")
    lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
    return json.loads(lines[-1]) if lines else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True, help="the song recording (wav/mp3/m4a), e.g. the platform's preview of the library sound")
    ap.add_argument("--lyric", default="", help="first words of the chorus, to find it (optional)")
    ap.add_argument("--from", dest="from_s", type=float, default=0.0, help="search for the chorus from this second")
    ap.add_argument("--frame", help="use this start frame instead of generating one (still gated)")
    ap.add_argument("--prompt", help="override render.prompt")
    ap.add_argument("--out")
    ap.add_argument("--config")
    ap.add_argument("--dry", action="store_true", help="hook window + frame only, no render")
    x = ap.parse_args()
    cfg = load(x.config)
    out = Path(x.out or Path(cfg["out_dir"]) / time.strftime("hook_%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    say(f"out: {out}")

    # 1. hook window
    say("1/5 hook window")
    run_tools(cfg, "lipsync_hook.hook_window", [x.song, out / "hook", "--lyric", x.lyric, "--from", x.from_s] + (["--config", x.config] if x.config else []))
    hook = json.loads((out / "hook" / "hook.json").read_text())
    say(f"hook: {hook['cut_start']}-{hook['cut_end']} s, {hook['bars']} bars, {hook['length_s']} s, residue {hook['residue_ms']} ms, pace {hook['pace']['why']}")

    # 2. start frame
    say("2/5 start frame")
    if x.frame:
        res = run_tools(cfg, "lipsync_hook.frame_gate", [x.frame, cfg["character"]["canon"], out / "start_frame.png", cfg["character"]["face_height_target"]])
        if not res.get("ok"):
            raise SystemExit(f"your frame fails the gate: {res.get('why')}")
        frame = out / "start_frame.png"
    else:
        from lipsync_hook import frame_gen
        frame, res = frame_gen.make_frame(cfg, out, log_fn=say)
    say(f"frame: {frame} (identity {res.get('ident')}, face {res.get('face_h')})")
    if x.dry:
        say("dry run: stopping before the render")
        return

    # 3. render
    say(f"3/5 render ({hook['length_s']} s single shot; ~6 min per second on a GB10)")
    from lipsync_hook import render
    review, silent, meta = render.render_clip(cfg, frame, out / "hook" / "hook_audio.wav", out, prompt=x.prompt, hook_meta=hook, log=say)

    # 4. check
    say("4/5 checks")
    chk = run_tools(cfg, "lipsync_hook.check_clip", [review, out / "start_9x16.png", "--identity-min", "0.50"])
    (out / "checks.json").write_text(json.dumps(chk, indent=1))
    say(f"checks: {chk['checks']} -> {'PASS' if chk['pass'] else 'FAIL'}")
    if not chk["pass"]:
        raise SystemExit("the clip failed its checks; see checks.json (re-run for a new seed / frame)")

    # 5. deliver
    delivery = {"clip_with_audio": str(review), "clip_silent_for_upload": str(silent), "start_frame": str(frame),
                "hook": {k: hook[k] for k in ("cut_start", "cut_end", "bars", "length_s", "bpm", "bar_s", "residue_ms")},
                "upload_instruction": (f"Upload clip_silent_for_upload and ADD THE SONG from the platform's library; trim the sound so that "
                                       f"song time {hook['cut_start']:.3f} s plays under clip time 0 (on a 1/30 s grid: frame "
                                       f"{int(round(hook['cut_start'] * 30))}). Place it exactly (zoom the timeline, compute pixels from "
                                       f"two ruler labels, read the position back); never by playing and measuring."),
                "checks": chk["checks"], "render": {k: meta[k] for k in ("frames_rendered", "fps", "render_s", "seed", "checkpoint")}}
    (out / "DELIVERY.json").write_text(json.dumps(delivery, indent=1))
    say(f"5/5 DELIVERED {silent}")
    print(json.dumps(delivery, indent=1))


if __name__ == "__main__":
    main()
