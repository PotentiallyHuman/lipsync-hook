"""One config file holds every path and setting. Nothing in this package hard-codes a location.

    cfg = load()                      # ./config.json, or $LIPSYNC_HOOK_CONFIG
    cfg["comfy"]["dir"], cfg["character"]["canon"], ...

See config.example.json for every key and what it means."""
import json
import os
from pathlib import Path

DEFAULTS = {
    "comfy": {
        "dir": "~/ComfyUI",                         # the ComfyUI install (main.py lives here)
        "python": "~/ComfyUI/venv/bin/python",      # the python that runs ComfyUI
        "port": 8188,
        "run_with_sudo": False,                     # true only if your ComfyUI tree is root-owned
        "hf_home": "",                              # HF_HOME for the server if it needs one
        "cuda_floor_gib": 8,                        # live CUDA cap: keep this much MemFree (0 = no cap)
        "memory_max_gib": 0,                        # systemd-run MemoryMax for the server (0 = none)
        "min_free_gib_to_launch": 60,               # MemFree after a cache drop needed before the server starts
        "input_dir": "",                            # default <dir>/input
        "output_dir": "",                           # default <dir>/output
        "log": "/tmp/lipsync_comfy.log",
    },
    "models": {
        "workflow": "workflows/ltx23_two_stage_av.json",
        "checkpoint": "ltx-2.3-22b-dev-distilled0.5-baked.safetensors",   # the BAKED checkpoint (see docs/SETTINGS.md)
        "baked": True,                              # the distilled LoRA (and optionally the character LoRA) is merged in
        "character_lora": "",                       # runtime character LoRA name if NOT baked ("" = none)
        "character_lora_strength_base": 0.5,
        "character_lora_strength_refine": 0.9,
    },
    "render": {
        "fps": 24,
        "cfg": 1.0,
        "i2v_strength": 0.92,
        "stage1_width": 544, "stage1_height": 960,  # portrait 9:16 stage 1; x2 upsampler -> 1088x1920
        "final_width": 1080, "final_height": 1920,
        "lead_in_s": 0.6,                           # rendered before the window and cut off on the frame grid
        "prompt": "She sings the line straight to the camera, natural lip movement, warm expression; the camera drifts slowly and steadily sideways.",
        "negative": ("a second face, second mouth, extra mouth, extra hand, disembodied hand, hand appearing, hand entering "
                     "the frame, extra limb, extra arm, object appearing, second composition, crossfade, double exposure, "
                     "duplicate person, another person, cut, scene change, wide toothy grin, open-mouth smile, laughing, "
                     "posed music-video smile, influencer camera smile, exaggerated expression, low quality, blurry, "
                     "distorted, deformed, disfigured, extra limbs, extra fingers, malformed hands, extra head, two heads, "
                     "multiple faces, multiple mouths, warped face, melting face, flickering face, identity drift, jitter, "
                     "static frozen still picture, text, watermark, logo, subtitles, cartoon, anime, cgi render, plastic skin"),
        "timeout_s": 10800,
        "seed": 0,                                  # 0 = random per render
        "cache_warden": {"free_min_gib": 30, "period_s": 5},
    },
    "character": {
        "name": "the character",
        "description": "<one sentence: hair, eyes, skin, age range; written so a stranger would recognise the references>",
        "canon": "",                                # one reference image (full identity), REQUIRED for Codex frames and the gate
        "references": [],                           # up to 4 more accepted frames (style + identity), optional
        "identity_min": 0.45,                       # cosine to the canon face, start-frame gate
        "face_height_target": 0.31,                 # face height / frame height after the 9:16 crop
    },
    "frames": {
        "generator": "codex",                       # codex (ChatGPT image_gen via the Codex CLI) | manual (--frame PATH)
        "codex_model": "gpt-5",
        "codex_effort": "low",
        "tries": 2,
        "history": "state/frames_history.json",     # variety: every new frame differs from the last 3
    },
    "hook": {
        "min_s": 8.0, "max_s": 20.0,
        "bars_options": [4, 8, 12, 16],
        "loop_residue_ms": 25,
        "words_per_s_max": 1.9, "words_per_s_min": 0.8,
        "rapid_run_gap_s": 0.28, "rapid_run_len": 4,
        "syllables_per_s_max": 3.0,
        "gap_min_s": 0.25,
        "demucs": "",                               # path to a demucs binary for vocal separation ("" = use the mix)
        "whisper_model": "small",
    },
    "tools_python": "",                             # python with faster_whisper, librosa, insightface, cv2 (default: this one)
    "out_dir": "out",
    "state_dir": "state",
}


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load(path=None):
    p = Path(path or os.environ.get("LIPSYNC_HOOK_CONFIG") or "config.json")
    cfg = json.loads(json.dumps(DEFAULTS))
    if p.exists():
        _merge(cfg, json.loads(p.read_text()))
    cfg["_file"] = str(p.resolve())
    cfg["_root"] = str(p.resolve().parent)
    for key in ("dir", "python"):
        cfg["comfy"][key] = str(Path(cfg["comfy"][key]).expanduser())
    cfg["comfy"]["input_dir"] = cfg["comfy"]["input_dir"] or str(Path(cfg["comfy"]["dir"]) / "input")
    cfg["comfy"]["output_dir"] = cfg["comfy"]["output_dir"] or str(Path(cfg["comfy"]["dir"]) / "output")
    for key in ("canon",):
        if cfg["character"][key]:
            cfg["character"][key] = str(Path(cfg["character"][key]).expanduser())
    cfg["character"]["references"] = [str(Path(r).expanduser()) for r in cfg["character"]["references"]]
    for key in ("out_dir", "state_dir"):
        q = Path(cfg[key]).expanduser()
        cfg[key] = str(q if q.is_absolute() else Path(cfg["_root"]) / q)
    wf = Path(cfg["models"]["workflow"]).expanduser()
    cfg["models"]["workflow"] = str(wf if wf.is_absolute() else Path(cfg["_root"]) / wf)
    h = Path(cfg["frames"]["history"]).expanduser()
    cfg["frames"]["history"] = str(h if h.is_absolute() else Path(cfg["_root"]) / h)
    return cfg


def tools_python(cfg):
    import sys
    return cfg.get("tools_python") or sys.executable
