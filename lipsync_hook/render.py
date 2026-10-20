"""ONE continuous lip-sync shot with LTX-2.3 (22B, distilled, two-stage): the start frame is the only image
conditioning, the hook audio is frozen (audio-locked) and only the picture is generated, so the mouth follows the
real song timing.

    from lipsync_hook import render
    render.render_clip(cfg, start_png, hook_wav, out_dir, prompt=None)

The proven settings (docs/SETTINGS.md): cfg 1.0, I2V strength 0.92, freeze mask 1.0, 8n+1 frames, 24 fps,
portrait stage-1 544x960 -> x2 upsampler -> 1088x1920 -> delivered 1080x1920, a 0.6 s lead-in rendered BEFORE the
window and cut off on the frame grid so delivered frame 0 = hook audio sample 0. No maximum shot length: the only
costs are time (about 6 min per second of shot on a GB10) and memory (the live cap + cache warden handle it).
A hook longer than one shot is NOT split: a split snaps the character back to the start pose at the cut.

Outputs in out_dir: clip.mp4 (with the hook audio, for review), clip_silent.mp4 (for upload), start_9x16.png,
render.json (settings + the placement numbers for the uploader)."""
import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

from . import comfy_server as srv

# node ids of the LTX two-stage audio-video graph (workflows/ltx23_two_stage_av.json)
N_LOADIMAGE, N_LATENT, N_NUM_FRAMES, N_FPS = "2004", "3059", "4988", "4989"
N_BYPASS_I2V, N_RESIZE, N_SAVE, N_I2V_STRENGTH = "4987", "4990", "4852", "3159"
N_LOAD_AUDIO, N_CREATE_VIDEO = "9001", "4849"
N_BASE_GUIDER, N_BASE_SAMPLER, N_REFINE_GUIDER = "4828", "4829", "4964"
N_CKPT, N_DISTILLED_LORA = "3940", "4922"
AV_CONCAT, COND, MODEL, VAE, AUDIO_VAE = ["4528", 0], "1241", ["4922", 0], ["3940", 2], ["4010", 0]
POS_NODES, NEG_NODES = ["4979", "4980", "2483"], ["2612", "4981"]


def lead_in_s(fps, lead=0.6):
    """The lead-in rounded UP to whole frames (24 fps: 15 frames = 0.625 s) so the cut lands on a frame."""
    return math.ceil(round(lead * fps, 6)) / fps


def frames_8n1(seconds, fps, min_frames=73):
    fr = int(seconds * fps) + 1
    return max(min_frames, ((fr + 6) // 8) * 8 + 1)


def ffprobe_duration(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                                capture_output=True, text=True).stdout.strip())


def crop_9x16(src, dst):
    from PIL import Image
    im = Image.open(src).convert("RGB")
    w, h = im.size
    tw = round(h * 9 / 16)
    if tw <= w:
        box = ((w - tw) // 2, 0, (w - tw) // 2 + tw, h)
    else:
        th = round(w * 16 / 9)
        box = (0, (h - th) // 2, w, (h - th) // 2 + th)
    im.crop(box).save(dst)
    return box


def _copy_into(path, dst, sudo=False):
    try:
        shutil.copyfile(path, dst)
    except PermissionError:
        if not sudo:
            raise
        subprocess.run(["sudo", "-n", "cp", str(path), str(dst)], check=True)
        subprocess.run(["sudo", "-n", "chmod", "644", str(dst)], check=False)


def apply_baked_ckpt(wf, baked):
    """Every ckpt_name -> the baked file; the distilled LoRA node is dropped and its consumers rewired onto the
    checkpoint's model output, so ComfyUI patches nothing (no second copy of the weights)."""
    for node in wf.values():
        if "ckpt_name" in node.get("inputs", {}):
            node["inputs"]["ckpt_name"] = baked
    for node in wf.values():
        for name, val in list(node.get("inputs", {}).items()):
            if isinstance(val, list) and len(val) == 2 and val[0] == N_DISTILLED_LORA:
                node["inputs"][name] = [N_CKPT, 0]
    wf.pop(N_DISTILLED_LORA, None)
    return wf


def build_workflow(cfg, still_name, audio_name, n_frames, prompt, negative, seed=0):
    r, m = cfg["render"], cfg["models"]
    wf = json.loads(Path(m["workflow"]).read_text())
    fps = float(r["fps"])
    dur = n_frames / fps
    wf[N_LOADIMAGE]["inputs"]["image"] = still_name
    wf[N_LATENT]["inputs"]["width"], wf[N_LATENT]["inputs"]["height"] = int(r["stage1_width"]), int(r["stage1_height"])
    wf[N_NUM_FRAMES]["inputs"]["value"] = int(n_frames)
    wf[N_FPS]["inputs"]["value"] = fps
    wf[N_BYPASS_I2V]["inputs"]["value"] = False
    wf[N_RESIZE]["inputs"]["resize_type.longer_size"] = 2 * max(int(r["stage1_width"]), int(r["stage1_height"]))
    wf[N_SAVE]["inputs"]["filename_prefix"] = "lipsync"
    wf[N_I2V_STRENGTH]["inputs"]["strength"] = float(r["i2v_strength"])
    for nid in POS_NODES:
        for k in ("prompt", "text", "value"):
            if nid in wf and k in wf[nid]["inputs"]:
                wf[nid]["inputs"][k] = prompt
    for nid in NEG_NODES:
        for k in ("prompt", "text", "value"):
            if nid in wf and k in wf[nid]["inputs"]:
                wf[nid]["inputs"][k] = negative
    wf[N_LOAD_AUDIO]["inputs"]["audio"] = audio_name
    # the audio-locked freeze: the fed audio latent is kept, only the video latent is generated
    wf["9600"] = {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {
        "av_latent": AV_CONCAT, "positive": [COND, 0], "negative": [COND, 1], "model": MODEL, "vae": VAE,
        "audio_vae": AUDIO_VAE, "start_time": 0.0, "end_time": float(dur), "video_fps": fps,
        "mask_video": False, "mask_audio": False, "mask_init_value_video": 1.0, "mask_init_value_audio": 0.0,
        "slope_len": 3}}
    wf[N_BASE_SAMPLER]["inputs"]["latent_image"] = ["9600", 2]
    wf[N_BASE_GUIDER]["inputs"]["positive"] = ["9600", 0]
    wf[N_BASE_GUIDER]["inputs"]["negative"] = ["9600", 1]
    for g in (N_BASE_GUIDER, N_REFINE_GUIDER):
        wf[g]["inputs"]["cfg"] = float(r["cfg"])
    wf[N_CREATE_VIDEO]["inputs"]["audio"] = [N_LOAD_AUDIO, 0]
    if seed:
        for node in wf.values():
            for k in ("noise_seed", "seed"):
                if k in node.get("inputs", {}):
                    node["inputs"][k] = int(seed)
    # character LoRA at runtime (only when it is NOT baked in): base pass low, refine pass high
    if m.get("character_lora"):
        s1, s2 = float(m["character_lora_strength_base"]), float(m["character_lora_strength_refine"])
        wf["8300"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": MODEL, "lora_name": m["character_lora"], "strength_model": s1}}
        wf["8400"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": MODEL, "lora_name": m["character_lora"], "strength_model": s2}}
        wf[N_BASE_GUIDER]["inputs"]["model"] = ["8300", 0]
        wf[N_REFINE_GUIDER]["inputs"]["model"] = ["8400", 0]
    if m.get("baked"):
        apply_baked_ckpt(wf, m["checkpoint"])
    else:
        wf[N_CKPT]["inputs"]["ckpt_name"] = m["checkpoint"]
    return wf


def submit_and_wait(cfg, wf, timeout_s, log=print):
    resp = srv.http(cfg, "/prompt", {"prompt": wf})
    pid = resp.get("prompt_id")
    if not pid:
        raise RuntimeError(f"submit failed: {resp}")
    log(f"[render] submitted prompt {pid}")
    t0 = time.time()
    while True:
        time.sleep(15)
        if time.time() - t0 > timeout_s:
            try:
                srv.http(cfg, "/interrupt", {})
            except Exception:
                pass
            raise RuntimeError(f"render exceeded {timeout_s / 60:.0f} min")
        try:
            hist = srv.http(cfg, f"/history/{pid}")
        except Exception as e:
            log(f"[render] poll error {e}")
            continue
        entry = hist.get(pid)
        if entry and entry.get("status", {}).get("status_str") == "error":
            raise RuntimeError("ComfyUI reported an error for the render (out of memory? see the server log)")
        if entry and entry.get("status", {}).get("completed"):
            for out in entry.get("outputs", {}).values():
                for v in out.get("images") or out.get("gifs") or out.get("videos") or []:
                    fn = v.get("filename") if isinstance(v, dict) else None
                    if fn and fn.endswith(".mp4"):
                        return Path(cfg["comfy"]["output_dir"]) / fn
            raise RuntimeError("render completed but produced no mp4")


def render_clip(cfg, start_png, hook_wav, out_dir, prompt=None, hook_meta=None, log=print, keep_server=False):
    r = cfg["render"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fps = int(r["fps"])
    lead = lead_in_s(fps, float(r["lead_in_s"]))
    win = ffprobe_duration(hook_wav)
    n = frames_8n1(lead + win, fps)
    still = out / "start_9x16.png"
    crop_9x16(start_png, still)
    # lead-in of silence + the exact window = what LTX hears; frame grid: delivered frame 0 == window sample 0
    stem = out / "audio_leadin.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-t", f"{lead:.6f}", "-i", "anullsrc=r=48000:cl=stereo",
                    "-i", str(hook_wav), "-filter_complex",
                    "[1:a]aresample=48000,aformat=channel_layouts=stereo[a1];[0:a][a1]concat=n=2:v=0:a=1", str(stem)], check=True)
    dur = n / fps
    slice_wav = out / "audio_slice.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(stem), "-af", f"apad,atrim=0:{dur:.4f}", "-ac", "2", "-ar", "48000",
                    str(slice_wav)], check=True)
    tag = f"lipsync_{int(time.time()) % 1000000}"
    still_name, audio_name = f"{tag}.png", f"{tag}.wav"
    sudo = bool(cfg["comfy"].get("run_with_sudo"))
    _copy_into(still, Path(cfg["comfy"]["input_dir"]) / still_name, sudo)
    _copy_into(slice_wav, Path(cfg["comfy"]["input_dir"]) / audio_name, sudo)
    seed = int(r.get("seed") or 0) or random.randint(1, 2**31)
    wf = build_workflow(cfg, still_name, audio_name, n, prompt or r["prompt"], r["negative"], seed)
    (out / "workflow_submitted.json").write_text(json.dumps(wf, indent=1))
    log(f"[render] {win:.2f} s window + {lead:.3f} s lead-in -> {n} frames at {fps} fps ({dur:.2f} s); seed {seed}")
    if not srv.start(cfg, log):
        raise RuntimeError("ComfyUI did not come up")
    warden = srv.CacheWarden(int(r["cache_warden"]["free_min_gib"]), int(r["cache_warden"]["period_s"]), log=log)
    warden.start()
    t0 = time.time()
    try:
        raw = submit_and_wait(cfg, wf, int(r["timeout_s"]), log)
    finally:
        warden.stop()
        if not keep_server:
            srv.stop(cfg, log)
    secs = int(time.time() - t0)
    raw_keep = out / "raw_render.mp4"
    _copy_into(raw, raw_keep)
    # cut the lead-in on the frame grid, deliver the exact window at the final size, constant frame rate
    W, H = int(r["final_width"]), int(r["final_height"])
    review, silent = out / "clip.mp4", out / "clip_silent.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_keep), "-ss", f"{lead:.4f}", "-t", f"{win:.4f}",
                    "-i", str(hook_wav), "-map", "0:v", "-map", "1:a", "-af", "apad", "-vf", f"scale={W}:{H}:flags=lanczos,fps={fps}",
                    "-fps_mode", "cfr", "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-crf", "16",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(review)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(review), "-an", "-c:v", "copy", "-movflags", "+faststart", str(silent)], check=True)
    meta = {"window_s": win, "fps": fps, "frames_rendered": n, "lead_in_s": lead, "prompt": prompt or r["prompt"], "cfg": r["cfg"],
            "i2v_strength": r["i2v_strength"], "checkpoint": cfg["models"]["checkpoint"], "baked": cfg["models"]["baked"],
            "seed": seed, "render_s": secs, "start_frame": str(start_png), "hook_audio": str(hook_wav),
            "placement": {"clip_time_0_is_song_time_s": (hook_meta or {}).get("cut_start"),
                          "rule": "the uploader starts the library sound at clip_time_0_is_song_time_s under clip time 0 "
                                  "(on a 1/30 s grid: round(cut_start x 30) frames); no play-and-measure loops"}}
    (out / "render.json").write_text(json.dumps(meta, indent=1))
    log(f"[render] DONE {review} in {secs} s")
    return review, silent, meta
