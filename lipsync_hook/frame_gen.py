"""A fresh start frame of the character from reference images, through the Codex CLI's image_gen tool
(ChatGPT subscription login, no browser), varied every time, gated by frame_gate before anything renders.

    python -m lipsync_hook.frame_gen <out_dir> [--config config.json]        -> <out_dir>/start_frame.png

Recipe (the one that produced the accepted frames): "the reference images show the character; keep her exact face,
skin tone and hair exactly as shown", a location, a vivid opaque outfit, the mood, a lip-sync framing (chest-up or
waist-up, face about a third of the height, lips closed, facing the camera), no phone, no mirror, no text.
Variety: setting, outfit, framing, mood, light, hair and camera position all differ from the last 3 frames made.
Codex: lean exec, image_gen only, the script reads prompt.txt/refs.txt itself and never prints the tool result
(printing it once cost 635k tokens). At most 5 reference paths."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import load, tools_python

SETTINGS = [
    "a sunlit park path with tall autumn trees", "a quiet cobbled old-town street at golden hour",
    "a rooftop terrace with the city skyline behind her", "a wooden pier at a calm lake", "a flower market stall full of tulips",
    "a cafe terrace with bistro tables on a city square", "a forest trail with sunbeams through the leaves",
    "a harbour promenade with boats behind her", "a summer meadow of tall grass and wildflowers", "a sandy beach with soft waves behind her",
    "an apartment balcony with potted plants and city lights behind her", "a bright kitchen with a window and morning light",
    "a cosy bedroom with soft bedding and a window", "the passenger seat of a parked car, window open", "a train window seat with countryside passing",
    "a library aisle with warm wooden shelves", "a bookshop corner", "a tennis court fence at dusk", "a snowy street with warm shop windows",
    "a vineyard path at late afternoon", "a rainy city street under a cafe awning", "a hotel lobby with marble and warm lamps",
]
OUTFITS = [
    "an emerald green knit sweater", "a navy blue wrap dress with short sleeves", "a mustard yellow jacket over a white top",
    "a teal blouse", "a burgundy wool coat, open, over a cream top", "a red dress with small white flowers",
    "a cobalt blue denim jacket over a yellow top", "a coral pink blouse", "a light blue gingham sundress", "a plain brown ribbed t-shirt",
    "a cream cable-knit cardigan", "a black turtleneck", "a white linen shirt", "a sage green hoodie", "a dark green blazer over a white tee",
    "a lavender satin blouse", "a striped navy-and-white long-sleeve top", "an orange puffer jacket over a grey top", "a rust-coloured corduroy jacket",
]
FRAMINGS = [
    ("chest-up, facing the camera straight on", "her face centred in the upper half and about one third of the picture height"),
    ("waist-up, shoulders square to the camera", "her face in the upper third, about a quarter of the picture height"),
    ("waist-up, body turned slightly to the side, face to the camera", "her face in the upper third, about a quarter of the picture height"),
    ("chest-up, leaning on a railing or ledge with her forearms, face to the camera", "her face centred in the upper half, about a third of the picture height"),
    ("waist-up, seated, elbows on a table, face to the camera", "her face in the upper third, about a quarter of the picture height"),
    ("chest-up, one hand tucking hair behind her ear, face to the camera", "her face centred in the upper half, about a third of the picture height"),
]
MOODS = ["calm and relaxed, lips closed, a soft pleasant look", "a small confident smile, lips closed", "playful, eyebrows slightly raised, lips closed",
         "dreamy and unhurried, lips closed", "warm and friendly, lips closed, chin slightly down", "cool and composed, lips closed"]
LIGHTS = ["natural daylight", "golden-hour sunlight from the side", "soft overcast light", "blue-hour dusk light",
          "warm indoor lamp light", "bright morning window light", "dappled sunlight through leaves"]
HAIR = ["worn down", "in a loose low ponytail", "half up", "tucked behind one ear", "over one shoulder"]
CAMERAS = ["a friend holding a phone at eye level", "a phone on a small tripod at chest height", "a friend holding a phone slightly above eye level"]

TASK = '''Make ONE picture. Run exactly this code-mode script once and nothing else (no view_image, no reading of skills or
documentation, no commentary). The script reads the job's own files so you never retype the prompt:

// @exec: {"yield_time_ms": 300000}
const j = "job";
const prompt = (await tools.exec_command({cmd: `cat ${j}/prompt.txt`, max_output_tokens: 4000})).output;
const refs = (await tools.exec_command({cmd: `cat ${j}/refs.txt`, max_output_tokens: 1000})).output
  .split("\\n").map(x => x.trim()).filter(x => x.startsWith("/"));
await tools.image_gen__imagegen({prompt, referenced_image_paths: refs.slice(0, 5)});
const saved = (await tools.exec_command({cmd: `ls -t "$HOME/.codex/generated_images"/*/*.png | head -1`, max_output_tokens: 200})).output.trim();
text("done " + saved);

Never print, store or image() the tool's result. When the script has run reply with exactly one line: DONE <saved path>,
or FAILED <reason>.
'''
LIMIT_MARKERS = ("out of credits", "usage limit", "rate limit", "quota", "insufficient", "limit reached", "too many requests", "upgrade your plan")


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def pick_variety(history_file):
    hist = json.loads(Path(history_file).read_text()) if Path(history_file).exists() else []
    recent = hist[-3:]

    def lru(key, values):
        last = {h.get(key): i for i, h in enumerate(hist)}
        banned = {h.get(key) for h in recent}
        cands = [v for v in values if v not in banned] or list(values)
        return min(cands, key=lambda v: (last.get(v, -1), values.index(v)))

    pick = {"setting": lru("setting", SETTINGS), "outfit": lru("outfit", OUTFITS), "framing": lru("framing", [f[0] for f in FRAMINGS]),
            "mood": lru("mood", MOODS), "light": lru("light", LIGHTS), "hair": lru("hair", HAIR), "camera": lru("camera", CAMERAS), "t": time.time()}
    hist.append(pick)
    Path(history_file).parent.mkdir(parents=True, exist_ok=True)
    Path(history_file).write_text(json.dumps(hist[-60:], indent=1))
    return pick


def prompt_text(cfg, v, retry_note=""):
    ch = cfg["character"]
    face_rule = next(f[1] for f in FRAMINGS if f[0] == v["framing"])
    return (
        f"The reference images all show the same character: {ch['description']}. Keep her exact face, skin tone and hair "
        f"colour exactly as shown in the references; do not change her appearance or age. Hair {v['hair']}. "
        f"Location: {v['setting']}. Outfit: {v['outfit']}, fully opaque, vivid colour. Light: {v['light']}. "
        f"Expression and body language: {v['mood']}, as if she is about to sing. "
        f"Framing: vertical 9:16 portrait photograph taken by {v['camera']}; {v['framing']}; her head and face fully inside "
        f"the frame with air above the hair; {face_rule}; not a face close-up and not a full-body shot. "
        "Photorealistic, sharp focus on the face, candid phone-camera look. There is no phone, no tripod and no mirror "
        "visible anywhere; she holds nothing. No other people. No text, no labels, no watermarks." + (" " + retry_note if retry_note else "")
    )


def codex_argv(cfg, job, last):
    f = cfg["frames"]
    return ["codex", "exec", "--ignore-user-config", "-m", f["codex_model"], "-c", f'model_reasoning_effort="{f["codex_effort"]}"',
            "-c", 'forced_login_method="chatgpt"', "--enable", "image_generation",
            "-c", "features.browser_use=false", "-c", "features.computer_use=false", "-c", "features.apps=false",
            "-c", "features.goals=false", "-c", "features.in_app_browser=false", "-c", "features.hooks=false",
            "-c", "features.view_image=false", "-c", "features.multi_agent=false", "-c", "features.skill_search=false",
            "-c", "features.sleep_tool=false", "-c", "features.tool_suggest=false", "-c", "features.plugins=false",
            "-c", "features.personality=false", "-c", "features.mentions_v2=false", "-c", "features.in_app_chat=false",
            "-c", "features.in_app_local_automation=false", "-c", "features.skill_mcp_dependency_install=false",
            "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "-C", str(job.parent),
            "--json", "-o", str(last), "-"]


def run_codex(cfg, job, timeout_s=480):
    events, last, err = job / "events.jsonl", job / "last_message.txt", job / "stderr.txt"
    env = dict(os.environ)
    for k in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
        env.pop(k, None)
    t0 = time.time()
    with (job / "TASK.txt").open("rb") as task, events.open("wb") as out, err.open("wb") as e:
        proc = subprocess.Popen(codex_argv(cfg, job, last), stdin=task, stdout=out, stderr=e, env=env, start_new_session=True)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    texts = []
    for line in events.read_text(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") in ("error", "turn.failed"):
            texts.append(str(ev.get("message") or ev.get("error") or ev))
        it = ev.get("item") or {}
        if ev.get("type") == "item.completed" and it.get("type") == "agent_message":
            texts.append(str(it.get("text") or ""))
    texts += err.read_text(errors="replace").splitlines()[-40:]
    limit = next((t.strip()[:300] for t in texts if any(m in t.lower() for m in LIMIT_MARKERS)), "")
    gen = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "generated_images"
    fresh = [p for p in gen.glob("*/*.png") if p.stat().st_mtime >= t0 - 2]
    newest = max(fresh, key=lambda p: p.stat().st_mtime) if fresh else None
    log(f"codex rc {proc.returncode} in {time.time() - t0:.0f} s, new picture: {newest}, limit: {limit!r}")
    return newest, limit


def gate_subprocess(cfg, raw, cropped):
    """Crop + score in the tools python (cv2/insightface may live in another venv)."""
    r = subprocess.run([tools_python(cfg), "-m", "lipsync_hook.frame_gate", str(raw), cfg["character"]["canon"], str(cropped),
                        str(cfg["character"]["face_height_target"])], capture_output=True, text=True, cwd=cfg["_root"],
                       env=dict(os.environ, PYTHONPATH=cfg["_root"]))
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if not line:
        raise RuntimeError(f"gate failed: {r.stderr[-500:]}")
    return json.loads(line[-1])


def make_frame(cfg, out_dir, tries=None, log_fn=log):
    out = Path(out_dir)
    job = out / "frame_job" / "job"
    job.mkdir(parents=True, exist_ok=True)
    canon = cfg["character"]["canon"]
    if not canon or not Path(canon).exists():
        raise SystemExit("config character.canon must point to the character's reference image")
    refs = [Path(canon)] + [Path(p) for p in cfg["character"]["references"] if Path(p).exists()]
    v = pick_variety(cfg["frames"]["history"])
    note = ""
    for attempt in range(1, (tries or int(cfg["frames"]["tries"])) + 1):
        (job / "prompt.txt").write_text(prompt_text(cfg, v, note))
        (job / "refs.txt").write_text("\n".join(str(p.resolve()) for p in refs[:5]) + "\n")
        (job / "TASK.txt").write_text(TASK)
        log_fn(f"frame attempt {attempt}: {v['setting']} / {v['outfit']} / {v['framing']} / {v['mood']} / {v['light']}")
        png, limit = run_codex(cfg, job)
        if limit:
            raise SystemExit(f"codex limit: {limit}")
        if not png:
            note = "The previous attempt produced no picture; produce exactly one picture."
            continue
        raw = out / f"frame_raw_{attempt}.png"
        shutil.copy(png, raw)
        cropped = out / f"frame_crop_{attempt}.png"
        row = gate_subprocess(cfg, raw, cropped)
        row.update(attempt=attempt, **{k: v[k] for k in ("setting", "outfit", "framing", "mood", "light")})
        (out / f"frame_score_{attempt}.json").write_text(json.dumps(row, indent=1))
        log_fn(f"frame attempt {attempt}: ident {row.get('ident')} face_h {row.get('face_h')} -> {row.get('why')}")
        if row.get("ok"):
            final = out / "start_frame.png"
            shutil.copy(cropped, final)
            return final, row
        why = row.get("why", "")
        fixes = []
        if "face small" in why:
            fixes.append("Frame her closer: waist-up with the face about one third of the picture height.")
        if "faces" in why and "no face" not in why:
            fixes.append("Exactly one person, alone, no reflections or posters with faces.")
        if "mouth open" in why:
            fixes.append("Her lips are closed.")
        if "turned" in why:
            fixes.append("She faces the camera straight on, head level.")
        if "identity" in why:
            fixes.append("Match the face in the references exactly: same eyes, nose, mouth and hair.")
        note = " ".join(fixes) or "Follow the framing exactly."
    raise SystemExit("no start frame passed the gate")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--config")
    ap.add_argument("--dry", action="store_true", help="print the prompt, call nothing")
    x = ap.parse_args()
    cfg = load(x.config)
    if x.dry:
        print(prompt_text(cfg, pick_variety(cfg["frames"]["history"])))
    else:
        final, row = make_frame(cfg, x.out_dir)
        print(json.dumps({"frame": str(final), **row}))
