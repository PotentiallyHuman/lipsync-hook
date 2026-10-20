---
name: lipsync-hook
description: Make a lip-sync hook clip of a character for a short-video platform, from a song recording to a verified clip ready to upload, in one command. Use when the user says "make a lip-sync hook", "make a hook of <character>", "render her singing the chorus", or wants the chorus window, start frame, render and checks done deterministically.
---

# lipsync-hook — the skill

## The one command
```
cd <repo> && bin/make_hook.py --song <recording.wav> [--lyric "<first words of the chorus>"] [--frame <picture.png>]
```
It prints each step, writes `out/hook_<stamp>/` and ends with `DELIVERY.json`. Read `DELIVERY.json` back to the user:
the clip path and the one placement number (song time under clip time 0).

## Preconditions (check before running, say which one is missing)
1. `config.json` exists (copy `config.example.json`), with `character.canon` = the character's reference picture and
   `comfy.dir` / `comfy.python` pointing at a ComfyUI that has the LTX nodes and the BAKED checkpoint (`setup/INSTALL.md`).
2. The box is idle: no other model loaded (ollama, other servers). `python -m lipsync_hook.comfy_server status` must
   show at least 60 GiB really free after a cache drop, or the server refuses to launch.
3. The Codex CLI is logged in (ChatGPT subscription) if `frames.generator` is `codex`; otherwise pass `--frame`.
4. `sudo -n sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches` works without a password (the cache warden needs it).

## What to verify after each step (the tool prints it; you confirm it)
- Hook: `hook/hook.json` has `bars` in {4,8,12,16}, `length_s` in [8,20], `residue_ms` < 25, `pace.ok` true.
- Frame: `frame_score_*.json` / the gate line says `ok`; identity >= 0.45; face_h >= 0.28.
- Render: `render.json` exists, `frames_rendered` = 8n+1, one shot; `clip_silent.mp4` has a video stream and no audio.
- Checks: `checks.json` `pass` true. If false, re-run (new seed / new frame); never loosen a threshold to pass.

## Never change (each was proven the hard way)
- cfg stays 1.0; never add face words to the prompt; never drive the render with a vocal stem; never split a long hook
  into shots; never cap the shot length to fix memory (fix memory: warden, fresh server, unload other models).
- Never place the added song by playing and measuring. The placement is a number: `cut_start` from `hook.json`.
- Lyrics never go into logs, messages or file names (the hook window finder uses them only to locate the chorus).

## When it fails
- "real free N GiB < 60": something else holds the memory; find and stop it, then retry.
- Render error from ComfyUI: open `comfy.log`; "out of memory" means the live cap refused an allocation; run again
  on an idle box (the first render after a reboot is the cleanest).
- No bar-aligned window: try `--lyric` with the chorus's first words, `--from <seconds>`, or another song. Rapid-fire
  choruses are rejected by design.

## Uploading is yours
This skill ends at the Post button. Upload `clip_silent.mp4`, add the song from the platform's library and fit it with
`docs/FIT.md` (the pin, the timeline zoom, the read-back). Do it on your own account, by hand or with your own tooling.
