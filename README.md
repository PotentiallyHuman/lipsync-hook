# lipsync-hook

From the order **"make a lip-sync hook"** to a clip you can upload and click Post on, in one command, with every
step verified by measurement. Built and proven on a DGX Spark (GB10, 121 GiB unified memory). Deterministic: the
same settings every time, no play-and-measure loops, no guessing.

```
bin/make_hook.py --song song.wav --lyric "first words of the chorus"
```

What comes out of `out/hook_<stamp>/`:

| file | what it is |
|---|---|
| `hook/hook_audio.wav`, `hook/hook.json` | the chorus window: starts on a downbeat, ends 4/8/12/16 bars later, loops |
| `start_frame.png` | a fresh picture of your character (Codex image_gen from your references), face-gated |
| `clip.mp4` | the one-shot lip-sync render WITH the hook audio, for review |
| `clip_silent.mp4` | the same picture without audio: this is what you upload, adding the song from the platform's library |
| `checks.json` | constant frame rate, no jumps, no duplicate frames, identity held, one face |
| `DELIVERY.json` | the one number the upload needs: song time under clip time 0 |

## The five steps

1. **Hook window** (`lipsync_hook/hook_window.py`). Whisper word timestamps + beat tracking + the vocal envelope
   pick a whole-bar chorus window of 8 to 20 s that starts on a downbeat and ends on a downbeat, never inside a
   word, with a loop residue under 25 ms, sung on held notes (pace rule). The hook audio is exactly that window.
2. **Start frame** (`lipsync_hook/frame_gen.py` + `frame_gate.py`). Codex's image tool draws your character from
   your reference images with a setting, outfit, framing, mood and light that all differ from the last three
   frames. The picture is cropped to 9:16 around the face at the right face size and gated: one face, identity to
   the canon >= 0.45, frontal, lips closed. A wrong face never reaches the renderer.
3. **Render** (`lipsync_hook/render.py`). LTX-2.3 22B distilled, two-stage, portrait 544x960 -> 1088x1920, cfg 1.0,
   image-to-video strength 0.92, the hook audio frozen (audio-locked) so only the picture is generated and the
   mouth follows the real song. ONE continuous shot whatever the length (there is no maximum; a split shot snaps
   the character back to the start pose at the cut). A 0.6 s lead-in is rendered before the window and cut off on
   the frame grid, so delivered frame 0 is hook sample 0.
4. **Checks** (`lipsync_hook/check_clip.py`). Measured, not eyeballed: constant 24 fps, zero jumps, duplicate
   frames under 15 %, identity to the start frame >= 0.50, no frame with a second face. Sync and expression you
   judge by eye; every automatic sync metric we tried scored a perfect clip 0.22.
5. **Delivery**. `DELIVERY.json` says: upload `clip_silent.mp4`, add the song from the library, and start the sound
   at `cut_start` seconds under clip time 0. See `docs/FIT.md` for how to place that exactly.

## Memory on a unified-memory box (the part that cost us most)

- The 22B weights (46 GiB) are held ONCE: the distilled LoRA is merged into the checkpoint (`setup/bake_lora.py`),
  ComfyUI runs `--highvram`, nothing is patched at runtime.
- The server runs through `comfy_capped.py`: a live per-process CUDA cap that keeps the last 8 GiB free, so an
  overrun fails cleanly inside the process instead of freezing the box.
- A cache warden drops the page cache whenever MemFree falls under 30 GiB during a render (the kernel does not
  evict cache for GPU allocations here).
- The server is started fresh for each render and stopped after it (an aged server gets 10x slower; an idle one
  holds 50 to 60 GiB).
- Never a length cap as a memory fix. A 20 s shot is about 500 frames and renders in about 2 hours.

## Use it from an agent

`SKILL.md` is written for Claude Code, Codex or any agent that reads skills: it names the one command, the
preconditions, what to verify after each step and what never to change. Put this folder in your skills path or
tell the agent to read `SKILL.md` first.

## Not in this repo

Uploading, scheduling and posting. Do those yourself on your own account. `docs/FIT.md` explains how to make the
added song fit the clip exactly.

See `setup/INSTALL.md` for the one-time setup (ComfyUI + LTX nodes + models + the Codex CLI), `docs/SETTINGS.md`
for every setting and why it is what it is.
