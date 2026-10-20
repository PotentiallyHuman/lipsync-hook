# One-time setup (DGX Spark / GB10, Ubuntu, 121 GiB unified memory)

## 1. ComfyUI with the LTX nodes
- ComfyUI (any recent main). Custom nodes: `ComfyUI-LTXVideo` (Lightricks). The graph also uses core nodes only
  otherwise (`CheckpointLoaderSimple`, `CFGGuider`, `SamplerCustomAdvanced`, `LatentUpscaleModelLoader`, `LoadAudio`,
  `CreateVideo`, `SaveVideo`, `ResizeImageMaskNode`).
- Models under `ComfyUI/models/`:
  - `checkpoints/ltx-2.3-22b-dev.safetensors` (base, 46 GiB)
  - `loras/ltxv/ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (distilled LoRA)
  - `latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
  - `text_encoders/comfy_gemma_3_12B_it.safetensors` (the LTX AV text encoder)
  - your character's LTX LoRA, if you trained one (`loras/<character>_ltx.safetensors`)
- Start ComfyUI once by hand and load `workflows/ltx23_two_stage_av.json` to see every node resolve.

## 2. Bake the checkpoint (once)
The distilled LoRA merged into the checkpoint, then (optionally) your character LoRA merged into that:
```
cd <ComfyUI dir>
<comfy python> <repo>/setup/bake_lora.py --ckpt ltx-2.3-22b-dev.safetensors \
    --lora ltxv/ltx-2.3-22b-distilled-lora-384-1.1.safetensors --strength 0.5 \
    --out models/checkpoints/ltx-2.3-22b-dev-distilled0.5-baked.safetensors
<comfy python> <repo>/setup/bake_lora.py --ckpt ltx-2.3-22b-dev-distilled0.5-baked.safetensors \
    --lora <character>_ltx.safetensors --strength 1.0 \
    --out models/checkpoints/ltx-2.3-22b-dev-distilled0.5-<character>1.0-baked.safetensors
```
About 25 s and 43 GiB of free memory each; the script streams one tensor at a time and spot-checks the result.
Put the final name in `config.json` `models.checkpoint`.

## 3. The tools python (word timestamps, beats, faces)
```
python3 -m venv ~/venvs/lipsync-tools && ~/venvs/lipsync-tools/bin/pip install -r requirements.txt
```
`requirements.txt`: faster-whisper, librosa, soundfile, numpy, opencv-python-headless, insightface, onnxruntime, pillow.
insightface downloads `buffalo_l` on first use. Optional: demucs (`pip install demucs`) for a vocal stem; set
`hook.demucs` to its binary. Put the venv's python in `config.json` `tools_python`.

## 4. Codex CLI (start frames)
`npm i -g @openai/codex`, then `codex login` with a ChatGPT subscription. The frame generator uses the built-in
`image_gen` tool (max 5 reference images). Alternative: pass your own picture with `--frame` (it is still gated).

## 5. Passwordless cache drop
The cache warden and the launch gate need:
```
echo "$USER ALL=(root) NOPASSWD: /usr/bin/sync, /usr/bin/tee /proc/sys/vm/drop_caches" | sudo tee /etc/sudoers.d/lipsync-hook
```

## 6. config.json
`cp config.example.json config.json`, set `comfy.dir`, `comfy.python`, `models.checkpoint`, `character.canon`
(one reference picture of your character), `character.references` (up to 4 accepted frames), `tools_python`.

## 7. Smoke test
```
python -m lipsync_hook.comfy_server status        # free memory, server state
bin/make_hook.py --song some_song.wav --lyric "first words" --dry   # hook window + start frame, no render
bin/make_hook.py --song some_song.wav --lyric "first words"         # the whole thing (~1 h for a 10 s hook)
```
