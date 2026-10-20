#!/usr/bin/env python3
"""bake_lora.py — write a checkpoint with a LoRA MERGED into the weights, so ComfyUI has nothing to patch.

    cd <comfyui dir> && <venv python> <this file> --ckpt ltx-2.3-22b-dev.safetensors \\
        --lora ltxv/ltx-2.3-22b-distilled-lora-384-1.1.safetensors --strength 0.5 \\
        --out models/checkpoints/ltx-2.3-22b-dev-distilled0.5-baked.safetensors

WHY (2026-09-25, after the 17:11 hard freeze): on the GB10's one 121 GiB pool every ComfyUI memory mode holds the 46 GiB
of LTX weights TWICE — dynamic VRAM keeps a staging copy to patch from, high-VRAM keeps a CPU BACKUP of every weight a
LoRA touches (comfy/model_patcher.py patch_weight_to_device), and memory-mapped files cannot be dropped while the
loader streams from them. The distilled LoRA (rank 384, 7.6 GB) touches the whole model, so the backups ARE a second
model. Baked in, there is nothing to back up: the weights sit on the GPU once and the 2x pass gets the rest of the pool.

HOW: a fully STREAMING merge — one tensor in memory at a time, from the source file straight into the output file.
Never a model object (ComfyUI's loader made a second copy and its CPU mode upcast 22B params to fp32: 0 GiB free) and
never a dict of the whole output (safetensors' save_file serialises a second 46 GiB blob: 0 GiB free again, 18:0x).
The safetensors container is written by hand: the header (shapes and dtypes are those of the source) then the tensor
bytes in the same order. Each tensor the LoRA touches is patched on the GPU with ComfyUI's OWN math (comfy.lora.load_lora
+ calculate_weight, the functions the runtime uses, fp32 intermediate, back to bf16); every other tensor (VAE, audio
VAE, vocoder, text projection, norms) is copied byte for byte, so CheckpointLoaderSimple / the audio-VAE and
text-encoder loaders read the new file like the original. Afterwards the file is re-opened and spot-checked against
a fresh patch of the same keys. Aborts itself when MemFree < --floor GiB. Run as the user (no sudo).
"""
import argparse
import json
import os
import struct
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="name under models/checkpoints")
ap.add_argument("--lora", required=True, help="name under models/loras")
ap.add_argument("--strength", type=float, required=True)
ap.add_argument("--out", required=True, help="output path (.safetensors)")
ap.add_argument("--floor", type=float, default=6.0, help="abort when MemFree < this many GiB")
ap.add_argument("--device", default="cuda")
ap.add_argument("--check", type=int, default=4, help="spot-check this many patched + unpatched keys after writing")
a = ap.parse_args()

sys.path.insert(0, os.getcwd())
import comfy.options                      # noqa: E402
comfy.options.enable_args_parsing()
sys.argv = ["main.py", "--disable-pinned-memory"]
import torch                              # noqa: E402
from safetensors import safe_open         # noqa: E402
import folder_paths                       # noqa: E402
import comfy.lora                         # noqa: E402
import comfy.utils                        # noqa: E402

ITEM = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def free_gib() -> float:
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemFree:"):
            return int(ln.split()[1]) / (1 << 20)
    return -1.0


def tensor_bytes(t: torch.Tensor) -> memoryview:
    t = t.contiguous().cpu()
    return memoryview(t.view(torch.uint8).numpy())     # same bytes, no dtype conversion


t0 = time.time()
ckpt = folder_paths.get_full_path_or_raise("checkpoints", a.ckpt)
lora_path = folder_paths.get_full_path_or_raise("loras", a.lora)
out_path = os.path.abspath(a.out)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
tmp_path = out_path + ".part"
print(f"[bake] {ckpt}\n[bake] + {lora_path} x {a.strength} -> {out_path}\n[bake] free {free_gib():.0f} GiB", flush=True)

lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
with safe_open(ckpt, "pt") as f:
    keys = list(f.keys())
    meta = dict(f.metadata() or {})
    key_map = {}     # exactly comfy.lora.model_lora_keys_unet's generic formats for the diffusion model
    for k in keys:
        if k.startswith("model.diffusion_model."):
            mk = k[len("model."):]
            if mk.endswith(".weight"):
                key_map["lora_unet_{}".format(mk[len("diffusion_model."):-len(".weight")].replace(".", "_"))] = mk
                key_map[mk[:-len(".weight")]] = mk
            else:
                key_map[mk] = mk
    patches = comfy.lora.load_lora(lora_sd, key_map, log_missing=True)
    del lora_sd
    # the header: source shapes and dtypes, offsets in key order
    header, off = {}, 0
    for k in keys:
        sl = f.get_slice(k)
        shape, dt = list(sl.get_shape()), sl.get_dtype()
        n = 1
        for d in shape:
            n *= d
        size = n * ITEM[dt]
        header[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + size]}
        off += size
    meta["baked_lora"] = (f"{a.lora} x {a.strength} merged into model.diffusion_model.* by setup/bake_lora.py "
                          f"(comfy.lora.calculate_weight) on {time.strftime('%Y-%m-%d %H:%M')}; other keys unchanged")
    header["__metadata__"] = meta
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    print(f"[bake] {len(keys)} tensors, {len(patches)} patched by the LoRA, {off / 2**30:.1f} GiB of data; streaming on {a.device}", flush=True)
    n_done = n_patched = 0
    with open(tmp_path, "wb") as o:
        o.write(struct.pack("<Q", len(hb)))
        o.write(hb)
        for k in keys:
            if free_gib() < a.floor:
                o.close()
                os.remove(tmp_path)
                sys.exit(f"[bake] ABORT: MemFree {free_gib():.1f} GiB < floor {a.floor}")
            w = f.get_tensor(k)
            mk = k[len("model."):] if k.startswith("model.") else None
            if mk in patches:
                p = [(a.strength, patches[mk], 1.0, None, None)]   # (strength_patch, patch, strength_model, offset, function)
                w32 = w.to(a.device, torch.float32)
                merged = comfy.lora.calculate_weight(p, w32, mk, intermediate_dtype=torch.float32).to(w.dtype)
                o.write(tensor_bytes(merged))
                del w32, merged
                n_patched += 1
            else:
                o.write(tensor_bytes(w))
            del w
            n_done += 1
            if n_done % 500 == 0:
                print(f"[bake] {n_done}/{len(keys)} ({n_patched} patched), free {free_gib():.0f} GiB, {time.time() - t0:.0f} s", flush=True)
    assert n_patched == len(patches), (n_patched, len(patches))
    os.replace(tmp_path, out_path)
    print(f"[bake] written {out_path} {os.path.getsize(out_path) / 2**30:.1f} GiB in {time.time() - t0:.0f} s; spot-checking", flush=True)
    # spot-check: re-open the new file; unpatched keys byte-equal, patched keys equal a fresh patch
    with safe_open(out_path, "pt") as g:
        assert list(g.keys()) == keys
        pk = [k for k in keys if k.startswith("model.") and k[len("model."):] in patches]
        uk = [k for k in keys if not (k.startswith("model.") and k[len("model."):] in patches)]
        step_p, step_u = max(1, len(pk) // a.check), max(1, len(uk) // a.check)
        for k in uk[::step_u][:a.check]:
            assert torch.equal(g.get_tensor(k), f.get_tensor(k)), f"unpatched key differs: {k}"
        worst = 0.0
        for k in pk[::step_p][:a.check]:
            mk = k[len("model."):]
            fresh = comfy.lora.calculate_weight([(a.strength, patches[mk], 1.0, None, None)], f.get_tensor(k).to(a.device, torch.float32), mk,
                                                intermediate_dtype=torch.float32).to(f.get_tensor(k).dtype).cpu()
            got = g.get_tensor(k)
            assert torch.equal(got, fresh), f"patched key differs from a fresh patch: {k}"
            worst = max(worst, (got.float() - f.get_tensor(k).float()).abs().max().item())
        print(f"[bake] spot-check OK: {a.check} unpatched keys byte-equal, {a.check} patched keys equal a fresh ComfyUI patch "
              f"(max |delta| vs the original {worst:.4g})", flush=True)
print(f"[bake] DONE in {time.time() - t0:.0f} s: {out_path}", flush=True)
