# Settings, and why each one is what it is

Every number here was reached by a failed render, not by taste. Change one only with a side-by-side on the same hook.

## Model and server
| setting | value | why |
|---|---|---|
| model | LTX-2.3 22B dev, distilled LoRA 384 at 0.5 **baked into the checkpoint** | on unified memory every ComfyUI mode holds the weights twice when a LoRA is patched at runtime (a CPU backup of every touched weight); merged in, they sit on the GPU once |
| character LoRA | baked into the same checkpoint at 1.0 (`setup/bake_lora.py` twice) | a runtime character LoRA at two strengths = two patched copies = out of memory on the refine pass; baked, identity holds in every frame |
| server flags | `--highvram --disable-pinned-memory --disable-async-offload`, run through `comfy_capped.py` | highvram keeps the weights resident; async offload deadlocks the latent upsampler; the capped wrapper fails cleanly instead of freezing the box |
| CUDA floor | 8 GiB | the live cap: allowed = reserved + (MemFree - 8 GiB), every second |
| launch gate | MemFree >= 60 GiB after a cache drop | MemAvailable lies on unified memory (GPU-resident weights count as page cache); only MemFree is real |
| cache warden | drop the page cache when MemFree < 30 GiB, every 5 s | the kernel does not evict cache for a GPU allocation; the default 14 GiB / 20 s was too slow for the refine pass's bursts |
| fresh server per render | yes | after ~90 min of process uptime the refine pass slows 10x; an idle server holds 50 to 60 GiB |

## Render
| setting | value | why |
|---|---|---|
| fps | 24 | the proven references; the lead-in cut lands on a whole frame (15 frames = 0.625 s) |
| cfg | 1.0 (both guiders) | raising it to fix a symptom made expression and motion worse every time |
| image-to-video strength | 0.92 | identity from the start frame, with room to move |
| freeze mask | video 1.0, audio 0.0 (`LTXVSetAudioVideoMaskByTime`) | the audio latent is the real song; only the picture is generated, so the mouth follows the song's timing |
| stage 1 | 544x960 portrait, x2 spatial upsampler to 1088x1920, delivered 1080x1920 | native portrait; the refine pass is where identity detail lands |
| frames | 8n+1 of (window + 0.6 s lead-in), min 73 | LTX's grid; the lead-in holds the low-content warm-up and is cut off |
| shot length | the whole hook, one shot | there is no maximum (the engine's own comment: "effectively unlimited; the costs are time and memory"); 33 s single shots exist; a 2-shot clip snaps back to the start pose at the cut |
| prompt | action + camera only: "She sings the line straight to the camera, natural lip movement, warm expression; the camera drifts slowly and steadily sideways." | face words, "static camera" and emotion adjectives all made it worse |
| negative | the quality/anatomy list in `config.example.json` | inert at cfg 1.0, kept for the record |
| time | about 6 min per second of shot on a GB10 (7.4 s = 38 min, 20 s = about 2 h) | |

## Start frame
| setting | value | why |
|---|---|---|
| source | Codex image_gen with up to 5 references (canon first) | the references are what keep the identity; a frame without them is a generic person |
| framing | chest-up to waist-up, face about a third of the height after the 9:16 crop, lips closed | a face under 0.28 of the height makes the detail pass invent skin (moles); an open mouth fights the audio |
| gate | one face, identity cosine >= 0.45 to the canon, yaw <= 25, pitch <= 20, mouth gap <= 0.04 | measured on the frames that rendered well vs the ones that drifted |
| variety | setting, outfit, framing, mood, light, hair, camera all differ from the last three | a feed of identical frames reads as one video |

## Hook window
| setting | value | why |
|---|---|---|
| length | 8 to 20 s | long enough for a whole chorus, loops on the platform |
| bars | 4, 8, 12 or 16, downbeat to downbeat | the clip loops over and over without a seam; at 100 bpm 4 bars = 9.6 s, 8 bars = 19.2 s |
| ends | start in a vocal gap or on the first word's onset; end in a gap or on a word onset; never inside a word | a cut word reads as a mistake |
| loop residue | < 25 ms | tracker jitter is about 20 ms; above that the loop audibly stutters |
| pace | words/s <= 1.9, no 4 words closer than 0.28 s, syllables/s <= 3.0, words/s >= 0.8 | rapid-fire choruses do not lip-sync well; held notes do |
| words | faster-whisper, VAD off, on the vocal stem if demucs is configured | every VAD finds "no speech" in sung music |

## Checks (accept/reject by measurement)
constant frame rate; zero jumps (frame diff > median + 6 MAD); duplicate frames < 15 %; identity to the start frame
mean >= 0.50; no frame with a second face (a "materialising" hand or mouth shows up as one). Sync and expression are
judged by eye: the mouth-openness-vs-audio metrics scored a perfect clip 0.22.
