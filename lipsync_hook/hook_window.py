"""Find the hook: a bar-aligned chorus window that LOOPS, from a song recording and (optionally) a lyric hint.

    python -m lipsync_hook.hook_window SONG.wav OUT_DIR [--lyric "first words of the chorus"] [--from 0] [--config ...]
        -> OUT_DIR/hook_audio.wav (exactly the window) + OUT_DIR/hook.json (cut_start, cut_end, bars, bpm, ...)

Rule (what works on short-video platforms and what the renderer wants):
- the window starts on a DOWNBEAT (the beat nearest the chorus's first sung onset) and ends on the downbeat exactly
  4, 8, 12 or 16 bars later, so the clip loops over and over; loop residue (end - start - bars x bar) < 25 ms;
- length 8-20 s; never starts or ends inside a sung word (start in a vocal gap or on the first word's onset; end in a
  gap or on a word onset);
- pace: sung on held notes, not chanted: words/s <= 1.9, no run of 4 words with onset gaps < 0.28 s, syllables/s <= 3.0,
  words/s >= 0.8 (a rapid-fire chorus never lip-syncs well).
Tools: faster_whisper (word timestamps, VAD off: VADs find no speech in sung music), librosa beat tracking on the mix,
optional demucs vocal stem for the gaps (configure hook.demucs); without it the mix envelope is used."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import load


def vocals(cfg, song, workdir):
    d = cfg["hook"].get("demucs")
    if not d:
        return song
    out = Path(workdir) / "stems"
    subprocess.run([d, "--two-stems=vocals", "-n", "htdemucs", "-o", str(out), str(song)], check=True, capture_output=True)
    cand = list(out.glob("htdemucs/*/vocals.wav"))
    return cand[0] if cand else song


def words_of(cfg, wav):
    from faster_whisper import WhisperModel
    import librosa
    y, sr = librosa.load(str(wav), sr=16000, mono=True)
    m = WhisperModel(cfg["hook"]["whisper_model"], device="cpu", compute_type="int8")
    segs, _ = m.transcribe(y, language="en", vad_filter=False, word_timestamps=True, beam_size=5)
    return [(w.start, w.end, w.word.strip()) for s in segs for w in (s.words or [])]


def envelope(wav):
    import librosa
    y, sr = librosa.load(str(wav), sr=16000, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=800, hop_length=80)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=80)
    level = rms / max(np.percentile(rms, 95), 1e-9)
    return t, level


def beats_of(song):
    import librosa
    y, sr = librosa.load(str(song), sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return float(np.atleast_1d(tempo)[0]), np.array(beats)


def in_gap(t, env_t, env_l, gap_min, thr=0.6):
    """Is time t inside a quiet stretch (level < thr of the loud reference) at least gap_min long?"""
    i = int(np.searchsorted(env_t, t))
    if i <= 0 or i >= len(env_l):
        return True
    lo = i
    while lo > 0 and env_l[lo - 1] < thr:
        lo -= 1
    hi = i
    while hi < len(env_l) - 1 and env_l[hi + 1] < thr:
        hi += 1
    return env_l[i] < thr and (env_t[hi] - env_t[lo]) >= gap_min


def syllables(word):
    return max(1, len(re.findall(r"[aeiouy]+", word.lower())))


def pace(words, t0, t1, h):
    ws = [w for w in words if t0 <= w[0] < t1]
    if len(ws) < 2:
        return {"ok": False, "why": "too few words"}
    span = max(ws[-1][1] - ws[0][0], 0.5)
    wps = len(ws) / span
    sps = sum(syllables(w[2]) for w in ws) / span
    run, longest = 1, 1
    for a, b in zip(ws, ws[1:]):
        run = run + 1 if (b[0] - a[1]) < h["rapid_run_gap_s"] else 1
        longest = max(longest, run)
    why = []
    if wps > h["words_per_s_max"]:
        why.append(f"words/s {wps:.2f} > {h['words_per_s_max']}")
    if wps < h["words_per_s_min"]:
        why.append(f"words/s {wps:.2f} < {h['words_per_s_min']}")
    if longest >= h["rapid_run_len"]:
        why.append(f"rapid run of {longest} words")
    if sps > h["syllables_per_s_max"]:
        why.append(f"syllables/s {sps:.2f} > {h['syllables_per_s_max']}")
    return {"ok": not why, "why": "; ".join(why) or "ok", "words_per_s": round(wps, 2), "syllables_per_s": round(sps, 2),
            "median_word_s": round(float(np.median([w[1] - w[0] for w in ws])), 2), "words": len(ws)}


def find_chorus_onset(words, lyric, from_s):
    """The first sung word onset of the chorus: by the lyric hint (fuzzy, first 3 words), else the first word after from_s."""
    ws = [w for w in words if w[0] >= from_s]
    if not ws:
        return None
    if lyric:
        key = [re.sub(r"[^a-z']", "", t.lower()) for t in lyric.split()][:3]
        for i in range(len(ws)):
            got = [re.sub(r"[^a-z']", "", w[2].lower()) for w in ws[i:i + len(key)]]
            if got == key:
                return ws[i][0]
    return ws[0][0]


def choose(cfg, song, out_dir, lyric="", from_s=0.0, log=print):
    h = cfg["hook"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    voc = vocals(cfg, Path(song), out)
    words = words_of(cfg, voc)
    env_t, env_l = envelope(voc)
    bpm, beats = beats_of(song)
    bar = 4 * 60 / bpm
    total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(song)],
                                 capture_output=True, text=True).stdout.strip())
    onset = find_chorus_onset(words, lyric, from_s)
    if onset is None:
        raise SystemExit("no sung words found")
    # the beat grid: a robust linear fit through the tracked beats around the onset; downbeat = beat nearest the onset
    near = beats[(beats > onset - 8 * bar) & (beats < onset + 8 * bar)]
    if len(near) < 8:
        near = beats
    k = np.arange(len(near))
    A = np.vstack([k, np.ones_like(k)]).T
    beat_s, b0 = np.linalg.lstsq(A, near, rcond=None)[0]
    bar = 4 * beat_s
    bpm = 60 / beat_s
    i0 = int(round((onset - b0) / beat_s))
    down = b0 + i0 * beat_s
    if onset <= 0.10:          # the recording starts ON the chorus: the recording start is the downbeat
        down = 0.0
    starts = [down, down - bar]  # the onset's downbeat, or the previous bar's downbeat (keeps a pickup)
    cands = []
    for s in starts:
        if s < 0 or s > onset + 0.08:
            continue
        s_clean = in_gap(s, env_t, env_l, 0.15) or abs(s - onset) <= 0.08 or s == 0.0
        if not s_clean:
            continue
        for nb in h["bars_options"]:
            e = s + nb * bar
            L = e - s
            if L < h["min_s"] or L > h["max_s"] or e > total - 0.05:
                continue
            onsets = [w[0] for w in words]
            e_on_onset = any(abs(e - o) <= 0.08 for o in onsets)
            inside_word = any(w[0] + 0.08 < e < w[1] for w in words)
            e_clean = in_gap(e, env_t, env_l, h["gap_min_s"]) or e_on_onset
            if not e_clean or inside_word:
                continue
            # loop residue on the raw tracker beats at both cut points
            res_ms = 1000 * ((min(abs(beats - e)) if len(beats) else 0) + (min(abs(beats - s)) if len(beats) and s > 0 else 0))
            if res_ms > h["loop_residue_ms"] * 2:
                continue
            p = pace(words, s, e, h)
            cands.append({"cut_start": round(float(s), 3), "cut_end": round(float(e), 3), "bars": nb, "length_s": round(float(L), 3),
                          "bpm": round(float(bpm), 1), "bar_s": round(float(bar), 3), "residue_ms": round(float(res_ms), 1),
                          "start_case": "recording start" if s == 0.0 else ("downbeat on onset" if abs(s - onset) <= 0.08 else "previous downbeat (pickup kept)"),
                          "end_case": "word onset" if e_on_onset else "vocal gap", "pace": p})
    ok = [c for c in cands if c["pace"]["ok"]]
    log(f"[hook] {bpm:.1f} bpm, bar {bar:.3f} s, chorus onset {onset:.2f} s, candidates {len(cands)}, pace-ok {len(ok)}")
    for c in cands:
        log(f"[hook]   {c['cut_start']:.2f}-{c['cut_end']:.2f} {c['bars']} bars {c['length_s']:.1f} s residue {c['residue_ms']:.0f} ms "
            f"start={c['start_case']} end={c['end_case']} pace={c['pace']['why']}")
    if not ok:
        raise SystemExit("no bar-aligned window passed (see the candidates above); try another song, a different --lyric hint or --from")
    best = sorted(ok, key=lambda c: (abs(c["cut_start"] - onset) > 0.08, -c["length_s"]))[0]
    wav = out / "hook_audio.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(song), "-ss", f"{best['cut_start']:.3f}", "-t", f"{best['length_s']:.3f}",
                    "-ar", "48000", "-ac", "2", str(wav)], check=True)
    best["song"] = str(song)
    best["hook_audio"] = str(wav)
    best["placement"] = {"clip_time_0_is_song_time_s": best["cut_start"], "frame_30fps": int(round(best["cut_start"] * 30))}
    (out / "hook.json").write_text(json.dumps(best, indent=1))
    log(f"[hook] chosen {best['cut_start']:.3f}-{best['cut_end']:.3f} s ({best['bars']} bars, {best['length_s']:.2f} s) -> {wav}")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("out_dir")
    ap.add_argument("--lyric", default="", help="first words of the chorus (used only to find it; never printed)")
    ap.add_argument("--from", dest="from_s", type=float, default=0.0)
    ap.add_argument("--config")
    x = ap.parse_args()
    choose(load(x.config), x.song, x.out_dir, x.lyric, x.from_s)
