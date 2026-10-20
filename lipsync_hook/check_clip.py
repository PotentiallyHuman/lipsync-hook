"""Measure a rendered clip on every row that decides accept/reject (never "it looks fine"):

    python -m lipsync_hook.check_clip CLIP.mp4 START_FRAME.png [--identity-min 0.50]

- container: constant frame rate, frame count, duration;
- stutter: exact-duplicate frames (fraction, longest run) and sudden jumps (frame diff > median + 6 MAD);
- identity: insightface cosine of the face in sampled frames to the start-frame face (min / mean), frames without a face;
- ghosts: frames with more than one face (a hand or mouth "materialising" shows up as a second face).
Pass = cfr, 0 jumps, dup_frac < 0.15, id_mean >= identity_min, no frame with extra faces. Lip-sync and expression are
judged BY EYE: every mouth-openness-vs-audio metric tried scored a perfect clip 0.22, so none is used here."""
import json
import subprocess
import sys

import cv2
import numpy as np


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=avg_frame_rate,r_frame_rate,nb_read_frames,width,height:format=duration", "-of", "json", path],
                         capture_output=True, text=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    def fr(x):
        from fractions import Fraction
        try:
            f = Fraction(x)
            return round(float(f), 3) if f else None
        except (ValueError, ZeroDivisionError):
            return None
    pts = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.split()
    t = np.array(sorted(float(x) for x in pts if x and x != "N/A"))
    dt = np.diff(t) if len(t) > 1 else np.array([0])
    return {"w": s.get("width"), "h": s.get("height"), "avg_fps": fr(s["avg_frame_rate"]), "frames": int(s.get("nb_read_frames", 0)),
            "duration": round(float(d["format"]["duration"]), 3), "cfr": bool(len(dt) and (dt.max() - dt.min()) < 1e-3)}


def frames_gray(path, size=(180, 320)):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), size).astype(np.float32))
    cap.release()
    return out


def stutter(gray):
    d = np.array([np.abs(gray[i] - gray[i - 1]).mean() for i in range(1, len(gray))])
    dup = d < 0.3
    run = best = 0
    for x in dup:
        run = run + 1 if x else 0
        best = max(best, run)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) + 1e-6
    return {"dup_frac": round(float(dup.mean()), 3), "longest_dup_run": best, "jumps": int((d > med + 6 * mad).sum()),
            "median_motion": round(med, 3)}


def faces(path, start_png, every=5):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    def emb(img):
        fs = [f for f in app.get(img) if f.det_score > 0.6]
        if not fs:
            return None, 0
        f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return f.normed_embedding, len(fs)
    ref, _ = emb(cv2.imread(start_png))
    if ref is None:
        return {"identity": "no face in start frame"}
    cap = cv2.VideoCapture(path)
    sims, i, miss, extra = [], 0, 0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            e, n = emb(f)
            if e is None:
                miss += 1
            else:
                sims.append(float(np.dot(ref, e)))
                if n > 1:
                    extra += 1
        i += 1
    cap.release()
    return {"id_min": round(min(sims), 3) if sims else None, "id_mean": round(float(np.mean(sims)), 3) if sims else None,
            "frames_without_face": miss, "frames_with_extra_faces": extra}


def check(clip, start_png, identity_min=0.50):
    p = probe(clip)
    res = {**p, **stutter(frames_gray(clip)), **faces(clip, start_png)}
    checks = {"cfr": bool(res["cfr"]), "no_jumps": res["jumps"] == 0, "dup_frac<0.15": res["dup_frac"] < 0.15,
              f"id_mean>={identity_min}": (res.get("id_mean") or 0) >= identity_min, "one_face": res.get("frames_with_extra_faces", 99) == 0}
    res["checks"] = checks
    res["pass"] = all(checks.values())
    return res


if __name__ == "__main__":
    clip, start = sys.argv[1:3]
    imin = float(sys.argv[sys.argv.index("--identity-min") + 1]) if "--identity-min" in sys.argv else 0.50
    print(json.dumps(check(clip, start, imin)))
