"""Start-frame gate: crop a generated picture to 9:16 around the face at the target face size, then score it
against the character's canon face. A frame that fails is never rendered (a wrong face would be a wasted hour).

Gate (measured on the music-video references): exactly one face; identity cosine >= identity_min (default 0.45);
face height >= 0.28 of the frame after the crop (small faces make the detail pass invent skin); frontal or slight 3/4
(|yaw| <= 25, |pitch| <= 20); lips relaxed (mouth gap <= 0.04 of face height) so the audio drives the opening.

Run with a python that has cv2 + insightface (buffalo_l, CPU is fine)."""
import json
import sys

import cv2
import numpy as np

_APP = None


def app():
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis
        _APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _APP.prepare(ctx_id=-1, det_size=(640, 640))
    return _APP


def biggest(fs):
    return max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def crop_to_face(src, dst, face_h_target=0.31, min_w=720):
    """9:16 window around the face so the face is face_h_target of the height, face centre ~36 % from the top;
    upscaled to min_w wide if the window is narrower (a mild upscale is what accepted frames had)."""
    img = cv2.imread(str(src))
    h, w = img.shape[:2]
    fs = app().get(img)
    if len(fs) != 1:
        tw = round(h * 9 / 16)
        out = img[:, max((w - tw) // 2, 0):][:, :tw]
        cv2.imwrite(str(dst), out)
        return out.shape[1], out.shape[0], len(fs)
    x0, y0, x1, y1 = fs[0].bbox
    cx, cy, fh = (x0 + x1) / 2, (y0 + y1) / 2, (y1 - y0)
    th = min(h, fh / face_h_target)
    tw = th * 9 / 16
    if tw > w:
        tw, th = w, w * 16 / 9
    cx0 = min(max(cx - tw / 2, 0), w - tw)
    cy0 = min(max(cy - th * 0.36, 0), h - th)
    out = img[int(cy0):int(cy0 + th), int(cx0):int(cx0 + tw)]
    if out.shape[1] < min_w:
        out = cv2.resize(out, (min_w, round(min_w * 16 / 9)), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(dst), out)
    return out.shape[1], out.shape[0], len(fs)


def score(frame, canon, identity_min=0.45):
    rf = app().get(cv2.imread(str(canon)))
    if not rf:
        raise SystemExit("no face in the canon reference")
    remb = biggest(rf).normed_embedding
    img = cv2.imread(str(frame))
    fs = app().get(img)
    if not fs:
        return {"why": "no face", "ok": False}
    f = biggest(fs)
    x0, y0, x1, y1 = [int(v) for v in f.bbox]
    h = img.shape[0]
    face_h = (y1 - y0) / h
    pitch, yaw, roll = [float(v) for v in f.pose]
    lm = f.landmark_3d_68
    mouth = float(np.linalg.norm(lm[66][:2] - lm[62][:2]) / max(y1 - y0, 1))
    ident = float(np.dot(remb, f.normed_embedding))
    why = []
    if len(fs) != 1:
        why.append(f"{len(fs)} faces")
    if face_h < 0.28:
        why.append(f"face small {face_h:.2f}")
    if abs(yaw) > 25 or abs(pitch) > 20:
        why.append(f"turned yaw {yaw:.0f} pitch {pitch:.0f}")
    if mouth > 0.04:
        why.append(f"mouth open {mouth:.3f}")
    if ident < identity_min:
        why.append(f"identity {ident:.2f}")
    return {"faces": len(fs), "face_h": round(face_h, 3), "yaw": round(yaw, 1), "pitch": round(pitch, 1), "mouth": round(mouth, 3),
            "ident": round(ident, 3), "w": img.shape[1], "h": h, "why": "; ".join(why) or "ok", "ok": not why}


if __name__ == "__main__":
    # frame_gate.py RAW.png CANON.png OUT.png [face_h_target]
    raw, canon, out = sys.argv[1:4]
    tgt = float(sys.argv[4]) if len(sys.argv) > 4 else 0.31
    w, h, n = crop_to_face(raw, out, tgt)
    print(json.dumps({"crop": [w, h, n], **score(out, canon)}))
