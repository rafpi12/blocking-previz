"""screen_comp.py - put UI onto blank monitor screens in AI-generated video so it looks filmed.

Pure CPU: numpy, opencv-python-headless, Pillow, onnxruntime (person matte), imageio-ffmpeg (if no ffmpeg).
No internet needed: the person-matte model ships in ../assets/modnet_fp16.onnx (MODNet, Apache-2.0).

Typical use (see references/ui_overlay.md):
    from screen_comp import *
    info = probe('screen_1.mp4')                       # fps, size, frames, cuts
    preview_detect('screen_1.mp4', 'check.png')         # look at the detected screen outline
    quads = track_screen('screen_1.mp4')                # per-frame corners (TL, TR, BR, BL)
    ui = VideoUI('UI_1.mp4')                            # or ImageUI / PlayerUI / any f(i, t) -> BGR image
    composite_video('screen_1.mp4', quads, ui, 'screen_1_UI.mp4')
"""
import os, math, json, shutil, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, '..', 'assets', 'modnet_fp16.onnx')


def gblur(img, sigma):
    """Gaussian blur; large sigmas run on a downscaled copy (same look, ~10-50x faster on 1080p)."""
    if sigma < 3: return cv2.GaussianBlur(img, (0, 0), sigma)
    f = max(2, int(sigma / 1.5)); h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# ============================================================================ video io
def _ffmpeg():
    exe = shutil.which('ffmpeg')
    if exe: return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise SystemExit('ffmpeg not found: pip install imageio-ffmpeg')


def probe(path):
    """fps, width, height, frame count, and likely hard cuts (frame indices where a new shot starts)."""
    c = cv2.VideoCapture(path); fps = c.get(cv2.CAP_PROP_FPS) or 30
    prev, cuts, n = None, [], 0
    while True:
        ok, f = c.read()
        if not ok: break
        g = cv2.resize(f, (96, 54)).astype(np.float32)
        if prev is not None and np.abs(g - prev).mean() > 28: cuts.append(n)
        prev = g; n += 1
    w, h = int(c.get(3)), int(c.get(4))
    return dict(fps=fps, width=w, height=h, frames=n, duration=n / fps, cuts=cuts)


def frames(path):
    c = cv2.VideoCapture(path)
    while True:
        ok, f = c.read()
        if not ok: return
        yield f


def get_frame(path, i):
    c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = c.read(); return f if ok else None


class Writer:
    """Pipe BGR frames to H.264. Audio: copy from `audio_from` (plate) or pass an ffmpeg filter via mix_audio()."""
    def __init__(self, path, w, h, fps):
        self.tmp = path + '.noaudio.mp4'; self.path, self.fps = path, fps
        self.p = subprocess.Popen([_ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
                                   '-r', f'{fps}', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', self.tmp],
                                  stdin=subprocess.PIPE)
    def write(self, f): self.p.stdin.write(np.ascontiguousarray(f).tobytes())
    def close(self, audio_from=None, extra_audio=None, filter_complex=None):
        """audio_from: plate video (keeps its sound). extra_audio: list of files mixed in with filter_complex
        (inputs: [1:a] = plate, [2:a].. = extras; must output [a])."""
        self.p.stdin.close(); self.p.wait()
        cmd = [_ffmpeg(), '-y', '-loglevel', 'error', '-i', self.tmp]
        if audio_from: cmd += ['-i', audio_from]
        for e in (extra_audio or []): cmd += ['-i', e]
        if filter_complex: cmd += ['-filter_complex', filter_complex, '-map', '0:v', '-map', '[a]']
        elif audio_from: cmd += ['-map', '0:v', '-map', '1:a?']
        cmd += ['-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', self.path] if (audio_from or filter_complex) else ['-c', 'copy', self.path]
        subprocess.run(cmd, check=True); os.remove(self.tmp)


# ============================================================================ screen detection & tracking
def order(p):
    p = np.asarray(p, float); s = p.sum(1); d = p[:, 0] - p[:, 1]
    return np.array([p[np.argmin(s)], p[np.argmax(d)], p[np.argmax(s)], p[np.argmin(d)]])   # TL, TR, BR, BL


def screen_mask(fr, seed, vfrac=0.72, sadd=45):
    """Bright, low-saturation region connected to `seed` (x, y): the glowing blank screen."""
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV); x, y = int(seed[0]), int(seed[1])
    patch = hsv[max(0, y - 15):y + 15, max(0, x - 15):x + 15].reshape(-1, 3)
    sv, ss = np.median(patch[:, 2]), np.median(patch[:, 1])
    m = ((hsv[..., 2] > sv * vfrac) & (hsv[..., 1] < ss + sadd)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(m)
    l = lab[min(y, m.shape[0] - 1), min(x, m.shape[1] - 1)]
    if l == 0:
        ys, xs = np.nonzero(m)
        if len(xs) == 0: return None
        k = np.argmin((xs - x) ** 2 + (ys - y) ** 2); l = lab[ys[k], xs[k]]
    mm = (lab == l).astype(np.uint8)
    return cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def quad_from_mask(mm):
    cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea); h = cv2.convexHull(c); peri = cv2.arcLength(h, True)
    for e in np.linspace(0.005, 0.08, 40):
        a = cv2.approxPolyDP(h, e * peri, True)
        if len(a) == 4:
            q = order(a.reshape(-1, 2))
            if min(np.linalg.norm(q[k] - q[(k + 1) % 4]) for k in range(4)) > 8: return q
    return order(cv2.boxPoints(cv2.minAreaRect(c)))


def _line_fit(pts):
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _inter(l1, l2):
    p1, d1 = l1; p2, d2 = l2; A = np.array([d1, -d2]).T; t = np.linalg.solve(A, p2 - p1); return p1 + t[0] * d1


def refine_edges(mask, q0, tol=10, min_pts=20):
    """Fit straight lines to the mask outline near each edge of q0 (sub-pixel corners, ignores rounded/occluded bits)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs: return q0
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float); lines = []
    for k in range(4):
        a, b = q0[k], q0[(k + 1) % 4]; d = b - a; L = np.linalg.norm(d); u = d / L; n = np.array([-u[1], u[0]])
        rel = c - a; t = rel @ u; dist = np.abs(rel @ n)
        sel = (dist < tol) & (t > 0.06 * L) & (t < 0.94 * L)
        lines.append(_line_fit(c[sel]) if sel.sum() > min_pts else (a, u))
    try: return np.array([_inter(lines[(k - 1) % 4], lines[k]) for k in range(4)])
    except np.linalg.LinAlgError: return q0


def detect_screen(fr, seed=None, exclude=None, hidden=None, ratio=1.6):
    """Find the blank glowing screen in a frame. seed=(x, y) inside the screen helps if there are several bright
    areas. exclude = 0..1 matte (e.g. person_matte) of things in front of the screen to ignore.
    hidden='left'/'right': that side of the screen is covered (over-the-shoulder). The covered corners are rebuilt
    along the visible top/bottom edges at width = ratio x screen height (1.6 ~ a 16:9 screen seen at an angle),
    so the quad follows the monitor, not the outline of the head."""
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    if exclude is not None:
        hsv = hsv.copy(); hsv[..., 2][exclude > 0.3] = 0
        fr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if seed is None:
        m = ((hsv[..., 2] > np.percentile(hsv[..., 2], 97) * 0.85) & (hsv[..., 1] < 70)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        best, score = None, 0
        for i in range(1, n):
            area = st[i, cv2.CC_STAT_AREA]
            if area < 0.004 * m.size: continue
            comp = (lab == i).astype(np.uint8)
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            (cx, cy), (rw, rh), _ = cv2.minAreaRect(max(cs, key=cv2.contourArea))
            rect = area / max(rw * rh, 1)                               # 1 = perfect rectangle
            bright = float(hsv[..., 2][comp > 0].mean()) / 255
            sc = area * rect ** 3 * bright ** 4                         # screens are big, rectangular and the brightest thing
            if sc > score: best, score = cen[i], sc
        if best is None: return None
        seed = best
    mm = screen_mask(fr, seed)
    if mm is None or mm.sum() < 2000: return None
    q = quad_from_mask(mm)
    r = refine_edges(mm, q)
    r = r if np.isfinite(r).all() else q
    if hidden in ('left', 'right'):
        r = _rebuild_hidden(mm, r, hidden, ratio)
    return r


def _rebuild_hidden(mm, q, hidden, ratio):
    cs, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    TL, TR, BR, BL = q
    if hidden == 'left': A_t, A_b = TR, BR          # visible corners
    else: A_t, A_b = TL, BL
    h = np.linalg.norm(A_b - A_t)
    def edge_dir(a, b_guess, other):
        d0 = (b_guess - a) / (np.linalg.norm(b_guess - a) + 1e-9); n = np.array([-d0[1], d0[0]])
        rel = pts - a; t = rel @ d0; dist = np.abs(rel @ n)
        sel = (dist < 5) & (t > 0.05 * h) & (t < 0.4 * np.linalg.norm(b_guess - a))   # only the part next to the visible corner
        if sel.sum() < 15: return d0, np.linalg.norm(b_guess - a)
        vx, vy, x0, y0 = cv2.fitLine(pts[sel].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        d = np.array([vx, vy]); d = d if d @ d0 > 0 else -d
        return d, float(np.max((pts[sel] - a) @ d))
    if hidden == 'left':
        dt, ext_t = edge_dir(TR, TL, None); db, ext_b = edge_dir(BR, BL, None)
        TLn = TR + dt * max(ratio * h, ext_t); BLn = BR + db * max(ratio * h * 1.02, ext_b)
        return np.array([TLn, TR, BR, BLn])
    dt, ext_t = edge_dir(TL, TR, None); db, ext_b = edge_dir(BL, BR, None)
    TRn = TL + dt * max(ratio * h, ext_t); BRn = BL + db * max(ratio * h * 1.02, ext_b)
    return np.array([TL, TRn, BRn, BL])


def preview_detect(video, out_png, frame=0, seed=None, quad=None):
    """Save a frame with the detected (or given) screen outline drawn on it, so you can check it by eye."""
    f = get_frame(video, frame); q = quad if quad is not None else detect_screen(f, seed)
    v = f.copy()
    if q is not None:
        cv2.polylines(v, [np.round(q).astype(np.int32)], True, (0, 0, 255), 2)
        for k, p in enumerate(q): cv2.putText(v, 'TL TR BR BL'.split()[k], tuple(int(x) for x in p), 0, 0.8, (0, 255, 255), 2)
    cv2.imwrite(out_png, v); return q


def _smooth(Q, sigma):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(Q, sigma, axis=0, mode='nearest') if len(Q) > 3 else Q


def track_screen(video, seed=None, start_quad=None, method='mask', static=False, smooth=2.0, occluder=None, frames_range=None,
                 min_segment=8, poly=3, hidden=None, ratio=1.6):
    """Per-frame screen corners -> {frame_index: 4x2 array (TL, TR, BR, BL)}.
    method='mask'   : re-detect the glowing screen every frame (best when the screen is mostly visible).
    method='feature': follow the monitor with feature tracking from `start_quad` on the first frame (use when a
                      person/arm covers a big part of the screen). `occluder(frame)->0..1` masks moving people out.
    static=True     : locked-off camera: one averaged quad for every frame (most stable).
    occluder(frame)->0..1 also masks people out of mask detection (over-the-shoulder shots).
    Hard cuts are detected and each shot segment is tracked/smoothed on its own. Segments shorter than
    min_segment frames (a cut to another shot at the very end) are skipped, so no UI lands on the wrong shot.
    Smoothing: outlier frames are dropped and each corner is fitted with a low-order curve (poly=3), so the UI
    stays locked to the monitor instead of jittering; poly=None uses gaussian smoothing instead."""
    info = probe(video); n = info['frames']
    lo, hi = frames_range or (0, n)
    starts = sorted(set([lo] + [c for c in info['cuts'] if lo < c < hi]))
    segs = {}                                     # frame -> segment id
    for si, a in enumerate(starts):
        b = starts[si + 1] if si + 1 < len(starts) else hi
        for i in range(a, b): segs[i] = si
    per = {}                                      # segment -> (idx list, quad list)
    s_pt, q, g0 = seed, None, None
    for i, f in enumerate(frames(video)):
        if i not in segs: continue
        new_seg = i in starts
        if method == 'mask':
            if new_seg: s_pt = seed
            if s_pt is not None and hidden: s_pt = None if new_seg else s_pt
            qi = detect_screen(f, s_pt, exclude=occluder(f) if occluder is not None else None, hidden=hidden, ratio=ratio)
            if qi is None: continue
            s_pt = qi.mean(0)
        else:
            g1 = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            if new_seg or q is None:
                q = np.asarray(start_quad if (start_quad is not None and i == lo) else detect_screen(f, seed), float)
            else:
                band = np.zeros(g0.shape, np.uint8); cv2.fillPoly(band, [np.round(expand_quad(q, 0.12)).astype(np.int32)], 255)
                if occluder is not None: band[occluder(f) > 0.3] = 0
                p0 = cv2.goodFeaturesToTrack(g0, 400, 0.01, 6, mask=band)
                if p0 is not None and len(p0) >= 8:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=3)
                    ok = st.ravel() == 1
                    if ok.sum() >= 8:
                        Hm, _ = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 2.0)
                        if Hm is not None: q = cv2.perspectiveTransform(q.reshape(1, 4, 2).astype(np.float32), Hm).reshape(4, 2).astype(float)
            g0 = g1; qi = q.copy()
        per.setdefault(segs[i], ([], []))
        per[segs[i]][0].append(i); per[segs[i]][1].append(qi)
    out = {}
    for si, (idx, Q) in per.items():
        seg_len = sum(1 for k, v in segs.items() if v == si)
        if seg_len < min_segment: continue
        Q = np.array(Q, float); idx = np.array(idx)
        if hidden in ('left', 'right'):              # monitor is rigid: one shape relative to the visible edge for the whole segment
            vi, hi_ = ((1, 2), (0, 3)) if hidden == 'left' else ((0, 3), (1, 2))
            A, Bv = Q[:, vi[0]], Q[:, vi[1]]
            ev = Bv - A; hgt = np.linalg.norm(ev, axis=1); ang_v = np.arctan2(ev[:, 1], ev[:, 0])
            rel = []
            for c_vis, c_hid in ((vi[0], hi_[0]), (vi[1], hi_[1])):
                d = Q[:, c_hid] - Q[:, c_vis]
                rel.append((np.median(np.arctan2(d[:, 1], d[:, 0]) - ang_v), np.median(np.linalg.norm(d, axis=1) / hgt)))
            for c_vis, c_hid, (da, lr) in ((vi[0], hi_[0], rel[0]), (vi[1], hi_[1], rel[1])):
                a_ = ang_v + da
                Q[:, c_hid] = Q[:, c_vis] + np.stack([np.cos(a_), np.sin(a_)], 1) * (lr * hgt)[:, None]
        med = np.median(Q, 0); err = np.abs(Q - med).reshape(len(Q), -1).max(1)
        good = err < max(6.0, 3 * np.median(err) + 2)                       # drop frames where detection jumped
        if static:
            Q = np.repeat(np.median(Q[good], 0)[None], len(Q), 0)
        elif poly is not None and good.sum() > poly + 2:
            if hidden: poly = min(poly, 1)                                   # estimated corners: keep their motion simple
            tt = idx.astype(float); fit = np.zeros_like(Q)
            for c in range(4):
                for a in range(2):
                    fit[:, c, a] = np.polyval(np.polyfit(tt[good], Q[good, c, a], poly), tt)
            Q = fit
        else:
            Q = _smooth(Q, smooth)
        for i, qq in zip(idx, Q): out[int(i)] = qq
    return out


def save_quads(quads, path): json.dump({str(k): np.asarray(v).tolist() for k, v in quads.items()}, open(path, 'w'))
def load_quads(path): return {int(k): np.array(v) for k, v in json.load(open(path)).items()}


# ============================================================================ person / arm matte (things in front of the screen)
_SESS = None
def person_matte(fr, keep_largest=True):
    """0..1 matte of people in the frame (MODNet). keep_largest drops small false positives (glare, cups)."""
    global _SESS
    import onnxruntime as ort
    if _SESS is None: _SESS = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
    h, w = fr.shape[:2]; rh, rw = 512, int(round(512 * w / h / 32) * 32)
    x = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (rw, rh)).astype(np.float32) / 127.5 - 1
    m = _SESS.run(None, {_SESS.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    m = np.clip((cv2.resize(m, (w, h)) - 0.12) / 0.55, 0, 1)
    if keep_largest:
        n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0.3).astype(np.uint8))
        if n > 1:
            big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            m = m * cv2.dilate((lab == big).astype(np.uint8), np.ones((9, 9), np.uint8))
    return m


def difference_matte(fr, clean_ref, screen_mask_, inside_only=True):
    """Matte of anything that changed vs a clean reference frame of the same locked-off shot, kept only where it
    connects to something outside the screen (an arm reaching in/out). Use when people-matting misses (fast blur)."""
    cur, ref = fr.astype(np.float32), clean_ref.astype(np.float32)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    sat = np.clip((hsv[..., 1].astype(np.float32) - 40) / 30, 0, 1)
    darker = np.clip((ref.mean(2) - cur.mean(2) - 15) / 25, 0, 1)
    dd = np.abs(cur - ref).max(2) / 255.0
    key = np.clip((dd - 0.07) / 0.08, 0, 1) * np.maximum(sat, darker)
    b = cv2.morphologyEx((key > 0.35).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab = cv2.connectedComponents(b)
    keep = np.isin(lab, np.unique(lab[(screen_mask_ == 0) & (b > 0) & (lab > 0)]))
    keep[lab == 0] = False
    kb = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    return cv2.GaussianBlur(np.maximum(kb.astype(np.float32), key * keep), (0, 0), 1.2)


# ============================================================================ lighting model of the blank screen
def expand_quad(q, frac=0.006):
    q = np.asarray(q, float); c = q.mean(0); e = frac * np.linalg.norm(q[2] - q[0]); lines = []
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]; d = (b - a) / np.linalg.norm(b - a); nrm = np.array([d[1], -d[0]])
        if np.dot(nrm, (a + b) / 2 - c) < 0: nrm = -nrm
        lines.append((a + nrm * e, d))
    return np.array([_inter(lines[(k - 1) % 4], lines[k]) for k in range(4)])


def _uv(quad, shape):
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
    h, w = shape; ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.perspectiveTransform(np.stack([xs, ys], -1).reshape(-1, 1, 2), H).reshape(h, w, 2)


def _basis(u, v): return np.stack([np.ones_like(u), u, v, u * u, v * v, u * v, u * u * v, u * v * v], -1)


def fit_white(plate, quad, valid):
    """Smooth model of the blank screen's own colour/brightness (glow, falloff, tint) in screen coordinates."""
    x0, y0 = np.maximum(np.floor(quad.min(0)).astype(int), 0); x1, y1 = np.ceil(quad.max(0)).astype(int)
    x1, y1 = min(plate.shape[1], x1), min(plate.shape[0], y1)
    uv = _uv(quad - [x0, y0], (y1 - y0, x1 - x0)); sub = plate[y0:y1, x0:x1].astype(np.float32); m = valid[y0:y1, x0:x1]
    u, v = uv[..., 0], uv[..., 1]
    inside = (u > 0.02) & (u < 0.98) & (v > 0.03) & (v < 0.97) & m
    ys, xs = np.nonzero(inside)
    if len(xs) < 50: return None
    if len(xs) > 6000:
        sel = np.random.default_rng(0).choice(len(xs), 6000, replace=False); ys, xs = ys[sel], xs[sel]
    A = _basis(u[ys, xs], v[ys, xs]); coefs = []
    for ch in range(3):
        y = sub[ys, xs, ch]; c = np.linalg.lstsq(A, y, rcond=None)[0]
        r = y - A @ c; keep = r > -2.5 * r.std()
        coefs.append(np.linalg.lstsq(A[keep], y[keep], rcond=None)[0])
    return np.array(coefs)


def eval_white(coefs, quad, shape):
    uv = np.clip(_uv(quad, shape), 0.0, 1.0); B = _basis(uv[..., 0], uv[..., 1])
    return np.clip(np.stack([B @ coefs[c] for c in range(3)], -1), 0, 255)


def lit_matte(plate, quad, white, grow=0.018):
    """Everything lit by the screen up to the dark bezel (finds the true display edge)."""
    H, W = plate.shape[:2]
    Mb = np.zeros((H, W), np.uint8); cv2.fillPoly(Mb, [np.round(expand_quad(quad, grow)).astype(np.int32)], 1)
    core = np.zeros((H, W), np.uint8); cv2.fillPoly(core, [np.round(expand_quad(quad, -0.01)).astype(np.int32)], 1)
    lum = plate.astype(np.float32).mean(2); lw = white.mean(2)
    ring = cv2.dilate(Mb, np.ones((9, 9), np.uint8)) - Mb
    Lb = float(np.median(lum[ring > 0])) if ring.any() else 20.0
    Lb = min(Lb, 0.5 * float(np.median(lw[core > 0])))
    key = np.clip((lum - (Lb + 10)) / 22.0, 0, 1) * Mb
    b = ((key > 0.5) | (core > 0)).astype(np.uint8)
    n, lab = cv2.connectedComponents(b); ids = np.unique(lab[core > 0]); ids = ids[ids > 0]
    return np.where(core > 0, 1.0, key * np.isin(lab, ids)).astype(np.float32)


def snap_to_bezel(plate, quad, white, occl=None):
    """Move each quad edge onto the real lit-display/bezel boundary (no light rim left around the UI)."""
    k = lit_matte(plate, quad, white)
    if occl is not None: k = k * (occl < 0.3)
    d = np.linalg.norm(quad[2] - quad[0]); tol = 0.02 * d
    cs, _ = cv2.findContours((k > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs: return quad
    pts = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64); lines = []
    for e in range(4):
        a, c = quad[e], quad[(e + 1) % 4]; dv = (c - a) / np.linalg.norm(c - a); nv = np.array([-dv[1], dv[0]])
        rel = pts - a; t = rel @ dv; L = np.linalg.norm(c - a); dist = rel @ nv
        sel = (np.abs(dist) < tol) & (t > 0.08 * L) & (t < 0.92 * L)
        lines.append(_line_fit(pts[sel]) if sel.sum() >= 25 else (a, dv))
    return np.array([_inter(lines[(e - 1) % 4], lines[e]) for e in range(4)])


# ============================================================================ the composite
_TEX = None
def _texture():
    global _TEX
    if _TEX is None:
        rng = np.random.default_rng(11); m = np.zeros((540, 960), np.float32)
        for sc, wgt in ((60, 1.0), (25, 0.6), (9, 0.3)):
            m += wgt * cv2.GaussianBlur(rng.standard_normal((540, 960)).astype(np.float32), (0, 0), sc) * sc
        m /= np.abs(m).max() + 1e-6
        gr = cv2.GaussianBlur(rng.standard_normal((540, 960)).astype(np.float32), (0, 0), 0.7); gr /= gr.std() * 3
        _TEX = np.dstack([m, gr]).astype(np.float32)
    return _TEX


def composite(plate, quad, ui, coefs, occl=None, reflection=0.0, soft=0.6, bloom=0.10, black=0.035,
              texture=True, veil=0.09, mottle=0.05, grain=0.016, edge_shade=0.22, dof=None, trim=3):
    """One frame. quad = snapped screen corners. ui = BGR image (any size, stretched to the screen).
    reflection: keep the plate's reflections (people/room) on top of the UI (0 = off, 1-2.5 typical).
    trim: pixels cut from each edge of the UI first (screen recordings often have a light 1-2 px border that
    shows up as a coloured line along the bezel).
    texture: matte LCD look (veil glow, raised blacks at angles, backlight mottling, fine grain).
    dof: optional 0..1 map of extra blur (1 = most blurred) to match a shallow-focus plate."""
    H, W = plate.shape[:2]; P = plate.astype(np.float32)
    if trim: ui = ui[trim:-trim, trim:-trim]
    white = eval_white(coefs, quad, (H, W)).astype(np.float32)
    k = np.zeros((H, W), np.float32)
    cv2.fillPoly(k, [np.round(expand_quad(quad, 0.0015) * 8).astype(np.int32)], 1.0, lineType=cv2.LINE_AA, shift=3)
    k = gblur(k, 0.6)
    h, w = ui.shape[:2]
    tw = int(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
    if tw < w:
        sc = min(1.0, 1.6 * tw / w); ui = cv2.resize(ui, (max(8, int(w * sc)), max(8, int(h * sc))), interpolation=cv2.INTER_AREA); h, w = ui.shape[:2]
    Mx = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad.astype(np.float32))
    U = cv2.warpPerspective(ui, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if soft > 0: U = gblur(U, soft)
    U = U.astype(np.float32) / 255.0
    lw = white.mean(2); lp = gblur(P.mean(2), 1.0)
    core = np.zeros((H, W), np.float32); cv2.fillPoly(core, [np.round(quad).astype(np.int32)], 1); core = gblur(core, 1.5)
    wfac = core + (1 - core) * np.clip(lp / np.maximum(lw, 1), 0, 1.0)
    sw = float(np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    if texture:
        sh = float(np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
        fore = np.clip((sw / max(sh, 1)) / (16 / 9), 0.2, 1.0)
        U = U * (1 - veil) + gblur(U, max(2.0, 0.035 * sw)) * veil
        black = black + 0.05 + 0.08 * (1 - fore)
        g = U.mean(2, keepdims=True); U = g + (U - g) * (0.92 - 0.12 * (1 - fore))
        tex = _texture()
        Mt = cv2.getPerspectiveTransform(np.float32([[0, 0], [tex.shape[1], 0], [tex.shape[1], tex.shape[0]], [0, tex.shape[0]]]), quad.astype(np.float32))
        tw_ = cv2.warpPerspective(tex, Mt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        U = U * (1 + mottle * tw_[..., 0:1]) + grain * tw_[..., 1:2]
    out = white * wfac[..., None] * (black + (1 - black) * U)
    if texture:
        hi = np.clip(out - 0.55 * white, 0, None)
        out = out + 0.22 * gblur(hi, max(1.5, 0.004 * sw)) + 0.10 * gblur(hi, max(4.0, 0.015 * sw))
    kb = gblur(k, max(1.5, 0.004 * np.linalg.norm(quad[2] - quad[0])))
    out *= (1 - edge_shade * (1 - np.clip(kb * 1.6 - 0.6, 0, 1)))[..., None]
    if reflection > 0:
        r = np.clip(lw - P.mean(2), 0, 255)
        e = max(3, int(0.02 * np.linalg.norm(quad[3] - quad[0])))
        M0 = np.zeros((H, W), np.uint8); cv2.fillPoly(M0, [np.round(quad).astype(np.int32)], 1)
        inner = cv2.erode(M0, np.ones((2 * e + 1, 2 * e + 1), np.uint8)).astype(np.float32)
        r = cv2.GaussianBlur(r * gblur(inner, e / 2), (0, 0), 2.0)
        out = out + reflection * r[..., None] * np.array([1.0, 0.98, 0.94], np.float32)
    gr = np.clip(P - gblur(P, 1.4), -4.0, 4.0)        # plate sensor noise only
    out = out + 0.85 * gr
    if dof is not None:
        x = out; o2 = x.copy()
        for sg, lo in ((1.6, 0.0), (3.2, 1 / 3), (5.5, 2 / 3)):
            wgt = np.clip((dof - lo) * 3, 0, 1)[..., None]; o2 = o2 * (1 - wgt) + gblur(x, sg) * wgt
        out = o2
    a = k.copy()
    if occl is not None: a = a * (1 - occl)
    scale = max(8.0, np.linalg.norm(quad[3] - quad[0]) / 40.0)
    g_new = gblur(out * k[..., None], scale); g_old = gblur(P * k[..., None], scale)
    halo = (gblur(k, scale * 1.5) - k).clip(0, 1)[..., None]
    if occl is not None: halo = halo * (1 - cv2.dilate(occl, np.ones((15, 15), np.uint8)))[..., None]
    base = P + bloom * 3.0 * halo * (g_new - g_old)
    out = out + bloom * 0.25 * g_new
    return np.clip(base * (1 - a[..., None]) + out * a[..., None], 0, 255).astype(np.uint8), a


def composite_roi(plate, quad, ui, coefs, occl=None, **kw):
    """composite() on a crop around the screen only (same result, much faster for screens that don't fill the frame)."""
    H, W = plate.shape[:2]; q = np.asarray(quad, float)
    m = int(0.18 * np.linalg.norm(q[2] - q[0])) + 80
    x0, y0 = np.maximum(np.floor(q.min(0)).astype(int) - m, 0); x1, y1 = np.minimum(np.ceil(q.max(0)).astype(int) + m, [W, H])
    if (x1 - x0) * (y1 - y0) > 0.7 * W * H:
        return composite(plate, q, ui, coefs, occl=occl, **kw)
    off = np.array([x0, y0], float)
    c2 = coefs.copy()
    sub = plate[y0:y1, x0:x1]
    o = None if occl is None else occl[y0:y1, x0:x1]
    if kw.get('dof') is not None: kw = dict(kw, dof=kw['dof'][y0:y1, x0:x1])
    img, a = composite(sub, q - off, ui, c2, occl=o, **kw)
    out = plate.copy(); out[y0:y1, x0:x1] = img
    A = np.zeros((H, W), np.float32); A[y0:y1, x0:x1] = a
    return out, A


def composite_video(plate_path, quads, ui, out_path, occluder=None, reflection=0.0, audio=True, filter_complex=None,
                    extra_audio=None, post=None, snap=True, white_from=None, edge_pad=3, **look):
    """Full-length composite of one plate. Frames without a quad are passed through untouched.
    ui(i, t) -> BGR image for frame i (t = seconds). occluder(i, frame) -> 0..1 matte or None.
    post(i, out_frame, plate_frame, alpha) -> frame, for per-shot effects (e.g. fade to the plate).
    edge_pad: pixels the UI tucks under the bezel after snapping. Raise it (4-5) if a thin light/blue line remains
    along an edge; lower it (1-2) if a bright band spills onto the bezel.
    white_from: list of frame indices to build one fixed lighting model from (use for screens that flash/
    change brightness during the shot); default = per-frame fit, temporally smoothed."""
    info = probe(plate_path); fps, n = info['fps'], info['frames']
    from scipy.ndimage import gaussian_filter1d
    # pass 1: lighting model + bezel snapping per frame (small data only)
    coefs, snapped = {}, {}
    if white_from:
        Cs = []
        for i in white_from:
            f = get_frame(plate_path, i); q = np.asarray(quads[i])
            M = np.zeros(f.shape[:2], np.uint8); cv2.fillPoly(M, [np.round(q).astype(np.int32)], 1)
            c = fit_white(f, q, M.astype(bool) & (f.mean(2) > 60))
            if c is not None: Cs.append(c)
        C0 = np.mean(Cs, 0)
    occ = {}                                          # occluder mattes are computed once (uint8) and reused in pass 2
    for i, f in enumerate(frames(plate_path)):
        if i not in quads: continue
        q = np.asarray(quads[i], float)
        o = occluder(i, f) if occluder else None
        if o is not None: occ[i] = (np.clip(o, 0, 1) * 255).astype(np.uint8)
        if white_from: c = C0
        else:
            M = np.zeros(f.shape[:2], np.uint8); cv2.fillPoly(M, [np.round(q).astype(np.int32)], 1)
            valid = M.astype(bool) & (f.mean(2) > 60)
            if o is not None: valid &= (o < 0.2)
            c = fit_white(f, q, valid)
            if c is None: continue
        coefs[i] = c
        if snap:                                      # snap on a crop around the screen (fast), then tuck under the bezel
            H, W = f.shape[:2]; m = int(0.12 * np.linalg.norm(q[2] - q[0])) + 40
            x0, y0 = np.maximum(np.floor(q.min(0)).astype(int) - m, 0); x1, y1 = np.minimum(np.ceil(q.max(0)).astype(int) + m, [W, H])
            off = np.array([x0, y0], float); fs = f[y0:y1, x0:x1]; qs = q - off
            sq = snap_to_bezel(fs, qs, eval_white(c, qs, fs.shape[:2]), None if o is None else o[y0:y1, x0:x1]) + off
            snapped[i] = expand_quad(sq, edge_pad / max(np.linalg.norm(sq[2] - sq[0]), 1.0))
        else: snapped[i] = q
    idx = sorted(coefs)
    if not white_from and len(idx) > 3:
        C = gaussian_filter1d(np.array([coefs[k] for k in idx]), 2.0, axis=0, mode='nearest')
        for j, k in enumerate(idx): coefs[k] = C[j]
    # smooth snapped corners within each continuous run
    runs, cur = [], [idx[0]] if idx else []
    for a, b in zip(idx[:-1], idx[1:]):
        if b == a + 1 and np.abs(snapped[b] - snapped[a]).max() < 40: cur.append(b)
        else: runs.append(cur); cur = [b]
    if cur: runs.append(cur)
    for r in runs:
        if len(r) > 3:
            S = gaussian_filter1d(np.array([snapped[k] for k in r]), 1.5, axis=0, mode='nearest')
            for j, k in enumerate(r): snapped[k] = S[j]
    # pass 2: render
    wr = Writer(out_path, info['width'], info['height'], fps)
    for i, f in enumerate(frames(plate_path)):
        if i in coefs:
            o = occ[i].astype(np.float32) / 255.0 if i in occ else None
            img, a = composite_roi(f, snapped[i], ui(i, i / fps), coefs[i], occl=o, reflection=reflection, **look)
            if post: img = post(i, img, f, a)
        else:
            img = f
        wr.write(img)
    if not audio: wr.close()
    else: wr.close(audio_from=plate_path, extra_audio=extra_audio, filter_complex=filter_complex)
    return out_path


# ============================================================================ UI sources
class VideoUI:
    """Screen recording as the UI. time_map(t_plate) -> t_ui lets you hold, delay or trim (natural speed = t).
    patches = [(x0, y0, x1, y1, (b, g, r)), ...] painted over every frame (e.g. hide a credit counter)."""
    def __init__(self, path, time_map=None, patches=None):
        self.path = path; c = cv2.VideoCapture(path); self.fps = c.get(cv2.CAP_PROP_FPS) or 30
        self.n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); self.map = time_map or (lambda t: t); self.patches = patches or []
        self._cap, self._pos, self._cache = None, -1, {}
    def frame(self, k):
        k = int(min(max(k, 0), self.n - 1))
        if k in self._cache: return self._cache[k]
        if self._cap is None or k < self._pos or k > self._pos + 60:            # seek only when jumping
            self._cap = cv2.VideoCapture(self.path); self._cap.set(cv2.CAP_PROP_POS_FRAMES, k); self._pos = k - 1
        f = None
        while self._pos < k:
            ok, f = self._cap.read(); self._pos += 1
            if not ok: f = self._cache.get(max(self._cache) if self._cache else None); break
        for (x0, y0, x1, y1, col) in self.patches: f[y0:y1, x0:x1] = col
        if len(self._cache) > 12: self._cache.pop(min(self._cache))
        self._cache[k] = f
        return f
    def __call__(self, i, t):
        return self.frame(round(min(max(self.map(t), 0), (self.n - 1) / self.fps) * self.fps))


class ImageUI:
    def __init__(self, path_or_img):
        self.img = cv2.imread(path_or_img) if isinstance(path_or_img, str) else path_or_img
    def __call__(self, i, t): return self.img


_FONTS = {}
def _font(size):
    if size not in _FONTS:
        try: _FONTS[size] = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
        except Exception: _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


class PlayerUI:
    """A video playing inside a screenshot of an app's video player.
    rect = (x0, y0, x1, y1) of the video area in the screenshot. crop = (x0, y0, x1, y1) of the screenshot to show
    (e.g. trim browser chrome to 16:9). Draws a simple progress bar + time; paused=True shows a frozen frame."""
    def __init__(self, screenshot, video, rect, crop=None, time_map=None, controls=True, paused=False):
        self.base = cv2.imread(screenshot); self.v = VideoUI(video, time_map); self.rect, self.crop = rect, crop
        self.controls, self.paused = controls, paused
        self.total = self.v.n / self.v.fps
    def __call__(self, i, t):
        can = self.base.copy(); x0, y0, x1, y1 = self.rect; vw, vh = x1 - x0, y1 - y0
        tv = min(max(self.v.map(t), 0), self.total)
        vid = self.v(i, t); h, w = vid.shape[:2]; s = max(vw / w, vh / h)
        r = cv2.resize(vid, (int(math.ceil(w * s)), int(math.ceil(h * s))), interpolation=cv2.INTER_AREA)
        oy, ox = (r.shape[0] - vh) // 2, (r.shape[1] - vw) // 2
        can[y0:y1, x0:x1] = r[oy:oy + vh, ox:ox + vw]
        if self.controls:
            reg = can[y0:y1, x0:x1].astype(np.float32); gy = np.linspace(0, 1, vh)[:, None, None]
            can[y0:y1, x0:x1] = (reg * (1 - np.clip((gy - 0.75) / 0.25, 0, 1) * 0.6)).astype(np.uint8)
            im = Image.fromarray(cv2.cvtColor(can, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im)
            bh = max(4, vh // 140); by = y1 - vh // 22; pad = vw // 40
            d.rounded_rectangle([x0 + pad, by - bh, x1 - pad, by + bh], radius=bh, fill=(110, 110, 112))
            d.rounded_rectangle([x0 + pad, by - bh, x0 + pad + max(2 * bh, (vw - 2 * pad) * tv / max(self.total, 1e-6)), by + bh], radius=bh, fill=(255, 255, 255))
            cy = by - vh // 16; sz = vh // 40
            if self.paused: d.polygon([(x0 + pad, cy - sz), (x0 + pad, cy + sz), (x0 + pad + 1.7 * sz, cy)], fill=(255, 255, 255))
            else:
                d.rectangle([x0 + pad, cy - sz, x0 + pad + sz * 0.55, cy + sz], fill=(255, 255, 255))
                d.rectangle([x0 + pad + sz, cy - sz, x0 + pad + sz * 1.55, cy + sz], fill=(255, 255, 255))
            fnt = _font(int(sz * 1.6))
            d.text((x0 + pad + 2.6 * sz, cy), f'0:{int(tv):02d} / 0:{int(round(self.total)):02d}', font=fnt, fill=(255, 255, 255), anchor='lm')
            can = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        if self.crop: x0, y0, x1, y1 = self.crop; can = can[y0:y1, x0:x1]
        return can


def contact_sheet(video, out_png, n=8, width=480):
    """Thumbnails across a video for quick review."""
    info = probe(video); ims = []
    for k in np.linspace(0, info['frames'] - 1, n).astype(int):
        f = get_frame(video, k); f = cv2.resize(f, (width, int(width * f.shape[0] / f.shape[1])))
        cv2.putText(f, f'{k} ({k / info["fps"]:.2f}s)', (6, 24), 0, 0.7, (0, 255, 255), 2); ims.append(f)
    while len(ims) % 4: ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(out_png, np.vstack([np.hstack(ims[k:k + 4]) for k in range(0, len(ims), 4)]))
