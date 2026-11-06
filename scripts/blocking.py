"""blocking.py - tiny pure-Python 3D blocking renderer (numpy + Pillow only, no GPU/browser/network).

Build a scene from primitives, animate with keyframe tracks, render shots to MP4 for use as a
video-to-video "previz" reference. See references/blocking.md for how to write a scene file.

Coordinates: metres. Y is up, -Z is "forward" for the default camera. Angles in degrees.
"""
import math, os, subprocess, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------- animation helpers
def ease(x, kind='inout'):
    x = min(max(x, 0.0), 1.0)
    if kind == 'linear': return x
    if kind == 'in': return x * x
    if kind == 'out': return 1 - (1 - x) * (1 - x)
    return x * x * (3 - 2 * x)


def track(*keys, kind='inout'):
    """track((t0, v0), (t1, v1), ...) -> f(t). Values may be numbers or tuples/lists (lerped per element)."""
    keys = sorted(keys, key=lambda k: k[0])
    def f(t):
        if t <= keys[0][0]: return keys[0][1]
        if t >= keys[-1][0]: return keys[-1][1]
        for (ta, va), (tb, vb) in zip(keys[:-1], keys[1:]):
            if ta <= t <= tb:
                u = ease((t - ta) / max(tb - ta, 1e-9), kind)
                if isinstance(va, (tuple, list)):
                    return tuple(a + (b - a) * u for a, b in zip(va, vb))
                return va + (vb - va) * u
    return f


def val(x, t):
    return x(t) if callable(x) else x


def shake(t, amp=0.02, freq=9.0, seed=0):
    """Small camera-shake offset (x, y, z)."""
    r = np.random.default_rng(seed)
    ph = r.uniform(0, 6.28, 6)
    return (amp * math.sin(t * freq + ph[0]) * math.cos(t * 1.7 * freq + ph[1]),
            amp * math.sin(t * 1.3 * freq + ph[2]) * math.cos(t * 0.7 * freq + ph[3]),
            amp * 0.5 * math.sin(t * 1.1 * freq + ph[4]))


# ----------------------------------------------------------------------------- math
def rot_matrix(rx, ry, rz):
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz          # yaw, then pitch, then roll


def mat4(pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
    M = np.eye(4)
    if np.isscalar(scale): scale = (scale, scale, scale)
    M[:3, :3] = rot_matrix(*rot) @ np.diag(scale)
    M[:3, 3] = pos
    return M


# ----------------------------------------------------------------------------- geometry
class Node:
    """A transform with optional mesh and children. pos/rot/scale/visible/color may be callables of t."""
    def __init__(self, verts=None, faces=None, color=(0.7, 0.7, 0.7), pos=(0, 0, 0), rot=(0, 0, 0), scale=1.0,
                 emissive=False, visible=True, children=None, name='', smooth=False):
        self.smooth = smooth
        self.verts = None if verts is None else np.asarray(verts, float)
        self.faces = None if faces is None else np.asarray(faces, int)
        self.color, self.pos, self.rot, self.scale = color, pos, rot, scale
        self.emissive, self.visible, self.name = emissive, visible, name
        self.children = list(children or [])

    def add(self, *nodes):
        self.children.extend(nodes); return self


def group(*children, pos=(0, 0, 0), rot=(0, 0, 0), scale=1.0, visible=True, name=''):
    return Node(pos=pos, rot=rot, scale=scale, visible=visible, children=children, name=name)


def box(size=(1, 1, 1), color=(0.7, 0.7, 0.7), pos=(0, 0, 0), rot=(0, 0, 0), anchor='center', **kw):
    """anchor: 'center' or 'bottom' (sits on y=0)."""
    w, h, d = size
    y0 = 0 if anchor == 'bottom' else -h / 2
    x0, z0 = -w / 2, -d / 2
    v = np.array([[x0, y0, z0], [x0 + w, y0, z0], [x0 + w, y0 + h, z0], [x0, y0 + h, z0],
                  [x0, y0, z0 + d], [x0 + w, y0, z0 + d], [x0 + w, y0 + h, z0 + d], [x0, y0 + h, z0 + d]])
    f = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]]
    return Node(v, f, color, pos, rot, **kw)


def cylinder(r=0.5, h=1.0, n=24, color=(0.7, 0.7, 0.7), pos=(0, 0, 0), rot=(0, 0, 0), r_top=None, anchor='bottom', smooth=True, **kw):
    """Vertical cylinder / cone (r_top) along +Y. Smooth-shaded sides, flat caps."""
    rt = r if r_top is None else r_top
    y0 = 0 if anchor == 'bottom' else -h / 2
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bot = np.stack([r * np.cos(a), np.full(n, y0), r * np.sin(a)], 1)
    top = np.stack([rt * np.cos(a), np.full(n, y0 + h), rt * np.sin(a)], 1)
    v = np.vstack([bot, top, bot, top, [[0, y0, 0], [0, y0 + h, 0]]])      # side ring verts, then separate cap ring verts
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + j], [i, n + j, n + i], [4 * n, 2 * n + j, 2 * n + i], [4 * n + 1, 3 * n + i, 3 * n + j]]
    return Node(v, f, color, pos, rot, smooth=smooth, **kw)


def capsule(r=0.05, h=0.3, color=(0.7, 0.7, 0.7), pos=(0, 0, 0), rot=(0, 0, 0), n=16, scale=1.0, **kw):
    """Rounded limb hanging DOWN from its origin (length h incl. the rounded ends). Good for arms/legs/torso."""
    body = cylinder(r, max(h - 2 * r, 0.001), n, color, pos=(0, -h + r, 0))
    return group(body, sphere(r, n, color, pos=(0, -r, 0)), sphere(r, n, color, pos=(0, -h + r, 0)), pos=pos, rot=rot, scale=scale)


def sphere(r=0.5, n=20, color=(0.7, 0.7, 0.7), pos=(0, 0, 0), rot=(0, 0, 0), scale=1.0, smooth=True, **kw):
    lat = np.linspace(0, np.pi, n // 2 + 1)
    lon = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = [[r * math.sin(a) * math.cos(b), r * math.cos(a), r * math.sin(a) * math.sin(b)] for a in lat for b in lon]
    f = []
    m = len(lon)
    for i in range(len(lat) - 1):
        for j in range(m):
            a, b, c, d = i * m + j, i * m + (j + 1) % m, (i + 1) * m + (j + 1) % m, (i + 1) * m + j
            f += [[a, c, b], [a, d, c]]
    return Node(v, f, color, pos, rot, scale=scale, smooth=smooth, **kw)


def plane(size=(10, 10), color=(0.6, 0.6, 0.6), pos=(0, 0, 0), rot=(0, 0, 0), div=8, **kw):
    """Flat XZ plane facing +Y, subdivided so near-camera clipping stays clean."""
    w, d = size
    xs, zs = np.linspace(-w / 2, w / 2, div + 1), np.linspace(-d / 2, d / 2, div + 1)
    v = np.array([[x, 0, z] for z in zs for x in xs])
    f = []
    for i in range(div):
        for j in range(div):
            a, b, c, e = i * (div + 1) + j, i * (div + 1) + j + 1, (i + 1) * (div + 1) + j + 1, (i + 1) * (div + 1) + j
            f += [[a, c, b], [a, e, c]]
    return Node(v, f, color, pos, rot, **kw)


def polygon_prism(points2d, depth=0.1, color=(0.7, 0.7, 0.7), pos=(0, 0, 0), rot=(0, 0, 0), **kw):
    """Extrude a convex-ish 2D outline in the XZ plane (top view) by `depth` along Y. Good for wings/hulls."""
    p = np.asarray(points2d, float); n = len(p)
    bot = np.stack([p[:, 0], np.full(n, -depth / 2), p[:, 1]], 1)
    top = np.stack([p[:, 0], np.full(n, depth / 2), p[:, 1]], 1)
    v = np.vstack([bot, top])
    f = []
    for i in range(1, n - 1):
        f += [[0, i + 1, i], [n, n + i, n + i + 1]]
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + j], [i, n + j, n + i]]
    return Node(v, f, color, pos, rot, **kw)


def beam(a, b, width=0.05, color=(0.2, 1, 0.9), emissive=True, **kw):
    """A glowing bolt/laser from point a to point b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a)) or 1e-6
    d = (b - a) / L
    yaw = math.degrees(math.atan2(d[0], d[2]))
    pitch = -math.degrees(math.asin(max(-1, min(1, d[1]))))
    n = box((width, width, L), color, pos=tuple((a + b) / 2), rot=(pitch, yaw, 0), emissive=emissive, **kw)
    return n


# ----------------------------------------------------------------------------- mannequin
POSES = {
    # Angles in degrees. spine/neck: + leans forward/looks down. sh=(pitch, roll): pitch - = arm forward/up
    # (-90 = straight forward, -170 = overhead), roll = out to the side. el: - = forearm bends forward.
    # hip: -90 = thigh forward (sitting). knee: + = shin bends down.
    'stand':     dict(spine=0, neck=0, head_yaw=0, shL=(0, 8), elL=-5, shR=(0, -8), elR=-5, hipL=0, hipR=0, kneeL=0, kneeR=0),
    'sit':       dict(spine=-5, neck=0, head_yaw=0, shL=(-20, 10), elL=-60, shR=(-20, -10), elR=-60, hipL=-90, hipR=-90, kneeL=90, kneeR=90),
    'sit_type':  dict(spine=8, neck=8, head_yaw=0, shL=(-35, 12), elL=-70, shR=(-35, -12), elR=-70, hipL=-90, hipR=-90, kneeL=90, kneeR=90),
    'sit_slump': dict(spine=18, neck=-12, head_yaw=0, shL=(-55, 25), elL=-130, shR=(-25, -10), elR=-70, hipL=-85, hipR=-85, kneeL=85, kneeR=85),
    'sit_stretch': dict(spine=-12, neck=-15, head_yaw=0, shL=(-170, 15), elL=-15, shR=(-170, -15), elR=-15, hipL=-90, hipR=-90, kneeL=90, kneeR=90),
    'sit_rub_eyes': dict(spine=12, neck=10, head_yaw=0, shL=(-95, 25), elL=-140, shR=(-95, -25), elR=-140, hipL=-90, hipR=-90, kneeL=90, kneeR=90),
    'sit_head_down': dict(spine=45, neck=25, head_yaw=0, shL=(-80, 30), elL=-110, shR=(-80, -30), elR=-110, hipL=-90, hipR=-90, kneeL=90, kneeR=90),
    'sit_shock': dict(spine=-20, neck=-10, head_yaw=0, shL=(-60, 30), elL=-40, shR=(-90, -10), elR=-110, hipL=-80, hipR=-80, kneeL=80, kneeR=80),
    'reach':     dict(spine=10, neck=0, head_yaw=0, shL=(0, 8), elL=-10, shR=(-90, 0), elR=0, hipL=-90, hipR=-90, kneeL=90, kneeR=90),
    'arms_up':   dict(spine=0, neck=-10, head_yaw=0, shL=(-170, 10), elL=-10, shR=(-170, -10), elR=-10, hipL=-90, hipR=-90, kneeL=90, kneeR=90),
}


def pose_track(*keys, kind='inout'):
    """pose_track((t, 'sit'), (1.2, 'sit_stretch'), (2.0, {'sit': .5, ...}? no - use names or dicts))."""
    keys = [(t, POSES[p] if isinstance(p, str) else p) for t, p in keys]
    names = keys[0][1].keys()
    tracks = {k: track(*[(t, p[k]) for t, p in keys], kind=kind) for k in names}
    return lambda t: {k: tr(t) for k, tr in tracks.items()}


def person(pose='stand', shirt=(0.92, 0.92, 0.92), pants=(0.12, 0.12, 0.14), skin=(0.85, 0.68, 0.55), hair=(0.08, 0.08, 0.08),
           tie=None, pos=(0, 0, 0), rot=(0, 0, 0), scale=1.0, extras=None):
    """Simple jointed mannequin (~1.75 m). `pose` = name, dict, or f(t)->dict. Origin = hips (seat height when sitting).
    The face side is -Z (looks toward -Z). A dark visor band marks the face direction."""
    P = (lambda t: POSES[pose]) if isinstance(pose, str) else (pose if callable(pose) else (lambda t: pose))
    g = lambda key, i=None: (lambda t: (P(t)[key] if i is None else P(t)[key][i]))
    head = group(sphere(0.11, 24, skin, scale=(1, 1.15, 1)),
                 sphere(0.118, 24, hair, pos=(0, 0.035, 0.03), scale=(1, 0.95, 1.02)),       # hair cap sits back: face shows the facing direction
                 pos=(0, 0.2, 0))
    neck = group(cylinder(0.055, 0.17, 16, skin), head, pos=(0, 0.46, 0), rot=lambda t: (-g('neck')(t), g('head_yaw')(t), 0))
    torso_parts = [cylinder(0.19, 0.42, 24, shirt, scale=(1, 1, 0.6)), sphere(0.19, 24, shirt, pos=(0, 0.42, 0), scale=(1, 0.42, 0.6)),
                   sphere(0.18, 24, shirt, pos=(0, 0.02, 0), scale=(1, 0.35, 0.62))]
    if tie: torso_parts.append(box((0.06, 0.38, 0.02), tie, pos=(0, 0.3, -0.112)))
    def arm(side):
        s = 1 if side == 'L' else -1
        fore = group(capsule(0.042, 0.29, shirt), sphere(0.048, 16, skin, pos=(0, -0.31, 0)),
                     pos=(0, -0.27, 0), rot=lambda t: (-g('el' + side)(t), 0, 0))
        upper = group(capsule(0.05, 0.3, shirt), fore,
                      pos=(s * 0.2, 0.47, 0), rot=lambda t: (-g('sh' + side, 0)(t), 0, g('sh' + side, 1)(t)))
        return upper
    def leg(side):
        s = 1 if side == 'L' else -1
        shin = group(capsule(0.055, 0.44, pants), sphere(0.06, 16, (0.05, 0.05, 0.05), pos=(0, -0.45, -0.06), scale=(1, 0.6, 2.0)),
                     pos=(0, -0.42, 0), rot=lambda t: (-g('knee' + side)(t), 0, 0))
        return group(capsule(0.07, 0.45, pants), shin, pos=(s * 0.1, 0, 0), rot=lambda t: (-g('hip' + side)(t), 0, 0))
    torso = group(*torso_parts, neck, arm('L'), arm('R'), rot=lambda t: (-g('spine')(t), 0, 0))
    root = group(sphere(0.18, 24, pants, scale=(1, 0.45, 0.65)), torso, leg('L'), leg('R'), *(extras or []), pos=pos, rot=rot, scale=scale, name='person')
    return root


# ----------------------------------------------------------------------------- camera
class Camera:
    def __init__(self, pos, target, lens=35, roll=0.0, flash=0.0, fade=0.0, fog=None, bg=None):
        """lens in mm (full-frame 36 mm wide sensor). pos/target/lens/roll/flash/fade may be callables of t.
        flash: 0..1 white overlay. fade: 0..1 black overlay. fog=(color, start_m, end_m) and bg=(r,g,b) override the renderer per shot."""
        self.pos, self.target, self.lens, self.roll, self.flash, self.fade = pos, target, lens, roll, flash, fade
        self.fog, self.bg = fog, bg


# ----------------------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, width=960, height=540, bg=(0.05, 0.06, 0.08), light_dir=(-0.4, -1.0, -0.6), ambient=0.38,
                 fog=None, ss=1):
        """fog = (color, start_m, end_m) for distance haze. ss = supersampling factor (2 = smoother edges, 4x slower)."""
        self.W, self.H, self.bg, self.ambient, self.fog, self.ss = width, height, np.array(bg, float), ambient, fog, ss
        l = np.asarray(light_dir, float); self.light = -l / np.linalg.norm(l)

    def _collect(self, node, t, parent, out):
        if not val(node.visible, t): return
        M = parent @ mat4(val(node.pos, t), val(node.rot, t), val(node.scale, t))
        if node.verts is not None and len(node.faces):
            v = node.verts @ M[:3, :3].T + M[:3, 3]
            T = v[node.faces]
            fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
            if node.smooth:
                vn = np.zeros_like(v); np.add.at(vn, node.faces.ravel(), np.repeat(fn, 3, 0))
                vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
                N = vn[node.faces]
            else:
                N = np.repeat(fn[:, None], 3, 1)
            out.append((T, np.asarray(val(node.color, t), float), node.emissive, N))
        for c in node.children: self._collect(c, t, M, out)

    def render(self, nodes, cam, t):
        W, H = self.W * self.ss, self.H * self.ss
        tris = []
        for n in nodes: self._collect(n, t, np.eye(4), tris)
        img = np.empty((H, W, 3), np.float32); img[:] = cam.bg if getattr(cam, 'bg', None) is not None else self.bg
        fog = cam.fog if getattr(cam, 'fog', None) is not None else self.fog
        zbuf = np.full((H, W), np.inf, np.float32)
        if tris:
            T = np.concatenate([x[0] for x in tris]); C = np.concatenate([np.repeat(x[1][None], len(x[0]), 0) for x in tris])
            E = np.concatenate([np.full(len(x[0]), x[2]) for x in tris])
            N = np.concatenate([x[3] for x in tris])                       # (n, 3 verts, 3) normals
            lam = np.abs(N @ self.light)                                    # (n, 3) two-sided Lambert per vertex
            shade = np.where(E[:, None, None], 1.0, (self.ambient + (1 - self.ambient) * lam)[..., None]) * C[:, None, :]
            # camera
            p = np.asarray(val(cam.pos, t), float); tg = np.asarray(val(cam.target, t), float)
            fwd = tg - p; fwd /= np.linalg.norm(fwd)
            up0 = np.array([0, 1.0, 0]) if abs(fwd[1]) < 0.99 else np.array([0, 0, 1.0])
            right = np.cross(fwd, up0); right /= np.linalg.norm(right); up = np.cross(right, fwd)
            rl = math.radians(val(cam.roll, t))
            right, up = right * math.cos(rl) + up * math.sin(rl), up * math.cos(rl) - right * math.sin(rl)
            V = T - p
            x, y, z = V @ right, V @ up, V @ fwd           # z = depth
            f = W * val(cam.lens, t) / 36.0
            near = 0.05
            # fog colour mix by distance
            if fog is not None:
                fc, f0, f1 = fog
                dist = np.clip((z.mean(1) - f0) / (f1 - f0), 0, 1)[:, None, None] * 0.92
                shade = shade * (1 - dist) + np.asarray(fc, float) * dist
            # triangles fully in front
            front = (z > near).all(1)
            part = (z > near).any(1) & ~front
            # cull triangles entirely outside the frame (big speed-up on crowded or wide sets)
            zs = np.where(z > near, z, near)
            sxa = W / 2 + f * x / zs; sya = H / 2 - f * y / zs
            off = (sxa.max(1) < 0) | (sxa.min(1) > W) | (sya.max(1) < 0) | (sya.min(1) > H)
            front &= ~off
            polys = [(np.stack([x[i], y[i], z[i]], 1), shade[i]) for i in np.nonzero(front)[0]]
            for i in np.nonzero(part)[0]:                     # clip against the near plane
                P3 = np.stack([x[i], y[i], z[i]], 1); Cv = shade[i]; outp, outc = [], []
                for a in range(3):
                    A, B = P3[a], P3[(a + 1) % 3]
                    if A[2] > near: outp.append(A); outc.append(Cv[a])
                    if (A[2] > near) != (B[2] > near):
                        u = (near - A[2]) / (B[2] - A[2]); outp.append(A + (B - A) * u); outc.append(Cv[a] + (Cv[(a + 1) % 3] - Cv[a]) * u)
                for k in range(1, len(outp) - 1):
                    polys.append((np.array([outp[0], outp[k], outp[k + 1]]), np.array([outc[0], outc[k], outc[k + 1]])))
            cx, cy = W / 2, H / 2
            for P3, col in polys:
                sx = cx + f * P3[:, 0] / P3[:, 2]; sy = cy - f * P3[:, 1] / P3[:, 2]; iz = 1.0 / P3[:, 2]
                x0, x1 = int(max(0, math.floor(sx.min()))), int(min(W - 1, math.ceil(sx.max())))
                y0, y1 = int(max(0, math.floor(sy.min()))), int(min(H - 1, math.ceil(sy.max())))
                if x0 > x1 or y0 > y1: continue
                den = (sy[1] - sy[2]) * (sx[0] - sx[2]) + (sx[2] - sx[1]) * (sy[0] - sy[2])
                if abs(den) < 1e-9: continue
                ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float32) + 0.5
                w0 = ((sy[1] - sy[2]) * (xs - sx[2]) + (sx[2] - sx[1]) * (ys - sy[2])) / den
                w1 = ((sy[2] - sy[0]) * (xs - sx[2]) + (sx[0] - sx[2]) * (ys - sy[2])) / den
                w2 = 1 - w0 - w1
                inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
                if not inside.any(): continue
                depth = 1.0 / (w0 * iz[0] + w1 * iz[1] + w2 * iz[2])
                zb = zbuf[y0:y1 + 1, x0:x1 + 1]
                m = inside & (depth < zb)
                zb[m] = depth[m]
                img[y0:y1 + 1, x0:x1 + 1][m] = (w0[m, None] * col[0] + w1[m, None] * col[1] + w2[m, None] * col[2])
        fl, fd = val(cam.flash, t), val(cam.fade, t)
        if fl > 0: img = img * (1 - fl) + fl
        if fd > 0: img = img * (1 - fd)
        out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if self.ss > 1:
            out = np.asarray(Image.fromarray(out).resize((self.W, self.H), Image.LANCZOS))
        return out


# ----------------------------------------------------------------------------- shots & output
class Shot:
    def __init__(self, id, dur, title, note='', lens=35, build=None):
        """build(t) -> (list_of_nodes, Camera). t runs 0..dur."""
        self.id, self.dur, self.title, self.note, self.lens, self.build = id, dur, title, note, lens, build


def _font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf'):
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _label(frame, shot, t_local, t_global):
    im = Image.fromarray(frame); d = ImageDraw.Draw(im); W, H = im.size
    f1, f2 = _font(max(12, H // 30)), _font(max(10, H // 40))
    top = f"{shot.id}  {shot.lens}mm  {shot.title}"
    tc = f"{t_global:6.2f}s  (shot {t_local:4.2f}/{shot.dur:.2f})"
    if W < H:   # vertical: title and timecode on two lines
        d.rectangle([0, 0, W, f1.size + f2.size + 14], fill=(0, 0, 0))
        d.text((10, 4), top[:int(W / (f1.size * 0.62))], font=f1, fill=(255, 255, 255))
        d.text((10, f1.size + 8), tc, font=f2, fill=(255, 220, 120))
    else:
        d.rectangle([0, 0, W, H // 15], fill=(0, 0, 0))
        d.text((10, 4), top, font=f1, fill=(255, 255, 255))
        d.text((W - 10, 4), tc, font=f2, fill=(255, 220, 120), anchor='ra')
    if shot.note:
        import textwrap
        cw = max(10, int((W - 20) / (f2.size * 0.62)))
        lines = textwrap.wrap(shot.note[:240], cw)[:4]
        lh = f2.size + 4; bh = lh * len(lines) + 8
        d.rectangle([0, H - bh, W, H], fill=(0, 0, 0))
        for i, ln in enumerate(lines):
            d.text((10, H - bh + 4 + i * lh), ln, font=f2, fill=(220, 220, 220))
    return np.asarray(im)


def _ffmpeg():
    exe = shutil.which('ffmpeg')
    if exe: return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise SystemExit('ffmpeg not found: run  pip install imageio-ffmpeg  (or install ffmpeg) and retry.')


class _Writer:
    def __init__(self, path, w, h, fps):
        self.p = subprocess.Popen([_ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                                   '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', path],
                                  stdin=subprocess.PIPE)
    def write(self, fr): self.p.stdin.write(fr.tobytes())
    def close(self): self.p.stdin.close(); self.p.wait()


def parse_selection(sel, shots):
    """'S01-S04', 'S01,S05,C2', 'all'."""
    ids = [s.id for s in shots]
    if sel in (None, '', 'all'): return list(shots)
    out = []
    for part in sel.split(','):
        part = part.strip()
        if '-' in part and part not in ids:
            a, b = part.split('-'); i, j = ids.index(a), ids.index(b); out += shots[i:j + 1]
        else: out.append(shots[ids.index(part)])
    return out


_MP = {}

def _mp_frame(job):
    si, k = job
    s, R, fps = _MP['shots'][si], _MP['R'], _MP['fps']
    t = k / fps
    nodes, cam = s.build(t)
    return R.render(nodes, cam, t)


def render_shots(shots, renderer, out_dir, name, fps=24, log=print, workers=None):
    """Renders the selected shots back to back. Writes <name>_clean.mp4, <name>_labeled.mp4, <name>_timing.txt.
    workers: number of processes (default = all CPU cores). Frames stay in order."""
    import multiprocessing as mp
    os.makedirs(out_dir, exist_ok=True)
    W, H = renderer.W, renderer.H
    clean = _Writer(os.path.join(out_dir, f'{name}_clean.mp4'), W, H, fps)
    lab = _Writer(os.path.join(out_dir, f'{name}_labeled.mp4'), W, H, fps)
    lines = [f'{name}  total {sum(s.dur for s in shots):.2f}s  @ {fps}fps', 'Shot   in-out (this clip)   lens   title | action']
    jobs, starts, tg = [], [], 0.0
    for si, s in enumerate(shots):
        n = max(1, int(round(s.dur * fps)))
        jobs += [(si, k) for k in range(n)]; starts.append(tg)
        lines.append(f'{s.id:<6} {tg:6.2f}-{tg + s.dur:6.2f}s  {s.lens:>3}mm  {s.title} | {s.note}')
        tg += s.dur
    workers = workers or os.cpu_count() or 1
    _MP.update(shots=shots, R=renderer, fps=fps)
    done = {}
    def emit(job, fr):
        si, k = job; s = shots[si]; t = k / fps
        clean.write(fr); lab.write(_label(fr, s, t, starts[si] + t))
        done[si] = done.get(si, 0) + 1
        if done[si] == max(1, int(round(s.dur * fps))): log(f'  {s.id} done ({done[si]} frames)')
    if workers > 1 and 'fork' in mp.get_all_start_methods():
        with mp.get_context('fork').Pool(workers) as pool:
            for job, fr in zip(jobs, pool.imap(_mp_frame, jobs, chunksize=2)): emit(job, fr)
    else:
        for job in jobs: emit(job, _mp_frame(job))
    clean.close(); lab.close()
    open(os.path.join(out_dir, f'{name}_timing.txt'), 'w').write('\n'.join(lines) + '\n')
    return os.path.join(out_dir, f'{name}_clean.mp4')


# ----------------------------------------------------------------------------- walk / run cycle (added for vertical previz)
def walk(steps_per_s=1.8, stride=28, run=False):
    """Returns f(t)->pose dict for a walk (or run) cycle. Use as person(pose=walk(), ...) and move pos along -Z (the face side)."""
    k = 1.6 if run else 1.0
    def f(t):
        s = math.sin(t * steps_per_s * math.pi)
        a = stride * k
        return dict(spine=8 if run else 3, neck=0, head_yaw=0,
                    shL=(0.8 * a * s, 8), elL=-15 - (40 if run else 0), shR=(-0.8 * a * s, -8), elR=-15 - (40 if run else 0),
                    hipL=-a * s, hipR=a * s,
                    kneeL=max(0, a * 1.2 * s) + 5, kneeR=max(0, -a * 1.2 * s) + 5)
    return f
