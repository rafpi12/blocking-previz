"""Example blocking scene: 'The Cubicle' (office opening + one chase shot).
Use as a pattern: build the sets once as functions, then each Shot.build(t) returns (nodes, Camera).
Render:  python render_blocking.py example_scene.py --shots S01-S04
"""
import math
import numpy as np
from blocking import *

RENDERER = Renderer(960, 540, bg=(0.04, 0.05, 0.08), ambient=0.42, ss=2)   # ss=2: smooth edges (use --ss 1 for quick drafts)

# ---------------------------------------------------------------- palette
WHITE, GREY, DARK, BLACK = (0.9, 0.9, 0.9), (0.55, 0.56, 0.58), (0.18, 0.19, 0.22), (0.06, 0.06, 0.07)
RED, SCREEN, CITY_LIGHT = (0.75, 0.12, 0.1), (0.92, 0.95, 1.0), (1.0, 0.8, 0.45)

# ---------------------------------------------------------------- office set
# Desk at the origin; the worker sits at +Z facing -Z (toward the monitor). Clock on the pillar behind the desk.
def clock(t):
    sec = -6.0 * int(t * 1)            # second hand ticks once per second
    face = cylinder(0.36, 0.04, 32, WHITE, rot=(90, 0, 0), anchor='center')
    rim = cylinder(0.39, 0.05, 32, BLACK, rot=(90, 0, 0), pos=(0, 0, -0.01), anchor='center')
    ticks = [box((0.02, 0.07, 0.01), BLACK, pos=(0.3 * math.sin(math.radians(a)), 0.3 * math.cos(math.radians(a)), 0.03), rot=(0, 0, -a)) for a in range(0, 360, 30)]
    hour = group(box((0.025, 0.17, 0.01), BLACK, pos=(0, 0.085, 0)), pos=(0, 0, 0.04), rot=(0, 0, -305))
    minute = group(box((0.02, 0.27, 0.01), BLACK, pos=(0, 0.135, 0)), pos=(0, 0, 0.045), rot=(0, 0, -330))
    second = group(box((0.008, 0.3, 0.01), RED, pos=(0, 0.11, 0)), pos=(0, 0, 0.05), rot=(0, 0, sec))
    return group(rim, face, *ticks, hour, minute, second, pos=(0, 2.25, -1.38))


def office(t, screen_on=True):
    nodes = [
        plane((30, 30), (0.22, 0.24, 0.26), div=12),                                   # floor
        box((1.4, 3.2, 0.5), (0.86, 0.87, 0.88), pos=(0, 0, -1.65), anchor='bottom'),   # pillar for the clock
        box((30, 3.2, 0.1), (0.05, 0.07, 0.12), pos=(0, 0, -9), anchor='bottom'),       # window wall at night
        box((30, 0.1, 30), (0.12, 0.12, 0.13), pos=(0, 3.2, 0)),                         # ceiling
        clock(t),
        # desk
        box((1.8, 0.04, 0.8), (0.93, 0.93, 0.93), pos=(0, 0.72, 0)),
        box((0.04, 0.72, 0.7), GREY, pos=(-0.86, 0, 0), anchor='bottom'), box((0.04, 0.72, 0.7), GREY, pos=(0.86, 0, 0), anchor='bottom'),
        # monitor (screen faces +Z toward the worker). Leave it a blank glow: UI is added later.
        box((0.22, 0.02, 0.18), BLACK, pos=(0, 0.75, -0.22)), box((0.05, 0.3, 0.04), BLACK, pos=(0, 0.75, -0.24), anchor='bottom'),
        box((0.66, 0.4, 0.04), BLACK, pos=(0, 1.08, -0.22)),
        box((0.62, 0.36, 0.005), SCREEN if screen_on else DARK, pos=(0, 1.08, -0.197), emissive=screen_on),
        # keyboard slopes slightly: the edge toward the monitor sits higher
        box((0.44, 0.025, 0.14), BLACK, pos=(0, 0.755, 0.08), rot=(5, 0, 0)),
        box((0.06, 0.03, 0.1), BLACK, pos=(0.34, 0.755, 0.1)),                                     # mouse
        cylinder(0.04, 0.1, 12, RED, pos=(0.55, 0.74, -0.05)),                                       # red mug
        box((0.22, 0.01, 0.3), (0.85, 0.75, 0.5), pos=(-0.6, 0.745, -0.05)),                          # folder
        # chair (seat + back behind the worker)
        box((0.5, 0.08, 0.5), DARK, pos=(0, 0.48, 0.66)), box((0.48, 0.6, 0.06), DARK, pos=(0, 0.55, 0.93), anchor='bottom'),
        cylinder(0.03, 0.44, 8, BLACK, pos=(0, 0, 0.66)),
    ]
    # background rows of desks + city lights through the window
    for i in range(-3, 4):
        for j in range(1, 4):
            if i == 0: continue
            nodes.append(box((1.8, 0.04, 0.8), (0.8, 0.8, 0.8), pos=(i * 2.6, 0.72, -j * 2.4)))
            nodes.append(box((0.6, 0.38, 0.04), BLACK, pos=(i * 2.6, 1.05, -j * 2.4 - 0.2)))
    for k in range(60):
        x = (k * 7.31) % 28 - 14; y = 0.6 + (k * 3.7) % 2.2
        nodes.append(box((0.08, 0.08, 0.02), CITY_LIGHT, pos=(x, y, -8.9), emissive=True))
    return nodes


def worker(pose):
    # hips sit on the chair seat; faces -Z toward the monitor
    return person(pose, shirt=(0.95, 0.95, 0.95), pants=(0.08, 0.08, 0.09), tie=BLACK, pos=(0, 0.58, 0.66))


# ---------------------------------------------------------------- city + ships (chase)
def city():
    nodes = [plane((400, 60), (0.33, 0.37, 0.33), pos=(0, -30, 0), div=10)]
    for side in (-1, 1):
        for k in range(30):
            x = -60 + k * 9 + (k % 3) * 1.5
            h = 40 + (k * 17) % 45
            w = 5 + (k * 5) % 3
            nodes.append(box((w, h, w), (0.55, 0.58, 0.55) if k % 2 else (0.48, 0.52, 0.5), pos=(x, -30, side * (14 + (k % 4) * 2)), anchor='bottom'))
    return nodes


def rebel_ship(pos, rot=(0, 0, 0)):
    """Hand-repainted off-white arrowhead, orange spine stripe, two-seat clear canopy. Nose points +X."""
    hull = polygon_prism([(2.6, 0), (-1.6, 1.9), (-1.2, 0), (-1.6, -1.9)], 0.35, (0.86, 0.84, 0.78))
    stripe = box((3.2, 0.02, 0.25), (0.85, 0.42, 0.12), pos=(0.2, 0.19, 0))
    canopy = sphere(0.45, 12, (0.55, 0.75, 0.85), pos=(0.6, 0.22, 0), scale=(1.6, 0.6, 1.1))
    exhaust = sphere(0.18, 8, (1.0, 0.55, 0.15), pos=(-1.3, 0, 0), emissive=True)
    patch = polygon_prism([(-0.4, 0.9), (-1.4, 1.75), (-1.1, 0.9)], 0.37, (0.3, 0.31, 0.33))
    return group(hull, stripe, canopy, exhaust, patch, pos=pos, rot=rot)


def interceptor(pos, rot=(0, 0, 0)):
    """White needle fuselage with two crescent wings and cyan edge lights. Nose points +X."""
    body = cylinder(0.22, 5.0, 10, (0.92, 0.93, 0.95), rot=(0, 0, -90), anchor='center', r_top=0.05)
    def crescent(s):
        pts = [(-1.5, 0.2 * s), (-0.5, 2.6 * s), (1.2, 3.0 * s), (2.6, 2.2 * s), (1.0, 2.5 * s), (-0.2, 2.0 * s), (-0.9, 0.3 * s)]
        if s < 0: pts = pts[::-1]
        return polygon_prism(pts, 0.08, (0.9, 0.92, 0.95))
    glow = [box((1.8, 0.05, 0.05), (0.3, 1.0, 1.0), pos=(0.5, 0.05, s * 2.35), emissive=True) for s in (-1, 1)]
    canopy = sphere(0.25, 10, (0.75, 0.78, 0.82), pos=(0.2, 0.18, 0), scale=(1.8, 0.6, 1))
    return group(body, crescent(1), crescent(-1), *glow, canopy, pos=pos, rot=rot)


# ---------------------------------------------------------------- shots
def s01(t):
    cam = Camera(pos=track((0, (0, 2.25, 0.2)), (3, (0, 2.25, -0.25))), target=(0, 2.25, -1.38), lens=50)
    return office(t), cam

def s02(t):
    pose = pose_track((0, 'sit_slump'), (1.0, 'sit_slump'), (1.6, 'sit_stretch'), (2.4, 'sit_stretch'), (3.0, 'sit'))(t)
    cam = Camera(pos=(-3.0, 1.15, 0.35), target=(0, 1.0, 0.25), lens=35)
    return office(t) + [worker(pose)], cam

def s03(t):
    cam = Camera(pos=(0.55, 1.42, -0.7), target=(0, 1.22, 0.66), lens=40)
    return office(t) + [worker('sit_type')], cam

def s04(t):
    cam = Camera(pos=track((0, (0.45, 1.5, 1.55)), (3, (0.38, 1.45, 1.35))), target=(0, 1.05, -0.22), lens=35)
    return office(t) + [worker('sit_type')], cam

def c1(t):
    # Both ships fly +X (screen right in every chase shot). Camera tracks alongside.
    x = -20 + 22 * t
    reb = rebel_ship((x + 6, 2 + math.sin(t * 3) * 0.8, 0), rot=(math.sin(t * 3) * 15, 0, -10 * math.cos(t * 3)))
    itc = interceptor((x - 4, 2.6 + math.sin(t * 3 + 1) * 0.6, 0.8), rot=(0, 0, 0))
    bolts = []
    for k in range(4):
        tf = (t * 3 + k * 0.25) % 1.0
        a = (x - 1.4, 2.6, 0.8); b = (x - 1.4 + 6 * tf + 1.5, 2.4, 0.4)
        if tf > 0.1: bolts.append(beam((a[0] + 7 * tf, a[1], a[2]), (b[0] + 7 * tf, b[1], b[2]), 0.08, (0.3, 1.0, 0.95)))
    sk = shake(t, 0.15)
    cam = Camera(pos=(x - 1 + sk[0], 7.5 + sk[1], 10), target=(x + 1.5, 2.0, 0), lens=32,
                 fog=((0.62, 0.66, 0.63), 10, 90), bg=(0.7, 0.73, 0.7))
    return city() + [reb, itc] + bolts, cam



# ---------------------------------------------------------------- C2: 7 s one-shot chase-to-crash (the tutorial's hero shot)
# Beats match the Seedance prompt (references/video.md, Example 5):
# 0.0-1.9 wide chase | 1.9-2.7 swing behind the interceptor's wing, it fires, wing hit | 2.7-3.9 overtake, bolts hit a
# building ahead | 3.9-5.4 swoop round the nose and push in through the canopy | 5.4-6.3 close on his face, direct hit
# (flash, sparks, red alarm), head snaps to the window | 6.3-6.96 camera swings behind his head, ship rolls and dives
# into a wall, cut to black. Ships always fly +X (screen right in the wide).

def _rebel_path(t):
    x = 34 * t
    y = 3.0 + 1.0 * math.sin(2.0 * t) - (0 if t < 6.2 else 9 * (t - 6.2) ** 2)
    z = 2.6 * math.sin(1.3 * t)
    return np.array([x, y, z])

def _frame(path, t):
    """Ship position + yaw/pitch/roll from its own path (nose = +X, banks into turns)."""
    p, p2 = path(t), path(t + 0.02); v = (p2 - p) / 0.02
    yaw = -math.degrees(math.atan2(v[2], v[0])); pitch = math.degrees(math.atan2(v[1], math.hypot(v[0], v[2])))
    dz = (path(t + 0.04)[2] - 2 * p[2] + path(t - 0.04)[2]) / 0.0016
    roll = max(-35, min(35, -dz * 1.2)) + (0 if t < 6.2 else 70 * ease((t - 6.2) / 0.5))
    return p, yaw, pitch, roll

def _to_world(fr, local, roll_w=1.0):
    p, yaw, pitch, roll = fr
    return tuple(p + rot_matrix(roll * roll_w, yaw, pitch) @ np.asarray(local, float))

def rebel_ship_big(t, hit_wing=False, hit_cockpit=0.0, man_pose='sit', head_yaw=0.0, canopy_glass=True):
    """Rebel two-seat interceptor (~11 m, nose +X, right wing +Z), matching the reference image: off-white hull,
    pyramid nose, swept wings (left wing has a grey replacement panel), dark engine + orange exhaust at the back,
    orange spine stripe. OPEN cockpit (side walls, no solid roof) with a few thin canopy arches, so the seats and both
    characters read clearly from outside. Fixed layout: seats side by side facing the nose, dashboard with amber
    gauges in front, bulkhead with toolbox + fire extinguisher behind, stick between the LEFT seat's knees,
    throttle on the left wall. (canopy_glass kept for API compatibility; the glass is not drawn.)"""
    HULL, DARK, ORANGE, SEAT = (0.81, 0.8, 0.76), (0.16, 0.17, 0.17), (0.79, 0.42, 0.2), (0.42, 0.23, 0.13)
    def B(sx, sy, sz, col, x, y, z, **kw):          # box in the original layout coords (nose -z) -> this frame (nose +X)
        return box((sz, sy, sx), col, pos=(-z, y, x), **kw)
    parts = [
        B(2.38, 0.8, 7.2, HULL, 0, -0.66, 0.4),                                   # lower hull (floor top y=-0.25)
        B(0.12, 0.77, 3.15, HULL, -1.15, 0.135, -0.8), B(0.12, 0.77, 3.15, HULL, 1.15, 0.135, -0.8),   # cockpit side walls
        B(2.2, 0.75, 0.5, HULL, 0, 0.125, -2.15),                                 # dash block
        B(2.0, 0.08, 0.35, DARK, 0, 0.475, -1.95),                                 # dash top
        B(2.4, 0.75, 3.2, HULL, 0, 0.125, 2.4),                                   # rear deck
        B(0.7, 0.03, 3.2, ORANGE, 0, 0.515, 2.4),                                 # spine stripe
        B(0.6, 0.02, 0.9, ORANGE, -0.75, 0.33, -2.9),
        B(0.5, 0.05, 0.12, (1.0, 0.7, 0.28), 0, 0.05, -5.55, emissive=True),
        polygon_prism([(2.4, -1.2), (5.6, 0.0), (2.4, 1.2)], 0.55, HULL, pos=(0, -0.3, 0)),                      # pointed nose (top view)
        cylinder(0.7, 1.6, 20, DARK, rot=(0, 0, 90), anchor='center', pos=(-4.6, -0.05, 0)),            # engine
        sphere(0.55, 16, (1.0, 0.69, 0.38), pos=(-5.4, -0.05, 0), scale=(0.15, 1, 1), emissive=True),  # exhaust glow
        B(1.4, 0.03, 1.0, (0.43, 0.44, 0.45), -3.1, -0.08, 1.7),                 # replacement panel (left wing)
    ]
    for k in range(7):                                                            # amber gauges
        parts.append(B(0.12, 0.08, 0.01, (1.0, 0.7 if k % 2 else 0.54, 0.28 if k % 2 else 0.23), -0.75 + k * 0.25, 0.32, -1.87, emissive=True))
    for sgn in (-1, 1):                                                           # swept wings + amber edge strips
        pts = [(1.15, -1.6), (5.2, 2.0), (5.2, 2.7), (1.15, 3.6)]
        q = [(-y, sgn * x) for x, y in pts]
        if sgn < 0: q = q[::-1]
        parts.append(polygon_prism(q, 0.14, HULL, pos=(0, -0.1, 0)))
        for k in range(3):
            f = 0.25 + k * 0.25
            parts.append(box((0.08, 0.05, 0.35), (1.0, 0.7, 0.28), pos=(1.6 - 3.6 * f, -0.12, sgn * (1.15 + 4.05 * f)), rot=(0, sgn * 42, 0), emissive=True))
    for sgn in (-1, 1):                                                           # seats exactly side by side
        parts += [B(0.5, 0.12, 0.5, SEAT, sgn * 0.55, 0.13, -0.6), B(0.5, 0.58, 0.12, SEAT, sgn * 0.55, 0.45, -0.24),
                  B(0.26, 0.16, 0.08, SEAT, sgn * 0.55, 0.8, -0.2)]
    parts += [B(0.28, 0.4, 1.3, DARK, 0, 0.0, -1.0),                              # centre console
              cylinder(0.025, 0.45, 10, DARK, pos=(1.15, -0.05, -0.55)), box((0.05, 0.1, 0.06), (0.07, 0.07, 0.07), pos=(1.15, 0.45, -0.55)),   # stick (left seat)
              cylinder(0.02, 0.25, 10, DARK, pos=(0.8, 0.3, -0.98)),            # throttle (left wall)
              B(2.2, 1.2, 0.1, (0.81, 0.8, 0.75), 0, 0.35, 0.68),               # bulkhead behind the seats
              B(0.6, 0.3, 0.25, (0.55, 0.56, 0.58), 0.05, 0.62, 0.5), B(0.9, 0.45, 0.03, (0.24, 0.23, 0.17), 0.05, 0.66, 0.36),   # toolbox + cargo net
              cylinder(0.06, 0.35, 12, (0.69, 0.16, 0.13), pos=(-0.55, 0.52, -0.6))]   # fire extinguisher
    for dz in ((-0.6, 0.4, 1.0) if canopy_glass else (1.0,)):                         # thin canopy arches; front ones hidden while the camera is inside
        ff = math.sqrt(max(0.05, 1 - (dz / 1.6) ** 2)); prev = None
        for a in np.linspace(0, math.pi, 13):
            p = np.array([-(dz - 1.6) - 0.8 * 0 , 0.5 + 0.85 * ff * math.sin(a), 1.15 * ff * math.cos(a)])
            p[0] = -(dz - 0.8)
            if prev is not None:
                d = p - prev; L = float(np.linalg.norm(d))
                parts.append(box((0.06, L + 0.02, 0.06), (0.48, 0.49, 0.5), pos=tuple((p + prev) / 2), rot=(math.degrees(math.atan2(d[2], d[1])), 0, 0)))
            prev = p
    pilot = person('sit', shirt=(0.56, 0.24, 0.13), pants=(0.62, 0.55, 0.42), skin=(0.78, 0.6, 0.45), hair=(0.16, 0.1, 0.06),
                   pos=(0.62, 0.2, -0.55), rot=(0, -90, 0), scale=0.92)
    man_p = POSES[man_pose] if isinstance(man_pose, str) else man_pose
    man = person(dict(man_p, head_yaw=head_yaw), shirt=(0.95, 0.95, 0.95), pants=(0.08, 0.08, 0.09), tie=BLACK,
                 pos=(0.62, 0.2, 0.55), rot=(0, -90, 0), scale=0.92)
    parts += [pilot, man]
    if hit_wing:                                                                  # right wing hit: fire trail
        parts += [sphere(0.18 + 0.07 * k, 12, (1.0, 0.45 + 0.1 * k, 0.1), pos=(-1.2 - 0.7 * k, 0.1, 3.6 + 0.05 * k), emissive=True) for k in range(4)]
    if hit_cockpit > 0:
        parts += [sphere(0.05, 8, (1.0, 0.85, 0.4), pos=(1.95, 0.5 + 0.35 * hit_cockpit * k / 5, -0.6 + 0.25 * k), emissive=True) for k in range(6)]
        if int(t * 8) % 2 == 0: parts.append(sphere(0.09, 12, (1.0, 0.1, 0.1), pos=(1.9, 0.6, 0.0), emissive=True))   # red alarm
    return parts


def _crescent(s):
    """One crescent wing in top view (s=+1 right, -1 left): grows from the rear of the fuselage, sweeps wide outward
    and curves forward to a sharp tip level with the nose. Built from tapered segments so it reads as a smooth arc."""
    W = (0.92, 0.93, 0.95); parts = []
    C, n = np.array([-0.2, 2.5]), 18
    phis = np.radians(np.linspace(-136, -136 - 236, n))          # sweep backward -> outward -> forward
    rads = np.linspace(3.0, 3.9, n)
    pts = [C + r * np.array([math.cos(p), math.sin(p)]) for p, r in zip(phis, rads)]
    for k in range(n - 1):
        u0, u1 = k / (n - 1), (k + 1) / (n - 1)
        w0, w1 = 0.12 + 0.7 * math.sin(math.pi * min(u0 * 1.15, 1)) , 0.12 + 0.7 * math.sin(math.pi * min(u1 * 1.15, 1))
        a, b = pts[k], pts[k + 1]
        na, nb = (a - C) / np.linalg.norm(a - C), (b - C) / np.linalg.norm(b - C)
        quad = [a, b, b - nb * w1, a - na * w0]                    # outer edge -> inner edge
        q = [(p[0], s * p[1]) for p in quad]
        if s < 0: q = q[::-1]
        parts.append(polygon_prism(q, 0.1, W))
        if 0.25 < u0 < 0.8:                                        # cyan strip on the inner edge
            ia, ib = a - na * w0, b - nb * w1; mid = (ia + ib) / 2; d = ib - ia
            parts.append(box((float(np.linalg.norm(d)), 0.05, 0.05), (0.35, 1.0, 1.0), pos=(mid[0], 0.07, s * mid[1]),
                             rot=(0, -math.degrees(math.atan2(s * d[1], d[0])), 0), emissive=True))
    return parts


def interceptor_big(t):
    """Hostile interceptor matching the reference: needle-thin white fuselage with a sharp nose and rear spike, two
    mirrored crescent wings forming a pincer around it, small chrome teardrop canopy, twin rail guns. Nose +X."""
    W = (0.92, 0.93, 0.95)
    parts = [cylinder(0.38, 4.8, 20, W, rot=(0, 0, -90), r_top=0.02, pos=(0, 0, 0)),          # front cone -> nose
             cylinder(0.38, 4.2, 20, W, rot=(0, 0, 90), r_top=0.04, pos=(0, 0, 0)),           # rear cone -> spike
             sphere(0.32, 20, (0.78, 0.8, 0.84), pos=(0.3, 0.28, 0), scale=(2.0, 0.55, 0.9)),  # chrome canopy
             ]
    for z in (-0.32, 0.32):                                                                     # twin rail guns
        parts += [cylinder(0.035, 3.2, 8, (0.7, 0.72, 0.74), rot=(0, 0, -90), pos=(1.6, -0.1, z)),
                  sphere(0.05, 8, (0.4, 1.0, 1.0), pos=(4.8, -0.1, z), emissive=True)]
    return parts + _crescent(1) + _crescent(-1)


_CITY_C2 = None
def city_c2():
    global _CITY_C2
    if _CITY_C2 is None:
        nodes = [plane((600, 120), (0.33, 0.37, 0.33), pos=(150, -25, 0), div=12)]
        for side in (-1, 1):
            for k in range(36):
                x = -60 + k * 9 + (k % 3) * 1.5; h = 45 + (k * 17) % 50; w = 6 + (k * 5) % 3
                col = (0.55, 0.58, 0.55) if k % 2 else (0.47, 0.51, 0.49)
                nodes.append(box((w, h, w), col, pos=(x, -25, side * (13 + (k % 4) * 2)), anchor='bottom'))
                if k % 5 != 2: nodes.append(box((w * 1.05, 2.5, w * 1.05), (0.25, 0.42, 0.21), pos=(x, -25 + h, side * (13 + (k % 4) * 2)), anchor='bottom'))   # overgrown roofs
                if k % 3 == 0: nodes.append(box((w * 0.9, h * 0.25, 0.4), (0.3, 0.42, 0.3), pos=(x, -25 + h * 0.3, side * (13 + (k % 4) * 2) - side * w / 2), anchor='bottom'))   # vines
        end = _rebel_path(6.96)
        nodes.append(box((10, 60, 30), (0.5, 0.53, 0.5), pos=(end[0] + 12, -25, end[2]), anchor='bottom'))   # the wall they crash into
        _CITY_C2 = nodes
    return _CITY_C2


def c2(t):
    fr = _frame(_rebel_path, t)
    lag = 1.0 - 0.3 * ease((t - 1.9) / 1.5)                                              # interceptor ~34 m behind, closing to ~24 m
    fi = _frame(lambda u: _rebel_path(u) + np.array([0, 2.5, 4.0 * math.sin(u * 1.7)]), t - lag)   # offset up + weaving: never overlaps
    man_pose = pose_track((0, 'sit'), (5.4, 'sit'), (5.6, 'sit_shock'), (6.35, 'sit_shock'), (6.6, 'arms_up'))(t)
    head = track((5.7, 0), (5.85, -70), kind='out')(t)                                       # snaps toward the side window
    reb = group(*rebel_ship_big(t, hit_wing=t > 2.35, hit_cockpit=max(0, min(1, (t - 5.75) / 0.3)), man_pose=man_pose, head_yaw=head,
                                canopy_glass=not (5.15 < t < 6.4)),                                                      # glass vanishes once the camera is through it
                pos=tuple(fr[0]), rot=(fr[3], fr[1], fr[2]))
    itc = group(*interceptor_big(t), pos=tuple(fi[0]), rot=(fi[3], fi[1], fi[2]))
    extra = []
    for k, t0 in enumerate([1.95, 2.1, 2.25, 2.95, 3.1, 3.25, 3.4]):                     # bolts from the interceptor's nose
        if t0 <= t < t0 + 0.35:
            u = (t - t0) / 0.35
            a = np.array(_to_world(fi, (4.2, 0, 0))); b = np.array(_to_world(fr, (2 + 10 * (k >= 3), 0.6, 2.4 if k == 2 else (-1.5 + k % 2 * 3))))
            pa, pb = a + (b - a) * u, a + (b - a) * min(1, u + 0.25)
            extra.append(beam(tuple(pa), tuple(pb), 0.12, (0.3, 1.0, 0.95)))
    if 3.3 < t < 4.2:                                                                       # bolts hit a building ahead: debris
        hit = np.array(_rebel_path(3.3)) + np.array([42, 8, -12]); u = t - 3.3
        extra += [sphere(0.5 + 0.2 * (k % 3), 8, (0.55, 0.55, 0.52), pos=tuple(hit + np.array([3 * math.cos(k) * u * 4, -9 * u * u + 2 * u * (k % 4), 3 * math.sin(k) * u * 4]))) for k in range(10)]
        if u < 0.4: extra.append(sphere(1.6 * (1 - u / 0.4) + 0.4, 12, (1.0, 0.7, 0.3), pos=tuple(hit), emissive=True))
    # camera: keyframes in the rebel ship's frame (x = forward, y = up, z = right)
    K = [(0.0, (-62, 11, 6), (-16, 1.5, 0)), (1.9, (-46, 6.5, 4.0), (-14, 1.5, 0)), (2.7, (-36, 4.5, 3.0), (0, 0.5, 0)),
         (3.4, (-18, 3.0, 3.0), (2, 0.6, 0)), (3.9, (-8.0, 2.2, 1.0), (2, 0.6, 0)), (4.6, (4.5, 2.6, 5.0), (0.6, 0.8, 0.2)), (5.4, (2.2, 1.15, 0.55), (0.6, 1.05, 0.55)),
         (6.3, (1.45, 1.12, 0.55), (0.6, 1.08, 0.55)), (6.55, (-0.35, 1.4, 0.35), (4, 0.75, 0.25)), (6.96, (-0.35, 1.4, 0.35), (4, 0.75, 0.25))]
    cpos = track(*[(k[0], k[1]) for k in K])(t); ctg = track(*[(k[0], k[2]) for k in K])(t)
    rw = ease((t - 4.6) / 0.8)                                                              # inside the ship, the camera rolls with it
    sk = shake(t, 0.25 if 5.7 < t < 6.2 else 0.06, 14)
    cam = Camera(pos=tuple(np.array(_to_world(fr, cpos, rw)) + np.array(sk) * (0.3 if t > 5.4 else 1)),
                 target=_to_world(fr, ctg, rw), lens=30, roll=fr[3] * rw * 0.0,
                 flash=max(0, 0.9 - (t - 5.72) * 3) if t > 5.72 else 0, fade=ease((t - 6.75) / 0.2),
                 fog=((0.62, 0.66, 0.63), 15, 140), bg=(0.7, 0.73, 0.7))
    return city_c2() + [reb, itc] + extra, cam


SHOTS = [
    Shot('S01', 3.0, 'ECU WALL CLOCK', 'Second hand ticks. Locked off, slow creep in.', 50, s01),
    Shot('S02', 3.0, 'SIDE PROFILE WIDE - BORED', 'Slumped, head on hand; big stretch, arms up; settles back.', 35, s02),
    Shot('S03', 2.0, 'FRONT MCU FROM BEHIND MONITOR', 'Tired, typing, lit by the blank screen.', 35, s03),
    Shot('S04', 3.0, 'OTS - LOOKING AT MONITOR', 'Over the shoulder; screen stays blank for UI later.', 35, s04),
    Shot('C1', 3.0, 'SIDE-TRACKING WIDE - CHASE', 'Rebel ship flees screen-right; interceptor behind firing bolts.', 35, c1),
    Shot('C2', 6.96, 'ONER - CHASE TO CRASH', 'Wide > behind interceptor wing (hit) > overtake > swoop round nose > into cockpit > direct hit > over-shoulder crash.', 30, c2),
]
