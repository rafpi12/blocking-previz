"""Example 9:16 blocking scene: a man walks down a Paris street at night, a car passes screen-right.
Render:  python render_blocking.py example_vertical.py --shots V01-V02 --size 9:16
Pattern: build the set once as a function, each Shot.build(t) returns (nodes, Camera).
Vertical framing: keep the subject in the central column, use height (low angles, tilts, people head-to-toe).
"""
import math
from blocking import *

RENDERER = Renderer(540, 960, bg=(0.03, 0.04, 0.07), ambient=0.40, ss=2)

ASPHALT, PAVE, WALL, WALL2 = (0.16, 0.17, 0.19), (0.42, 0.42, 0.40), (0.62, 0.58, 0.50), (0.55, 0.52, 0.46)
WINDOW, LAMP, CAR = (1.0, 0.78, 0.45), (1.0, 0.85, 0.55), (0.55, 0.08, 0.08)

def street(t):
    """Street runs along Z. Road centre x=0, pavements at |x|>4, Haussmann-ish facades at |x|=6."""
    n = [plane((14, 80), ASPHALT, pos=(0, 0, -30), div=10)]
    for s in (-1, 1):
        n.append(box((2.2, 0.15, 80), PAVE, pos=(s * 5.1, 0, -30), anchor='bottom'))
        for k in range(12):
            z = 5 - k * 6
            n.append(box((0.4, 18, 5.8), WALL if k % 2 else WALL2, pos=(s * 6.4, 0, z), anchor='bottom'))
            for fl in range(1, 6):
                for w in (-1.5, 0, 1.5):
                    lit = (k * 7 + fl * 3 + int(w)) % 4 == 0
                    n.append(box((0.05, 1.4, 0.8), WINDOW if lit else (0.08, 0.09, 0.12), pos=(s * 6.18, fl * 3.1, z + w), emissive=lit))
        for k in range(6):   # street lamps
            z = 2 - k * 12
            n += [cylinder(0.06, 4.2, 8, (0.1, 0.1, 0.1), pos=(s * 4.4, 0.15, z)),
                  sphere(0.22, 10, LAMP, pos=(s * 4.4, 4.4, z), emissive=True)]
    return n

def car(z, x=1.6, heading=180):
    """Car body; heading 180 = drives toward -Z (same way he walks). Lane x=1.6 = near lane."""
    body = box((1.8, 0.7, 4.3), CAR, pos=(0, 0.35, 0), anchor='bottom')
    cab = box((1.6, 0.55, 2.2), (0.15, 0.18, 0.22), pos=(0, 1.05, -0.2), anchor='bottom')
    wheels = [cylinder(0.33, 0.25, 14, (0.05, 0.05, 0.05), rot=(0, 0, 90), anchor='center', pos=(sx * 0.9, 0.33, sz * 1.4)) for sx in (-1, 1) for sz in (-1, 1)]
    lights = [box((0.35, 0.15, 0.05), (1, 0.95, 0.8), pos=(sx * 0.6, 0.75, 2.16), emissive=True) for sx in (-1, 1)]
    return group(body, cab, *wheels, *lights, pos=(x, 0, z), rot=(0, heading, 0))

def man(t):
    # walks toward -Z at 1.4 m/s on the right pavement
    return person(walk(), shirt=(0.93, 0.93, 0.93), pants=(0.93, 0.93, 0.93), hair=(0.9, 0.9, 0.88),
                  pos=(4.9, 1.0, -1.4 * t), rot=(0, 0, 0))

def v01(t):
    # Low-angle dolly back in front of him, slight rise. He stays in the central column.
    z = -1.4 * t
    cam = Camera(pos=(4.7, track((0, 0.5), (4, 0.9))(t), z - 3.2), target=(4.9, 1.7, z), lens=24)
    return street(t) + [man(t)], cam

def v02(t):
    # Starts on his face (side), orbits to his back as the car passes; car crosses the frame.
    t0 = 4.0 + t
    z = -1.4 * t0
    ang = track((0, -100), (4, 15))(t)   # road side (x<4.9) to behind him; never through the facade at x=6.2
    r = 2.2
    cam = Camera(pos=(4.9 + r * math.sin(math.radians(ang)), 1.6, z + r * math.cos(math.radians(ang))),
                 target=(4.9, 1.55, z), lens=32)
    return street(t0) + [man(t0), car(4 - 5 * t)], cam

SHOTS = [
    Shot('V01', 4.0, 'Low dolly back', 'He walks toward camera on the right pavement. Low angle, slow rise.', 24, v01),
    Shot('V02', 4.0, 'Orbit + car pass', 'Camera orbits from his profile to his back while the red car overtakes him in the near lane.', 32, v02),
]
