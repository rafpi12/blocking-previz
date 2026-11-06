#!/usr/bin/env python3
"""Render shots from a blocking scene file.

  python render_blocking.py scene.py                       # all shots
  python render_blocking.py scene.py --shots S01-S04        # a group (render 4-5 shots per AI video generation)
  python render_blocking.py scene.py --shots S02 --still 1.2   # one PNG to check a pose/framing quickly
  options: --name NAME  --out DIR  --size 9:16 | 9:16hd | 16:9 | WxH  --fps 24  --ss 1  --workers N (default: all cores)

The scene file must define SHOTS (list of blocking.Shot) and may define RENDERER (blocking.Renderer).
Outputs: <name>_clean.mp4 (feed to the video model as @video1), <name>_labeled.mp4 (shot id, lens,
timecode, action), <name>_timing.txt.
"""
import argparse, importlib.util, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocking as B
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scene'); ap.add_argument('--shots', default='all'); ap.add_argument('--name')
    ap.add_argument('--out', default='blocking_out'); ap.add_argument('--size', default=None)
    ap.add_argument('--fps', type=int, default=24); ap.add_argument('--ss', type=int, default=None)
    ap.add_argument('--still', type=float, default=None); ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('scene', a.scene); sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
    R = getattr(sc, 'RENDERER', B.Renderer())
    PRESETS = {'9:16': (540, 960), '9:16hd': (720, 1280), '16:9': (960, 540), '16:9hd': (1280, 720)}
    if a.size: R.W, R.H = PRESETS.get(a.size.lower(), None) or tuple(map(int, a.size.lower().split('x')))
    if a.ss: R.ss = a.ss
    shots = B.parse_selection(a.shots, sc.SHOTS)
    os.makedirs(a.out, exist_ok=True)
    if a.still is not None:
        s = shots[0]; nodes, cam = s.build(a.still)
        p = os.path.join(a.out, f'{s.id}_{a.still:.2f}s.png'); Image.fromarray(R.render(nodes, cam, a.still)).save(p); print(p); return
    name = a.name or (a.shots.replace(',', '_') if a.shots != 'all' else 'all_shots')
    t0 = time.time()
    p = B.render_shots(shots, R, a.out, name, fps=a.fps, workers=a.workers)
    print(f'done in {time.time() - t0:.0f}s ->', p)


if __name__ == '__main__':
    main()
