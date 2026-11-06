# blocking-previz

Scripts for the `blocking-previz` skill (Raph). Adapted from the "AI Video Realism" skill (Tao Prompts).

- `scripts/blocking.py` + `scripts/render_blocking.py`: pure-Python 3D blocking renderer (numpy + Pillow + ffmpeg). Renders previz clips to feed Seedance as a video reference.
  Added: `--size 9:16 | 9:16hd | 16:9 | 16:9hd` presets, label layout for vertical frames, `walk()` walk/run cycle.
- `scripts/example_scene.py`: original 16:9 example (office + aircraft chase oner).
- `scripts/example_vertical.py`: 9:16 example (street at night, low dolly back, orbit, car overtakes).
- `scripts/screen_comp.py` + `assets/modnet_fp16.onnx`: composite UI / content onto blank screens in generated videos (MODNet, Apache-2.0).

Setup: `pip install numpy pillow imageio-ffmpeg` (blocking), plus `opencv-python-headless scipy onnxruntime` (screen comp).
