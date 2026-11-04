# RUNLOG

Dated log of every command that matters: runtime and headline output. Becomes the methods section.

## Environment (2026-11-04)
- Host: Windows 11, NVIDIA RTX 4050 Laptop GPU (6 GB), driver 592.82 (CUDA 13.1 driver API), 16 GB RAM.
- WSL2 Ubuntu 24.04.4, Python 3.12.3; WSL memory limit currently 7.6 GB (default; see note below).
- torch 2.10.0+cu128, vllm 0.19.1, transformers 5.22.1, datasets 5.1.0, math-verify 0.9.0, pillow 13.0.0,
  imageio-ffmpeg (ffmpeg 9.0.2 static build, libx264). Full pin list: requirements.lock.txt.
- Project lives at ~/wsp-r1 inside WSL (ext4), not /mnt/c, for I/O speed.

## Log
- 2026-11-04 Kit unpacked from wsp-r1-starter-kit.zip; file modes reset to 644 (zip had them read-only).
- 2026-11-04 pip install -r requirements.txt (+ imageio-ffmpeg, pillow for the demo video). OK.
- 2026-11-04 Verified: torch.cuda.is_available() True (RTX 4050 Laptop GPU); vllm 0.19.1 imports.
- 2026-11-04 PLAN.md Phase 6 video spec revised by Dev (30 s loopable cut + 60 s explainer); make_demo_video.py written.
- NOTE: wsp.load_base() builds a ~7 GB fp32 copy; WSL has 7.6 GB. Needs a larger WSL memory limit
  (.wslconfig) before the editor run (Phase 5).
