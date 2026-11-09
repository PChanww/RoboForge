"""Render an aligned replay (front+wrist frames with state/action overlay) to MP4.

Reads a staging episode npz and writes datasets/replay/ep{index}.mp4.
"""
import argparse
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAB = Path(__file__).resolve().parents[1]
W, H, FPS = 960, 480, 20


def font(size, bold=False):
    for p in (Path("C:/Windows/Fonts") / ("msyhbd.ttc" if bold else "msyh.ttc"),
              Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")):
        if p.is_file():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=19)
    ap.add_argument("--staging", default=str(LAB / "datasets" / "staging"))
    ap.add_argument("--out", default=str(LAB / "datasets" / "replay"))
    args = ap.parse_args()
    z = np.load(Path(args.staging) / "episodes" / f"ep{args.episode:03d}.npz", allow_pickle=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ep{args.episode:03d}_replay.mp4"
    f_big, f_small = font(20, True), font(14)
    n = z["joint_pos"].shape[0]
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
         "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)
    try:
        for i in range(n):
            frame = Image.new("RGB", (W, H), "#101a20")
            front = Image.open(io.BytesIO(bytes(z["jpeg_front"][i]))).resize((420, 420))
            wrist = Image.open(io.BytesIO(bytes(z["jpeg_wrist"][i]))).resize((420, 420))
            frame.paste(front, (20, 40))
            frame.paste(wrist, (460, 40))
            d = ImageDraw.Draw(frame)
            d.text((20, 8), f"Episode {args.episode}  ·  {str(z['asset'][0])}  ·  seed {int(z['seed'][0])}",
                   fill="#85c9bd", font=f_small)
            a = z["action"][i]
            eef = z["eef_pos"][i]
            d.text((20, 462), f"t={z['frame_time'][i]:.2f}s  eef_z={eef[2]:.3f}m  gripper={z['gripper'][i]:.2f}  "
                              f"action=[{a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f},{a[3]:+.2f}]",
                   fill="#d9e5e3", font=f_small)
            img = Image.new("RGB", (W, H), "#101a20")
            img.paste(front, (20, 30))
            img.paste(wrist, (500, 30))
            d = ImageDraw.Draw(img)
            d.text((20, 4), f"回放 · Episode {args.episode} · {str(z['asset'][0])} · seed {int(z['seed'][0])}",
                   fill="#85c9bd", font=f_small)
            d.text((20, 454), f"frame {i:03d}/{n}  t={z['frame_time'][i]:.2f}s  eef_z={eef[2]:.3f}m  "
                              f"gripper={z['gripper'][i]:.2f}  action=[{a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f},{a[3]:+.2f}]",
                   fill="#d9e5e3", font=f_small)
            if bool(z["success"][0]) and i == n - 1:
                d.text((720, 4), "成功", fill="#81d2bd", font=f_big)
            proc.stdin.write(img.tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
    print(f"wrote {path} ({n} frames)" if code == 0 else f"ffmpeg failed {code}")


if __name__ == "__main__":
    main()