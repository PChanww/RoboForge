"""Run paired MuJoCo controls and export their evidence. ffmpeg is needed for MP4."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim_env import BUILTINS, TabletopEnv

ROOT = Path(__file__).resolve().parent
TICKS = 180
SEEDS = range(10)
FPS = 20
FRAME_SIZE = (1280, 720)
ASSET_LABELS = {"standard": "标准杯", "wide": "宽口杯", "tall": "高杯"}


def advance(env, open_only):
    action = env.expert_action()
    if open_only:
        action[-1] = 1
    env.step(action)


def score(asset, seed, open_only):
    env = TabletopEnv(asset, seed)
    try:
        initial_z = env.metrics()["object_xyz"][2]
        peak_rise, success_at = 0.0, None
        for tick in range(1, TICKS + 1):
            advance(env, open_only)
            m = env.metrics()
            peak_rise = max(peak_rise, m["object_xyz"][2] - initial_z)
            if success_at is None and m["success"]:
                success_at = tick
        return success_at, round(1000 * peak_rise, 1), m
    finally:
        env.close()


def run_controls():
    rows = {}
    for asset in BUILTINS:
        successes, heights, open_successes, open_heights, open_lifts = [], [], [], [], []
        for seed in SEEDS:
            success_at, height, _ = score(asset, seed, False)
            open_at, open_height, open_final = score(asset, seed, True)
            successes.append(success_at)
            heights.append(height)
            open_successes.append(open_at)
            open_heights.append(open_height)
            open_lifts.append(open_final["lifted"])
        rows[asset] = {
            "baseline_success_tick_by_seed": successes,
            "baseline_max_rise_mm_by_seed": heights,
            "open_control_success_tick_by_seed": open_successes,
            "open_control_max_rise_mm_by_seed": open_heights,
            "open_control_lifts": sum(open_lifts),
        }
    baseline = sum(t is not None for row in rows.values() for t in row["baseline_success_tick_by_seed"])
    negative = sum(t is not None for row in rows.values() for t in row["open_control_success_tick_by_seed"])
    return {
        "task": "Place the cup in the tray.",
        "controls": {
            "baseline": "Privileged pose-aware scripted controller, not VLA inference.",
            "open_control": "Same target XYZ at every tick, jaws commanded fully open throughout.",
        },
        "assets": "Procedurally generated built-in cups. No Lux3D geometry in this evaluation.",
        "simulator": "MuJoCo",
        "mujoco_version": mujoco.__version__,
        "sim_env_sha256": hashlib.sha256((ROOT / "sim_env.py").read_bytes()).hexdigest(),
        "seeds_in_array_order": list(SEEDS),
        "steps_per_run": TICKS,
        "control_hz": 20,
        "success": ("Lift the cup at least 31 mm with bilateral contact, then release upright "
                    "within the tray footprint, in tray-floor contact and at rest for six "
                    "successive control ticks. Success remains latched for the episode."),
        "summary": {"runs_per_control": len(SEEDS) * len(BUILTINS),
                    "baseline_successes": baseline, "open_control_successes": negative},
        "results": rows,
    }


def font_family(font_path, bold=False):
    if font_path:
        return font_path
    files = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / ("msyhbd.ttc" if bold else "msyh.ttc"),
        Path(f"/usr/share/fonts/opentype/noto/NotoSansCJK-{'Bold' if bold else 'Regular'}.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    for path in files:
        if path.is_file():
            return str(path)
    raise RuntimeError("Install a CJK font or pass --font with its path.")


def fonts(font_path):
    regular, bold = font_family(font_path), font_family(font_path, True)
    return {key: ImageFont.truetype(path, size) for key, path, size in [
        ("large", bold, 36), ("panel", bold, 24), ("body", regular, 21),
        ("small", regular, 18), ("meta", regular, 16)]}


def render_pair(envs, comparison, initial_z, font):
    frame = Image.new("RGB", FRAME_SIZE, "#101a20")
    draw = ImageDraw.Draw(frame)
    part = 2 if comparison else 1
    tick = envs[0].tick
    draw.text((36, 18), f"MUJOCO     /     桌面抓放实验     /     {part:02}  ·  02",
              fill="#85c9bd", font=font["meta"])
    heading = "同一条运动轨迹，夹爪开合会改变结果吗？" if comparison else "同一个抓放任务，换一种杯型"
    draw.text((34, 45), heading, fill="#f0f5f4", font=font["large"])
    draw.text((1244, 62), f"随机种子  0    /    仿真  {tick / 20:.2f} s",
              fill="#bbcacb", font=font["small"], anchor="ra")
    for i, (x, env) in enumerate(zip((36, 650), envs)):
        m = env.metrics()
        panel_width = 594
        draw.rounded_rectangle((x, 118, x + panel_width, 552), radius=11, fill="#1b2a33")
        label = (("正常闭爪", "始终开爪")[i] if comparison else ASSET_LABELS[env.asset])
        draw.text((x+15, 128), label, fill="#edf5f3", font=font["panel"])
        status = ("已放稳" if m["success"] else "未完成" if tick == TICKS else "接触托盘" if m["tray_contact"] else
                  "已抬起" if m["lifted"] else "双指接触" if m["contacts"] == 2 else "运行中")
        draw.text((x+panel_width-15, 137), status, fill="#8ad8c6", anchor="ra", font=font["small"])
        frame.paste(Image.fromarray(env.render(panel_width, 335)), (x, 176))
        rise = 1000 * (m["object_xyz"][2] - initial_z[i])
        draw.text((x+15, 520), f"接触杯身的手指   {m['contacts']}/2",
                  fill="#d1dbdc", font=font["meta"])
        draw.text((x+panel_width-15, 520), f"杯身升高   {max(0, rise):.0f} mm",
                  fill="#d1dbdc", font=font["meta"], anchor="ra")
    if comparison:
        caption = ("两侧执行相同的位移指令，右侧夹爪始终打开。" if tick < 65 else
                   "右侧夹爪按相同路径移动，杯子留在桌面上。")
    else:
        caption = ("两款杯子从相同位置出发，控制器按各自的杯高调整夹取位置。" if tick < 64 else
                   "杯子由双指接触带动，位移由物理引擎计算。" if tick < 150 else
                   "松爪后，杯子须在托盘内直立静置至少 0.3 秒。")
    draw.text((36, 573), caption, fill="#d9e5e3", font=font["body"])
    draw.rounded_rectangle((36, 625, 1244, 630), radius=2, fill="#30414b")
    if tick:
        draw.rounded_rectangle((36, 625, 36 + int(1208 * tick / TICKS), 630), radius=2, fill="#81d2bd")
    draw.text((36, 661), "控制器  ·  脚本基线（读取初始位姿，无 VLA 推理）", fill="#b1c0c3", font=font["small"])
    draw.text((1244, 661), "资产  ·  程序生成杯型（非 Lux3D）", fill="#b1c0c3",
              font=font["small"], anchor="ra")
    return frame


def write_video(path, font_path):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Install ffmpeg for video generation, or use --no-video.")
    font = fonts(font_path)
    args = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", "1280x720", "-r", str(FPS), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path)]
    with subprocess.Popen(args, stdin=subprocess.PIPE) as proc:
        try:
            for comparison in (False, True):
                envs = [TabletopEnv("standard", 0), TabletopEnv("standard" if comparison else "wide", 0)]
                try:
                    initial = [e.metrics()["object_xyz"][2] for e in envs]
                    for tick in range(TICKS + 1):
                        if tick:
                            advance(envs[0], False)
                            advance(envs[1], comparison)
                        pixels = render_pair(envs, comparison, initial, font).tobytes()
                        # Freeze initial and final frames long enough to read both panels.
                        repeats = 18 if tick == 0 else 26 if tick == TICKS else 1
                        for _ in range(repeats):
                            proc.stdin.write(pixels)
                finally:
                    for e in envs:
                        e.close()
        finally:
            proc.stdin.close()
        if proc.wait():
            raise RuntimeError("ffmpeg failed to encode the video")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evidence")
    parser.add_argument("--no-video", action="store_true", help="Only run the controls and save JSON")
    parser.add_argument("--font", help="CJK TrueType/OpenType font for the video captions")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_controls()
    with tempfile.TemporaryDirectory(dir=args.output_dir) as scratch:
        scratch = Path(scratch)
        if not args.no_video:
            video = scratch / "baseline-comparison.mp4"
            write_video(video, args.font)
            video.replace(args.output_dir / video.name)
        target = scratch / "baseline-results.json"
        target.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        target.replace(args.output_dir / target.name)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
