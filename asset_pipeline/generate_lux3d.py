"""Generate a cup via Aholo's China API, download its mesh, and register it."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from import_asset import MAX_FILE, check_scale, output_path, register

PROMPT = ("A single upright handleless ceramic tumbler, straight cylindrical sides, an open circular "
          "rim, flat base, matte ivory glaze. Height 9 cm, outside diameter 6 cm. "
          "Isolated object, no saucer, no table, no liquid, no text.")


def download_mesh(outputs, directory):
    urls = [s["content"] for s in outputs if isinstance(s.get("content"), str)
            and urlsplit(s["content"]).scheme == "https"]
    # URLs can be signed; check file signatures instead of trusting extensions.
    for url in sorted(urls, key=lambda u: not urlsplit(u).path.lower().endswith(".glb")):
        with urlopen(url, timeout=90) as response:
            header = response.read(4)
            if header == b"glTF":
                path = directory / "result.glb"
            elif header == b"PK\x03\x04":
                path = directory / "result.zip"
            else:
                continue
            with path.open("wb") as dest:
                dest.write(header)
                while chunk := response.read(1024 * 1024):
                    if dest.tell() + len(chunk) > MAX_FILE:
                        raise ValueError("Generated mesh exceeds 256 MiB.")
                    dest.write(chunk)
            return path
    raise ValueError("Provider returned no GLB/ZIP mesh. Check the task's available exports.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slug", default="lux3d-cup")
    p.add_argument("--prompt", help="Custom text-to-3D prompt; default generates a handleless cup.")
    p.add_argument("--task-id", help="Resume an existing generation; does not create a new task.")
    p.add_argument("--height-m", type=float, default=0.09)
    p.add_argument("--radius-m", type=float, default=None)
    p.add_argument("--mass-kg", type=float, default=0.12)
    a = p.parse_args()
    check_scale(a.height_m, a.radius_m, a.mass_kg)
    out = output_path(a.slug)
    if out.exists():
        p.exit(1, f"Already registered: {out}. Use a different slug.\n")
    key = (os.environ.get("AHOLO_API_KEY") or os.environ.get("LUX3D_CN_API_KEY")
           or os.environ.get("LUX3D_API_KEY"))
    if not key:
        p.exit(1, "Set AHOLO_API_KEY locally or import a GLB downloaded from your account.\n")
    try:
        from manycore.aholo_sdk_core import AholoClientConfig
        from manycore.aholo_sdk_lux3d import create_lux3d_client
    except ImportError:
        p.exit(1, "Install the optional dependency: python -m pip install manycore-aholo-sdk-lux3d==1.9.0\n")
    client = create_lux3d_client(AholoClientConfig(api_key=key, region="cn"))
    jobs = out.parent.parent / "generation_jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    receipt = jobs / (out.name + ".json")
    if receipt.exists():
        job = json.loads(receipt.read_text(encoding="utf-8"))
        if not job["task_id"]:
            if not a.task_id:
                p.exit(1, "Creation status unknown. Check the Aholo task history. Pass its --task-id "
                       "to resume, or choose a new slug if no task was created.\n")
            job["task_id"] = a.task_id
        if (a.task_id and str(job["task_id"]) != a.task_id) or (a.prompt and job["prompt"] != a.prompt):
            p.exit(1, "Slug already has a different task or prompt. Use a new slug.\n")
    else:
        prompt = a.prompt if a.task_id else a.prompt or PROMPT
        job = {"task_id": a.task_id, "prompt": prompt}
        # Save intent first, so a retry after network failure doesn't create another billable task.
        with receipt.open("x", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False)
        if not a.task_id:
            job["task_id"] = str(client.text_to_3d.create(prompt=prompt, version="G1-Turbo",
                style="photorealistic", face_count=20000, output_format=["glb"], enable_pbr=True))
    # Use a unique temporary file for concurrent polling/resume processes.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=jobs, delete=False) as f:
        json.dump(job, f, ensure_ascii=False, indent=2)
        receipt_tmp = Path(f.name)
    receipt_tmp.replace(receipt)
    print(f"Aholo task {job['task_id']}. Restart with the same slug to resume.", flush=True)
    result = client.tasks.wait_for(job["task_id"], interval_ms=10000, timeout_ms=900000)
    with tempfile.TemporaryDirectory() as tmp:
        file = download_mesh(result["outputs"], Path(tmp))
        print(register(file, out, height=a.height_m, radius=a.radius_m, mass=a.mass_kg, source="lux3d",
                       source_detail="downloaded from authenticated Aholo API task",
                       task_id=job["task_id"], prompt=job["prompt"]))


if __name__ == "__main__":
    main()
