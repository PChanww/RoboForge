"""Generate a cup via Aholo's China API, download its mesh, and register it."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from import_asset import MAX_FILE, check_scale, download_mesh, output_path, register

PROMPT = ("A single upright handleless ceramic tumbler, straight cylindrical sides, an open circular "
          "rim, flat base, matte ivory glaze. Height 9 cm, outside diameter 6 cm. "
          "Isolated object, no saucer, no table, no liquid, no text.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slug", default="lux3d-cup")
    p.add_argument("--prompt", help="Custom text-to-3D prompt; default generates a handleless cup.")
    p.add_argument("--task-id", help="Resume an existing generation; does not create a new task.")
    p.add_argument("--height-m", type=float, default=0.09)
    p.add_argument("--radius-m", type=float, default=None)
    p.add_argument("--mass-kg", type=float, default=0.12)
    p.add_argument("--kind", choices=["task_object", "static_prop", "robot_visual"],
                   help="Generic asset kind; when set, uses the generic importer (box/cylinder proxy by spec).")
    p.add_argument("--role", choices=["visual_only", "static", "dynamic"], default="dynamic")
    p.add_argument("--width-m", type=float, default=None)
    p.add_argument("--depth-m", type=float, default=None)
    a = p.parse_args()
    if not a.kind:
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
        if a.kind and not a.prompt and not a.task_id:
            p.exit(1, "Generic --kind requires an explicit --prompt (no cup default).\n")
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
        if a.kind:
            from import_object import register_object
            print(register_object(file, out, kind=a.kind, role=a.role, height=a.height_m,
                                  width=a.width_m, depth=a.depth_m, radius=a.radius_m, mass=a.mass_kg,
                                  source="lux3d", source_detail="downloaded from authenticated Aholo API task",
                                  task_id=job["task_id"], prompt=job["prompt"]))
        else:
            print(register(file, out, height=a.height_m, radius=a.radius_m, mass=a.mass_kg, source="lux3d",
                           source_detail="downloaded from authenticated Aholo API task",
                           task_id=job["task_id"], prompt=job["prompt"]))


if __name__ == "__main__":
    main()
