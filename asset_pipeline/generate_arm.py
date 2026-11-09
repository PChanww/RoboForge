"""Create or resume the single Lux3D arm task. The source stays in its own asset directory."""
import getpass
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from manycore.aholo_sdk_core import AholoClientConfig
from manycore.aholo_sdk_lux3d import create_lux3d_client
from generate_lux3d import download_mesh
from import_asset import read_glb

OUT = Path(__file__).resolve().parents[1] / "assets" / "robot"
PROMPT = ("A single realistic compact industrial articulated tabletop robot arm, in its fully extended "
          "vertical calibration pose. One fixed heavy circular bolted pedestal at the bottom, two long "
          "successive slim rigid links stacked almost straight upwards, shoulder and elbow hinges with "
          "circular graphite bearing caps, short wrist housing, small parallel fingers at the very top. "
          "Total height 55 cm, base diameter 13 cm. Upper arm and forearm have equal length, brushed "
          "warm off-white cast aluminum shells with a narrow burnt-orange accent, dark joint seals and "
          "subtle panel seams. Clean product CAD mesh, physically plausible joints, no cables connecting "
          "different links, no scenery, no extra limbs, no logo, no turntable, no rails or gantry.")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    meta = OUT / "source.json"
    if (OUT / "original.glb").exists():
        raise SystemExit("Original arm already exists")
    client = create_lux3d_client(AholoClientConfig(api_key=os.getenv("AHOLO_API_KEY") or getpass.getpass("Credential: "),region="cn"))
    if meta.exists():
        record = json.loads(meta.read_text())
        if not record.get("task_id"):
            raise SystemExit("Inspect Aholo task history before retrying the pending submission.")
    else:
        record = dict(prompt=PROMPT,version="G1-Turbo",task_id=None)
        with meta.open("x") as f:
            json.dump(record,f,indent=2)
        record["task_id"] = str(client.text_to_3d.create(prompt=PROMPT,version="G1-Turbo",style="photorealistic",face_count=30000,output_format=["glb"],enable_pbr=True))
        meta.write_text(json.dumps(record,indent=2))
    print("Arm task", record["task_id"],flush=True)
    result = client.tasks.wait_for(record["task_id"],interval_ms=10000,timeout_ms=900000)
    with tempfile.TemporaryDirectory() as temp:
        glb = read_glb(download_mesh(result["outputs"], Path(temp)))
        record["sha256"] = hashlib.sha256(glb).hexdigest()
        (OUT/"original.glb").write_bytes(glb)
        meta.write_text(json.dumps(record,indent=2))
