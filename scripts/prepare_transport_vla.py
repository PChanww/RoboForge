"""Download the published SmolVLA checkpoint and processor, checking source hashes.

Weights live under ignored work/, never in the source tree or a dataset export.
No paid inference/generation service is involved.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
OPENER = build_opener(ProxyHandler({}))


def prepare(repo, directory, *, processor=False):
    url = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
    with OPENER.open(url, timeout=60) as response:
        rows = json.load(response)["Data"]["Files"]
    rows = [row for row in rows if row["Type"] == "blob" and
            (row["Path"].endswith((".json", ".txt")) or not processor and row["Path"].endswith(".safetensors"))]
    directory.mkdir(parents=True, exist_ok=True)

    def download(row):
        path = directory / row["Path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = row["Sha256"]
        if path.is_file():
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == expected:
                    return row
        source = f"https://modelscope.cn/models/{repo}/resolve/master/{quote(row['Path'])}"
        temporary = path.with_suffix(path.suffix + ".part")
        # Some download gateways truncate large single responses. Strict byte
        # ranges make interruption resumable without accepting a partial model.
        offset = temporary.stat().st_size if temporary.is_file() else 0
        if offset > row["Size"]:
            offset = 0
        if row["Size"] > 32*1024*1024 and offset < row["Size"]:
            ranges = directory / (row["Path"] + ".ranges")
            ranges.mkdir(exist_ok=True)
            starts = list(range(offset, row["Size"], 2*1024*1024))

            def fetch_range(start):
                end = min(start+2*1024*1024, row["Size"])-1
                part = ranges / f"{start}.part"
                if part.is_file() and part.stat().st_size == end-start+1:
                    return part
                for attempt in range(4):
                    try:
                        request = Request(source, headers={"Range": f"bytes={start}-{end}"})
                        with OPENER.open(request, timeout=60) as response:
                            data = response.read()
                            if response.headers.get("Content-Range") != f"bytes {start}-{end}/{row['Size']}" or len(data) != end-start+1:
                                raise ValueError("Incomplete model range")
                        part.write_bytes(data)
                        return part
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(1)

            with ThreadPoolExecutor(max_workers=8) as pool, temporary.open("ab" if offset else "wb") as output:
                for index, part in enumerate(pool.map(fetch_range, starts)):
                    output.write(part.read_bytes())
                    output.flush()
                    if index % 32 == 0:
                        print(json.dumps({"file": row["Path"], "downloaded": output.tell(), "total": row["Size"]}), flush=True)
            offset = temporary.stat().st_size
        with temporary.open("ab" if offset else "wb") as output:
            while offset < row["Size"]:
                end = min(offset + 8*1024*1024, row["Size"]) - 1
                request = Request(source, headers={"Range": f"bytes={offset}-{end}"})
                with OPENER.open(request, timeout=120) as response:
                    data = response.read()
                    if response.status == 206:
                        if response.headers.get("Content-Range") != f"bytes {offset}-{end}/{row['Size']}" or len(data) != end-offset+1:
                            raise ValueError(f"Invalid source byte range: {row['Path']}")
                    elif response.status != 200 or offset or len(data) != row["Size"]:
                        raise ValueError(f"Invalid source response: {row['Path']}")
                output.write(data)
                output.flush()
                offset += len(data)
        with temporary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256")
        if temporary.stat().st_size != row["Size"] or digest.hexdigest() != expected:
            raise ValueError(f"Download hash/size mismatch: {repo}/{row['Path']}")
        temporary.replace(path)
        ranges = directory / (row["Path"] + ".ranges")
        if ranges.is_dir():
            for chunk in ranges.glob("*.part"):
                chunk.unlink()
            ranges.rmdir()
        print(json.dumps({"file": str(path.relative_to(ROOT)), "bytes": row["Size"], "verified": True}), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(download, rows))
    (directory / "download-provenance.json").write_text(json.dumps({"repo": repo, "provider": "modelscope",
        "files": [{k: row[k] for k in ("Path", "Size", "Sha256", "Revision")} for row in receipts]}, indent=2), encoding="utf8")


if __name__ == "__main__":
    prepare("lerobot/smolvla_libero", ROOT / "work/smolvla-libero")
    prepare("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", ROOT / "work/smolvlm-processor", processor=True)
