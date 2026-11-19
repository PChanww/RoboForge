"""Register a cup GLB (or a ZIP containing one) for use by the MuJoCo lab."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import struct
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
import trimesh

REGISTRY = Path(__file__).resolve().parents[1] / "assets" / "imported"
MAX_FILE = 256 * 1024**2


def download_mesh(outputs, directory):
    """Download the first GLB/ZIP mesh from provider outputs, validating by magic bytes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
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


def read_glb(path: Path, member: str | None = None) -> bytes:
    if path.stat().st_size > MAX_FILE:
        raise ValueError("Asset exceeds 256 MiB.")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            choices = [i for i in zf.infolist() if i.filename.lower().endswith(".glb")
                       and not i.filename.startswith("__MACOSX/")]
            if member:
                choices = [i for i in choices if i.filename == member]
            if len(choices) != 1:
                raise ValueError("Select exactly one GLB in the ZIP with --member. Candidates: "
                                 + ", ".join(i.filename for i in choices))
            if choices[0].file_size > MAX_FILE:
                raise ValueError("Unpacked GLB exceeds 256 MiB.")
            return zf.read(choices[0])
    if path.suffix.lower() != ".glb":
        raise ValueError("Expected a .glb or a .zip containing a .glb.")
    if member:
        raise ValueError("--member is only valid for ZIP input.")
    return path.read_bytes()


def sample_color(mesh: trimesh.Trimesh):
    """Approximate the visible surface by one color for the simulator material."""
    visual = mesh.visual
    if visual.kind is None or not np.isfinite(mesh.area) or mesh.area <= 0:
        return None
    count = min(2048, len(mesh.faces))
    positions = (np.arange(count) + 0.5) * mesh.area / count
    ids = np.searchsorted(np.cumsum(mesh.area_faces), positions).clip(0, len(mesh.faces)-1)
    if visual.kind == "face":
        return visual.face_colors[ids, :3].mean(axis=0) / 255
    if visual.kind == "vertex":
        return visual.vertex_colors[mesh.faces[ids], :3].mean(axis=(0, 1)) / 255
    mat = visual.material
    factor = getattr(mat, "baseColorFactor", None)
    texture = getattr(mat, "baseColorTexture", None)
    rgb = np.asarray(factor[:3], dtype=float) / 255 if factor is not None else np.ones(3)
    if texture is not None and visual.uv is not None:
        uv = visual.uv[mesh.faces[ids]].mean(axis=1)
        # glTF color textures are sRGB while baseColorFactor is linear.
        colors = np.asarray(mat.to_color(uv), dtype=float)[:, :3] / 255
        linear = np.where(colors <= .04045, colors / 12.92, ((colors + .055) / 1.055) ** 2.4)
        rgb = rgb * linear.mean(axis=0)
    elif factor is None:
        return None
    return rgb


def parse_geometry(data: bytes):
    if len(data) < 20:
        raise ValueError("Incomplete GLB header.")
    magic, version, length, json_size, tag = struct.unpack_from("<4sIII4s", data)
    if (magic, version, length, tag) != (b"glTF", 2, len(data), b"JSON") or json_size > length - 20:
        raise ValueError("Expected a glTF 2.0 binary mesh with a JSON header.")
    gltf = json.loads(data[20:20 + json_size])
    for item in [*gltf.get("buffers", []), *gltf.get("images", [])]:
        if item.get("uri") and not item["uri"].startswith("data:"):
            raise ValueError("Export a self-contained GLB with embedded buffers and images.")
    scene = trimesh.load_scene(io.BytesIO(data), file_type="glb", process=False)
    # Bake every visible instance's world transform before merging primitives.
    geometries = []
    colors, weights = [], []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces):
            mesh = mesh.copy()
            mesh.apply_transform(transform)
            rgb = sample_color(mesh)
            if rgb is not None:
                colors.append(rgb)
                weights.append(mesh.area)
            mesh.visual = trimesh.visual.ColorVisuals(mesh)
            geometries.append(mesh)
    if not geometries:
        raise ValueError("No triangle mesh found. Select the GLB mesh, not the splat preview.")
    mesh = trimesh.util.concatenate(geometries)
    if not np.isfinite(mesh.vertices).all() or len(mesh.faces) > 1_000_000:
        raise ValueError("Invalid vertices or over one million triangles. Export a smaller mesh.")
    color = np.average(colors, axis=0, weights=weights).clip(0, 1).tolist() if colors else None
    return mesh, color


def check_scale(height, radius, mass):
    if not 0 < height < 1 or (radius is not None and not 0 < radius < 1) or not 0 < mass < 10:
        raise ValueError("This tabletop simulator accepts dimensions below 1 m and mass below 10 kg; all must be positive.")


def register(path: Path, output: Path, *, height: float, mass: float,
             radius: float | None, source: str, task_id: str | None,
             prompt: str | None = None, member: str | None = None, up: str = "y",
             source_detail: str = "user-supplied export") -> Path:
    if output.exists():
        raise ValueError(f"Asset already exists: {output}. Use a new slug.")
    if source == "lux3d" and not (task_id or "").strip():
        raise ValueError("For Lux3D, provide the original generation task or asset ID.")
    check_scale(height, radius, mass)
    data = read_glb(path, member)
    mesh, color = parse_geometry(data)
    if up == "y":
        mesh.apply_transform([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    low, high = mesh.bounds
    if np.any(high - low < 1e-9):
        raise ValueError("Mesh has a flat or zero-volume bounding box.")
    mesh.apply_scale(height / (high[2] - low[2]))
    low, high = mesh.bounds
    bottom = mesh.vertices[mesh.vertices[:, 2] <= low[2] + 0.12 * height, :2]
    # Center the cylinder under the cup's foot so a projecting handle doesn't offset it.
    center = (bottom.min(axis=0) + bottom.max(axis=0)) / 2
    mesh.apply_translation([-center[0], -center[1], -low[2]])
    proxy_radius = float(radius if radius is not None else min(mesh.extents[:2]) / 2)
    check_scale(height, proxy_radius, mass)
    manifest = {
        "obj": "visual.obj", "radius": proxy_radius, "height": height, "mass": mass,
        "radius_m": proxy_radius, "height_m": height, "mass_kg": mass,
        "bounds_m": mesh.bounds.tolist(), "up_axis": "Z", "triangles": int(len(mesh.faces)),
        "collision_proxy": "upright cylinder, body only; no handle/rim/interior contacts",
        "radius_basis": "user-specified" if radius is not None else "half of shorter horizontal extent",
        "source": source, "source_detail": source_detail, "generation_id": task_id, "prompt": prompt,
        "source_mesh": "original.glb", "sha256_glb": hashlib.sha256(data).hexdigest(),
        "scale_basis": "user-specified height, not an automatic physical measurement",
    }
    if color is not None:
        manifest["color_rgb"] = color
        manifest["color_basis"] = "area-weighted mean of material or indexed vertex colors; uniform simulator tint"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".building-", dir=output.parent))
    try:
        (temp / "original.glb").write_bytes(data)
        # The simulator supplies the cup's material. Keep source PBR data intact in original.glb.
        with (temp / "visual.obj").open("w", encoding="ascii", newline="\n") as f:
            np.savetxt(f, mesh.vertices, fmt="v %.9g %.9g %.9g")
            np.savetxt(f, mesh.faces + 1, fmt="f %d %d %d")
        (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.rename(output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return output / "manifest.json"


def output_path(slug: str, registry: Path = REGISTRY) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", slug):
        raise ValueError("Use a 1–64 character slug with Latin letters, digits, '_' or '-'.")
    return registry / slug


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--slug", required=True)
    p.add_argument("--source", choices=["lux3d", "other"], default="other")
    p.add_argument("--task-id", help="Original provider generation task / asset ID")
    p.add_argument("--prompt", help="Original generation prompt, if available")
    p.add_argument("--height-m", type=float, default=0.09)
    p.add_argument("--radius-m", type=float, help="Cup body radius, excluding handle")
    p.add_argument("--mass-kg", type=float, default=0.12)
    p.add_argument("--up", choices=["y", "z"], default="y", help="Source up axis, glTF standard is Y")
    p.add_argument("--member", help="Exact GLB member name if ZIP contains multiple variants")
    a = p.parse_args()
    try:
        print(register(a.input, output_path(a.slug), height=a.height_m, radius=a.radius_m, mass=a.mass_kg,
                       source=a.source, task_id=a.task_id, prompt=a.prompt, up=a.up, member=a.member))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        p.exit(1, f"Import failed: {exc}\n")


if __name__ == "__main__":
    main()
