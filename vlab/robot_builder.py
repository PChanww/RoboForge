"""Articulated robot assembly from split parts + standard URDF export.

Honest engineering scope:
- Does NOT fabricate unverified physics/kinematics from a single unrigged silhouette.
- Exports a valid ROS/MuJoCo URDF XML for any RobotConfig link/joint tree.
- Assembles split visual part meshes (e.g. from part_split) onto the validated 5R
  kinematic chain (or builds a custom multi-link visual shell replacement) with safe
  capsule collision geoms and analytical IK preserved.
"""
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .scene_assets import valid_id
from .schema import RobotConfig


def _fmt(vals) -> str:
    return " ".join(f"{float(v):.6g}" for v in vals)


def robot_to_urdf(robot: RobotConfig, mesh_uri_prefix: str = "meshes") -> str:
    """Serialize any RobotConfig tree into a standard ROS-compatible URDF XML string."""
    root = ET.Element("robot", {"name": robot.id})
    for link in robot.links:
        el = ET.SubElement(root, "link", {"name": link.name})
        total_mass = sum(float(g.mass_kg) for g in link.collisions)
        diag = link.inertia_diag or [1e-4, 1e-4, 1e-4]
        if total_mass > 0:
            inert = ET.SubElement(el, "inertial")
            ET.SubElement(inert, "origin", {"xyz": _fmt(link.com or [0.0, 0.0, 0.0]), "rpy": "0 0 0"})
            ET.SubElement(inert, "mass", {"value": f"{total_mass:.6g}"})
            ET.SubElement(inert, "inertia", {
                "ixx": f"{diag[0]:.6g}", "ixy": "0", "ixz": "0",
                "iyy": f"{diag[1]:.6g}", "iyz": "0", "izz": f"{diag[2]:.6g}",
            })
        for v in link.visuals:
            vis = ET.SubElement(el, "visual")
            ET.SubElement(vis, "origin", {"xyz": _fmt(v.pos), "rpy": "0 0 0"})
            geom = ET.SubElement(vis, "geometry")
            ET.SubElement(geom, "mesh", {"filename": f"{mesh_uri_prefix}/{v.mesh}.obj"})
            mat = ET.SubElement(vis, "material", {"name": f"{v.mesh}_mat"})
            ET.SubElement(mat, "color", {"rgba": _fmt(v.rgba)})
        for g in link.collisions:
            col = ET.SubElement(el, "collision")
            if g.type == "capsule" and g.fromto:
                a = np.asarray(g.fromto[:3], dtype=float)
                b = np.asarray(g.fromto[3:], dtype=float)
                mid = 0.5 * (a + b)
                length = float(np.linalg.norm(b - a))
                ET.SubElement(col, "origin", {"xyz": _fmt(mid.tolist()), "rpy": "0 0 0"})
                geom = ET.SubElement(col, "geometry")
                ET.SubElement(geom, "cylinder", {"radius": f"{g.size[0]:.6g}", "length": f"{max(length, 1e-3):.6g}"})
            elif g.type == "sphere":
                ET.SubElement(col, "origin", {"xyz": _fmt(g.pos), "rpy": "0 0 0"})
                geom = ET.SubElement(col, "geometry")
                ET.SubElement(geom, "sphere", {"radius": f"{g.size[0]:.6g}"})
            else:
                ET.SubElement(col, "origin", {"xyz": _fmt(g.pos), "rpy": "0 0 0"})
                geom = ET.SubElement(col, "geometry")
                sz = g.size if len(g.size) == 3 else [g.size[0], g.size[0], g.size[0]]
                ET.SubElement(geom, "box", {"size": _fmt([2.0 * x for x in sz])})

    links_by_name = {l.name: l for l in robot.links}
    act_by_joint = {a.joint: a for a in robot.actuators}
    for j in robot.joints:
        if j.child not in links_by_name:
            continue
        child_link = links_by_name[j.child]
        parent_name = child_link.parent
        if not parent_name or parent_name not in links_by_name:
            continue
        jtype = "revolute" if j.type == "revolute" else ("prismatic" if j.type == "prismatic" else "fixed")
        je = ET.SubElement(root, "joint", {"name": j.name, "type": jtype})
        ET.SubElement(je, "parent", {"link": parent_name})
        ET.SubElement(je, "child", {"link": j.child})
        ET.SubElement(je, "origin", {"xyz": _fmt(child_link.origin), "rpy": "0 0 0"})
        if jtype != "fixed":
            ET.SubElement(je, "axis", {"xyz": _fmt(j.axis)})
            rng = j.range or [-3.14159, 3.14159]
            effort = act_by_joint[j.name].force_limit if j.name in act_by_joint else 20.0
            ET.SubElement(je, "limit", {
                "lower": f"{rng[0]:.6g}", "upper": f"{rng[1]:.6g}",
                "effort": f"{effort:.6g}", "velocity": "3.14",
            })
            ET.SubElement(je, "dynamics", {"damping": f"{j.damping:.6g}"})

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def _normalize_part_obj(src_obj: Path, dst_obj: Path, target_height: float, anchor: str = "bottom") -> None:
    verts = []
    faces = []
    for line in src_obj.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append(line)
    if not verts:
        raise ValueError(f"Part OBJ {src_obj} has no vertices")
    v = np.asarray(verts, dtype=np.float64)
    low, high = v.min(axis=0), v.max(axis=0)
    ext = np.maximum(high - low, 1e-6)
    longest = float(ext.max())
    scale = float(target_height) / longest
    v = v * scale
    low, high = v.min(axis=0), v.max(axis=0)
    cx = 0.5 * (low[0] + high[0])
    cy = 0.5 * (low[1] + high[1])
    cz = low[2] if anchor == "bottom" else 0.5 * (low[2] + high[2])
    v -= np.array([cx, cy, cz])
    with dst_obj.open("w", encoding="ascii", newline="\n") as f:
        np.savetxt(f, v, fmt="v %.9g %.9g %.9g")
        if faces:
            f.write("\n".join(faces) + "\n")


# Map robot link mesh names -> expected nominal height/extent in metres so articulated visual
# shells fit joints proportionally while preserving kinematic capsule collision & analytic IK.
_SLOT_NOMINAL_EXTENTS = {
    "base": ("pedestal", 0.08),
    "upper": ("upper_arm", 0.23),
    "fore": ("forearm", 0.23),
    "wrist": ("tool", 0.08),
}


def assemble_robot_from_parts(
    base_robot: RobotConfig,
    part_objs: dict[str, Path],
    lab_root: Path,
    robot_id: str,
) -> RobotConfig:
    """Copy and scale split part OBJs onto articulated robot links in assets/robots/<robot_id>/."""
    rid = valid_id(robot_id)
    if not part_objs:
        raise ValueError("Provide at least one part mapping (e.g. base, upper, fore, wrist)")
    for slot in part_objs:
        if slot not in _SLOT_NOMINAL_EXTENTS:
            raise ValueError(f"Unsupported robot part slot {slot!r}; expected one of {sorted(_SLOT_NOMINAL_EXTENTS)}")
    src_dir = Path(lab_root) / base_robot.mesh_dir
    out_dir = Path(lab_root) / "assets" / "robots" / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    # Seed all required mesh OBJs from the base robot first (palm, finger, unreplaced slots)
    for p in src_dir.glob("*.obj"):
        shutil.copy2(p, out_dir / p.name)
    # Replace specified slots with normalized split part meshes
    for slot, src_obj in part_objs.items():
        src_obj = Path(src_obj)
        if not src_obj.is_file():
            raise FileNotFoundError(f"Part OBJ not found: {src_obj}")
        _, target_h = _SLOT_NOMINAL_EXTENTS[slot]
        anchor = "center" if slot == "wrist" else "bottom"
        _normalize_part_obj(src_obj, out_dir / f"{slot}.obj", target_height=target_h, anchor=anchor)
    return base_robot.model_copy(update={
        "id": rid,
        "mesh_dir": f"assets/robots/{rid}",
    })
