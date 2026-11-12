"""Build MuJoCo XML from a RobotConfig link/joint tree. No fixed DOF assumptions."""
from __future__ import annotations

from .schema import CollisionGeom, ProjectConfig, RobotConfig, SensorConfig


def _fmt(values):
    return " ".join(f"{v:.6g}" for v in values)


def _geom_xml(g: CollisionGeom):
    name = f' name="{g.name}"' if g.name else ""
    common = (f'{name} mass="{g.mass_kg:.6g}" friction="{_fmt(g.friction)}" '
              f'contype="{g.contype}" conaffinity="{g.conaffinity}" group="{g.group}" '
              f'rgba="{_fmt(g.rgba)}" solref="{_fmt(g.solref)}"')
    if g.type == "capsule" and g.fromto:
        return f'<geom type="capsule" fromto="{_fmt(g.fromto)}" size="{g.size[0]:.6g}" {common}/>'
    if g.type == "mesh":
        return f'<geom type="mesh" mesh="{g.mesh}" pos="{_fmt(g.pos)}" {common}/>'
    return f'<geom type="{g.type}" pos="{_fmt(g.pos)}" size="{_fmt(g.size)}" {common}/>'


def _visual_xml(v):
    return (f'<geom class="visual" type="mesh" mesh="{v.mesh}" pos="{_fmt(v.pos)}" '
            f'quat="{_fmt(v.quat)}" rgba="{_fmt(v.rgba)}"/>')


def _camera_xml(s: SensorConfig):
    w, h = s.resolution
    # Config stores corner-based pixels; MuJoCo 4.0.0 renders principalpixel as a
    # center-relative offset with the opposite sign (verified in tests/test_camera_projection.py).
    px, py = w / 2 - s.intrinsics.cx, h / 2 - s.intrinsics.cy
    return (f'<camera name="{s.id}" pos="{_fmt(s.position)}" xyaxes="{_fmt(s.xyaxes)}" '
            f'sensorsize=".032 .024" focalpixel="{s.intrinsics.fx:.6g} {s.intrinsics.fy:.6g}" '
            f'principalpixel="{px:.6g} {py:.6g}" resolution="{w} {h}"/>')


def robot_xml(robot: RobotConfig, sensors: list[SensorConfig]):
    links = {l.name: l for l in robot.links}
    joint_by_child = {j.child: j for j in robot.joints}
    children = {l.name: [] for l in robot.links}
    for l in robot.links:
        if l.parent:
            children[l.parent].append(l.name)
    cams_by_parent: dict[str, list[str]] = {}
    for s in sensors:
        cams_by_parent.setdefault(s.parent, []).append(_camera_xml(s))
    mesh_names = set()

    def build(name):
        l = links[name]
        j = joint_by_child.get(name)
        joint = ""
        if j and j.type != "fixed":
            kind = "hinge" if j.type == "revolute" else "slide"
            rng = f' range="{_fmt(j.range)}"' if j.range else ""
            joint = (f'<joint name="{j.name}" type="{kind}" axis="{_fmt(j.axis)}"{rng} '
                     f'damping="{j.damping:.6g}" armature="{j.armature:.6g}"/>')
        geoms = []
        for g in l.collisions:
            geoms.append(_geom_xml(g))
            if g.type == "mesh" and g.mesh:
                mesh_names.add(g.mesh)
        for v in l.visuals:
            geoms.append(_visual_xml(v))
            mesh_names.add(v.mesh)
        inertial = ""
        if l.inertia_diag is not None:
            mass = sum(g.mass_kg for g in l.collisions)
            pos = _fmt(l.com or [0, 0, 0])
            inertial = f'<inertial pos="{pos}" mass="{mass:.6g}" diaginertia="{_fmt(l.inertia_diag)}"/>'
        gc = "1" if l.gravcomp else "0"
        inner = joint + inertial + "".join(geoms) + "".join(cams_by_parent.get(name, []))
        inner += "".join(build(c) for c in children[name])
        return f'<body name="{name}" pos="{_fmt(l.origin)}" gravcomp="{gc}">{inner}</body>'

    body = build(robot.root_link)
    actuators = "".join(
        f'<position name="{a.joint}" joint="{a.joint}" kp="{a.kp:.6g}" kv="{a.kv:.6g}" '
        f'forcerange="-{a.force_limit:.6g} {a.force_limit:.6g}"/>'
        for a in robot.actuators)
    world_cams = "".join(cams_by_parent.get("world", []))
    return body, actuators, sorted(mesh_names), world_cams


def mesh_files(config: ProjectConfig, robot: RobotConfig, mesh_names, lab_root):
    base = lab_root / robot.mesh_dir
    return {f"{n}.obj": (base / f"{n}.obj").read_bytes() for n in mesh_names}