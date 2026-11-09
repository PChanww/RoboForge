"""Five revolute joints, analytic fixed-attitude IK and Lux3D visual links."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent / "assets" / "robot"
L1, L2 = .23, .23
BASE_Y, SHOULDER_Z, TOOL = .205, .08, .075
NAMES = ("waist","shoulder","elbow","wrist_pitch","wrist_yaw")
def ik(point):
    x, y, z = point
    radial = np.hypot(x,y-BASE_Y)
    h = z + TOOL - SHOULDER_Z
    bend = np.arccos(np.clip((radial**2+h**2-L1**2-L2**2)/(2*L1*L2),-1,1))
    yaw = np.arctan2(y-BASE_Y,x)
    pitch = np.arctan2(radial,h)-np.arctan2(L2*np.sin(bend),L1+L2*np.cos(bend))
    return [yaw,pitch,bend,-pitch-bend,-yaw]

def mesh_files():
    return {p.name:p.read_bytes() for p in ROOT.glob("*.obj")}

def arm_xml(fingers, spec):
    shell=lambda name: f'<geom class="visual" type="mesh" mesh="{name}" rgba=".84 .73 .52 1"/>'
    def hinge(i,axis,limits):
        return f'<joint name="{NAMES[i]}" axis="{axis}" type="hinge" range="{limits}" damping=".8" armature=".012"/>'
    # Collision capsules stop short of the joint housings so adjacent links never overlap.
    # contype=2/conaffinity=1: links collide with the world, cup and gripper, not each other.
    def rod(length,mass):
        return f'<geom type="capsule" fromto="0 0 .02 0 0 {length-.07}" size=".017" mass="{mass}" group="3" rgba="0 0 0 0" contype="2" conaffinity="1"/>'
    # Joint housings keep their mass but are non-colliding: the rods already cover link volume.
    hub='<geom type="sphere" size=".027" mass=".08" material="rubber" contype="0" conaffinity="0"/>'
    return f"""
    <body name="pedestal" pos="0 {BASE_Y} 0">{shell('base')}
    <body name="turret" pos="0 0 {SHOULDER_Z}" gravcomp="1">{hinge(0,'0 0 1','-3.14 3.14')}{hub}
    <body name="upper_arm" gravcomp="1">{hinge(1,'0 1 0','-1.7 2.7')}{rod(L1,spec['upper_mass_kg'])}{shell('upper')}
    <body name="forearm" pos="0 0 {L1}" gravcomp="1">{hinge(2,'0 1 0','.05 3.08')}{rod(L2,spec['forearm_mass_kg'])}{shell('fore')}{hub}
    <body name="wrist" pos="0 0 {L2}" gravcomp="1">{hinge(3,'0 1 0','-3.45 1.7')}{hub}
    <body name="tool" gravcomp="1">{hinge(4,'0 0 1','-3.14 3.14')}{shell('wrist')}
    <geom type="sphere" size=".018" mass=".06" material="steel" contype="0" conaffinity="0"/>
    <body name="eef" pos="0 0 {-TOOL}" gravcomp="1">
    <geom type="box" pos="0 0 .060" size=".016 .040 .008" material="rubber" mass=".09"/>
    <camera name="wrist" pos=".062 0 .075" xyaxes="0 1 0 -1 0 .72" fovy="87"/>
    {''.join(fingers)}
    </body></body></body></body></body></body></body>"""
def actuators(spec):
    return ''.join(f'<position joint="{n}" kp="{kp}" kv="{kv}" forcerange="-{force} {force}"/>' for n,kp,kv,force in zip(NAMES,[320,370,290,150,100],[15,17,12,6,4],[spec["shoulder_torque_nm"]]*3+[8,5]))

def parameters(candidate=None):
    p=json.loads((candidate or (ROOT/"physics.json")).read_text())
    bounds={"upper_mass_kg":(.1,1.5),"forearm_mass_kg":(.08,1.2),"shoulder_torque_nm":(8,80)}
    if set(p)!=set(bounds) or any(type(p[k]) not in (int,float) or not lo<=p[k]<=hi for k,(lo,hi) in bounds.items()):
        raise ValueError("Invalid arm mass / torque proposal")
    return p
