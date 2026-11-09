"""Fit longitudinal regions of the generated calibration mesh to local rigid-link frames."""
from pathlib import Path
import numpy as np
from import_asset import parse_geometry
root=Path(__file__).resolve().parents[1]/"assets/robot"
mesh,_=parse_geometry((root/"original.glb").read_bytes())
# glTF +Y is up. Express vertices in simulator coordinates.
v=mesh.vertices[:,[0,2,1]].copy()
v[:,1]*=-1
z=(v[:,2]-v[:,2].min())/np.ptp(v[:,2])
for name,lo,hi,width,length in [('base',0,.18,.108,.08),('upper',.18,.51,.062,.23),('fore',.51,.82,.056,.23),('wrist',.82,1.001,.041,.045)]:
    center_z=z[mesh.faces].mean(axis=1)
    faces=mesh.faces[(center_z>=lo)&(center_z<hi)]
    ids,idx=np.unique(faces,return_inverse=True)
    pts=v[ids].copy()
    pts-=pts.min(axis=0)
    pts[:,:2]-=np.ptp(pts[:,:2],axis=0)/2
    pts[:,:2]*=width/np.ptp(pts[:,:2],axis=0).max()
    pts[:,2]*=length/np.ptp(pts[:,2])
    if name=='wrist':pts[:,2]*=-1
    with (root/(name+'.obj')).open('w') as f:
        np.savetxt(f,pts,fmt="v %.8f %.8f %.8f")
        np.savetxt(f,idx.reshape(-1,3)+1,fmt="f %d %d %d")
