"""An accretion disk around the hole: Doppler beaming makes the approaching
side brighter, and light bending lifts the far side into arcs above and
below the shadow.

Usage:  python examples/03_disk.py     ->  disk.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from PIL import Image

from bh.render import deflection_table
from bh.disk import trajectory_table, disk_layer, add_bloom
from bh.sky import StarSky

r0, W, H, fov = 42.0, 1280, 720, 26.0
sky = StarSky()
psi_t, A_t = deflection_table(r0)
traj = trajectory_table(r0)

a = np.array([0.0, 0.0, -1.0], dtype=np.float32)
t = np.tan(np.radians(fov) / 2)
xs = (((np.arange(W, dtype=np.float32) + 0.5) / W) * 2 - 1) * t
ys = (1 - ((np.arange(H, dtype=np.float32) + 0.5) / H) * 2) * t * H / W
PX, PY = np.meshgrid(xs, ys)
d = np.stack([PX, PY, -np.ones_like(PX)], axis=-1)
d /= np.linalg.norm(d, axis=-1, keepdims=True)
cospsi = np.clip(d @ a, -1, 1)
sinpsi = np.sqrt(np.maximum(1 - cospsi ** 2, 1e-18))
p = (d - cospsi[..., None] * a) / sinpsi[..., None]
psi = np.arccos(cospsi)

A = np.interp(psi, psi_t, A_t).astype(np.float32)
sdir = np.cos(A)[..., None] * (-a) + np.sin(A)[..., None] * p
col = sky(sdir)
col[psi < psi_t[0]] = 0.0
disk_rgb, hit = disk_layer(psi, p, r0, incl_deg=83.0, traj=traj)
col = np.where(hit[..., None], disk_rgb, col)
col = add_bloom(col)
col = col / (1.0 + 0.35 * col)
Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8)).save("disk.png")
print("saved disk.png")
