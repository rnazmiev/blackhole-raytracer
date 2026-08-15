"""Backward ray tracer for a static camera near a Schwarzschild black hole.

The camera sits on the +z axis at radius r0 and by default looks at the BH.
By spherical symmetry the deflection depends only on the angle psi between a
pixel's view ray and the BH direction, so we integrate a dense 1D table of
geodesics and map every pixel through it. Sampling is refined near the
critical angle so higher-order photon rings survive interpolation.
"""

import numpy as np

from .geodesic import M, trace, shadow_angle


def deflection_table(r0, n_base=1600, n_crit=900, eps_min=1e-8,
                     dphi=3e-3, phi_max=50.0):
    """Sorted (psi, A) samples for escaped rays; psi[0] is the shadow edge."""
    psic = shadow_angle(r0)
    parts = [np.linspace(1e-4, np.pi - 0.031, n_base),
             psic + np.geomspace(eps_min, 0.4, n_crit)]
    psi = np.unique(np.concatenate(parts))
    psi = psi[(psi > 0) & (psi < np.pi - 0.03)]
    res = trace(psi, r0, dphi=dphi, phi_max=phi_max)
    esc = res["escaped"]
    # nearly-backwards rays leave radially with negligible bending
    tail_psi = np.linspace(np.pi - 0.03, np.pi, 32)
    tail_A = np.pi - tail_psi
    psi_e = np.concatenate([psi[esc], tail_psi])
    A_e = np.concatenate([res["A"][esc], tail_A])
    o = np.argsort(psi_e)
    return psi_e[o], A_e[o]


def render_frame(r0, sky, width=1280, height=720, fov_deg=60.0,
                 look_offset=0.0, supersample=2, table_kw=None,
                 table=None, v=0.0, doppler=False, sky_rot=0.0):
    """Render one frame; returns uint8 (height, width, 3).

    look_offset tilts the camera away from the BH direction (radians, about
    the x-axis, towards +y) — used for the photon-ring close-ups.
    table    — precomputed (psi_t, A_t) to reuse across frames at fixed r0.
    v        — radial infall speed (fraction of c) of the camera relative to
               a static observer; applies special-relativistic aberration.
               For free fall from rest at infinity v = sqrt(2M/r0).
    """
    psi_t, A_t = table if table is not None else \
        deflection_table(r0, **(table_kw or {}))

    W, H = width * supersample, height * supersample
    a = np.array([0.0, 0.0, -1.0], dtype=np.float32)       # camera -> BH
    co, so = np.cos(look_offset), np.sin(look_offset)
    fwd = np.array([0.0, so, -co], dtype=np.float32)
    rgt = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up = np.cross(rgt, fwd).astype(np.float32)

    t = np.tan(np.radians(fov_deg) / 2)
    xs = (((np.arange(W, dtype=np.float32) + 0.5) / W) * 2 - 1) * t
    ys = (1 - ((np.arange(H, dtype=np.float32) + 0.5) / H) * 2) * t * H / W
    PX, PY = np.meshgrid(xs, ys)

    d = fwd[None, None, :] + PX[..., None] * rgt + PY[..., None] * up
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    cospsi = np.clip(d @ a, -1.0, 1.0)
    sinpsi = np.sqrt(np.maximum(1.0 - cospsi ** 2, 1e-18))
    p = (d - cospsi[..., None] * a) / sinpsi[..., None]
    if v > 0.0:
        # aberration: falling-frame view angle -> static-frame view angle
        # (angles measured from the inward radial direction = motion direction)
        cospsi = np.clip((cospsi - v) / (1.0 - v * cospsi), -1.0, 1.0)
    psi = np.arccos(cospsi)

    A = np.interp(psi, psi_t, A_t).astype(np.float32)
    er = -a
    sdir = np.cos(A)[..., None] * er + np.sin(A)[..., None] * p

    if sky_rot != 0.0:
        # slide the celestial sphere about the vertical axis — reads as a
        # slow orbital drift while the camera (and the shadow) hold still
        ca, sa = np.cos(sky_rot), np.sin(sky_rot)
        x, z = sdir[..., 0].copy(), sdir[..., 2].copy()
        sdir[..., 0] = ca * x + sa * z
        sdir[..., 2] = -sa * x + ca * z

    col = sky(sdir)
    if doppler and v > 0.0:
        # frequency ratio falling-frame / infinity for infall from rest at
        # infinity: D = (1 + v*cos(psi_static)) / (1 - 2M/r0).
        # Ahead of us the sky runs blue and bright, behind it dims and reddens.
        f0 = 1.0 - 2.0 * M / r0
        D = np.clip((1.0 + v * cospsi) / f0, 0.25, 4.0).astype(np.float32)
        col = col * (D[..., None] ** 1.2)          # beaming
        col[..., 0] *= D ** -0.9                    # red channel down when blue
        col[..., 2] *= D ** 0.9                     # blue channel up
        col = col / (1.0 + 0.25 * col)              # soft tonemap
    col[psi < psi_t[0]] = 0.0

    if supersample > 1:
        col = col.reshape(height, supersample, width, supersample, 3)
        col = col.mean(axis=(1, 3))
    return (np.clip(col, 0, 1) * 255).astype(np.uint8)
