"""The rain frame: a radially free-falling observer (from rest at infinity),
valid on BOTH sides of the horizon (Painlevé–Gullstrand viewpoint).

For view angle psi (measured from the inward/motion direction) the arriving
photon, traced into the past, is an orbit with

    u0 = 1/r0,   du/dphi|0 = (cos psi - v) / (r0 sin psi),   v = sqrt(2M/r0)

and conserved energy-at-infinity  E = 1 - v cos psi  (per unit local
frequency). Pixels are black when the past of the ray never reaches the
outside sky: either E <= 0 (the ray's past hugs the horizon — for a real,
collapsed-star black hole that light does not exist), or the traced orbit
falls back through the horizon. Outside the horizon this construction is
algebraically identical to the static-frame table + special-relativistic
aberration used so far.

The observed frequency ratio (Doppler + gravity combined) is simply
    D = 1 / (1 - v cos psi) = 1/E,
which reduces to the exterior formula (1 + v cos psi_static)/f.
"""

import numpy as np

from .geodesic import M, R_H, B_CRIT

U_ESC = 1.0 / 500.0


def rain_speed(r0):
    return np.sqrt(2.0 * M / r0)


def rain_shadow_angle(r0):
    """Angular radius (from the inward direction) of the black region."""
    v = rain_speed(r0)
    lo = np.arccos(min(1.0, 1.0 / v)) + 1e-9 if v > 1.0 else 1e-6
    ps = np.linspace(lo + 1e-6, np.pi - 1e-4, 4000)
    bs = ps * 0.0
    denom = 1.0 - v * np.cos(ps)
    good = denom > 1e-12
    bs[good] = r0 * np.sin(ps[good]) / denom[good]
    bs[~good] = np.inf
    above = bs > B_CRIT
    idx = np.nonzero(above[:-1] & ~above[1:])[0]
    return float(ps[idx[0]]) if idx.size else float(lo)


def rain_table(r0, n_base=1600, n_crit=900, eps_min=1e-8,
               dphi=2.5e-3, phi_max=50.0):
    """Sorted (psi, A) samples of escaped rays for a rain observer at r0,
    plus the shadow edge angle. psi is measured from the inward direction."""
    v = rain_speed(r0)
    psic = rain_shadow_angle(r0)

    parts = [np.linspace(1e-4, np.pi - 0.031, n_base),
             psic + np.geomspace(eps_min, 0.4, n_crit)]
    psi = np.unique(np.concatenate(parts))
    psi = psi[(psi > 0) & (psi < np.pi - 0.03)]
    n = psi.size

    E = 1.0 - v * np.cos(psi)
    u = np.full(n, 1.0 / r0)
    w = (np.cos(psi) - v) / (r0 * np.sin(psi))
    phi = np.zeros(n)

    active = E > 1e-12          # E <= 0 can never have come from the sky
    escaped = np.zeros(n, bool)
    uf = np.zeros(n)
    wf = np.zeros(n)
    phif = np.zeros(n)
    u_h = 1.0 / R_H

    def g(uu):
        return 3.0 * M * uu * uu - uu

    for _ in range(int(phi_max / dphi)):
        if not active.any():
            break
        idx = np.nonzero(active)[0]
        uu = u[idx]
        ww = w[idx]
        k1u = ww
        k1w = g(uu)
        k2u = ww + 0.5 * dphi * k1w
        k2w = g(uu + 0.5 * dphi * k1u)
        k3u = ww + 0.5 * dphi * k2w
        k3w = g(uu + 0.5 * dphi * k2u)
        k4u = ww + dphi * k3w
        k4w = g(uu + dphi * k3u)
        u[idx] = uu + dphi / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
        w[idx] = ww + dphi / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
        phi[idx] += dphi

        # dead: heading inward at or inside the horizon — its past (or the
        # forward-traced exterior ray) never reaches the outside sky
        dead = (w[idx] > 0) & (u[idx] >= u_h)
        esc = (u[idx] <= U_ESC) & (w[idx] < 0)
        if dead.any():
            active[idx[dead]] = False
        if esc.any():
            j = idx[esc]
            escaped[j] = True
            uf[j] = u[j]
            wf[j] = w[j]
            phif[j] = phi[j]
            active[j] = False

    A = np.zeros(n)
    jj = escaped
    if jj.any():
        r = 1.0 / np.maximum(uf[jj], 1e-12)
        drdphi = -wf[jj] * r * r
        dx = drdphi * np.cos(phif[jj]) - r * np.sin(phif[jj])
        dy = drdphi * np.sin(phif[jj]) + r * np.cos(phif[jj])
        Aw = np.arctan2(dy, dx)
        A[jj] = Aw + 2 * np.pi * np.round((phif[jj] - Aw) / (2 * np.pi))

    tail_psi = np.linspace(np.pi - 0.03, np.pi, 32)
    tail_A = np.pi - tail_psi
    psi_e = np.concatenate([psi[jj], tail_psi])
    A_e = np.concatenate([A[jj], tail_A])
    o = np.argsort(psi_e)
    return psi_e[o], A_e[o], psic


def render_rain_frame(r0, sky, width=1280, height=720, fov_deg=70.0,
                      look_offset=0.0, supersample=2, table=None,
                      doppler=True):
    """Frame seen by the rain observer at r0 (any r0 > 0), looking towards
    the hole (or tilted by look_offset). Returns (uint8 image, black_frac)."""
    psi_t, A_t, psic = table if table is not None else rain_table(r0)
    v = rain_speed(r0)

    W, H = width * supersample, height * supersample
    a = np.array([0.0, 0.0, -1.0], dtype=np.float32)
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

    cospsi = np.clip(d @ a, -1.0, 1.0)      # psi from the inward direction
    sinpsi = np.sqrt(np.maximum(1.0 - cospsi ** 2, 1e-18))
    p = (d - cospsi[..., None] * a) / sinpsi[..., None]
    psi = np.arccos(cospsi)

    A = np.interp(psi, psi_t, A_t).astype(np.float32)
    er = -a
    sdir = np.cos(A)[..., None] * er + np.sin(A)[..., None] * p
    col = sky(sdir)

    if doppler:
        D = np.clip(1.0 / np.maximum(1.0 - v * cospsi, 1e-3),
                    0.05, 6.0).astype(np.float32)
        col = col * (D[..., None] ** 1.2)
        col[..., 0] *= D ** -0.9
        col[..., 2] *= D ** 0.9
        col = col / (1.0 + 0.25 * col)

    shadow = psi < psi_t[0]
    col[shadow] = 0.0
    black_frac = float(shadow.mean())

    if supersample > 1:
        col = col.reshape(height, supersample, width, supersample, 3)
        col = col.mean(axis=(1, 3))
    return (np.clip(col, 0, 1) * 255).astype(np.uint8), black_frac
