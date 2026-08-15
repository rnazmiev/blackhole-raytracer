"""Thin accretion disk, rendered through the same geodesic machinery.

A pixel's ray lives in the plane spanned by the camera axis (z) and the
pixel's perpendicular direction p. A point at in-plane angle phi along the
trajectory sits at  X(phi) = cos(phi)*z + sin(phi)*p  (times r). The disk
plane has normal n; the ray crosses it where X(phi).n = 0, i.e. at
phi* + k*pi with a closed-form phi* per pixel. So one 2D table r(psi, phi)
turns disk rendering into lookups — no per-pixel integration.

Doppler boost + gravitational redshift give the classic bright/dim asymmetry;
higher-k crossings automatically produce the lensed arch above and below.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from .geodesic import M, R_H, shadow_angle


def trajectory_table(r0, n_base=1400, n_crit=700, eps_min=1e-7,
                     dphi=3e-3, phi_max=25.0, store_every=10):
    """Integrate rays and store r on a regular phi grid.

    Returns (psi_sorted, phi_grid, r_grid) with r_grid[i, j] = r of ray
    psi[i] at phi_grid[j]; NaN once the ray is captured or has escaped.
    """
    psic = shadow_angle(r0)
    parts = [np.linspace(1e-4, np.pi - 0.031, n_base),
             psic + np.geomspace(eps_min, 0.4, n_crit)]
    psi = np.unique(np.concatenate(parts))
    psi = psi[(psi > 0) & (psi < np.pi - 0.03)]
    n = psi.size

    f0 = 1.0 - 2.0 * M / r0
    b = r0 * np.sin(psi) / np.sqrt(f0)
    u = np.full(n, 1.0 / r0)
    w2 = 1.0 / np.maximum(b, 1e-12) ** 2 - u ** 2 * (1.0 - 2.0 * M * u)
    w = np.sqrt(np.maximum(w2, 0.0)) * np.where(psi <= np.pi / 2, 1.0, -1.0)

    active = np.ones(n, bool)
    n_steps = int(phi_max / dphi)
    n_cols = n_steps // store_every + 1
    r_grid = np.full((n, n_cols), np.nan, dtype=np.float32)
    r_grid[:, 0] = r0
    phi_grid = np.arange(n_cols) * (dphi * store_every)

    u_cap = 1.0 / R_H
    u_esc = 1.0 / 500.0

    def g(uu):
        return 3.0 * M * uu * uu - uu

    for step in range(1, n_steps + 1):
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

        stop = (u[idx] >= u_cap) | ((u[idx] <= u_esc) & (w[idx] < 0)) | \
               (u[idx] <= 0)
        if stop.any():
            active[idx[stop]] = False
        if step % store_every == 0:
            col = step // store_every
            live = np.nonzero(active)[0]
            r_grid[live, col] = 1.0 / u[live]
    return psi, phi_grid, r_grid


def _bilinear_r(psi_t, phi_grid, r_grid, psi_px, phi_px):
    """Sample r(psi, phi); NaN where any supporting node is NaN/out of range."""
    dphi_g = phi_grid[1] - phi_grid[0]
    i = np.clip(np.searchsorted(psi_t, psi_px), 1, len(psi_t) - 1)
    t = ((psi_px - psi_t[i - 1]) / (psi_t[i] - psi_t[i - 1])).astype(np.float32)
    jf = phi_px / dphi_g
    j = np.clip(jf.astype(int), 0, len(phi_grid) - 2)
    s = (jf - j).astype(np.float32)
    r00 = r_grid[i - 1, j]
    r01 = r_grid[i - 1, j + 1]
    r10 = r_grid[i, j]
    r11 = r_grid[i, j + 1]
    return ((1 - t) * ((1 - s) * r00 + s * r01)
            + t * ((1 - s) * r10 + s * r11))


def disk_layer(psi_px, p_px, r0, incl_deg=80.0, r_in=6.0, r_out=15.0,
               traj=None, k_max=6, beta_cap=0.72, az_phase=0.0):
    """Emission of the disk along each pixel ray (first crossing wins).

    psi_px : (...,) static-frame view angles;  p_px : (..., 3) in-plane
    perpendicular units;  traj : output of trajectory_table(r0).
    Returns (rgb, hit_mask).
    """
    psi_t, phi_grid, r_grid = traj if traj is not None else trajectory_table(r0)
    i_rad = np.radians(incl_deg)
    nrm = np.array([0.0, np.sin(i_rad), np.cos(i_rad)], dtype=np.float32)
    zax = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    c1 = np.float32(nrm @ zax)                      # cos(incl)
    c2 = (p_px @ nrm).astype(np.float32)            # sin(incl) * p_y
    phi1 = np.mod(np.arctan2(-c1, c2), np.pi).astype(np.float32)
    phi1 = np.where(phi1 < 1e-4, phi1 + np.pi, phi1)

    shape = psi_px.shape
    hit = np.zeros(shape, bool)
    r_hit = np.zeros(shape, np.float32)
    phi_hit = np.zeros(shape, np.float32)
    for k in range(k_max):
        phik = phi1 + k * np.pi
        rk = _bilinear_r(psi_t, phi_grid, r_grid, psi_px, phik)
        ok = np.isfinite(rk) & (rk >= r_in) & (rk <= r_out) & ~hit
        r_hit = np.where(ok, rk, r_hit)
        phi_hit = np.where(ok, phik, phi_hit)
        hit |= ok

    rgb = np.zeros(shape + (3,), np.float32)
    if not hit.any():
        return rgb, hit

    hp = hit
    r = r_hit[hp]
    ph = phi_hit[hp]
    p = p_px[hp]

    # local photon direction (towards the camera) from the trajectory tangent
    dphi_g = phi_grid[1] - phi_grid[0]
    r_p = _bilinear_r(psi_t, phi_grid, r_grid, psi_px[hp],
                      np.minimum(ph + dphi_g, phi_grid[-1]))
    r_m = _bilinear_r(psi_t, phi_grid, r_grid, psi_px[hp],
                      np.maximum(ph - dphi_g, phi_grid[0]))
    drdphi = np.nan_to_num((r_p - r_m) / (2 * dphi_g))
    cph = np.cos(ph)[:, None]
    sph = np.sin(ph)[:, None]
    Xhat = cph * zax[None, :] + sph * p
    tang = drdphi[:, None] * Xhat + r[:, None] * (-sph * zax[None, :] + cph * p)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    k_ph = -tang                                     # propagation: disk -> camera

    # Keplerian speed measured by a local static observer, capped near ISCO
    beta = np.minimum(np.sqrt(M / np.maximum(r - 2 * M, 0.35)), beta_cap)
    vdir = np.cross(np.broadcast_to(nrm, Xhat.shape), Xhat)
    vdir /= np.linalg.norm(vdir, axis=1, keepdims=True)
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    doppler = 1.0 / (gamma * (1.0 - beta * np.einsum('ij,ij->i', vdir, k_ph)))
    gfac = doppler * np.sqrt(np.maximum(1.0 - 2.0 * M / r, 0.0))

    # mild procedural texture: radial rings + azimuthal streaks
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e2 = np.cross(nrm, e1)
    az = np.arctan2(Xhat @ e2, Xhat @ e1) - az_phase
    texture = (1.0
               + 0.22 * np.sin(4.1 * r + 1.7)
               + 0.13 * np.sin(3.0 * az + 2.1 * r))

    temp = np.clip(gfac * (r_in / r) ** 0.75, 0.0, 2.5)
    inten = 0.8 * (r_in / r) ** 2.1 * gfac ** 3.3 * texture
    # soften inner/outer cutoffs
    inten *= np.clip((r_out - r) / 1.6, 0, 1) * np.clip((r - r_in) / 0.7, 0, 1)

    R = np.clip(temp * 2.4, 0, 1.0)
    G = np.clip(temp * 1.25 - 0.18, 0, 1.0) ** 1.05
    B = np.clip(temp * 0.95 - 0.32, 0, 1.0) ** 1.2
    col = np.stack([R, G, B], axis=1) * inten[:, None]
    rgb[hp] = col
    return rgb, hit


def add_bloom(img, sigma=6.0, strength=0.5, threshold=0.55):
    """Cheap glow: blur the bright part and add it back."""
    bright = np.clip(img - threshold, 0, None)
    blur = gaussian_filter(bright, (sigma, sigma, 0))
    return img + strength * blur
