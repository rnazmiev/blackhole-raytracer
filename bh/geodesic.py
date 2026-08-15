"""Null geodesics in Schwarzschild spacetime, G = c = 1, M = 1.

Ray geometry lives in the plane spanned by the camera position and the launch
direction. The Binet equation d2u/dphi2 = 3*M*u^2 - u (u = 1/r) is integrated
with fixed-step RK4, phi increasing along the ray.

Conventions:
  * psi  — angle between the launch direction and the *inward* radial
           direction (towards the BH). psi = 0 aims at the centre,
           psi = pi aims straight away.
  * The camera sits at phi = 0. In the 2D ray plane the x-axis points from
    the BH to the camera, the y-axis is the perpendicular launch component.
  * A    — unwrapped asymptotic direction angle of the escaped ray in that
           plane (atan2 of the final velocity, continued past 2*pi so that
           windings are preserved). For an undeflected ray A = pi - psi.
"""

import numpy as np

M = 1.0
R_H = 2.0 * M            # horizon
R_PH = 3.0 * M           # photon sphere
B_CRIT = 3.0 * np.sqrt(3.0) * M   # critical impact parameter


def shadow_angle(r0):
    """Angular radius of the black-hole shadow for a static observer at r0.

    Outside the photon sphere the shadow subtends arcsin(b_c*sqrt(f)/r0);
    between the horizon and the photon sphere the escape cone shrinks instead,
    and the shadow takes up the complementary angle (more than a hemisphere).
    """
    arg = B_CRIT * np.sqrt(np.maximum(1.0 - 2.0 * M / r0, 0.0)) / r0
    a = np.arcsin(np.clip(arg, -1.0, 1.0))
    return np.where(r0 >= R_PH, a, np.pi - a) if np.ndim(r0) else \
        (a if r0 >= R_PH else np.pi - a)


def trace(psi, r0, dphi=3e-3, phi_max=50.0, r_esc=500.0, record=False):
    """Integrate a batch of rays launched from radius r0 at angles psi.

    Returns a dict with boolean masks 'escaped'/'captured', unwrapped final
    direction 'A', impact parameters 'b', and (optionally) sampled paths as
    arrays of (r, phi).
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    n = psi.size
    f0 = 1.0 - 2.0 * M / r0
    b = r0 * np.sin(psi) / np.sqrt(f0)

    u = np.full(n, 1.0 / r0)
    w2 = 1.0 / np.maximum(b, 1e-12) ** 2 - u ** 2 * (1.0 - 2.0 * M * u)
    w = np.sqrt(np.maximum(w2, 0.0)) * np.where(psi <= np.pi / 2, 1.0, -1.0)
    phi = np.zeros(n)

    active = np.ones(n, bool)
    captured = np.zeros(n, bool)
    escaped = np.zeros(n, bool)
    uf = np.zeros(n)
    wf = np.zeros(n)
    phif = np.zeros(n)
    paths = [[(r0, 0.0)] for _ in range(n)] if record else None

    u_cap = 1.0 / R_H
    u_esc = 1.0 / r_esc
    n_steps = int(phi_max / dphi)

    def g(uu):
        return 3.0 * M * uu * uu - uu

    for step in range(n_steps):
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

        cap = u[idx] >= u_cap
        esc = (u[idx] <= u_esc) & (w[idx] < 0.0)
        if cap.any():
            j = idx[cap]
            captured[j] = True
            active[j] = False
        if esc.any():
            j = idx[esc]
            escaped[j] = True
            uf[j] = u[j]
            wf[j] = w[j]
            phif[j] = phi[j]
            active[j] = False
        if record and step % 5 == 0:
            for i in idx:
                if 0.0 < u[i]:
                    paths[i].append((1.0 / u[i], phi[i]))

    captured[active] = True  # ran out of phi budget => hugging the critical orbit

    A = np.zeros(n)
    j = escaped
    if j.any():
        r = 1.0 / np.maximum(uf[j], 1e-12)
        drdphi = -wf[j] * r * r
        dx = drdphi * np.cos(phif[j]) - r * np.sin(phif[j])
        dy = drdphi * np.sin(phif[j]) + r * np.cos(phif[j])
        Aw = np.arctan2(dy, dx)
        A[j] = Aw + 2 * np.pi * np.round((phif[j] - Aw) / (2 * np.pi))

    out = {"escaped": escaped, "captured": captured, "A": A,
           "phi_end": phif, "b": b}
    if record:
        out["paths"] = [np.array(p) for p in paths]
    return out


def path_from_point(x0, y0, dphi=2e-3, phi_max=50.0, record_every=1):
    """Trace a single ray launched from (x0, y0) moving in the -x direction.

    Returns an (N, 2) array of cartesian points in the BH-centred frame, plus
    a flag whether the ray was captured.
    """
    r0 = float(np.hypot(x0, y0))
    phi0 = float(np.arctan2(y0, x0))
    # launch direction -x  =>  psi equals the position angle phi0
    res = trace(np.array([abs(phi0)]), r0, dphi=dphi, phi_max=phi_max,
                record=True)
    path = res["paths"][0]
    sign = 1.0 if phi0 >= 0 else -1.0
    ang = sign * path[:, 1] + phi0
    xs = path[:, 0] * np.cos(ang)
    ys = path[:, 0] * np.sin(ang)
    pts = np.stack([xs, ys], axis=1)
    return pts[::record_every], bool(res["captured"][0])


def deflection_angle(b, r0=400.0):
    """Total bending angle alpha of a ray with impact parameter b (from far away).

    alpha = 0 for a straight ray; alpha -> inf as b -> B_CRIT from above.
    Returns np.inf for captured rays.
    """
    f0 = 1.0 - 2.0 * M / r0
    psi = float(np.arcsin(np.clip(b * np.sqrt(f0) / r0, -1.0, 1.0)))
    res = trace(np.array([psi]), r0, dphi=2e-3, phi_max=50.0)
    if not res["escaped"][0]:
        return np.inf
    return float(res["A"][0] - (np.pi - psi))


def find_b_for_deflection(alpha_target, lo=None, hi=None, iters=40):
    """Impact parameter whose bending angle equals alpha_target (bisection)."""
    lo = B_CRIT + 1e-6 if lo is None else lo
    hi = B_CRIT + 3.0 if hi is None else hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        a = deflection_angle(mid)
        if a > alpha_target:   # too much bending -> move away from critical
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
