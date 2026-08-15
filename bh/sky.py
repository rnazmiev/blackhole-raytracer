"""Celestial-sphere backgrounds: procedural starfield and a lat/long grid."""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def make_starmap(H=1024, W=2048, seed=3):
    """Equirectangular starfield texture, float RGB in [0, 1]."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), dtype=np.float32)
    layers = [  # (sigma, count, brightness)
        (0.7, 9000, 1.0),
        (1.1, 2200, 2.8),
        (1.9, 260, 7.0),
    ]
    for sigma, count, bright in layers:
        phi = rng.uniform(0, 2 * np.pi, count)
        z = rng.uniform(-1, 1, count)
        theta = np.arccos(z)
        rows = (theta / np.pi * (H - 1)).astype(int)
        cols = (phi / (2 * np.pi) * (W - 1)).astype(int)
        temp = rng.uniform(0, 1, count).astype(np.float32)
        colors = np.stack([0.65 + 0.35 * temp,
                           0.75 + 0.20 * temp * (1 - temp) * 4 * 0.5 + 0.05,
                           1.00 - 0.35 * temp], axis=1)
        amp = bright * (rng.uniform(0.05, 1.0, count) ** 3).astype(np.float32)
        layer = np.zeros_like(img)
        np.add.at(layer, (rows, cols), colors * amp[:, None])
        img += gaussian_filter(layer, (sigma, sigma, 0))
    # faint blue ambient so 'space' is not pure black
    img += np.array([0.004, 0.005, 0.010], dtype=np.float32)
    img /= np.percentile(img, 99.98) * 0.9
    return np.clip(img, 0, 1)


class StarSky:
    def __init__(self, H=1024, W=2048, seed=3):
        tex = make_starmap(H, W, seed)
        # pad one wrapped column so bilinear sampling never crosses the seam
        self.tex = np.concatenate([tex, tex[:, :1]], axis=1)
        self.H, self.W = H, W

    def __call__(self, s):
        """s: (..., 3) unit vectors -> (..., 3) RGB."""
        theta = np.arccos(np.clip(s[..., 2], -1.0, 1.0))
        phi = np.mod(np.arctan2(s[..., 1], s[..., 0]), 2 * np.pi)
        r = theta / np.pi * (self.H - 1)
        c = phi / (2 * np.pi) * self.W
        coords = np.stack([r.ravel(), c.ravel()])
        out = np.empty(s.shape[:-1] + (3,), dtype=np.float32)
        for ch in range(3):
            out[..., ch] = map_coordinates(
                self.tex[..., ch], coords, order=1, mode='nearest'
            ).reshape(s.shape[:-1])
        return out


class ImageSky:
    """Sky from an equirectangular panorama (e.g. the ESO Milky Way image)."""

    def __init__(self, path, gain=1.0, gamma=1.0):
        from PIL import Image
        im = Image.open(path).convert("RGB")
        tex = np.asarray(im, dtype=np.float32) / 255.0
        if gamma != 1.0:
            tex = tex ** gamma
        tex *= gain
        self.tex = np.concatenate([tex, tex[:, :1]], axis=1)
        self.H, self.W = tex.shape[0], tex.shape[1]

    def __call__(self, s):
        theta = np.arccos(np.clip(s[..., 2], -1.0, 1.0))
        phi = np.mod(np.arctan2(s[..., 1], s[..., 0]), 2 * np.pi)
        r = theta / np.pi * (self.H - 1)
        c = phi / (2 * np.pi) * self.W
        coords = np.stack([r.ravel(), c.ravel()])
        out = np.empty(s.shape[:-1] + (3,), dtype=np.float32)
        for ch in range(3):
            out[..., ch] = map_coordinates(
                self.tex[..., ch], coords, order=1, mode='nearest'
            ).reshape(s.shape[:-1])
        return out


class GridSky:
    """Lat/long grid of the celestial sphere — the 'bending graph paper' look."""

    def __init__(self, spacing_deg=10.0, width_deg=0.30,
                 bg=(0.010, 0.016, 0.045), line=(0.25, 0.75, 1.0)):
        self.sp = spacing_deg
        self.w = width_deg
        self.bg = np.array(bg, dtype=np.float32)
        self.line = np.array(line, dtype=np.float32)

    def __call__(self, s):
        lat = np.degrees(np.arcsin(np.clip(s[..., 2], -1.0, 1.0)))
        lon = np.degrees(np.arctan2(s[..., 1], s[..., 0]))
        d_lat = np.abs(np.mod(lat + self.sp / 2, self.sp) - self.sp / 2)
        d_lon = np.abs(np.mod(lon + self.sp / 2, self.sp) - self.sp / 2)
        d_lon = d_lon * np.maximum(np.cos(np.radians(lat)), 0.05)
        i_lat = np.exp(-(d_lat / self.w) ** 2)
        i_lon = np.exp(-(d_lon / self.w) ** 2)
        inten = np.maximum(i_lat, i_lon).astype(np.float32)
        # emphasise the equator so orientation is readable
        eq = np.exp(-(np.abs(lat) / (2.5 * self.w)) ** 2).astype(np.float32)
        col = (self.bg[None] + inten[..., None] * self.line[None]
               + eq[..., None] * np.array([1.0, 0.55, 0.2], dtype=np.float32) * 0.9)
        return np.clip(col, 0, 1)
