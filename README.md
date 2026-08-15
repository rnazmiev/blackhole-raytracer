# What You Actually See Falling Into a Black Hole — the code

The complete physics behind the video: a Schwarzschild ray tracer in ~360
lines of NumPy. Every frame in the video was produced by this code — nothing
was drawn by hand.

One equation does all the work — the shape of a light ray in Schwarzschild
spacetime (u = 1/r):

```
d²u/dφ² = 3Mu² − u
```

## What's inside

| file | what it does | lines |
|---|---|---|
| `bh/geodesic.py` | null geodesic integrator (RK4 over the Binet equation) | ~140 |
| `bh/render.py` | backward ray tracer for a static camera: one 1D deflection table per radius, every pixel mapped through it | ~80 |
| `bh/rain.py` | the same, for a free-falling camera — valid on **both** sides of the horizon (Painlevé–Gullstrand rain frame), Doppler included | ~140 |
| `bh/disk.py` | thin accretion disk: closed-form plane crossings from one 2D table, Doppler beaming + gravitational redshift | ~150 |
| `bh/sky.py` | celestial backgrounds: procedural star field, lat/long grid, equirectangular panoramas | ~90 |

## Quick start

```
pip install numpy scipy pillow
python examples/01_shadow.py    # the shadow, 2.6x wider than the horizon
python examples/02_inside.py    # the view from INSIDE the horizon
python examples/03_disk.py      # the accretion disk with its lensed arcs
```

Units: G = c = M = 1. The horizon is at r = 2, the photon sphere at r = 3,
the shadow edge at impact parameter b = 3√3 ≈ 5.196.

## Notes

- The star map in the video uses the ESO Milky Way panorama
  (ESO/S. Brunier, CC BY 4.0): https://www.eso.org/public/images/eso0932a/ —
  drop it into `assets/` and use `bh.sky.ImageSky` to reproduce those shots.
- The interior view answers the classic questions honestly: inside the
  horizon the hole's darkness fills only ~19% of the sky at r = 1M, you can
  still see your own feet, and near the centre the darkness flattens into a
  black "floor" below you.

## License

MIT. If this helped you, a link back to the video is appreciated.
