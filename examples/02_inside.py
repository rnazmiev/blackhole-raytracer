"""What a free-falling observer sees INSIDE the horizon (r = 1M): the
darkness is a bounded patch ahead, stars fill the rest of the sky.

Usage:  python examples/02_inside.py     ->  inside.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from PIL import Image

from bh.rain import render_rain_frame
from bh.sky import StarSky

img, black_frac = render_rain_frame(r0=1.0, sky=StarSky(), width=1280,
                                    height=720, fov_deg=120,
                                    look_offset=0.9)
Image.fromarray(img).save("inside.png")
cap = np.degrees(np.arccos(1 - 2 * black_frac))
print(f"saved inside.png  (the hole fills only part of the sky)")
