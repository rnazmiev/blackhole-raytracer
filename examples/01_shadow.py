"""Render the classic view: a black hole shadow in front of a star field.

Usage:  python examples/01_shadow.py     ->  shadow.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from PIL import Image

from bh.render import render_frame
from bh.sky import StarSky

img = render_frame(r0=30.0, sky=StarSky(), width=1280, height=720,
                   fov_deg=45)
Image.fromarray(img).save("shadow.png")
print("saved shadow.png")
