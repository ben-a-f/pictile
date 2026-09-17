# pictile

Align photos, split them into tiles, and create tile-swap animations with Python and OpenCV.

Install with `pip install pictile`.

Place JPG or PNG photos of the same scene in an `images/` folder, then:

```python
from pictile import Tileset
from pictile.animate import make_random_mp4

tileset = Tileset("images", reference=0)
tileset.align_sift()
tileset.crop_aligned()
tileset.make_tiles(3, 3)  # rows, columns

make_random_mp4(tileset, "animation.mp4")
```

The reference defaults to the first image sorted by filename. You can also select it by filename. Images that cannot be aligned are skipped; details are available in `tileset.diagnostics`. Animations need at least two successfully aligned images, including the reference.
