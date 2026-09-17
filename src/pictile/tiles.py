import cv2
import numpy as np
from pathlib import Path


def make_tiles(
        aligned: list[tuple[str, cv2.typing.MatLike | None]],
        n: int,
        m: int,
        ) -> list[tuple[str, list[list[cv2.typing.MatLike]]]]:
    """
    Split each image into n rows and m columns of equal-sized tiles.

    Each entry is a (filename, grid) pair, with tiles accessed as 
    grid[row][column].

    All images must have the same height and width. Remainder pixels on
    the bottom and right edges are discarded without resizing. 
    """

    if (not isinstance(n, int) or n <= 0 
        or not isinstance(m, int) or m <= 0):
        raise ValueError("n and m must be positive integers")

    images = [(name, image) for name, image in aligned if image is not None]
    if not images:
        raise ValueError("No aligned images found. Align using "
        ".align_sift() before making tiles.")

    height, width = images[0][1].shape[:2]
    if any(image.shape[:2] != (height, width) for _, image in images):
        raise ValueError("Images must have the same dimensions. Crop " \
        "aligned images using .crop_aligned() first.")

    tile_height, tile_width = height // n, width // m
    if tile_height == 0 or tile_width == 0:
        raise ValueError("The tile grid cannot have more rows or " \
        "columns than image pixels.")

    return [
        (name, [
            [
                image[
                    row * tile_height:(row + 1) * tile_height,
                    column * tile_width:(column + 1) * tile_width,
                ].copy()
                for column in range(m)
            ]
            for row in range(n)
        ])
        for name, image in images
    ]


def stitch_tiles(tiles: list[list[cv2.typing.MatLike]]
                 ) -> cv2.typing.MatLike:
    """
    Stitch a rectangular grid of tiles into a new image.

    Pass n rows of m tiles in their desired output positions.
    """
    if not tiles or not tiles[0]:
        raise ValueError(
            "The tile grid must contain at least one tile."
            )

    columns = len(tiles[0])
    if any(len(row) != columns for row in tiles):
        raise ValueError(
            "Every row must contain the same number of tiles."
            )

    first = tiles[0][0]
    for row in tiles:
        for tile in row:
            if not isinstance(tile, np.ndarray) or tile.shape != first.shape:
                raise ValueError("All tiles must have the same shape.")
            if tile.dtype != first.dtype:
                raise ValueError("All tiles must have the same dtype.")

    return np.concatenate(
        [np.concatenate(row, axis=1) for row in tiles], axis=0,
    )


def export_png(image: cv2.typing.MatLike, 
               output_path: str | Path) -> Path:
    """
    Save a stitched image as a single PNG and return its output path.

    Creates parent directories if needed. Existing files at
    the output path are overwritten.
    """

    output_path = Path(output_path)

    if not output_path.suffix:
        output_path = output_path.with_suffix(".png")
    elif output_path.suffix.lower() != ".png":
        raise ValueError("The output path must have a .png extension.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write PNG image to {output_path}")
    
    return output_path
