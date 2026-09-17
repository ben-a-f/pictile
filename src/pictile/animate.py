from pathlib import Path
import math
import cv2
import numpy as np
import random
from typing import Literal
import warnings
from .tileset import Tileset
from .tiles import stitch_tiles


def enforce_size(
        images: list[cv2.typing.MatLike],
        vertical: Literal["top", "bottom", "split"] = "split",
        horizontal: Literal["left", "right", "split"] = "split",
        ) -> list[cv2.typing.MatLike]:
    """
    Crop images to the smallest height and width in the input list.

    vertical and horizontal specify the edges to REMOVE pixels from.
    "split" removes pixels evenly from opposing edges; an odd extra
    pixel is removed from the bottom or right. 
    """

    if vertical not in ("top", "bottom", "split"):
        raise ValueError("vertical must be 'top', 'bottom', or 'split'.")
    if horizontal not in ("left", "right", "split"):
        raise ValueError("horizontal must be 'left', 'right', or 'split'.")
    if not images:
        return []
    for image in images:
        if (not isinstance(image, np.ndarray) or image.ndim not in (2, 3)
                or image.size == 0):
            raise ValueError("Images must be nonempty grayscale or colour arrays.")

    # Get minimum dimensions
    height = min(image.shape[0] for image in images)
    width = min(image.shape[1] for image in images)
    cropped = []

    # Crop each image from specified edges
    for image in images:
        extra_height = image.shape[0] - height
        extra_width = image.shape[1] - width
        top = (extra_height if vertical == "top" else
               extra_height // 2 if vertical == "split" else 0)
        left = (extra_width if horizontal == "left" else
                extra_width // 2 if horizontal == "split" else 0)
        cropped.append(image[top:top + height, left:left + width].copy())

    return cropped


def export_video(
        images: list[cv2.typing.MatLike],
        durations: list[float],
        output_path: str | Path,
        fps: float = 30.0,
        ) -> Path:
    """
    Write images in order to an MP4, displaying each for its duration
    in seconds (minimum 1 frame per image).

    Images with different dimensions are automatically cropped to the
    smallest height and width.

    Creates parent directories if needed. Existing files at the output 
    path are overwritten.
    """
    # Validate duration calculations
    if not images or len(images) != len(durations):
        raise ValueError("Provide one duration for each image, and at " \
        "least one image.")
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite.")

    # Calculate number of frames per image
    frame_counts = []
    for duration in durations:
        seconds = float(duration)
        frames = seconds * fps
        if seconds <= 0 or not math.isfinite(frames):
            raise ValueError("Durations must be positive and finite.")
        frame_counts.append(max(1, math.floor(frames + 0.5)))

    # Validate image format
    for image in images:
        if (not isinstance(image, np.ndarray) or image.size == 0
                or image.ndim not in (2, 3)
                or (image.ndim == 3 and image.shape[2] not in (1, 3, 4))
                or image.dtype != np.uint8):
            raise ValueError("Images must be nonempty uint8 grayscale, " \
            "BGR, or BGRA arrays.")

    # Crop images to smallest shared size if any differ
    if any(image.shape[:2] != images[0].shape[:2] for image in images):
        warnings.warn(
            "Supplied images are of different sizes. All images will be "
            "automatically cropped to the same size.",
            stacklevel=2,
        )
        images = enforce_size(images)

    # Validate output path
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".mp4")
    elif output_path.suffix.lower() != ".mp4":
        raise ValueError("The output path must have a .mp4 extension.")

    # Write frames to MP4
    height, width = images[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (width + width % 2, height + height % 2),
    )
    try:
        if not writer.isOpened():
            raise OSError(f"Could not open video writer for {output_path}")
        for image, count in zip(images, frame_counts):
            if image.ndim == 2 or image.shape[2] == 1:
                frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                frame = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                frame = image
            if height % 2 or width % 2:
                frame = cv2.copyMakeBorder(
                    frame, 0, height % 2, 0, width % 2,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0),
                )
            for _ in range(count):
                writer.write(frame)
    finally:
        writer.release()

    return output_path


def make_window_mp4(
        tileset: Tileset,
        output_path: str | Path,
        max_duration: float = 0.5,
        min_duration: float = 0.1,
        reverse: bool = False,
        fps: float = 30.0,
        ) -> tuple[list[cv2.typing.MatLike], list[float]]:
    """
    Export a video that replaces the centre tile of the tileset 
    reference image with the centre tile of each other image at
    increasing speed.

    Requires tiles to have been created with Tileset.make_tiles().

    max_duration sets the initial image duration and both reference holds
    in seconds. Each successive image lasts min_duration seconds less,
    down to a minimum of min_duration (0 < min_duration <= max_duration).
    reverse=True mirrors the timings to slow down symmetrically in the
    second half, without reversing image order.
    """

    # Validate input parameters
    max_duration = float(max_duration)
    min_duration = float(min_duration)
    if (not math.isfinite(max_duration) or not math.isfinite(min_duration)
            or not 0 < min_duration <= max_duration):
        raise ValueError("Durations must be finite and satisfy "
                         "0 < min_duration <= max_duration.")
    
    # Check tiles exist
    if not tileset.tiles:
        raise ValueError("Tileset.tiles is empty. Call " \
        "Tileset.make_tiles(n, m) first.")

    # Get reference image
    reference_grid = next((grid for name, grid in tileset.tiles
                           if name == tileset.reference_name), None)
    if reference_grid is None:
        raise ValueError("The Tileset must contain reference tiles.")
    reference = stitch_tiles(reference_grid)
    rows, columns = len(reference_grid), len(reference_grid[0])

    # Get all other images as tile donors
    others = [grid for name, grid in tileset.tiles
              if name != tileset.reference_name]
    if not others:
        raise ValueError("At least one other tile grid is required.")
    if any(len(grid) != rows or any(len(row) != columns for row in grid)
           for grid in others):
        raise ValueError("Tile grids must have matching row and " \
        "column counts.")

    # Stitch donor centre tiles onto reference image
    centre_row, centre_column = rows // 2, columns // 2
    images = [reference]
    for grid in others:
        composite = [row.copy() for row in reference_grid]
        composite[centre_row][centre_column] = grid[centre_row][centre_column]
        images.append(stitch_tiles(composite))
    images.append(reference)

    # Symmetric timing indices if reverse, else linear
    count = len(others)
    indices = [min(i, count - 1 - i) if reverse else i
               for i in range(count)]
    # Reference image bookends the video
    durations = [max_duration]
    durations.extend(max(max_duration - min_duration * i, min_duration)
                     for i in indices)
    durations.append(max_duration)

    export_video(images, durations, output_path, fps=fps)

    return images, durations

def make_random_mp4(
        tileset: Tileset,
        output_path: str | Path,
        n: int = 1,
        num_draws: int = 24,
        duration: float = 0.25,
        persist: int = 0,
        fps: float = 30.0,
        ) -> tuple[list[cv2.typing.MatLike], list[float]]:
    """
    Export a video of random tile replacements, starting with the
    reference image.

    Requires tiles to have been created with Tileset.make_tiles().

    n is the number of tiles replaced in each draw.
    num_draws is the number of draws (excluding the initial reference 
    image).
    duration is the number of seconds each draw is shown for.
    persist controls the number of draws each tile is preserved for 
    after being replaced.
    """
    # Validate input parameters
    for name, value, minimum in (("n", n, 1), ("num_draws", num_draws, 0),
                                  ("persist", persist, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite.")
    
    # Check tiles exist
    if not tileset.tiles:
        raise ValueError("Tileset.tiles is empty. Call "
                         "Tileset.make_tiles(n, m) first.")

    # Get reference image
    reference_grid = next((grid for name, grid in tileset.tiles
                           if name == tileset.reference_name), None)
    if reference_grid is None:
        raise ValueError("The Tileset must contain reference tiles.")

    # Get all other images as tile donors
    reference = stitch_tiles(reference_grid)
    rows, columns = len(reference_grid), len(reference_grid[0])
    others = [grid for name, grid in tileset.tiles
              if name != tileset.reference_name]
    if num_draws and not others:
        raise ValueError("At least one other tile grid is required.")
    for grid in others:
        if len(grid) != rows or any(len(row) != columns for row in grid):
            raise ValueError("Tile grids must have matching row and column counts.")
        for row in grid:
            for tile in row:
                if (not isinstance(tile, np.ndarray)
                        or tile.shape != reference_grid[0][0].shape
                        or tile.dtype != reference_grid[0][0].dtype):
                    raise ValueError("All tiles must have matching shapes and dtypes.")

    # Verify the specified parameters are possible with given tiles
    if n * min(num_draws, persist + 1) > rows * columns:
        raise ValueError("Too few tile positions for n replacements per draw "
                         "with this persist and num_draws.")

    # Iteratively draw tiles to replace and stitch them
    positions = [(row, column) for row in range(rows)
                 for column in range(columns)]
    next_available = dict.fromkeys(positions, 0)
    composite = [row.copy() for row in reference_grid]
    images = [reference]
    for draw in range(num_draws):
        eligible = [position for position in positions
                    if next_available[position] <= draw]
        for row, column in random.sample(eligible, n):
            donor = random.choice(others)
            composite[row][column] = donor[row][column]
            next_available[row, column] = draw + persist + 1
        images.append(stitch_tiles(composite))

    durations = [duration] * len(images)
    export_video(images, durations, output_path, fps=fps)

    return images, durations