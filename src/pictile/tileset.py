import cv2
import numpy as np
from pathlib import Path
from .align import load_images, align_sift, crop_aligned, sift_features
from .tiles import make_tiles

class Tileset:
    """
    Load a folder of images, align them to a reference image, and
    export aligned images.

    ".images" and ".aligned" contain (filename, BGR image) pairs.
    """

    def __init__(self, folder: str | Path, reference: int | str = 0):
        # Load original images
        self.images = load_images(folder)
        if not self.images:
            raise ValueError(f"No readable images found in {folder}")
        
        # Set reference name and image
        if isinstance(reference, int):
            self.reference_name, self.reference = self.images[reference]
        elif isinstance(reference, str):
            self.reference_name = reference
            for filename, image in self.images:
                if filename == reference:
                    self.reference = image
                    break
            else:
                raise ValueError(f"Reference image not found: {reference}")

        self.aligned: list[tuple[str, cv2.typing.MatLike | None]] = []
        self.diagnostics: dict[str, dict] = {}
        self.tiles: list[tuple[str, list[list[cv2.typing.MatLike]]]] = []
        self.tile_dimensions: tuple[int, int] = (-1, -1)


    def align_sift(self, verbose=False):
        """
        Align all images to the reference and store in ".aligned".

        Failed alignments are omitted from ".aligned" and recorded in 
        ".diagnostics".
        """

        self.aligned = [(self.reference_name, self.reference)]
        self.tiles = []
        self.diagnostics = {}
        reference_features = sift_features(self.reference)
        for name, image in self.images:
            if name == self.reference_name:
                continue
            if verbose:
                print(f"Aligning {name}...")

            aligned, info = align_sift(
                self.reference, image, reference_features=reference_features
            )
            self.diagnostics[name] = info
            if info["success"]:
                self.aligned.append((name, aligned))

        if verbose:
            print("Alignment Diagnostics:")
            print(self.diagnostics)


    def crop_aligned(self):
        """
        Crop aligned images to their maximum shared extent.
        """
        if self.aligned:
            self.aligned = crop_aligned(self.aligned)
            self.tiles = []
        else:
            ValueError(
                "No aligned images found. Images must be aligned " \
                "before cropping."
                )


    def export_aligned(self, output_dir: str | Path):
        """
        Write all aligned images as PNG to output directory.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        for name, image in self.aligned:
            cv2.imwrite(str(output_dir / name), image) 


    def make_tiles(self, n: int, m: int):
        """
        Store (n x m) tiles per aligned image in ".tiles".

        Call after align_sift() and crop_aligned(). Each entry is a
        (filename, grid) pair, with tiles accessed as grid[row][column].
        """
        self.tile_dimensions = (n,m)
        self.tiles = make_tiles(self.aligned, n, m)
        

    
