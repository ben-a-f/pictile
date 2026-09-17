from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from pictile import Tileset
from pictile.align import crop_aligned
from pictile.animate import export_video, make_random_mp4, make_window_mp4
from pictile.tiles import export_png, make_tiles, stitch_tiles


class CoreTests(unittest.TestCase):
    """
    Methods beginning with "test_" are automatically treated as tests.
    """
    def setUp(self):
        """
        Override inherited setUp method.
        Runs before every test to create a new temp directory to hold
        image files etc. during testing.
        """

        temporary = TemporaryDirectory()
        # Remove temp directory after test, even if test fails
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)


    def write_image(self, name, image):
        # Helper: Write an image, and check the write succeeds
        
        self.assertTrue(cv2.imwrite(str(self.folder / name), image))


    def tiled_pair(self):
        # Helper: Create two same-sized images, treat them as aligned,
        # and make tiles. These solid colour images have no SIFT 
        # features.

        for name, value in [("a.png", 40), ("b.png", 200)]:
            # numpy arrays represent pixel data
            # (height, width, channels)
            # value fills all channels, setting the colour
            self.write_image(name, 
                             np.full((12, 18, 3), value, np.uint8))
        tileset = Tileset(self.folder)

        # Treat images as already aligned
        tileset.aligned = [(name, image) for name, image in tileset.images]
        tileset.make_tiles(3, 3)
        
        return tileset


    def test_loading_and_reference_selection(self):

        # Create two image files
        image = np.full((8, 10, 3), (10, 20, 30), np.uint8)
        self.write_image("b.JPG", image)
        self.write_image("a.png", image)
        # Create two non-image files
        (self.folder / "broken.png").write_bytes(b"not an image")
        (self.folder / "notes.txt").write_text("ignored")

        tileset = Tileset(self.folder)
        # Check only image files are loaded into Tileset
        self.assertEqual([name for name, _ in tileset.images], 
                         ["a.png", "b.JPG"])
        # The first image sorted by filename is the default reference
        self.assertEqual(tileset.reference_name, "a.png")
        assert_array_equal(tileset.reference, image)
        # Test specifying a different reference
        self.assertEqual(
            Tileset(self.folder, reference=1).reference_name, 
            "b.JPG"
            )
        self.assertEqual(
            Tileset(self.folder, reference="b.JPG").reference_name, 
            "b.JPG"
            )
        # Check error is raised when an unknown reference is specified
        with self.assertRaises(ValueError):
            Tileset(self.folder, reference="missing.png")


    def test_empty_folder_is_rejected(self):
        with self.assertRaises(ValueError):
            Tileset(self.folder)


    def test_align_crop_tile_and_export_workflow(self):

        # Use random seed to get a repeatable image for SIFT 
        # to match
        # Values 1,..,255 avoids black pixels which could be affected 
        # by cropping
        reference = np.random.default_rng(42).integers(
            1, 256, size=(192, 192, 3), dtype=np.uint8
        )
        # Mimic a new image of the same object by translating 8px right
        # and 6px down, maintaining width=height=192
        # Exposed top/left edges become black; right/bottom pixels are
        # lost out of frame
        moving = cv2.warpAffine(
            reference, np.float32([[1, 0, 8], [0, 1, 6]]), (192, 192)
        )
        self.write_image("a.png", reference)
        self.write_image("b.png", moving)
        # Create a blank image
        self.write_image("blank.png", np.zeros_like(reference))

        # Test SIFT alignment
        # Translated image "b" should align
        # Blank image "blank" should fail with a recorded reason
        tileset = Tileset(self.folder)
        tileset.align_sift()
        self.assertEqual([name for name, _ in tileset.aligned], 
                         ["a.png", "b.png"])
        self.assertTrue(tileset.diagnostics["b.png"]["success"])
        self.assertFalse(tileset.diagnostics["blank.png"]["success"])
        self.assertTrue(tileset.diagnostics["blank.png"]["reason"])

        # Estimated homography should undo the translation, allowing
        # small tolerance for estimation/numerical error
        assert_allclose(
            tileset.diagnostics["b.png"]["homography"],
            [[1, 0, -8], [0, 1, -6], [0, 0, 1]], atol=0.15,
        )

        # Crop should leave images equally sized and nonempty
        tileset.crop_aligned()
        first, second = [image for _, image in tileset.aligned]
        self.assertEqual(first.shape, second.shape)
        self.assertGreater(first.size, 0)
        # CHeck two images are near-identical within small tolerance
        self.assertLess(float(np.abs(first.astype(float)-second).mean()),
                        5)
        
        # Check make_tiles() makes tiles for each successful image
        tileset.make_tiles(3, 3)
        self.assertEqual(tileset.tile_dimensions, (3, 3))
        self.assertEqual(len(tileset.tiles), 2)

        # Check export preserves images exactly
        output = self.folder / "aligned"
        tileset.export_aligned(output)
        for name, image in tileset.aligned:
            assert_array_equal(cv2.imread(str(output / name)), image)


    def test_crop_removes_shared_borders_but_preserves_interior_black(self):

        # Create two identical images
        first = np.full((8, 10, 3), 100, np.uint8)
        second = first.copy()
        # Create black edge regions
        second[:2] = 0
        second[:, :1] = 0
        # Create interior black pixel
        second[4, 5] = 0

        # Check both images lose first 2 rows and 1 col, and that the
        # interior black pixel remains
        cropped = crop_aligned([("a", first), ("b", second)])
        for (_, result), original in zip(cropped, [first, second]):
            assert_array_equal(result, original[2:, 1:])


    def test_tiles_stitch_composite_and_png_round_trip(self):

        # Create 7x11 image where every px has a distinct channel value
        original = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)

        # Tile and verify tile dimensions are 2x3
        grid = make_tiles([("a", original)], 2, 3)[0][1]
        self.assertEqual((len(grid), len(grid[0])), (2, 3))

        # Each tile is (7//2) x (11//3) = 3x3 px making 6x9 px total
        # Leftover bottom/right pixels are discarded.
        # Check stitched image matches that area of the original
        assert_array_equal(stitch_tiles(grid), original[:6, :9])

        # Set a tile to white and verify it corresponds to expected 
        # pixels after stitching
        grid[0][1][:] = 255
        expected = original[:6, :9].copy()
        expected[:3, 3:6] = 255
        composite = stitch_tiles(grid)
        assert_array_equal(composite, expected)
        # Check tile is stored separately and can't modify the original
        self.assertFalse(np.shares_memory(grid[0][1], original))

        # Verify stitched image exports correctly
        output = export_png(composite, self.folder / "nested" / "composite")
        self.assertEqual(output.suffix, ".png")
        assert_array_equal(cv2.imread(str(output)), expected)


    def test_invalid_tile_inputs(self):
        image = np.ones((6, 6, 3), np.uint8)

        # Test various invalid tile dimensions in one go with subTest
        for rows, columns in [(0, 2), (2, -1), (1.5, 2), (7, 2)]:
            with self.subTest(rows=rows, columns=columns), self.assertRaises(ValueError):
                make_tiles([("a", image)], rows, columns)
        # Test other invalid inputs
        with self.assertRaises(ValueError):
            make_tiles([], 2, 2)
        with self.assertRaises(ValueError):
            make_tiles([("a", image), ("b", image[:3])], 2, 2)
        for grid in [[], [[image], [image, image]]]:
            with self.subTest(grid=repr(grid)), self.assertRaises(ValueError):
                stitch_tiles(grid)


    # Replace export_video which is called in *_mp4() functions with mock
    # patch passes the mock as 'export' arg
    @patch("pictile.animate.export_video")
    def test_window_animation_replaces_only_centre(self, export):
        # Use helper to set up tiled images
        tileset = self.tiled_pair()
        
        # Generate frames and timings; mocked exporter writes no video
        output = self.folder / "window.mp4"
        images, durations = make_window_mp4(tileset, output)

        # Manually construct expected image with centre tile changed
        reference = tileset.reference.copy()
        expected = reference.copy()
        # 3x3 grid on an image of dimension (12,18) means the centre 
        # tile will be pixels at coords (4:8, 6:12)
        expected[4:8, 6:12] = 200
        self.assertEqual(len(images), 3)
        assert_array_equal(images[0], reference)
        assert_array_equal(images[1], expected)
        assert_array_equal(images[2], reference)
        self.assertEqual(durations, [0.5, 0.5, 0.5])

        # Check the mock was called exactly once with correct args
        export.assert_called_once_with(
            images, durations, output, fps=30.0
            )


    @patch("pictile.animate.export_video")
    def test_random_animation_replacements_and_persistence(self, export):
        # Use helper to set up tiled images
        tileset = self.tiled_pair()

        # Generate frames and timings; mocked exporter writes no video
        output = self.folder / "random.mp4"
        images, durations = make_random_mp4(tileset, output, n=2,
                                            num_draws=3, persist=2, 
                                            duration=0.1)
        self.assertEqual(len(images), 4)
        self.assertEqual(durations, [0.1] * 4)

        # persist=2 prevents reusing a position during the next two draws
        # Check this correctly forces two new 4x6 donor tiles that don't
        # match reference image by counting over one channel value
        # Channel values are known from tiled_pair()
        for draw, image in enumerate(images):
            self.assertEqual(np.count_nonzero(image[:, :, 0] == 200),
                             draw * 2 * 4 * 6)
            self.assertTrue(np.isin(image, [40, 200]).all())
        # Check animation generation hasn't modified any tiles in the 
        # source grid
        # Use stitch_tiles() to check all tiles at once
        assert_array_equal(stitch_tiles(tileset.tiles[0][1]), 
                           tileset.reference)
        
        # Check the mock was called exactly once with correct args
        export.assert_called_once_with(images, durations, 
                                       output, fps=30.0)


    def test_animations_require_another_aligned_image(self):
        # Use helper to set up tiled images 
        tileset = self.tiled_pair()
        # Drop one set of tiles
        tileset.tiles = tileset.tiles[:1]
        # Check videos can't be created because both default animations 
        # need a donor
        for animate in [make_random_mp4, make_window_mp4]:
            with self.subTest(animate=animate.__name__), self.assertRaises(ValueError):
                animate(tileset, self.folder / "invalid.mp4")


    # Mock the OpenCV writer so no video codec is needed
    @patch("pictile.animate.cv2.VideoWriter")
    def test_video_export_frame_order_timing_and_padding(self, 
                                                         writer_class):
        # Pretend the writer opened successfully so we can check
        # inputs and output directory creation
        writer = writer_class.return_value
        writer.isOpened.return_value = True

        # Declare the list element type to match export_video
        images: list[cv2.typing.MatLike] = [
            np.full((5, 7, 3), value, np.uint8)
            for value in [40, 200]
            ]

        # Export a video:
        # At 10 fps, 0.2 seconds gives two frames
        # 0.01 seconds is rounded up to the minimum one frame
        # => Expect channel values 40, 40, then 200
        output = export_video(images, [0.2, 0.01], 
                              self.folder / "nested" / "video", fps=10)
        self.assertEqual(output, self.folder / "nested" / "video.mp4")
        self.assertTrue(output.parent.is_dir())

        # Check odd dimensions were padded to even by writer (5,7)->(6,8)
        self.assertEqual(writer_class.call_args.args[2:], (10.0, (8, 6)))
        # Check writer wrote 3 frames
        self.assertEqual(writer.write.call_count, 3)
        for call, value in zip(writer.write.call_args_list, [40, 40, 200]):
            frame = call.args[0]
            # verify frame dimensions
            self.assertEqual(frame.shape, (6, 8, 3))
            # Check frames have expected channel values
            assert_array_equal(frame[:5, :7], 
                               np.full((5, 7, 3), value, np.uint8))
            # Check padding is just black bars
            self.assertFalse(frame[5, :].any())
            self.assertFalse(frame[:, 7].any())

        writer.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
