from pathlib import Path
from typing import Any
from collections.abc import Sequence
import cv2
import numpy as np


def load_images(
        folder: str | Path
        ) -> list[tuple[str, cv2.typing.MatLike]]:
    """
    Load all JPG/JPEG/PNG images from a folder in alphabetical order
    as BGR arrays for use in OpenCV.
    """

    paths = sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is not None:
            images.append((path.name, image))

    return images


def sift_features(
        image: cv2.typing.MatLike
        ) -> tuple[Sequence[cv2.KeyPoint], cv2.typing.MatLike | None]:
    """Detect SIFT keypoints and descriptors for a BGR image."""
    
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(grey, None)


def align_sift(
        reference: cv2.typing.MatLike, 
        moving: cv2.typing.MatLike,
        *,
        reference_features: tuple[
            Sequence[cv2.KeyPoint], cv2.typing.MatLike | None
        ] | None = None,
        ) -> tuple[cv2.typing.MatLike | None, dict[str, Any]]:
    """
    Align 'moving' image to 'reference' image using
      1. SIFT to find feature keypoints
      2. Lowe ratio test to match features
      3. RANSAC homography to map 'moving' features to 'reference' features

    Returns aligned image and diagnostic information.
    Pass sift_features(reference) to reuse reference features across calls.
    """

    # Detect reference features only when they have not been supplied.
    if reference_features is None:
        reference_features = sift_features(reference)
    kp_ref, des_ref = reference_features
    kp_mov, des_mov = sift_features(moving)
    # Exit if nothing detected
    if des_ref is None or des_mov is None:
        return None, {
            "success": False,
            "reason": "No descriptors found",
        }

    # Match descriptors by exhaustive comparison (brute force)
    # NORM_L2 for SIFT
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    # Return k best matches for each descriptor
    matches = matcher.knnMatch(
        des_mov,
        des_ref,
        k=2,
    )
    # Lowe ratio test to find good feature matches
    good_matches = [
        m for m, n in matches
        if m.distance < 0.75 * n.distance
    ]
    # Exit if too few unambiguous matches
    if len(good_matches) < 4:
        return None, {
            "success": False,
            "reason": "Too few matches",
            "matches": len(good_matches),
        }

    # Get pixel coordinates of good matches, reshaped for transform
    # 'moving' image as source (src)
    src_pts = np.float32([
        kp_mov[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)
    # 'reference' image as destination (dst)
    dst_pts = np.float32([
        kp_ref[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    # Estimate homography (transformation matrix) with RANSAC
    H, inlier_mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0,  # reprojection threshold in pixels
    )
    # Exit if unable to find a homography
    if H is None:
        return None, {
            "success": False,
            "reason": "Homography estimation failed",
        }

    # Transform moving image to reference coordinates with homography H
    height, width = reference.shape[:2]
    aligned = cv2.warpPerspective(
        moving,
        H,
        (width, height),
    )

    # Evaluate alignment
    # inlier_mask == 1 where each matching point agrees with estimated
    # homography H, else 0
    inlier_mask = inlier_mask.ravel().astype(bool)

    # Number of matches that passes Lowe ratio test
    num_matches = len(good_matches)
    # Number of those accepted by RANSAC
    num_inliers = int(inlier_mask.sum())

    inlier_ratio = num_inliers / num_matches

    # Transform 'reference' image points with H
    projected = cv2.perspectiveTransform(src_pts, H)
    # Calculate distance of transformed point from matching destination point
    errors = np.linalg.norm(
        projected.reshape(-1, 2)
        - dst_pts.reshape(-1, 2),
        axis=1,
    )

    # Only evaluate RANSAC inlier error
    inlier_errors = errors[inlier_mask]

    # Summarise evaluation
    mean_error = float(np.mean(inlier_errors))
    median_error = float(np.median(inlier_errors))

    diagnostics = {
        "success": True,
        "matches": num_matches,
        "inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
        "mean_reprojection_error": mean_error,
        "median_reprojection_error": median_error,
        "homography": H,
    }

    return aligned, diagnostics


def crop_aligned(
        aligned: list[tuple[str, cv2.typing.MatLike | None]]
        ) -> list[tuple[str, cv2.typing.MatLike | None]]:

    # Get valid images
    images = [image for _, image in aligned if image is not None]
    if not images:
        return aligned

    # Get pixel coords that exist in all images
    height = min(image.shape[0] for image in images)
    width = min(image.shape[1] for image in images)
    shared = np.ones((height, width), dtype=bool)

    # Build mask of valid pixel coords
    for image in images:
        pixels = image[:height, :width]

        # Get all nonblack pixels, permitting colour and greyscale images
        nonblack = np.any(pixels != 0, axis=2) if pixels.ndim == 3 else pixels != 0

        # Flood from outside the frame to identify black border regions
        # while permitting interior black pixels
        border_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        border_mask[1:-1, 1:-1] = nonblack

        # Flood fill assigns reachable pixels (black border regions) 
        # value 2 to make them easily identifiable for removal
        cv2.floodFill(border_mask, None, (0, 0), 2)

        # Remove the artificial border to isolate the image
        # Create mask excluding black border regions
        # Combine that mask with shared to build a combined mask of 
        # the viable region of all images
        shared &= border_mask[1:-1, 1:-1] != 2


    ## To find the largest all-valid rectangle from "shared" we search
    ## row-by-row

    # Track how many consecutive valid pixels have appeared in each
    # column up to the current search row
    # Extra zero column at the end ensures final rectangle is evaluated
    heights = np.zeros(width + 1, dtype=np.intp)

    # Store the dimensions of the best rectangle found so far
    best_area = 0
    top = bottom = left = right = 0

    # For each row of shared: y=row index, row=row values
    for y, row in enumerate(shared):

        # For each image col add 1 to cumulative height if the pixel 
        # is valid, else set cumulative height to 0
        heights[:width] += 1
        heights[:width] *= row

        # stack stores candidate rectangle heights along with the
        # leftmost x-position from which each height can extend
        stack = []

        # Get indices of "heights" where the histogram height changes
        # Treat blocks of equal height as a single stack for efficiency
        changes = np.flatnonzero(heights[1:] != heights[:-1]) + 1

        # For each change in height
        for x in np.concatenate(([0], changes)):

            # Current possible rectangle height and start index
            current = int(heights[x])
            start = int(x)

            # If the last stack height > this height then a candidate 
            # rectangle ends and we calculate its area
            while stack and stack[-1][1] > current:

                # Pop the rectangle height and the leftmost column from
                # which that height is able to extend
                start, rectangle_height = stack.pop()

                # Calculate area of candidate rectangle
                area = rectangle_height * (int(x) - start)
                if area > best_area:
                    best_area = area
                    top, bottom = y + 1 - rectangle_height, y + 1
                    left, right = start, int(x)

            # If this height is greater than the current stack height,
            # store it as a new possible rectangle height
            if current and (not stack or stack[-1][1] < current):
                stack.append((start, current))

    # Apply largest rectangle crop to all images
    return [
        (name, image[top:bottom, left:right] if image is not None else None)
        for name, image in aligned
    ]

