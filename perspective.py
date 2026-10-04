"""Geometry shared by board detection and the top-down view."""

import cv2
import numpy as np


BOARD_SIZE = 1024

ROTATIONS = {
    "none": None,
    "cw": cv2.ROTATE_90_CLOCKWISE,
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "half": cv2.ROTATE_180,
}


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return a convex quadrilateral as TL, TR, BR, BL in camera coordinates."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    ordered = np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)
    # Sorting by angle in image coordinates normally gives TL, TR, BR, BL.
    if ordered[1, 0] < ordered[3, 0]:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32)


def warp_board(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    target = np.float32(
        [[0, 0], [BOARD_SIZE - 1, 0],
         [BOARD_SIZE - 1, BOARD_SIZE - 1], [0, BOARD_SIZE - 1]]
    )
    matrix = cv2.getPerspectiveTransform(order_corners(corners), target)
    return cv2.warpPerspective(frame, matrix, (BOARD_SIZE, BOARD_SIZE),
                               flags=cv2.INTER_CUBIC)


def rotate_board(image: np.ndarray, rotation: str) -> np.ndarray:
    """Put the chosen player's side at the top of the board image."""
    if rotation not in ROTATIONS:
        raise ValueError(f"Unknown board rotation: {rotation}")
    code = ROTATIONS[rotation]
    return image if code is None else cv2.rotate(image, code)


def manual_corners(points, frame_shape) -> np.ndarray:
    """Validate four clicked outer board corners and order them for the warp."""
    corners = order_corners(np.asarray(points, np.float32))
    height, width = frame_shape[:2]
    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners):
        raise ValueError("Clicked corners must outline a convex board")
    if cv2.contourArea(corners) < 0.015 * height * width:
        raise ValueError("Clicked board outline is too small")
    if (np.any(corners[:, 0] < 0) or np.any(corners[:, 0] >= width) or
            np.any(corners[:, 1] < 0) or np.any(corners[:, 1] >= height)):
        raise ValueError("Clicked corners must be inside the camera image")
    return corners
