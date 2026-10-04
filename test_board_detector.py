"""Lighting and clutter regression tests for physical-board localization."""

import os
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import unittest

import cv2
import numpy as np

from board_detector import BoardDetector, BoardTracker
from perspective import manual_corners, order_corners


TRUE_CORNERS = np.float32([[170, 80], [720, 120], [780, 630], [95, 575]])


def scene(heavy_pieces=False):
    board = np.empty((640, 640, 3), np.uint8)
    for row in range(8):
        for col in range(8):
            color = (73, 106, 134) if (row + col) % 2 else (173, 202, 224)
            board[row * 80:(row + 1) * 80, col * 80:(col + 1) * 80] = color
    rows = (0, 1, 3, 4, 6, 7) if heavy_pieces else (0, 1, 6, 7)
    radius = 49 if heavy_pieces else 31
    for row in rows:
        for col in range(8):
            center = (col * 80 + 40, row * 80 + 38)
            cv2.circle(board, center, radius, (10, 10, 10), -1)
            cv2.circle(board, center, radius - 4,
                       (30, 30, 30) if row < 2 else (220, 220, 220), -1)
    image = np.full((720, 1100, 3), (115, 105, 98), np.uint8)
    cv2.rectangle(image, (8, 8), (1080, 680), (70, 75, 80), 5)
    cv2.rectangle(image, (830, 180), (1030, 500), (212, 212, 212), -1)
    source = np.float32([[0, 0], [639, 0], [639, 639], [0, 639]])
    matrix = cv2.getPerspectiveTransform(source, TRUE_CORNERS)
    warped = cv2.warpPerspective(board, matrix, (1100, 720))
    mask = cv2.warpPerspective(np.full((640, 640), 255, np.uint8), matrix, (1100, 720))
    image[mask > 0] = warped[mask > 0]
    return image


def lighting_variant(image, kind):
    rng = np.random.default_rng(19)
    value = image.astype(np.float32)
    if kind == "dim":
        return np.clip(value * .18 + 3 + rng.normal(0, 4, value.shape), 0, 255).astype(np.uint8)
    if kind == "very_dim":
        return np.clip(value * .10 + 1 + rng.normal(0, 3, value.shape), 0, 255).astype(np.uint8)
    if kind == "flat":
        return np.clip((value - 110) * .16 + 90 + rng.normal(0, 2, value.shape), 0, 255).astype(np.uint8)
    if kind == "gradient":
        return np.clip(value * np.linspace(.12, 1.5, image.shape[1])[None, :, None], 0, 255).astype(np.uint8)
    if kind == "glare":
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.ellipse(mask, (410, 340), (240, 170), -20, 0, 360, 255, -1)
        alpha = cv2.GaussianBlur(mask, (0, 0), 30).astype(np.float32) / 255 * .78
        return np.clip(value * (1 - alpha[:, :, None]) + 255 * alpha[:, :, None],
                       0, 255).astype(np.uint8)
    return image


class BoardLightingTests(unittest.TestCase):
    def test_manual_corners_accept_any_click_order_and_reject_small_outline(self):
        clicked = TRUE_CORNERS[[2, 0, 3, 1]]
        np.testing.assert_allclose(manual_corners(clicked, scene().shape),
                                   order_corners(TRUE_CORNERS))
        with self.assertRaises(ValueError):
            manual_corners(np.float32([[20, 20], [30, 20], [30, 30], [20, 30]]),
                           scene().shape)

    def assert_detected_near(self, frame, tolerance):
        result = BoardDetector().detect(frame)
        self.assertIsNotNone(result.corners, result.reason)
        error = np.mean(np.linalg.norm(result.corners - order_corners(TRUE_CORNERS), axis=1))
        self.assertLess(error, tolerance, f"{result.method}: {error:.1f}px")
        if result.method.endswith("(lighting)"):
            self.assertIsNotNone(result.lighting_view)

    def test_board_under_lighting_changes(self):
        image = scene()
        for kind in ("normal", "dim", "very_dim", "flat", "gradient", "glare"):
            with self.subTest(kind=kind):
                self.assert_detected_near(lighting_variant(image, kind), 12)

    def test_pieces_occlude_grid_under_poor_light(self):
        image = scene(heavy_pieces=True)
        for kind in ("normal", "dim", "flat", "gradient", "glare"):
            with self.subTest(kind=kind):
                self.assert_detected_near(lighting_variant(image, kind), 19)

    def test_clutter_without_board_is_rejected(self):
        image = np.full((720, 1100, 3), (90, 100, 110), np.uint8)
        cv2.rectangle(image, (90, 80), (710, 620), (20, 20, 20), 6)
        cv2.rectangle(image, (770, 120), (1040, 520), (230, 230, 230), -1)
        cv2.circle(image, (450, 350), 130, (20, 150, 200), -1)
        for kind in ("normal", "dim", "flat", "glare"):
            with self.subTest(kind=kind):
                self.assertIsNone(BoardDetector().detect(lighting_variant(image, kind)).corners)

    def test_low_light_detection_can_seed_tracking(self):
        frame = lighting_variant(scene(), "very_dim")
        detected = BoardDetector().detect(frame)
        tracker = BoardTracker()
        self.assertEqual(tracker.update(frame, detected.corners)[1], "live")
        covered = frame.copy()
        cv2.rectangle(covered, (300, 220), (550, 490), (40, 40, 40), -1)
        for _ in range(8):
            corners, state = tracker.update(covered, None)
            self.assertEqual(state, "tracked")
            self.assertIsNotNone(corners)


if __name__ == "__main__":
    unittest.main()
