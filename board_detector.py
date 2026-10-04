"""Detect the full 8x8 playing surface without using chess pieces."""

from dataclasses import dataclass
import math
import os
import time

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import cv2
import numpy as np

from perspective import order_corners


@dataclass
class DetectionResult:
    corners: np.ndarray | None
    method: str
    reason: str
    edges: np.ndarray
    lines_view: np.ndarray
    candidates_view: np.ndarray
    lighting_view: np.ndarray | None = None


@dataclass
class GridLine:
    position: float
    line: np.ndarray  # a*x + b*y = c, with (a,b) a unit normal
    strength: float


@dataclass
class GridBand:
    start: float
    end: float
    support: int
    strength: float


def _text(image: np.ndarray, message: str, row: int = 0) -> None:
    y = 25 + 26 * row
    cv2.putText(image, message, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, message, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 1, cv2.LINE_AA)


def _quad_ok(quad: np.ndarray, shape: tuple[int, ...]) -> bool:
    h, w = shape[:2]
    if not np.isfinite(quad).all():
        return False
    polygon = quad.astype(np.float32)
    if not cv2.isContourConvex(polygon):
        return False
    area = cv2.contourArea(polygon)
    if area < 0.025 * w * h or area > 0.9 * w * h:
        return False
    sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    if sides.min() < min(h, w) * 0.09 or sides.max() / sides.min() > 7:
        return False
    margin = max(w, h) * 0.12
    return bool(np.all(quad[:, 0] > -margin) and np.all(quad[:, 0] < w + margin)
                and np.all(quad[:, 1] > -margin) and np.all(quad[:, 1] < h + margin))


def _checker_score(gray: np.ndarray, quad: np.ndarray) -> tuple[float, float, float]:
    """Score cell alternation and repeated grid edges after rectification."""
    dst = np.float32([[0, 0], [399, 0], [399, 399], [0, 399]])
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    board = cv2.warpPerspective(gray, matrix, (400, 400))
    cells = np.empty((8, 8), np.float32)
    for row in range(8):
        for col in range(8):
            patch = board[row * 50 + 17:row * 50 + 33,
                          col * 50 + 17:col * 50 + 33]
            cells[row, col] = np.median(patch)

    # Adjacent cells should usually alternate, while a lighting gradient is slow.
    parity = np.fromfunction(lambda r, c: (r + c) % 2, (8, 8)).astype(bool)
    light = float(np.median(cells[parity]))
    dark = float(np.median(cells[~parity]))
    contrast = abs(light - dark)
    predicted = np.where(parity, light, dark)
    residual = float(np.median(np.abs(cells - predicted)))
    checker = max(0.0, (contrast - 0.6 * residual) / 40.0)

    blurred = cv2.GaussianBlur(board, (3, 3), 0)
    dx = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    dy = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))
    vertical = np.mean([np.mean(dx[12:388, x - 2:x + 3]) for x in range(50, 400, 50)])
    horizontal = np.mean([np.mean(dy[y - 2:y + 3, 12:388]) for y in range(50, 400, 50)])
    off_v = np.mean([np.mean(dx[12:388, x - 2:x + 3]) for x in range(25, 400, 50)])
    off_h = np.mean([np.mean(dy[y - 2:y + 3, 12:388]) for y in range(25, 400, 50)])
    grid = max(0.0, (vertical + horizontal) / (off_v + off_h + 1.0) - 1.0)
    return checker + 0.65 * grid, checker, grid


def _line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    matrix = np.array([first[:2], second[:2]], np.float64)
    determinant = np.linalg.det(matrix)
    if abs(determinant) < 0.15:
        return None
    return np.linalg.solve(matrix, np.array([first[2], second[2]])).astype(np.float32)


def _normalize_lighting(gray: np.ndarray) -> np.ndarray:
    """Remove slow illumination changes while preserving the square boundaries."""
    low, median, high = np.percentile(gray, (2, 50, 98))
    if median < 55 or high - low < 45:
        # CLAHE alone amplifies sensor noise in a dim or nearly flat image.
        smooth = cv2.fastNlMeansDenoising(gray, h=5)
    else:
        smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    illumination = cv2.GaussianBlur(smooth, (0, 0), 35)
    flattened = np.clip((smooth.astype(np.float32) + 8) /
                        (illumination.astype(np.float32) + 8) * 128,
                        0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flattened)


class BoardDetector:
    def __init__(self, max_width: int = 960):
        self.max_width = max_width
        self.frame_number = 0

    def detect(self, frame: np.ndarray, full_search: bool = True) -> DetectionResult:
        self.frame_number += 1
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_width / w)
        small = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
        camera_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(camera_gray)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 125)
        lines_view = small.copy()
        candidates_view = small.copy()
        if not full_search:
            _text(candidates_view, "Tracking board; full search every 10 frames")
            return DetectionResult(None, "tracking", "Full search skipped while tracking",
                                   edges, lines_view, candidates_view)

        sb_quad, sb_points, sb_reason = self._from_internal_corners(gray)
        if sb_points is not None:
            cv2.drawChessboardCorners(candidates_view, (7, 7),
                                      sb_points.reshape(-1, 1, 2), True)
        if sb_quad is not None:
            _, checker, grid = _checker_score(gray, sb_quad)
            if _quad_ok(sb_quad, gray.shape) and (checker > 0.12 or grid > 0.14):
                cv2.polylines(candidates_view, [sb_quad.astype(np.int32)],
                              True, (0, 255, 0), 2)
                _text(candidates_view, f"SB: checker {checker:.2f}, grid {grid:.2f}")
                return DetectionResult(sb_quad / scale, "internal corners", "OK",
                                       edges, lines_view, candidates_view)
            sb_reason = f"internal corners rejected: checker {checker:.2f}, grid {grid:.2f}"

        lighting = _normalize_lighting(camera_gray)
        raw_low, raw_median, raw_high = np.percentile(camera_gray, (2, 50, 98))
        lighting_debug = lighting.copy()
        _text(lighting_debug, f"Raw light: median {raw_median:.0f}, range {raw_high - raw_low:.0f}")
        lighting_edges = cv2.Canny(cv2.GaussianBlur(lighting, (5, 5), 0), 20, 55)
        light_quad, light_points, light_reason = self._from_internal_corners(
            lighting, exhaustive=True)
        if light_points is not None:
            cv2.drawChessboardCorners(candidates_view, (7, 7),
                                      light_points.reshape(-1, 1, 2), True)
        if light_quad is not None:
            _, checker, grid = _checker_score(lighting, light_quad)
            if _quad_ok(light_quad, lighting.shape) and (checker > 0.12 or grid > 0.14):
                cv2.polylines(candidates_view, [light_quad.astype(np.int32)],
                              True, (0, 255, 0), 2)
                _text(candidates_view, f"Lighting-normalized SB: checker {checker:.2f}, grid {grid:.2f}")
                return DetectionResult(light_quad / scale, "internal corners (lighting)",
                                       "OK", lighting_edges, lines_view, candidates_view,
                                       lighting_debug)
            light_reason = (f"lighting corners rejected: checker {checker:.2f}, "
                            f"grid {grid:.2f}")

        quad, line_reason = self._from_grid_lines(gray, edges, lines_view, candidates_view)
        if quad is not None:
            return DetectionResult(quad / scale, "Hough grid", "OK", edges,
                                   lines_view, candidates_view, lighting_debug)
        lighting_lines = small.copy()
        lighting_candidates = small.copy()
        light_line_quad, light_line_reason = self._from_grid_lines(
            lighting, lighting_edges, lighting_lines, lighting_candidates)
        if light_line_quad is not None:
            return DetectionResult(light_line_quad / scale, "Hough grid (lighting)", "OK",
                                   lighting_edges, lighting_lines, lighting_candidates,
                                   lighting_debug)
        reason = (f"standard: {sb_reason}; {line_reason}; "
                  f"lighting: {light_reason}; {light_line_reason}")
        candidates_view = lighting_candidates
        _text(candidates_view, f"DIRECT DETECTION FAILED - raw median {raw_median:.0f}, range {raw_high - raw_low:.0f}")
        _text(candidates_view, f"Standard: {sb_reason}; {line_reason}"[:95], 1)
        _text(candidates_view, f"Lighting: {light_reason}; {light_line_reason}"[:95], 2)
        return DetectionResult(None, "none", reason, lighting_edges, lighting_lines,
                               candidates_view, lighting_debug)

    def _from_internal_corners(self, gray: np.ndarray, exhaustive: bool = False):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if exhaustive or self.frame_number % 8 == 0:
            flags |= cv2.CALIB_CB_EXHAUSTIVE
        try:
            found, corners = cv2.findChessboardCornersSB(gray, (7, 7), flags=flags)
        except cv2.error as exc:
            return None, None, f"SB error: {str(exc).splitlines()[0]}"
        if not found or corners is None:
            return None, None, "7x7 internal corners absent/occluded"
        points = corners.reshape(7, 7, 2)
        source = np.array([[c, r] for r in range(1, 8) for c in range(1, 8)], np.float32)
        homography, mask = cv2.findHomography(source, points.reshape(-1, 2), cv2.RANSAC, 3.0)
        if homography is None or mask is None or int(mask.sum()) < 42:
            return None, points, "internal corner homography inconsistent"
        projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), homography).reshape(-1, 2)
        if np.median(np.linalg.norm(projected - points.reshape(-1, 2), axis=1)) > 2.5:
            return None, points, "internal corner reprojection too large"
        outer = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])
        quad = cv2.perspectiveTransform(outer.reshape(-1, 1, 2), homography).reshape(4, 2)
        return order_corners(quad), points, "OK"

    def _from_grid_lines(self, gray, edges, lines_view, candidates_view):
        h, w = gray.shape
        minimum = max(35, int(min(h, w) * 0.08))
        raw = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=45,
                              minLineLength=minimum, maxLineGap=18)
        if raw is None:
            return None, "Hough found no long lines"
        segments = raw.reshape(-1, 4).astype(np.float32)
        if len(segments) > 450:
            lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
            segments = segments[np.argsort(lengths)[-450:]]
        angles = np.mod(np.arctan2(segments[:, 3] - segments[:, 1],
                                   segments[:, 2] - segments[:, 0]), np.pi)
        lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
        histogram, _ = np.histogram(angles, bins=180, range=(0, np.pi), weights=lengths)
        smooth = np.convolve(np.r_[histogram[-5:], histogram, histogram[:5]],
                             np.ones(11), mode="same")[5:-5]
        first = int(np.argmax(smooth))
        separation = np.abs(np.arange(180) - first)
        separation = np.minimum(separation, 180 - separation)
        allowed = (separation >= 45) & (separation <= 135)
        second = int(np.argmax(np.where(allowed, smooth, -1)))
        if smooth[second] < 0.12 * smooth[first]:
            return None, "only one strong line direction"
        families = []
        colors = [(255, 180, 0), (255, 0, 255)]
        for family_index, peak in enumerate((first, second)):
            angle = np.deg2rad(peak + 0.5)
            distance = np.abs(angles - angle)
            distance = np.minimum(distance, np.pi - distance)
            selected = segments[distance < np.deg2rad(36)]
            selected_lengths = lengths[distance < np.deg2rad(36)]
            for segment in selected:
                cv2.line(lines_view, tuple(segment[:2].astype(int)),
                         tuple(segment[2:].astype(int)), colors[family_index], 1)
            lines = self._cluster_lines(selected, selected_lengths, angle, min(h, w))
            bands = self._grid_bands(lines, min(h, w))
            for line in lines:
                self._draw_full_line(lines_view, line.line, colors[family_index], 2)
            families.append((lines, bands))
        _text(lines_view, f"clustered lines: {len(families[0][0])} + {len(families[1][0])}")
        if not families[0][1] or not families[1][1]:
            return None, (f"grid support insufficient: {len(families[0][0])}/"
                          f"{len(families[1][0])} clustered lines")

        best = None
        best_score = 0.0
        count = 0
        for left_band in families[0][1][:8]:
            for right_band in families[1][1][:8]:
                a0 = self._line_at(families[0][0], left_band.start)
                a1 = self._line_at(families[0][0], left_band.end)
                b0 = self._line_at(families[1][0], right_band.start)
                b1 = self._line_at(families[1][0], right_band.end)
                intersections = [_line_intersection(a, b) for a, b in
                                 ((a0, b0), (a0, b1), (a1, b1), (a1, b0))]
                if any(point is None for point in intersections):
                    continue
                quad = order_corners(np.array(intersections))
                if not _quad_ok(quad, gray.shape):
                    continue
                count += 1
                score, checker, grid = _checker_score(gray, quad)
                cv2.polylines(candidates_view, [quad.astype(np.int32)],
                              True, (80, 80, 230), 1)
                # A partial board can repeat colors locally. Demand a strong
                # full-width grid, or exceptionally clear 8x8 cell alternation.
                if grid < 5.0 and checker < 2.0:
                    continue
                score += 0.025 * (left_band.support + right_band.support)
                if score > best_score:
                    best_score, best = score, quad
        if best is None:
            return None, f"{count} grid quads tested, none passed checker/grid validation"
        cv2.polylines(candidates_view, [best.astype(np.int32)], True, (0, 255, 0), 3)
        _text(candidates_view, f"Hough grid: {count} candidates, best {best_score:.2f}")
        return best, "OK"

    @staticmethod
    def _cluster_lines(segments, lengths, angle, image_size):
        if len(segments) == 0:
            return []
        normal = np.array([-math.sin(angle), math.cos(angle)], np.float32)
        centers = (segments[:, :2] + segments[:, 2:]) * 0.5
        positions = centers @ normal
        order = np.argsort(positions)
        groups = []
        tolerance = max(5.0, image_size * 0.009)
        for idx in order:
            value = float(positions[idx])
            if not groups or value - np.mean([positions[j] for j in groups[-1]]) > tolerance:
                groups.append([])
            groups[-1].append(int(idx))
        output = []
        for group in groups:
            strength = float(lengths[group].sum())
            if strength < image_size * 0.11:
                continue
            endpoints = segments[group].reshape(-1, 2)
            direction = cv2.fitLine(endpoints, cv2.DIST_L2, 0, 0.01, 0.01).reshape(4)
            vx, vy, x0, y0 = map(float, direction)
            n = np.array([-vy, vx], np.float64)
            if np.dot(n, normal) < 0:
                n = -n
            c = n[0] * x0 + n[1] * y0
            output.append(GridLine(float(np.average(positions[group], weights=lengths[group])),
                                   np.array([n[0], n[1], c]), strength))
        strongest = sorted(output, key=lambda line: line.strength, reverse=True)[:36]
        return sorted(strongest, key=lambda line: line.position)

    @staticmethod
    def _grid_bands(lines, image_size):
        if len(lines) < 6:
            return []
        positions = np.array([line.position for line in lines])
        candidates = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                distance = positions[j] - positions[i]
                for steps in range(5, 9):
                    unit = distance / steps
                    if unit < image_size * 0.017 or unit > image_size * 0.20:
                        continue
                    for offset in range(9 - steps):
                        # A one-dimensional projective grid permits near squares
                        # to appear wider than far squares in the camera image.
                        for curvature in (-0.35, 0.0, 0.35):
                            t = np.arange(9, dtype=np.float32) / 8.0
                            projective = t / (1.0 + curvature * (1.0 - t))
                            span = distance / (projective[offset + steps] - projective[offset])
                            start = positions[i] - span * projective[offset]
                            targets = start + span * projective
                            nearest = np.min(np.abs(targets[:, None] - positions[None, :]), axis=1)
                            support = int(np.sum(nearest < max(4.0, unit * 0.20)))
                            if support < 6:
                                continue
                            strength = support - float(np.mean(np.minimum(nearest / unit, 1)))
                            candidates.append(GridBand(float(start), float(start + span),
                                                       support, strength))
        candidates.sort(key=lambda band: band.strength, reverse=True)
        unique = []
        for band in candidates:
            if all(abs(band.start - old.start) > image_size * 0.025 or
                   abs(band.end - old.end) > image_size * 0.025 for old in unique):
                unique.append(band)
            if len(unique) == 8:
                break
        return unique

    @staticmethod
    def _line_at(lines, position):
        values = np.array([line.position for line in lines])
        index = int(np.searchsorted(values, position))
        lo = max(0, min(index - 1, len(lines) - 2))
        hi = lo + 1
        blend = (position - values[lo]) / (values[hi] - values[lo])
        result = (1 - blend) * lines[lo].line + blend * lines[hi].line
        result /= np.linalg.norm(result[:2])
        return result

    @staticmethod
    def _draw_full_line(image, line, color, thickness):
        a, b, c = line
        center = np.array([a, b]) * c
        direction = np.array([-b, a]) * max(image.shape[:2]) * 2
        p0 = tuple(np.round(center - direction).astype(int))
        p1 = tuple(np.round(center + direction).astype(int))
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


class BoardTracker:
    """Carry a verified board through occlusion using visible board features."""

    def __init__(self, hold_seconds: float = 3.0, track_width: int = 960):
        self.corners = None
        self.last_seen = 0.0
        self.hold_seconds = hold_seconds
        self.track_width = track_width
        self.pending = None
        self.pending_count = 0
        self.previous_gray = None
        self.points = None
        self.point_scale = 1.0
        self.feature_count = 0
        self.visible_points = None
        self.reference_gray = None
        self.reference_corners = None

    def update(self, frame: np.ndarray, detection: np.ndarray | None):
        now = time.monotonic()
        gray, scale = self._gray(frame)
        tracked, tracked_points = self._track(gray, scale)

        if detection is not None:
            detection = order_corners(detection)
            if self.corners is None:
                self.corners = detection.copy()
            else:
                reference = tracked if tracked is not None else self.corners
                size = np.mean(np.linalg.norm(np.roll(self.corners, -1, axis=0) - self.corners, axis=1))
                jump = float(np.mean(np.linalg.norm(detection - reference, axis=1)))
                limit = max(18.0, 0.05 * size) if tracked is not None else max(25.0, 0.22 * size)
                if jump > limit:
                    if tracked is not None:
                        return self._accept_track(gray, scale, tracked, tracked_points, now)
                    if self.pending is not None and np.mean(np.linalg.norm(detection - self.pending, axis=1)) < max(20.0, 0.12 * size):
                        self.pending_count += 1
                    else:
                        self.pending_count = 1
                    self.pending = detection.copy()
                    if self.pending_count < 3:
                        if tracked is not None:
                            return self._accept_track(gray, scale, tracked, tracked_points, now)
                        return self.current(now)
                    self.corners = detection.copy()
                else:
                    correction = 0.10 if tracked is not None else 0.35
                    self.corners = ((1.0 - correction) * reference +
                                    correction * detection).astype(np.float32)
                self.pending = None
                self.pending_count = 0
            self.last_seen = now
            self._seed(gray, scale)
            return self.corners.copy(), "live"

        if tracked is not None:
            return self._accept_track(gray, scale, tracked, tracked_points, now)
        return self.current(now)

    def current(self, now=None):
        now = time.monotonic() if now is None else now
        if self.corners is not None and now - self.last_seen <= self.hold_seconds:
            return self.corners.copy(), "held"
        self.corners = None
        self.pending = None
        self.pending_count = 0
        self.previous_gray = None
        self.points = None
        self.visible_points = None
        self.feature_count = 0
        self.reference_gray = None
        self.reference_corners = None
        return None, "lost"

    def _gray(self, frame):
        scale = min(1.0, self.track_width / frame.shape[1])
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scale

    def _seed(self, gray, scale):
        """Choose image features across the board, without identifying pieces."""
        self.previous_gray = gray
        self.point_scale = scale
        quad = (self.corners * scale).astype(np.int32)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, quad, 255)
        mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
        found = cv2.goodFeaturesToTrack(gray, maxCorners=320, qualityLevel=0.012,
                                       minDistance=7, mask=mask, blockSize=7)
        if found is None:
            self.points = None
            self.feature_count = 0
            self.visible_points = None
            return
        features = found.reshape(-1, 2)
        board_coordinates = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])
        to_board = cv2.getPerspectiveTransform(quad.astype(np.float32), board_coordinates)
        mapped = cv2.perspectiveTransform(features.reshape(-1, 1, 2), to_board).reshape(-1, 2)
        selected = []
        buckets = np.zeros((4, 4), np.int32)
        for point, (x, y) in zip(features, mapped):
            col, row = int(x // 2), int(y // 2)
            if 0 <= row < 4 and 0 <= col < 4 and buckets[row, col] < 16:
                selected.append(point)
                buckets[row, col] += 1
        self.points = np.asarray(selected, np.float32).reshape(-1, 1, 2)
        self.feature_count = len(selected)
        self.visible_points = self.points.reshape(-1, 2) / scale
        if self.reference_gray is None or self.reference_gray.shape != gray.shape:
            self.reference_gray = gray.copy()
            self.reference_corners = self.corners.copy() * scale

    def _track(self, gray, scale):
        if (self.corners is None or self.previous_gray is None or
                self.points is None or len(self.points) < 8 or
                self.previous_gray.shape != gray.shape or abs(scale - self.point_scale) > 1e-5):
            return None, None
        next_points, forward, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, self.points, None, winSize=(21, 21), maxLevel=3)
        if next_points is None:
            return None, None
        back_points, backward, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous_gray, next_points, None, winSize=(21, 21), maxLevel=3)
        if back_points is None:
            return None, None
        error = np.linalg.norm(self.points.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1)
        good = (forward.ravel() != 0) & (backward.ravel() != 0) & (error < 1.5)
        old = self.points.reshape(-1, 2)[good]
        new = next_points.reshape(-1, 2)[good]
        if len(old) < 10:
            return None, None
        transform, mask = cv2.findHomography(old, new, cv2.RANSAC, 3.0)
        if transform is None or mask is None:
            return None, None
        inliers = mask.ravel().astype(bool)
        if inliers.sum() < 10 or inliers.mean() < 0.45:
            return None, None
        old, new = old[inliers], new[inliers]
        board_width = np.mean(np.linalg.norm(np.roll(self.corners, -1, axis=0) - self.corners, axis=1)) * scale
        spread = np.ptp(old, axis=0)
        if min(spread) < 0.20 * board_width:
            return None, None
        fitted = cv2.perspectiveTransform(old.reshape(-1, 1, 2), transform).reshape(-1, 2)
        if np.median(np.linalg.norm(fitted - new, axis=1)) > 2.0:
            return None, None
        quad = cv2.perspectiveTransform((self.corners * scale).reshape(-1, 1, 2),
                                        transform).reshape(4, 2)
        if not _quad_ok(quad, gray.shape):
            return None, None
        motion = np.mean(np.linalg.norm(quad - self.corners * scale, axis=1))
        if motion > 0.18 * board_width:
            return None, None
        return order_corners(quad / scale), new.reshape(-1, 1, 2).astype(np.float32)

    def _accept_track(self, gray, scale, corners, points, now):
        self.corners = corners
        self.last_seen = now
        self.previous_gray = gray
        self.point_scale = scale
        self.points = points
        self.feature_count = len(points)
        self.visible_points = points.reshape(-1, 2) / scale
        if self.feature_count < 120:
            self._replenish(gray, scale)
        return self.corners.copy(), "tracked"

    def _replenish(self, gray, scale):
        """Add points from uncovered squares, rejecting changed image regions."""
        if self.reference_gray is None or self.reference_gray.shape != gray.shape:
            return
        current_quad = (self.corners * scale).astype(np.float32)
        reference_to_current = cv2.getPerspectiveTransform(self.reference_corners,
                                                             current_quad)
        aligned = cv2.warpPerspective(self.reference_gray, reference_to_current,
                                      (gray.shape[1], gray.shape[0]))
        difference = cv2.absdiff(cv2.GaussianBlur(gray, (5, 5), 0),
                                 cv2.GaussianBlur(aligned, (5, 5), 0))
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, current_quad.astype(np.int32), 255)
        mask[difference > 40] = 0
        mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
        found = cv2.goodFeaturesToTrack(gray, maxCorners=320, qualityLevel=0.012,
                                       minDistance=7, mask=mask, blockSize=7)
        if found is None:
            return
        existing = self.points.reshape(-1, 2)
        candidates = np.vstack((existing, found.reshape(-1, 2)))
        board_coordinates = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])
        to_board = cv2.getPerspectiveTransform(current_quad, board_coordinates)
        mapped = cv2.perspectiveTransform(candidates.reshape(-1, 1, 2), to_board).reshape(-1, 2)
        selected = []
        buckets = np.zeros((4, 4), np.int32)
        for point, (x, y) in zip(candidates, mapped):
            col, row = int(x // 2), int(y // 2)
            if not (0 <= row < 4 and 0 <= col < 4 and buckets[row, col] < 16):
                continue
            if selected and np.min(np.linalg.norm(np.asarray(selected) - point, axis=1)) < 6:
                continue
            selected.append(point)
            buckets[row, col] += 1
        if len(selected) > len(existing):
            self.points = np.asarray(selected, np.float32).reshape(-1, 1, 2)
            self.feature_count = len(selected)
            self.visible_points = self.points.reshape(-1, 2) / scale
