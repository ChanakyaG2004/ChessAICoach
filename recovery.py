"""Short motion gate for a user-confirmed image baseline or undo."""

import time

import numpy as np

from move_detector import Orientation, align_image, prepare_image, square_scores


class StableFrameGate:
    def __init__(self, stable_seconds: float = 0.8, motion_threshold: float = 0.10):
        self.stable_seconds = stable_seconds
        self.motion_threshold = motion_threshold
        self.orientation = Orientation("black")  # Only max score is used.
        self.reset()

    def reset(self) -> None:
        self.anchor = None
        self.samples = []
        self.stable_since = 0.0
        self.last_time = 0.0
        self.last_motion = 0.0

    def update(self, warped, reliable: bool, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if warped is None or not reliable:
            self.reset()
            return False
        image = prepare_image(warped)
        if self.anchor is None:
            self.anchor = image.copy()
            self.samples = [image]
            self.stable_since = now
        else:
            motion = float(square_scores(self.anchor, image, self.orientation).max())
            # The detected outline moves by a few pixels as corners are refreshed.
            # Correct that camera jitter before judging physical movement.
            if motion > self.motion_threshold:
                aligned, usable = align_image(self.anchor, image)
                if usable:
                    image = aligned
                    motion = float(square_scores(self.anchor, image, self.orientation).max())
            self.last_motion = motion
            if motion > self.motion_threshold:
                self.anchor = image.copy()
                self.samples = [image]
                self.stable_since = now
            else:
                self.samples.append(image)
                self.samples = self.samples[-7:]
        self.last_time = now
        return self.ready(now)

    def ready(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return (len(self.samples) >= 4 and
                now - self.stable_since >= self.stable_seconds and
                now - self.last_time <= 0.5)

    def snapshot(self, now: float | None = None):
        if not self.ready(now):
            raise ValueError("Wait for a stable, unobstructed board before confirming")
        return np.median(np.stack(self.samples), axis=0).astype(np.uint8)
