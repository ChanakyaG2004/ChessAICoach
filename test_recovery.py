"""A resync needs several quiet frames after the last physical movement."""

import unittest

import numpy as np

from recovery import StableFrameGate


class RecoveryTests(unittest.TestCase):
    def test_confirmation_waits_after_motion_and_uses_stable_frames(self):
        gate = StableFrameGate(stable_seconds=.6)
        before = np.zeros((512, 512, 3), np.uint8)
        after = before.copy()
        after[128:192, 128:192] = 255
        for i in range(4):
            self.assertFalse(gate.update(before, True, i * .1))
        self.assertFalse(gate.update(after, True, .4))
        with self.assertRaisesRegex(ValueError, "stable"):
            gate.snapshot(.4)
        for i in range(1, 8):
            gate.update(after, True, .4 + i * .1)
        self.assertTrue(gate.ready(1.1))
        np.testing.assert_array_equal(gate.snapshot(1.1), after)
        self.assertFalse(gate.update(None, False, 1.2))
        self.assertFalse(gate.ready(1.2))


if __name__ == "__main__":
    unittest.main()
