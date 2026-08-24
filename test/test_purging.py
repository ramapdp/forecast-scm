import unittest

import numpy as np
import pandas as pd

from utils.modelling import purging


def _rows(dates, lead_times):
    return pd.DataFrame({
        "Tanggal": pd.to_datetime(dates),
        "lead_time_days": lead_times,
    })


class TestLookaheadSafeMask(unittest.TestCase):
    """target_lead_time_cumulative sums demand over H+1..H+lead_time_days.

    Near a boundary that window reaches past it, so a row dated before the
    cutoff can carry a label built from days on the other side. The mask marks
    which rows keep their whole window on the near side.
    """

    BOUNDARY = pd.Timestamp("2025-12-01")

    def test_window_ending_before_the_boundary_is_safe(self):
        df = _rows(["2025-11-25"], [4])  # covers 26-29 Nov
        self.assertTrue(purging.lookahead_safe_mask(df, self.BOUNDARY).iloc[0])

    def test_window_ending_the_day_before_the_boundary_is_safe(self):
        df = _rows(["2025-11-26"], [4])  # covers 27-30 Nov
        self.assertTrue(purging.lookahead_safe_mask(df, self.BOUNDARY).iloc[0])

    def test_window_reaching_the_boundary_itself_is_unsafe(self):
        df = _rows(["2025-11-27"], [4])  # covers 28 Nov - 1 Dec
        self.assertFalse(purging.lookahead_safe_mask(df, self.BOUNDARY).iloc[0])

    def test_window_crossing_well_past_the_boundary_is_unsafe(self):
        df = _rows(["2025-11-30"], [3])  # covers 1-3 Dec
        self.assertFalse(purging.lookahead_safe_mask(df, self.BOUNDARY).iloc[0])

    def test_shorter_lead_time_survives_where_a_longer_one_does_not(self):
        df = _rows(["2025-11-30", "2025-11-30"], [1, 2])
        mask = purging.lookahead_safe_mask(df, self.BOUNDARY)
        self.assertFalse(mask.iloc[0])  # 1 Dec
        self.assertFalse(mask.iloc[1])  # 1-2 Dec

    def test_null_lead_time_is_treated_as_safe(self):
        """No lead time means no lead-time target either, so there is nothing
        for the boundary to contaminate."""
        df = _rows(["2025-11-30"], [np.nan])
        self.assertTrue(purging.lookahead_safe_mask(df, self.BOUNDARY).iloc[0])

    def test_rows_far_from_the_boundary_are_untouched(self):
        df = _rows(["2024-05-01"] * 3, [1, 2, 4])
        self.assertTrue(purging.lookahead_safe_mask(df, self.BOUNDARY).all())

    def test_mask_keeps_the_frames_index(self):
        df = _rows(["2024-05-01", "2025-11-30"], [1, 4])
        df.index = [7, 9]
        self.assertEqual(list(purging.lookahead_safe_mask(df, self.BOUNDARY).index), [7, 9])


if __name__ == "__main__":
    unittest.main()
