import unittest

import numpy as np
import pandas as pd

from utils import modeling_prep


def _event_items(rows):
    return pd.DataFrame(rows, columns=["Kode Barang", "is_event_driven"])


class TestAddEventFlag(unittest.TestCase):
    def test_marks_true_for_listed_event_sku(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00002"]})
        items = _event_items([["PCG-00002", "true"], ["PCG-00001", "false"]])
        result = modeling_prep.add_event_flag(df, items)
        self.assertTrue(bool(result.iloc[0]["is_event_driven"]))

    def test_marks_false_for_ordinary_sku(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00001"]})
        items = _event_items([["PCG-00002", "true"], ["PCG-00001", "false"]])
        result = modeling_prep.add_event_flag(df, items)
        self.assertFalse(bool(result.iloc[0]["is_event_driven"]))

    def test_is_case_and_whitespace_insensitive(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00002"]})
        items = _event_items([["PCG-00002", "  TRUE "]])
        result = modeling_prep.add_event_flag(df, items)
        self.assertTrue(bool(result.iloc[0]["is_event_driven"]))

    def test_raises_when_a_sku_is_missing_from_the_list(self):
        df = pd.DataFrame({"Kode Barang": ["FGS-99999"]})
        items = _event_items([["PCG-00002", "true"]])
        with self.assertRaisesRegex(ValueError, "FGS-99999"):
            modeling_prep.add_event_flag(df, items)

    def test_does_not_mutate_the_input_frame(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00001"]})
        items = _event_items([["PCG-00001", "false"]])
        modeling_prep.add_event_flag(df, items)
        self.assertNotIn("is_event_driven", df.columns)


if __name__ == "__main__":
    unittest.main()
