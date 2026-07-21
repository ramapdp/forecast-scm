import unittest

import normalize_items


class TestStripXxxPrefix(unittest.TestCase):
    def test_strips_lowercase_prefix(self):
        self.assertEqual(normalize_items.strip_xxx_prefix("xxx.FGS-00003"), "FGS-00003")

    def test_strips_uppercase_prefix_defensively(self):
        self.assertEqual(normalize_items.strip_xxx_prefix("XXX.FGS-00003"), "FGS-00003")

    def test_leaves_code_without_prefix_unchanged(self):
        self.assertEqual(normalize_items.strip_xxx_prefix("FGS-00003"), "FGS-00003")


class TestUnifySeparator(unittest.TestCase):
    def test_converts_letter_dot_digit_separator(self):
        self.assertEqual(normalize_items.unify_separator("FGS.00047"), "FGS-00047")

    def test_leaves_code_without_dot_unchanged(self):
        self.assertEqual(normalize_items.unify_separator("FGS-00047"), "FGS-00047")

    def test_preserves_xxx_prefix_dot_since_followed_by_letter(self):
        # The dot in "xxx." is followed by a letter (F), not a digit, so it
        # must NOT be converted — only the FGS.00069 -> FGS-00069 part is a
        # true code separator. This preserves the xxx. marker as recognizable
        # even after separator normalization.
        self.assertEqual(normalize_items.unify_separator("xxx.FGS.00069"), "xxx.FGS-00069")

    def test_converts_multiple_valid_separators(self):
        self.assertEqual(normalize_items.unify_separator("WIP.00005"), "WIP-00005")


if __name__ == "__main__":
    unittest.main()
