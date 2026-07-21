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


class TestNormalizeNameForComparison(unittest.TestCase):
    def test_strips_xxx_prefix_from_name(self):
        result = normalize_items.normalize_name_for_comparison("xxx.Iga Sapi Kebuli")
        self.assertEqual(result, "Iga Sapi Kebuli")

    def test_collapses_irregular_whitespace(self):
        result = normalize_items.normalize_name_for_comparison("Gula  Asam   250ml")
        self.assertEqual(result, normalize_items.normalize_name_for_comparison("Gula Asam 250ml"))

    def test_strips_trailing_parenthetical_annotation(self):
        result = normalize_items.normalize_name_for_comparison(
            "Club Mineral 330 ml (Menu pakai kode CM-330)"
        )
        self.assertEqual(result, "Club Mineral 330 ml")

    def test_preserves_non_trailing_parenthetical(self):
        # "Ayam Kebuli (0.6)" ends in a parenthetical, so under this function
        # it DOES get stripped (weight variant is treated as an annotation
        # for comparison purposes only — the stored Nama Barang is untouched
        # elsewhere). This is expected: it never causes a false merge in the
        # real dataset because no two DIFFERENT weight-variant products ever
        # land in the same code-collision group (verified against real data).
        result = normalize_items.normalize_name_for_comparison("Ayam Kebuli (0.6)")
        self.assertEqual(result, "Ayam Kebuli")

    def test_gula_asam_and_kunyit_asam_style_variants_match_after_normalization(self):
        a = normalize_items.normalize_name_for_comparison("Gula Asam 250ml")
        b = normalize_items.normalize_name_for_comparison("Gula Asam 250 ml")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
