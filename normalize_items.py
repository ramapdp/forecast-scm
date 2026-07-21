import re

RAW_DATA_FILE = "dataset/dataset.csv"
DATE_FORMAT = "%d %b %Y"

XXX_PREFIX_RE = re.compile(r"^xxx\.\s*", re.IGNORECASE)
SEPARATOR_RE = re.compile(r"(?<=[A-Za-z])\.(?=\d)")
TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")
WHITESPACE_RE = re.compile(r"(?<=\d)(?=[a-z])|(\s+)")


def strip_xxx_prefix(value: str) -> str:
    return XXX_PREFIX_RE.sub("", value)


def unify_separator(code: str) -> str:
    return SEPARATOR_RE.sub("-", code)


def normalize_name_for_comparison(name: str) -> str:
    name = strip_xxx_prefix(name)
    name = TRAILING_PAREN_RE.sub("", name)
    name = WHITESPACE_RE.sub(" ", name).strip()
    return name
