import re

RAW_DATA_FILE = "dataset/dataset.csv"
DATE_FORMAT = "%d %b %Y"

XXX_PREFIX_RE = re.compile(r"^xxx\.\s*", re.IGNORECASE)
SEPARATOR_RE = re.compile(r"(?<=[A-Za-z])\.(?=\d)")


def strip_xxx_prefix(value: str) -> str:
    return XXX_PREFIX_RE.sub("", value)


def unify_separator(code: str) -> str:
    return SEPARATOR_RE.sub("-", code)
