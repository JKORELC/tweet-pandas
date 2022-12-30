"""
Various text-cleansing routines for Pandas series objects.
"""
# pylint: disable=anomalous-backslash-in-string
from typing import NoReturn
from ftfy import fix_text
from pandas import Series


DOUBLE_QUOTES = r"\\xe2\\x80(\\x9c|\\x9d)"
SINGLE_QUOTES = r"\\xe2\\x80(\\x98|\\x99)"
ELIPSES_RULE = r"(\.{2,}|\\xe2\\x80\\xa6|…)"
URLLINK_RULE = (
    "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)


class CleanseText:
    """Methods for cleaning text data."""

    def __init__(self, series: Series) -> NoReturn:
        self._obj: Series = series

    def fix_text(self):
        """User-friendly alias."""
        return self.strip_mojibake()

    def get_hyperlinks(self) -> Series:
        """Retrieve link-like elements."""
        return self._obj.str.findall(URLLINK_RULE)

    def strip_elipses(self, token: str = "") -> Series:
        """Retrieve elipses-like elements."""
        return self._obj.str.replace(ELIPSES_RULE, token, regex=True)

    def strip_encoding(self) -> Series:
        """Retrieve character encodings."""
        return self._obj.str.encode("utf-8").str.decode("ascii", errors="ignore")

    def strip_hyperlink(self, token: str = "") -> Series:
        """Remove link-like elements."""
        return self._obj.str.replace(URLLINK_RULE, token, regex=True)

    def strip_mojibake(self) -> Series:
        """Remove mojibake."""
        return self._obj.apply(fix_text)

    def strip_newline(self) -> Series:
        """Remove newlines."""
        return self._obj.str.replace(r"(\\n)+", " ", regex=True)

    def strip_punctuation(self) -> Series:
        """Remove strict-set punctuation."""
        punctuation = "+_:!,."
        return self._obj.str.replace(punctuation, "", regex=False)

    def strip_quotes(self) -> Series:
        """Remove various quotes."""
        return (
            self._obj.str.replace(DOUBLE_QUOTES, '"', regex=True)
            .str.replace(SINGLE_QUOTES, "'", regex=True)
            .str.replace("(^['\"]|['\"]$)", "", regex=True)
        )

    def strip_whitespace(self, token: str = " ") -> Series:
        """Remove beginning, ending, and multiple whitespace."""
        return self._obj.str.replace("\s+", token, regex=True).str.replace(
            "^\s+|\s+$", "", regex=True
        )
