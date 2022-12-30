"""
Tweet cleansing routines for Pandas series objects.
"""
# pylint: disable=anomalous-backslash-in-string
from typing import NoReturn

from collections import Counter
from pandas import Series
from pandas.api.extensions import register_series_accessor

from .cleanse import CleanseText


HANDLE_RULE = "@([a-zA-Z0-9_]{2,15})"
MENTION_RULE = f"(?<!RT\s){HANDLE_RULE}"
RETWEET_RULE = f"(?<=RT\s)?RT\s{HANDLE_RULE}:"
HASHTAG_RULE = "#([a-zA-Z0-9_]{1,})"

EMOJI_EMOTICONS = "\U0001F600-\U0001F64F"  # emoticons
EMOJI_PICTOGRAM = "\U0001F300-\U0001F5FF"  # symbols & pictograms
EMOJI_TRANS_MAP = "\U0001F680-\U0001F6FF"  # transport & map symbols
EMOJI_FLAGS_IOS = "\U0001F1E0-\U0001F1FF"  # flags (iOS)
EMOJI_EMOJI_ALL = r"[^\w\s,. ]"
EMOJI_EMOJI_SET = (
    f"[{EMOJI_EMOTICONS}{EMOJI_PICTOGRAM}{EMOJI_TRANS_MAP}{EMOJI_FLAGS_IOS}]"
)


@register_series_accessor("tweet")
class TweetParser(CleanseText):
    """Methods for cleansing Tweets in Pandas series objects."""

    def __init__(self, series: Series) -> NoReturn:
        super().__init__(series)
        self._obj = series

    def count_elements(self, sep: str = ";") -> Series:
        """
        Return a count of delimited elements in string.

        Args:
            sep (str): String separate (default: ;).

        Returns:
            counts (pandas.Series[int]): A count of each delimited element in a list.
        """
        return self._obj.apply(lambda x: len(x.split(sep) if x != "" else ""))

    def get_emojis(self, counts: bool = False, all_emojis: bool = False) -> Series:
        """
        Retrieve emojis.

        Args:
            counts (bool): Return a dictionary with counts of emojis.
            all_amojis (bool): Return all emojies (not just unicode).

        Returns:
            emojis (pandas.Series): A series of string or dictionary elements.
        """
        emoji_rule = EMOJI_EMOJI_ALL if all_emojis else EMOJI_EMOJI_SET
        emojis = self._obj.str.findall(emoji_rule)

        if counts:
            return emojis.apply(Counter)

        return emojis

    def get_hashtags(self, sep: str = "") -> Series:
        """
        Retrieve hashtags.

        Args:
            sep (str): Hashtag element delimiter.

        Returns:
            hashtags (pandas.Series): A series of hashtags, optionally delimited.
        """
        hashtags = self._obj.str.findall(HASHTAG_RULE)
        return hashtags if not sep else hashtags.str.join(sep)

    def get_mentions(self, sep: str = "") -> Series:
        """
        Retrieve mentions.

        Args:
            sep (str): Mentions element delimiter.

        Returns:
            mentions (pandas.Series): A series of mentions, optionally delimited.
        """
        mentions = self._obj.str.findall(MENTION_RULE)
        return mentions if not sep else mentions.str.join(sep)

    def get_retweets(self) -> Series:
        """
        Retrieve retweets.

        Args:
            None

        Returns:
            retweets (pandas.Series): A series of retweet mentions.
        """
        return self._obj.str.findall(RETWEET_RULE)

    def has_hashtags(self, case_sensitive: bool = False) -> bool:
        """
        Flag hashtags.

        Args:
            case_sensitive (bool): Indicate use of case-sensitive match.

        Returns:
            flag (pandas.Series): A series of booleans indicating on hashtags.
        """
        return self._obj.str.contains(HASHTAG_RULE, case=case_sensitive)

    def has_mentions(self, case_sensitive: bool = False) -> bool:
        """
        Flag mentions.

        Args:
            case_sensitive (bool): Indicate use of case-sensitive match.

        Returns:
            flag (pandas.Series): A series of booleans indicating on mentions.
        """
        return self._obj.str.contains(MENTION_RULE, case=case_sensitive)

    def has_retweets(self, case_sensitive: bool = False) -> bool:
        """
        Flag retweets.

        Args:
            case_sensitive (bool): Indicate use of case-sensitive match.

        Returns:
            flag (pandas.Series): A series of booleans indicating retweets.
        """
        return self._obj.str.contains(RETWEET_RULE, case=case_sensitive)

    def strip_hashtags(self, token: str = "") -> Series:
        """
        Remove hashtags.

        Args:
            token (str): A replacement token.

        Returns:
            placeholder (pandas.Series): A series of placeholders.
        """
        return self._obj.str.replace(HASHTAG_RULE, token, regex=True)

    def strip_mentions(self, token: str = "") -> Series:
        """
        Remove mentions.

        Args:
            token (str): A replacement token.

        Returns:
            placeholder (pandas.Series): A series of placeholders.
        """
        return self._obj.str.replace(MENTION_RULE, token, regex=True)

    def strip_retweets(self, token="") -> Series:
        """
        Remove retweets.

        Args:
            token (str): A replacement token.

        Returns:
            placeholder (pandas.Series): A series of placeholders.
        """
        return self._obj.str.replace(RETWEET_RULE, token, regex=True)
