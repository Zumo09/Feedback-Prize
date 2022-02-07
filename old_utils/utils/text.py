import re
from typing import List, Callable
from functools import reduce

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@_-]')
GOOD_SYMBOLS_RE = re.compile('[^0-9a-z .,;!?]')

def lower(text: str) -> str:
    return text.lower()

def replace_special_characters(text: str) -> str:
    return REPLACE_BY_SPACE_RE.sub(' ', text)

def filter_out_uncommon_symbols(text: str) -> str:
    return GOOD_SYMBOLS_RE.sub('', text)

def strip_text(text: str) -> str:
    return text.strip()

PREPROC_PIPELINE = [
    lower,
    replace_special_characters,
    filter_out_uncommon_symbols,
    strip_text
]

# Anchor method

def text_prepare(text: str,
                 filter_methods: List[Callable[[str], str]] = None) -> str:
    """
    Applies a list of pre-processing functions in sequence (reduce).
    Note that the order is important here!
    """
    filter_methods = PREPROC_PIPELINE if filter_methods is None else filter_methods
    return reduce(lambda txt, f: f(txt), filter_methods, text)
