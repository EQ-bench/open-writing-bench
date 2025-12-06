"""Core utilities for slop score verifiers.

Ports the slop detection logic from slop-score/js/ to Python.
Uses the reference implementation from not-x-but-y-bench/src/ for contrast detection.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_left, bisect_right
from functools import lru_cache
from pathlib import Path
from typing import Any

# Import reference implementation modules
from . import _regexes_v3 as regexes_v3
from . import _regexes_pos as regexes_pos
from ._pos_tagger import tag_stream_with_offsets

# =============================================================================
# Data Loading
# =============================================================================

_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data"


def _normalize_quotes(s: str) -> str:
    """Normalize various quote characters to ASCII equivalents."""
    # Single quotes: ' ' ‛ ‚ ′ ʼ ＇ `
    s = re.sub(r"[\u2018\u2019\u201A\u201B\u2032\u02BC\uFF07`]", "'", s)
    # Double quotes: " " „ ‟ ″ « » ＂
    s = re.sub(r'[\u201C\u201D\u201E\u201F\u2033\u00AB\u00BB\uFF02]', '"', s)
    return s


def _normalize_text(text: str) -> str:
    """Normalize text for contrast detection (same as JS normalizeText)."""
    replacements = {
        '\u201c': '"', '\u201d': '"',  # " "
        '\u2018': "'", '\u2019': "'",  # ' '
        '\u2014': '-', '\u2013': '-'   # — –
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def words_only_lower(s: str) -> list[str]:
    """
    Tokenize text to lowercase words (same as JS wordsOnlyLower).

    Returns lowercase words containing only a-z and apostrophes,
    with leading/trailing apostrophes stripped.
    """
    txt = _normalize_quotes(s.lower())
    toks = re.findall(r"[a-z']+", txt)
    # Strip leading/trailing apostrophes from each token
    return [t.strip("'") for t in toks if t.strip("'")]


@lru_cache(maxsize=1)
def load_slop_words() -> set[str]:
    """Load slop words from JSON file."""
    path = _DATA_DIR / "slop_list_words.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result = set()
    for item in data:
        if not item or not item[0]:
            continue
        # Extract phrase using same regex as JS
        phrase_match = re.search(r"[a-z]+(?:'[a-z]+)?(?:\s+[a-z]+(?:'[a-z]+)?)*", str(item[0]).lower())
        if phrase_match:
            result.add(phrase_match.group(0))
    return result


@lru_cache(maxsize=1)
def load_slop_bigrams() -> set[str]:
    """Load slop bigrams from JSON file."""
    path = _DATA_DIR / "slop_list_bigrams.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result = set()
    for item in data:
        if not item or not item[0]:
            continue
        phrase_match = re.search(r"[a-z]+(?:'[a-z]+)?(?:\s+[a-z]+(?:'[a-z]+)?)*", str(item[0]).lower())
        if phrase_match:
            result.add(phrase_match.group(0))
    return result


@lru_cache(maxsize=1)
def load_slop_trigrams() -> set[str]:
    """Load slop trigrams from JSON file."""
    path = _DATA_DIR / "slop_list_trigrams.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result = set()
    for item in data:
        if not item or not item[0]:
            continue
        phrase_match = re.search(r"[a-z]+(?:'[a-z]+)?(?:\s+[a-z]+(?:'[a-z]+)?)*", str(item[0]).lower())
        if phrase_match:
            result.add(phrase_match.group(0))
    return result


# =============================================================================
# Slop Index Computation
# =============================================================================

def compute_slop_word_hits(tokens: list[str]) -> tuple[float, dict[str, int]]:
    """
    Compute slop word hits per 1000 words.

    Args:
        tokens: List of lowercase word tokens

    Returns:
        Tuple of (hits per 1k words, dict of word -> count)
    """
    n = len(tokens)
    if n == 0:
        return 0.0, {}

    slop_words = load_slop_words()
    hit_counts: dict[str, int] = {}

    for t in tokens:
        if t in slop_words:
            hit_counts[t] = hit_counts.get(t, 0) + 1

    total_hits = sum(hit_counts.values())
    score = (total_hits / n) * 1000

    return score, hit_counts


def compute_slop_bigram_hits(tokens: list[str]) -> tuple[float, dict[str, int]]:
    """
    Compute slop bigram hits per 1000 words.

    Args:
        tokens: List of lowercase word tokens

    Returns:
        Tuple of (hits per 1k words, dict of bigram -> count)
    """
    n = len(tokens)
    if n < 2:
        return 0.0, {}

    slop_bigrams = load_slop_bigrams()
    hit_counts: dict[str, int] = {}

    for i in range(n - 1):
        bigram = tokens[i] + " " + tokens[i + 1]
        if bigram in slop_bigrams:
            hit_counts[bigram] = hit_counts.get(bigram, 0) + 1

    total_hits = sum(hit_counts.values())
    score = (total_hits / n) * 1000

    return score, hit_counts


def compute_slop_trigram_hits(tokens: list[str]) -> tuple[float, dict[str, int]]:
    """
    Compute slop trigram hits per 1000 words.

    Args:
        tokens: List of lowercase word tokens

    Returns:
        Tuple of (hits per 1k words, dict of trigram -> count)
    """
    n = len(tokens)
    if n < 3:
        return 0.0, {}

    slop_trigrams = load_slop_trigrams()
    hit_counts: dict[str, int] = {}

    for i in range(n - 2):
        trigram = tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2]
        if trigram in slop_trigrams:
            hit_counts[trigram] = hit_counts.get(trigram, 0) + 1

    total_hits = sum(hit_counts.values())
    score = (total_hits / n) * 1000

    return score, hit_counts


# =============================================================================
# Contrast Pattern Detection ("not X, but Y")
# Ported from not-x-but-y-bench/src/scorer.py
# =============================================================================

# Stage-1 regexes (surface patterns)
STAGE1_REGEXES: dict[str, re.Pattern] = regexes_v3.compiled

# Stage-2 regexes (POS-based patterns)
STAGE2_REGEXES: dict[str, re.Pattern] = {
    "POS_DOESNT_VERB": regexes_pos.RE_POS_DOESNT_VERB,
    "POS_DONT_JUST_VERB": regexes_pos.RE_POS_DONT_JUST_VERB,
    "POS_GERUND_FRAGMENT": regexes_pos.RE_POS_GERUND_FRAGMENT,
    "POS_NOT_ADJ": regexes_pos.RE_POS_NOT_ADJ,
    "POS_DASH_VERB": regexes_pos.RE_POS_DASH_VERB,
    "POS_NOT_JUST_VERB_PAST": regexes_pos.RE_POS_NOT_JUST_VERB_PAST,
    "POS_COLON_VERB": regexes_pos.RE_POS_COLON_VERB,
    "POS_ISNT_JUST_VERB": regexes_pos.RE_POS_ISNT_JUST_VERB,
    "POS_QUOTE_MULTI_VERB": regexes_pos.RE_POS_QUOTE_MULTI_VERB,
    "POS_ELLIPSIS_VERB": regexes_pos.RE_POS_ELLIPSIS_VERB,
    "POS_NOT_NOUN": regexes_pos.RE_POS_NOT_NOUN,
    "POS_DOESNT_VERB_EMPHASIS": regexes_pos.RE_POS_DOESNT_VERB_EMPHASIS,
    "POS_DASH_VERB_BROAD": regexes_pos.RE_POS_DASH_VERB_BROAD,
    "POS_ELLIPSIS_BROAD": regexes_pos.RE_POS_ELLIPSIS_BROAD,
    "POS_NOT_BECAUSE": regexes_pos.RE_POS_NOT_BECAUSE,
    "POS_GERUND_BROAD": regexes_pos.RE_POS_GERUND_BROAD,
    "POS_QUOTE_VERBING": regexes_pos.RE_POS_QUOTE_VERBING,
    "POS_DOESNT_LITERAL": regexes_pos.RE_POS_DOESNT_LITERAL,
    "POS_DASH_NOUN_SWAP": regexes_pos.RE_POS_DASH_NOUN_SWAP,
    "POS_ISNT_DASH_EMPHASIS": regexes_pos.RE_POS_ISNT_DASH_EMPHASIS,
    "POS_THATS_NOT_NOUN": regexes_pos.RE_POS_THATS_NOT_NOUN,
    "POS_GERUND_EMPHASIS": regexes_pos.RE_POS_GERUND_EMPHASIS,
    "POS_QUOTE_ATTRIBUTION_VERB": regexes_pos.RE_POS_QUOTE_ATTRIBUTION_VERB,
    "POS_ISNT_NOUN": regexes_pos.RE_POS_ISNT_NOUN,
    "POS_ITS_NOT_JUST": regexes_pos.RE_POS_ITS_NOT_JUST,
    "POS_DASH_GERUND_OBJ": regexes_pos.RE_POS_DASH_GERUND_OBJ,
    "POS_ELLIPSIS_DIALOGUE": regexes_pos.RE_POS_ELLIPSIS_DIALOGUE,
    "POS_SEMI_NOUN": regexes_pos.RE_POS_SEMI_NOUN,
    "POS_ISNT_ADJ_NOUN": regexes_pos.RE_POS_ISNT_ADJ_NOUN,
    "POS_DIALOGUE_ATTR": regexes_pos.RE_POS_DIALOGUE_ATTR,
    "POS_TO_VERB_ISNT": regexes_pos.RE_POS_TO_VERB_ISNT,
    "POS_I_AM_NOT_SEMI": regexes_pos.RE_POS_I_AM_NOT_SEMI,
    "POS_NOT_ANYMORE_ITS": regexes_pos.RE_POS_NOT_ANYMORE_ITS,
    "POS_AINT_SIMPLE": regexes_pos.RE_POS_AINT_SIMPLE,
    "LEMMA_SAME_VERB": regexes_pos.RE_LEMMA_SAME_VERB,
}


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """
    Get sentence spans from text.

    Returns list of (start, end) tuples for each sentence.
    """
    _SENT_SPLIT = re.compile(r'[^.!?]*[.!?]', flags=re.S)
    spans = []
    last_end = 0
    for m in _SENT_SPLIT.finditer(text):
        spans.append((m.start(), m.end()))
        last_end = m.end()
    if last_end < len(text):
        spans.append((last_end, len(text)))
    return spans


def _covered_sentence_range(spans: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    """Find which sentences a match covers."""
    if not spans or start >= end:
        return None
    starts = [s for s, _ in spans]
    ends = [e for _, e in spans]
    lo = bisect_right(ends, start)
    hi = bisect_left(starts, end) - 1
    if lo >= len(spans) or hi < 0 or lo > hi:
        return None
    return lo, hi


def _merge_intervals(items: list[dict]) -> list[dict]:
    """Merge overlapping intervals."""
    if not items:
        return []
    items_sorted = sorted(items, key=lambda d: (d['lo'], d['hi'], d['raw_start']))
    merged = []
    cur = items_sorted[0].copy()
    for it in items_sorted[1:]:
        if it['lo'] <= cur['hi']:
            cur['hi'] = max(cur['hi'], it['hi'])
            cur['raw_end'] = max(cur['raw_end'], it['raw_end'])
        else:
            merged.append(cur)
            cur = it.copy()
    merged.append(cur)
    return merged


def extract_contrast_matches(text: str) -> list[dict[str, Any]]:
    """
    Extract "not X, but Y" contrast pattern matches from text.

    This implements both Stage 1 (surface patterns) and Stage 2 (POS-based patterns)
    from the reference implementation.

    Returns list of dicts with:
        - sentence: The matched sentence(s)
        - pattern_name: Which regex matched
        - match_text: The actual matched text
        - sentence_count: Number of sentences spanned
    """
    t_norm = _normalize_text(text)
    spans = sentence_spans(t_norm)
    candidates: list[dict] = []

    # Stage 1: Run surface regexes on raw normalized text
    for pname, pregex in STAGE1_REGEXES.items():
        for m in pregex.finditer(t_norm):
            rs, re_ = m.start(), m.end()
            rng = _covered_sentence_range(spans, rs, re_)
            if rng is None:
                continue
            lo, hi = rng
            candidates.append({
                "lo": lo, "hi": hi,
                "raw_start": rs, "raw_end": re_,
                "pattern_name": f"S1_{pname}",
                "match_text": m.group(0).strip(),
            })

    # Stage 2: Run POS-based regexes on tagged stream (map back to raw)
    if STAGE2_REGEXES:
        stream, pieces = tag_stream_with_offsets(t_norm, 'verb')
        stream_starts = [p[0] for p in pieces]
        stream_ends = [p[1] for p in pieces]

        def _stream_to_raw(ss: int, se: int) -> tuple[int, int] | None:
            i = bisect_right(stream_ends, ss)
            j = bisect_left(stream_starts, se) - 1
            if i >= len(pieces) or j < i:
                return None
            raw_s = min(p[2] for p in pieces[i:j+1])
            raw_e = max(p[3] for p in pieces[i:j+1])
            return raw_s, raw_e

        for pname, pregex in STAGE2_REGEXES.items():
            for m in pregex.finditer(stream):
                mapres = _stream_to_raw(m.start(), m.end())
                if not mapres:
                    continue
                rs, re_ = mapres
                rng = _covered_sentence_range(spans, rs, re_)
                if rng is None:
                    continue
                lo, hi = rng
                candidates.append({
                    "lo": lo, "hi": hi,
                    "raw_start": rs, "raw_end": re_,
                    "pattern_name": f"S2_{pname}",
                    "match_text": t_norm[rs:re_].strip(),
                })

    # Merge overlapping intervals
    merged = _merge_intervals(candidates)

    # Build results
    results: list[dict[str, Any]] = []
    for it in merged:
        s_lo, s_hi = it["lo"], it["hi"]
        sentence_span = s_hi - s_lo + 1

        if spans:
            block_start = spans[s_lo][0]
            block_end = spans[s_hi][1]
            sentence_text = t_norm[block_start:block_end].strip()
        else:
            sentence_text = it["match_text"]

        results.append({
            "sentence": sentence_text,
            "pattern_name": it["pattern_name"],
            "match_text": it["match_text"],
            "sentence_count": sentence_span,
        })

    return results


def compute_contrast_score(text: str) -> tuple[float, list[dict[str, Any]]]:
    """
    Compute "not X, but Y" contrast pattern score.

    Returns:
        Tuple of (hits per 1k chars, list of match details)
    """
    hits = extract_contrast_matches(text)
    chars = len(text)
    rate = (len(hits) * 1000.0 / chars) if chars > 0 else 0.0

    return rate, hits


# =============================================================================
# Slop Score Normalization and Combination
# =============================================================================

# Normalization ranges (from your provided values with 10% buffer)
NORMALIZATION_RANGES = {
    "slop_words": {
        "min": 6.9 * 0.9,   # 6.21
        "max": 41.0 * 1.1,  # 45.1
    },
    "slop_trigrams": {
        "min": 0.09 * 0.9,  # 0.081
        "max": 1.12 * 1.1,  # 1.232
    },
    "contrast": {
        "min": 0.04 * 0.9,  # 0.036
        "max": 0.81 * 1.1,  # 0.891
    },
}


def normalize_value(value: float, range_dict: dict[str, float]) -> float:
    """Normalize a value to 0-1 range."""
    normalized = (value - range_dict["min"]) / (range_dict["max"] - range_dict["min"])
    return max(0.0, min(1.0, normalized))


def compute_slop_score(
    slop_word_score: float,
    slop_trigram_score: float,
    contrast_score: float,
) -> float:
    """
    Compute combined slop score (0-1 range, higher = more slop).

    Weighted formula: 60% slop words + 25% not-x-but-y + 15% slop trigrams

    Returns score in 0-1 range.
    """
    norm_words = normalize_value(slop_word_score, NORMALIZATION_RANGES["slop_words"])
    norm_trigrams = normalize_value(slop_trigram_score, NORMALIZATION_RANGES["slop_trigrams"])
    norm_contrast = normalize_value(contrast_score, NORMALIZATION_RANGES["contrast"])

    # Weighted formula (result is 0-100, divide by 100 for 0-1)
    raw_score = (norm_words * 0.6 + norm_contrast * 0.25 + norm_trigrams * 0.15) * 100

    return raw_score / 100.0