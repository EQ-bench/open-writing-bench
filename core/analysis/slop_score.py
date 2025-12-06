"""Combined slop score metric.

Combines slop word hits, slop trigram hits, and "not X, but Y" contrast patterns
into a single slop score (0-1, higher = more slop/AI-like).
"""

from __future__ import annotations

from typing import Any

from liquid_verifiers.core import EvalResult
from liquid_verifiers.registry import register

from ._core import compute_reward, make_result, parse_target_config
from ._slop_core import (
    compute_contrast_score,
    compute_slop_trigram_hits,
    compute_slop_word_hits,
    compute_slop_score,
    words_only_lower,
)

METRIC_NAME = "slop_score"


@register("lexical|EN|slop_score")
def slop_score_verifier(
    solution: str,
    ground_truth: str,
    config: dict[str, Any] | None = None,
) -> EvalResult:
    """
    Compute combined slop score.

    Combines three indicators of AI-generated text:
    - Slop word hits (60% weight): Words commonly overused by AI
    - "Not X, but Y" patterns (25% weight): Rhetorical contrast patterns
    - Slop trigram hits (15% weight): Three-word phrases common in AI text

    The score is normalized to 0-1 range where:
    - 0 = Very human-like (low slop indicators)
    - 1 = Very AI-like (high slop indicators)

    For the reward score, lower slop_score values (more human-like) get higher rewards.

    Config/ground_truth options:
        domain: Domain for default targets (e.g., "creative_fiction")
        mean: Override target mean
        std: Override target std

    Returns:
        EvalResult with score based on distance from target
    """
    # Tokenize for word/trigram analysis
    tokens = words_only_lower(solution)

    # Compute individual metrics
    slop_word_score, word_hits = compute_slop_word_hits(tokens)
    slop_trigram_score, trigram_hits = compute_slop_trigram_hits(tokens)
    contrast_score, contrast_matches = compute_contrast_score(solution)

    # Compute combined slop score (0-1, higher = more slop)
    value = compute_slop_score(slop_word_score, slop_trigram_score, contrast_score)

    target_mean, target_std = parse_target_config(METRIC_NAME, ground_truth, config)
    score = compute_reward(value, target_mean, target_std)

    # Get top hits for details
    top_word_hits = sorted(word_hits.items(), key=lambda x: x[1], reverse=True)[:10]
    top_trigram_hits = sorted(trigram_hits.items(), key=lambda x: x[1], reverse=True)[:10]

    return make_result(
        metric_name=METRIC_NAME,
        value=value,
        score=score,
        target_mean=target_mean,
        target_std=target_std,
        extra_details={
            "slop_words_per_1k": slop_word_score,
            "slop_trigrams_per_1k": slop_trigram_score,
            "contrast_patterns_per_1k_chars": contrast_score,
            "total_tokens": len(tokens),
            "total_chars": len(solution),
            "word_hits_count": sum(word_hits.values()),
            "trigram_hits_count": sum(trigram_hits.values()),
            "contrast_hits_count": len(contrast_matches),
            "top_word_hits": top_word_hits,
            "top_trigram_hits": top_trigram_hits,
        },
    )