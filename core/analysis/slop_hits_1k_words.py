"""Slop list word hits per 1000 words metric."""

from __future__ import annotations

from typing import Any

from liquid_verifiers.core import EvalResult
from liquid_verifiers.registry import register

from ._core import compute_reward, make_result, parse_target_config
from ._slop_core import compute_slop_word_hits, words_only_lower

METRIC_NAME = "slop_list_hits_1k_words"


@register("lexical|EN|slop_list_hits_1k_words")
def slop_list_hits_1k_words_verifier(
    solution: str,
    ground_truth: str,
    config: dict[str, Any] | None = None,
) -> EvalResult:
    """
    Verify slop word hits per 1000 words.

    Counts occurrences of words from the slop word list (common AI-generated
    words like "ached", "arcane", "ethereal", etc.) per 1000 words.

    Lower values indicate more human-like text.

    Config/ground_truth options:
        domain: Domain for default targets (e.g., "creative_fiction")
        mean: Override target mean
        std: Override target std

    Returns:
        EvalResult with score based on distance from target
    """
    tokens = words_only_lower(solution)
    value, hit_counts = compute_slop_word_hits(tokens)

    target_mean, target_std = parse_target_config(METRIC_NAME, ground_truth, config)
    score = compute_reward(value, target_mean, target_std)

    # Get top hits for details
    top_hits = sorted(hit_counts.items(), key=lambda x: x[1], reverse=True)[:20]

    return make_result(
        metric_name=METRIC_NAME,
        value=value,
        score=score,
        target_mean=target_mean,
        target_std=target_std,
        extra_details={
            "total_tokens": len(tokens),
            "total_hits": sum(hit_counts.values()),
            "top_hits": top_hits,
        },
    )