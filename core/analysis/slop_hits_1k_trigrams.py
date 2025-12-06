"""Slop list trigram hits per 1000 words metric."""

from __future__ import annotations

from typing import Any

from liquid_verifiers.core import EvalResult
from liquid_verifiers.registry import register

from ._core import compute_reward, make_result, parse_target_config
from ._slop_core import compute_slop_trigram_hits, words_only_lower

METRIC_NAME = "slop_list_hits_1k_trigrams"


@register("lexical|EN|slop_list_hits_1k_trigrams")
def slop_list_hits_1k_trigrams_verifier(
    solution: str,
    ground_truth: str,
    config: dict[str, Any] | None = None,
) -> EvalResult:
    """
    Verify slop trigram hits per 1000 words.

    Counts occurrences of trigrams from the slop trigram list (common AI-generated
    phrases like "voice barely whisper", "took deep breath", "heart pounding chest",
    etc.) per 1000 words.

    Lower values indicate more human-like text.

    Config/ground_truth options:
        domain: Domain for default targets (e.g., "creative_fiction")
        mean: Override target mean
        std: Override target std

    Returns:
        EvalResult with score based on distance from target
    """
    tokens = words_only_lower(solution)
    value, hit_counts = compute_slop_trigram_hits(tokens)

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