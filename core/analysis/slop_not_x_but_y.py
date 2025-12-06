"""Slop "not X, but Y" contrast pattern hits per 1000 characters metric."""

from __future__ import annotations

from typing import Any

from liquid_verifiers.core import EvalResult
from liquid_verifiers.registry import register

from ._core import compute_reward, make_result, parse_target_config
from ._slop_core import compute_contrast_score

METRIC_NAME = "slop_not_x_but_y_1k"


@register("lexical|EN|slop_not_x_but_y_1k")
def slop_not_x_but_y_1k_verifier(
    solution: str,
    ground_truth: str,
    config: dict[str, Any] | None = None,
) -> EvalResult:
    """
    Verify "not X, but Y" contrast pattern hits per 1000 characters.

    Detects rhetorical contrast patterns commonly overused in AI-generated text,
    such as "It wasn't just a noise. It was a warning." or
    "This isn't a problem—it's an opportunity."

    Lower values indicate more human-like text.

    Config/ground_truth options:
        domain: Domain for default targets (e.g., "creative_fiction")
        mean: Override target mean
        std: Override target std

    Returns:
        EvalResult with score based on distance from target
    """
    value, matches = compute_contrast_score(solution)

    target_mean, target_std = parse_target_config(METRIC_NAME, ground_truth, config)
    score = compute_reward(value, target_mean, target_std)

    # Extract pattern names and sample matches for details
    pattern_counts: dict[str, int] = {}
    sample_matches: list[dict] = []

    for match in matches:
        pname = match.get("pattern_name", "unknown")
        pattern_counts[pname] = pattern_counts.get(pname, 0) + 1

        if len(sample_matches) < 10:
            sample_matches.append({
                "pattern": pname,
                "text": match.get("match_text", "")[:100],
            })

    return make_result(
        metric_name=METRIC_NAME,
        value=value,
        score=score,
        target_mean=target_mean,
        target_std=target_std,
        extra_details={
            "total_chars": len(solution),
            "total_hits": len(matches),
            "pattern_counts": pattern_counts,
            "sample_matches": sample_matches,
        },
    )