"""Format lexical analysis stats for judge prompts with rating cutoffs.

Provides functions to format model-level lexical stats with human-readable
ratings (extremely low/low/average/high/extremely high) for inclusion in
judge prompts.
"""

from typing import Dict, Any, Optional


# Rating cutoffs for each metric
# Format: (extremely_low_max, low_max, average_max, high_max)
# Values above high_max are "extremely high"
METRIC_CUTOFFS = {
    "slop_score": (0.1, 0.3, 0.45, 0.6),  # higher = more AI-like cliches
    "vocab_level": (5.0, 5.5, 5.75, 6.0),  # extremely high or low may hurt readability
    "avg_sentence_length": (8.5, 9.5, 18.0, 25.0),
    "avg_paragraph_length": (2.0, 2.5, 3.5, 4.5),
    "mattr_500": (0.42, 0.45, 0.58, 0.63),  # lexical diversity
}


def get_rating(value: float, cutoffs: tuple) -> str:
    """Convert a metric value to a rating string based on cutoffs.

    Args:
        value: The metric value
        cutoffs: Tuple of (extremely_low_max, low_max, average_max, high_max)

    Returns:
        Rating string: "extremely low", "low", "average", "high", or "extremely high"
    """
    extremely_low_max, low_max, average_max, high_max = cutoffs

    if value <= extremely_low_max:
        return "extremely low"
    elif value <= low_max:
        return "low"
    elif value <= average_max:
        return "average"
    elif value <= high_max:
        return "high"
    else:
        return "extremely high"


def format_stats_for_judge(
    lexical_analysis: Dict[str, Any],
    model_label: Optional[str] = None
) -> str:
    """Format lexical analysis stats for inclusion in judge prompts.

    Args:
        lexical_analysis: Dictionary with lexical analysis metrics (from run.results["lexical_analysis"])
        model_label: Optional label for the model (e.g., "A0493" for pairwise, or model name)

    Returns:
        Formatted string with stats and ratings, ready for judge prompt inclusion
    """
    if not lexical_analysis:
        return ""

    # Extract relevant metrics
    slop_score = lexical_analysis.get("slop_score")
    vocab_level = lexical_analysis.get("vocab_level")
    avg_sentence_length = lexical_analysis.get("avg_sentence_length")
    avg_paragraph_length = lexical_analysis.get("avg_paragraph_length")
    mattr_500 = lexical_analysis.get("mattr_500")

    # Build the stats block
    lines = []

    if model_label:
        lines.append(f"[LEXICAL STATISTICS FOR {model_label}]")
    else:
        lines.append("[LEXICAL STATISTICS FOR THE ABOVE MODEL]")

    lines.append("The following statistics represent overall averages across all outputs from this model, not just this specific piece:")
    lines.append("")

    if slop_score is not None:
        rating = get_rating(slop_score, METRIC_CUTOFFS["slop_score"])
        lines.append(f"- Slop Score: {slop_score:.3f} ({rating}) - higher values indicate more AI-like cliches and overused phrases")

    if vocab_level is not None:
        rating = get_rating(vocab_level, METRIC_CUTOFFS["vocab_level"])
        lines.append(f"- Vocabulary Level: {vocab_level:.2f} ({rating}) - average word frequency; extremely high or low may affect readability")

    if avg_sentence_length is not None:
        rating = get_rating(avg_sentence_length, METRIC_CUTOFFS["avg_sentence_length"])
        lines.append(f"- Avg Sentence Length: {avg_sentence_length:.1f} words ({rating})")

    if avg_paragraph_length is not None:
        rating = get_rating(avg_paragraph_length, METRIC_CUTOFFS["avg_paragraph_length"])
        lines.append(f"- Avg Paragraph Length: {avg_paragraph_length:.1f} sentences ({rating})")

    if mattr_500 is not None:
        rating = get_rating(mattr_500, METRIC_CUTOFFS["mattr_500"])
        lines.append(f"- Lexical Diversity (MATTR-500): {mattr_500:.3f} ({rating}) - higher values indicate more varied vocabulary")

    if model_label:
        lines.append(f"[/LEXICAL STATISTICS FOR {model_label}]")
    else:
        lines.append("[/LEXICAL STATISTICS]")

    return "\n".join(lines)
