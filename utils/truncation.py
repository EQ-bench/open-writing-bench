# utils/truncation.py

"""
Text truncation utilities for the Creative Writing Benchmark.
Provides functions for truncating text at the beginning, end, or middle.
"""

from typing import Literal


def truncate_text(
    text: str,
    max_chars: int,
    mode: Literal["end", "middle", "start"] = "end",
    ellipsis: str = "\n\n[...truncated...]\n\n"
) -> str:
    """
    Truncate text to a maximum number of characters.

    Args:
        text: The text to truncate
        max_chars: Maximum number of characters (including ellipsis if applied)
        mode: Where to truncate:
            - "end": Keep the beginning, cut the end (default)
            - "start": Keep the end, cut the beginning
            - "middle": Keep beginning and end, cut the middle
        ellipsis: String to insert at truncation point

    Returns:
        Truncated text, or original if already within limit
    """
    if not text or len(text) <= max_chars:
        return text

    ellipsis_len = len(ellipsis)
    available_chars = max_chars - ellipsis_len

    if available_chars <= 0:
        # Edge case: max_chars is too small to even fit ellipsis
        return text[:max_chars]

    if mode == "end":
        return text[:available_chars] + ellipsis

    elif mode == "start":
        return ellipsis + text[-available_chars:]

    elif mode == "middle":
        # Split available chars between start and end
        # Give slightly more to the start for better context
        start_chars = (available_chars + 1) // 2
        end_chars = available_chars // 2

        return text[:start_chars] + ellipsis + text[-end_chars:]

    else:
        raise ValueError(f"Invalid truncation mode: {mode}")


def truncate_chapters_for_judging(
    chapters: list[str],
    max_chars_per_chapter: int,
    mode: Literal["end", "middle"] = "middle"
) -> list[str]:
    """
    Truncate a list of chapter texts for judging.

    Args:
        chapters: List of chapter text strings
        max_chars_per_chapter: Maximum characters per chapter
        mode: Truncation mode ("end" or "middle")

    Returns:
        List of truncated chapter texts
    """
    return [truncate_text(ch, max_chars_per_chapter, mode=mode) for ch in chapters]


def truncate_for_pairwise_comparison(
    text: str,
    max_chars: int = 4500,
    mode: Literal["end", "middle"] = "middle"
) -> str:
    """
    Truncate text for pairwise ELO comparison.

    Args:
        text: The full text to truncate
        max_chars: Maximum characters (default 4500 per ELO config)
        mode: Truncation mode (default "middle" for longform)

    Returns:
        Truncated text suitable for pairwise judging
    """
    return truncate_text(text, max_chars, mode=mode)


def combine_chapters_for_comparison(
    chapters: list[str],
    max_total_chars: int = 4500,
    mode: Literal["end", "middle"] = "middle"
) -> str:
    """
    Combine and truncate multiple chapters for pairwise comparison.

    This creates a single text block from all chapters, then truncates
    the combined result for pairwise judging.

    Args:
        chapters: List of chapter text strings
        max_total_chars: Maximum total characters for combined output
        mode: Truncation mode for the combined text

    Returns:
        Combined and truncated text with chapter markers
    """
    if not chapters:
        return ""

    # Combine chapters with markers
    parts = []
    for i, chapter in enumerate(chapters, 1):
        parts.append(f"# Chapter {i}\n\n{chapter}")

    combined = "\n\n---\n\n".join(parts)

    return truncate_text(combined, max_total_chars, mode=mode)
