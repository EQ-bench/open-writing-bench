"""Lexical analysis module for creative writing benchmark.

Provides metrics for analyzing generated text including:
- Slop word hits per 1k words
- Slop trigram hits per 1k words
- "Not X but Y" contrast pattern rate per 1k chars
- Combined slop score (0-1)
- Vocabulary level (using wordfreq zipf scores)
- Sentence length statistics
- Paragraph length statistics
- MATTR-500 (Moving Average Type-Token Ratio)
- Average turn length (for multi-turn responses)
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from wordfreq import zipf_frequency

from .slop_analysis_shared import (
    words_only_lower,
    compute_slop_word_hits,
    compute_slop_bigram_hits,
    compute_slop_trigram_hits,
    compute_contrast_score,
    compute_slop_score,
)


class LexicalAnalysis(TypedDict):
    """Result of lexical analysis on text."""
    # Slop metrics
    slop_words_per_1k: float
    slop_trigrams_per_1k: float
    not_x_but_y_per_1k_chars: float
    slop_score: float

    # Vocabulary metrics
    vocab_level: float  # Average zipf score (higher = more common words)

    # Length metrics
    avg_sentence_length: float  # words per sentence
    avg_paragraph_length: float  # words per paragraph

    # Lexical diversity
    mattr_500: float  # Moving Average Type-Token Ratio with window 500

    # Turn metrics (for multi-turn only)
    avg_turn_length: float | None  # chars per turn
    num_turns: int | None

    # Counts
    total_words: int
    total_chars: int


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex."""
    # Split on sentence-ending punctuation followed by whitespace or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs (separated by blank lines)."""
    # Split on 2+ newlines
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def compute_vocab_level(tokens: list[str]) -> float:
    """
    Compute average vocabulary level using wordfreq zipf scores.

    Higher scores (1-7) = more common words
    Lower scores (0-2) = rarer words

    Returns average zipf score across all tokens.
    """
    if not tokens:
        return 0.0

    scores = []
    for token in tokens:
        # Get zipf frequency (returns 0.0 for unknown words)
        score = zipf_frequency(token, 'en')
        if score > 0:
            scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def compute_avg_sentence_length(text: str) -> float:
    """Compute average sentence length in words."""
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0

    word_counts = []
    for sentence in sentences:
        words = words_only_lower(sentence)
        word_counts.append(len(words))

    return sum(word_counts) / len(word_counts) if word_counts else 0.0


def compute_avg_paragraph_length(text: str) -> float:
    """Compute average paragraph length in words."""
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return 0.0

    word_counts = []
    for para in paragraphs:
        words = para.split()
        word_counts.append(len(words))

    return sum(word_counts) / len(word_counts) if word_counts else 0.0


def compute_mattr(tokens: list[str], window_size: int = 500) -> float:
    """
    Compute Moving Average Type-Token Ratio (MATTR).

    MATTR is less sensitive to text length than simple TTR.
    Uses a sliding window to compute local TTRs and averages them.

    Args:
        tokens: List of lowercase word tokens
        window_size: Size of sliding window (default 500)

    Returns:
        MATTR score between 0 and 1 (higher = more lexical diversity)
    """
    n = len(tokens)
    if n == 0:
        return 0.0

    # If text is shorter than window, compute simple TTR
    if n <= window_size:
        return len(set(tokens)) / n

    # Compute TTR for each window position
    ttrs = []
    for i in range(n - window_size + 1):
        window = tokens[i:i + window_size]
        ttr = len(set(window)) / window_size
        ttrs.append(ttr)

    return sum(ttrs) / len(ttrs)


def compute_avg_turn_length(turns: list[dict[str, Any]]) -> tuple[float, int]:
    """
    Compute average turn length for multi-turn responses.

    Only counts chapter turns, not planning.

    Args:
        turns: List of turn dictionaries from model_responses

    Returns:
        Tuple of (avg chars per turn, number of turns)
    """
    chapter_lengths = []
    for turn in turns:
        if turn.get("turn_type") == "chapter" and turn.get("assistant_response"):
            chapter_lengths.append(len(turn["assistant_response"]))

    if not chapter_lengths:
        return 0.0, 0

    return sum(chapter_lengths) / len(chapter_lengths), len(chapter_lengths)


def analyze_text(
    text: str,
    turns: list[dict[str, Any]] | None = None
) -> LexicalAnalysis:
    """
    Perform full lexical analysis on text.

    Args:
        text: The text to analyze (for multi-turn, concatenated chapters only)
        turns: Optional list of turn dictionaries for turn-level metrics

    Returns:
        LexicalAnalysis dictionary with all metrics
    """
    # Tokenize
    tokens = words_only_lower(text)

    # Slop metrics
    slop_words_per_1k, _ = compute_slop_word_hits(tokens)
    slop_trigrams_per_1k, _ = compute_slop_trigram_hits(tokens)
    not_x_but_y_per_1k, _ = compute_contrast_score(text)
    slop_score = compute_slop_score(slop_words_per_1k, slop_trigrams_per_1k, not_x_but_y_per_1k)

    # Vocabulary level
    vocab_level = compute_vocab_level(tokens)

    # Length metrics
    avg_sentence_length = compute_avg_sentence_length(text)
    avg_paragraph_length = compute_avg_paragraph_length(text)

    # Lexical diversity
    mattr = compute_mattr(tokens, window_size=500)

    # Turn metrics
    avg_turn_length = None
    num_turns = None
    if turns:
        avg_turn_length, num_turns = compute_avg_turn_length(turns)

    return LexicalAnalysis(
        slop_words_per_1k=round(slop_words_per_1k, 2),
        slop_trigrams_per_1k=round(slop_trigrams_per_1k, 3),
        not_x_but_y_per_1k_chars=round(not_x_but_y_per_1k, 3),
        slop_score=round(slop_score, 3),
        vocab_level=round(vocab_level, 2),
        avg_sentence_length=round(avg_sentence_length, 1),
        avg_paragraph_length=round(avg_paragraph_length, 1),
        mattr_500=round(mattr, 3),
        avg_turn_length=round(avg_turn_length, 0) if avg_turn_length else None,
        num_turns=num_turns,
        total_words=len(tokens),
        total_chars=len(text),
    )


def get_text_for_analysis(task) -> tuple[str, list[dict[str, Any]] | None]:
    """
    Extract text for lexical analysis from a task.

    For multi-turn tasks: concatenates all chapter responses (excludes planning)
    For single-turn tasks: uses model_response

    Args:
        task: Task database object

    Returns:
        Tuple of (text to analyze, turns list or None)
    """
    # Check for multi-turn
    if task.model_responses and len(task.model_responses) > 0:
        chapters = []
        for turn in task.model_responses:
            if turn.get("turn_type") == "chapter" and turn.get("assistant_response"):
                chapters.append(turn["assistant_response"])

        if chapters:
            text = "\n\n".join(chapters)
            return text, task.model_responses

    # Single-turn fallback
    if task.model_response:
        return task.model_response, None

    return "", None


def analyze_task(task) -> LexicalAnalysis | None:
    """
    Analyze a task's generated text.

    Args:
        task: Task database object

    Returns:
        LexicalAnalysis dict or None if no text available
    """
    text, turns = get_text_for_analysis(task)
    if not text:
        return None

    return analyze_text(text, turns)


def format_analysis_summary(analysis: LexicalAnalysis) -> str:
    """
    Format lexical analysis results for display.

    Args:
        analysis: LexicalAnalysis dictionary

    Returns:
        Formatted string for console output
    """
    lines = [
        "Lexical Analysis:",
        f"  Slop Score: {analysis['slop_score']:.3f}",
        f"    - Words/1k: {analysis['slop_words_per_1k']:.1f}",
        f"    - Trigrams/1k: {analysis['slop_trigrams_per_1k']:.3f}",
        f"    - Not-X-But-Y/1k chars: {analysis['not_x_but_y_per_1k_chars']:.3f}",
        f"  Vocab Level (zipf): {analysis['vocab_level']:.2f}",
        f"  Avg Sentence Length: {analysis['avg_sentence_length']:.1f} words",
        f"  Avg Paragraph Length: {analysis['avg_paragraph_length']:.1f} words",
        f"  MATTR-500: {analysis['mattr_500']:.3f}",
    ]

    if analysis['avg_turn_length'] is not None:
        lines.append(f"  Avg Turn Length: {analysis['avg_turn_length']:.0f} chars ({analysis['num_turns']} turns)")

    lines.append(f"  Total: {analysis['total_words']} words, {analysis['total_chars']} chars")

    return "\n".join(lines)


def aggregate_analyses(analyses: list[LexicalAnalysis]) -> LexicalAnalysis:
    """
    Aggregate multiple lexical analyses into a single summary.

    Computes means across all analyses.

    Args:
        analyses: List of LexicalAnalysis dictionaries

    Returns:
        Aggregated LexicalAnalysis
    """
    if not analyses:
        return LexicalAnalysis(
            slop_words_per_1k=0.0,
            slop_trigrams_per_1k=0.0,
            not_x_but_y_per_1k_chars=0.0,
            slop_score=0.0,
            vocab_level=0.0,
            avg_sentence_length=0.0,
            avg_paragraph_length=0.0,
            mattr_500=0.0,
            avg_turn_length=None,
            num_turns=None,
            total_words=0,
            total_chars=0,
        )

    n = len(analyses)

    # Average numeric fields
    def avg(key: str) -> float:
        vals = [a[key] for a in analyses if a[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    # Sum count fields
    def total(key: str) -> int:
        return sum(a[key] for a in analyses if a[key] is not None)

    # Average turn length only if available
    turn_lengths = [a['avg_turn_length'] for a in analyses if a['avg_turn_length'] is not None]
    avg_turn = sum(turn_lengths) / len(turn_lengths) if turn_lengths else None

    turn_counts = [a['num_turns'] for a in analyses if a['num_turns'] is not None]
    total_turns = sum(turn_counts) if turn_counts else None

    return LexicalAnalysis(
        slop_words_per_1k=round(avg('slop_words_per_1k'), 2),
        slop_trigrams_per_1k=round(avg('slop_trigrams_per_1k'), 3),
        not_x_but_y_per_1k_chars=round(avg('not_x_but_y_per_1k_chars'), 3),
        slop_score=round(avg('slop_score'), 3),
        vocab_level=round(avg('vocab_level'), 2),
        avg_sentence_length=round(avg('avg_sentence_length'), 1),
        avg_paragraph_length=round(avg('avg_paragraph_length'), 1),
        mattr_500=round(avg('mattr_500'), 3),
        avg_turn_length=round(avg_turn, 0) if avg_turn else None,
        num_turns=total_turns,
        total_words=total('total_words'),
        total_chars=total('total_chars'),
    )
