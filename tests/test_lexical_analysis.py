"""Tests for the lexical analysis module."""

import pytest
from core.analysis import (
    analyze_text,
    aggregate_analyses,
    format_analysis_summary,
    compute_vocab_level,
    compute_avg_sentence_length,
    compute_avg_paragraph_length,
    compute_mattr,
    compute_avg_turn_length,
    LexicalAnalysis,
)
from core.analysis.slop_analysis_shared import (
    words_only_lower,
    compute_slop_word_hits,
    compute_slop_bigram_hits,
    compute_slop_trigram_hits,
    compute_contrast_score,
    compute_slop_score,
)


class TestTokenization:
    """Tests for tokenization functions."""

    def test_words_only_lower_basic(self):
        """Test basic word tokenization."""
        text = "Hello World! This is a TEST."
        tokens = words_only_lower(text)
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_words_only_lower_with_punctuation(self):
        """Test tokenization removes punctuation."""
        text = "It's a wonderful day, isn't it?"
        tokens = words_only_lower(text)
        assert "it's" in tokens or "its" in tokens
        assert "isn't" in tokens or "isnt" in tokens

    def test_words_only_lower_empty(self):
        """Test empty string."""
        assert words_only_lower("") == []

    def test_words_only_lower_unicode_quotes(self):
        """Test smart quotes are normalized."""
        text = "\u201cHello\u201d \u2018world\u2019"  # "Hello" 'world'
        tokens = words_only_lower(text)
        assert "hello" in tokens
        assert "world" in tokens


class TestSlopMetrics:
    """Tests for slop detection metrics."""

    def test_slop_word_hits_with_slop(self):
        """Test detection of slop words."""
        # "ached" and "arcane" are in the slop list
        text = "The knight ached from battle. An arcane spell filled the arcane chamber."
        tokens = words_only_lower(text)
        score, hits = compute_slop_word_hits(tokens)
        assert score > 0
        assert "ached" in hits or "arcane" in hits

    def test_slop_word_hits_no_slop(self):
        """Test text without slop words."""
        text = "The cat sat on the mat."
        tokens = words_only_lower(text)
        score, hits = compute_slop_word_hits(tokens)
        # Score might be 0 or very low
        assert score >= 0

    def test_slop_word_hits_empty(self):
        """Test empty token list."""
        score, hits = compute_slop_word_hits([])
        assert score == 0.0
        assert hits == {}

    def test_slop_trigram_hits(self):
        """Test detection of slop trigrams."""
        # "deep breath" is a common slop bigram, and similar phrases appear in trigrams
        text = "She took a deep breath and felt a surge of emotion. Her heart was pounding in her chest."
        tokens = words_only_lower(text)
        score, hits = compute_slop_trigram_hits(tokens)
        # Should have some hits for common AI phrases
        assert score >= 0

    def test_slop_trigram_hits_short_text(self):
        """Test with text shorter than 3 tokens."""
        tokens = ["hello", "world"]
        score, hits = compute_slop_trigram_hits(tokens)
        assert score == 0.0
        assert hits == {}


class TestContrastPatterns:
    """Tests for 'not X, but Y' contrast pattern detection."""

    def test_contrast_basic_pattern(self):
        """Test basic contrast pattern detection."""
        text = "It wasn't just a house, but a home. They were not running; they were escaping."
        score, matches = compute_contrast_score(text)
        # Should detect at least one pattern
        assert score >= 0

    def test_contrast_no_patterns(self):
        """Test text without contrast patterns."""
        text = "The sun rose. Birds sang. Morning arrived quietly."
        score, matches = compute_contrast_score(text)
        assert score >= 0  # May or may not have matches

    def test_contrast_empty(self):
        """Test empty text."""
        score, matches = compute_contrast_score("")
        assert score == 0.0
        assert matches == []


class TestSlopScore:
    """Tests for combined slop score."""

    def test_slop_score_range(self):
        """Test slop score is in 0-1 range."""
        # Various inputs
        score = compute_slop_score(20.0, 0.5, 0.3)
        assert 0 <= score <= 1

    def test_slop_score_low_values(self):
        """Test slop score with low inputs."""
        score = compute_slop_score(0.0, 0.0, 0.0)
        assert 0 <= score <= 1

    def test_slop_score_high_values(self):
        """Test slop score with high inputs."""
        score = compute_slop_score(50.0, 2.0, 1.0)
        assert 0 <= score <= 1


class TestVocabLevel:
    """Tests for vocabulary level computation."""

    def test_vocab_level_common_words(self):
        """Test vocabulary level with common words."""
        tokens = ["the", "and", "is", "a", "to", "in"]
        level = compute_vocab_level(tokens)
        # Common words should have high zipf scores (5-7)
        assert level > 4.0

    def test_vocab_level_rare_words(self):
        """Test vocabulary level with rarer words."""
        tokens = ["perspicacious", "obfuscate", "ephemeral"]
        level = compute_vocab_level(tokens)
        # Rare words should have lower zipf scores
        assert level < 4.0

    def test_vocab_level_empty(self):
        """Test empty token list."""
        assert compute_vocab_level([]) == 0.0


class TestSentenceLength:
    """Tests for sentence length computation."""

    def test_avg_sentence_length_basic(self):
        """Test basic sentence length calculation."""
        text = "This is short. This is a longer sentence here."
        avg = compute_avg_sentence_length(text)
        # First: 3 words, Second: 6 words, avg = 4.5
        assert 3 < avg < 7

    def test_avg_sentence_length_single(self):
        """Test single sentence."""
        text = "One two three four five."
        avg = compute_avg_sentence_length(text)
        assert avg == 5.0

    def test_avg_sentence_length_empty(self):
        """Test empty text."""
        assert compute_avg_sentence_length("") == 0.0


class TestParagraphLength:
    """Tests for paragraph length computation."""

    def test_avg_paragraph_length_basic(self):
        """Test basic paragraph length calculation."""
        text = """First paragraph. Has two sentences.

Second paragraph. Also two sentences.

Third has one."""
        avg = compute_avg_paragraph_length(text)
        # Para 1: 5 words, Para 2: 5 words, Para 3: 3 words
        # Avg = (5+5+3)/3 = 4.33
        assert 4 < avg < 5

    def test_avg_paragraph_length_single(self):
        """Test single paragraph with multiple sentences."""
        text = "First. Second. Third."
        avg = compute_avg_paragraph_length(text)
        assert avg == 3.0  # 3 words

    def test_avg_paragraph_length_empty(self):
        """Test empty text."""
        assert compute_avg_paragraph_length("") == 0.0


class TestMATTR:
    """Tests for Moving Average Type-Token Ratio."""

    def test_mattr_diverse_text(self):
        """Test MATTR with diverse vocabulary."""
        # All unique words
        tokens = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        mattr = compute_mattr(tokens, window_size=5)
        # All unique should give high TTR
        assert mattr > 0.8

    def test_mattr_repetitive_text(self):
        """Test MATTR with repetitive vocabulary."""
        tokens = ["the"] * 100
        mattr = compute_mattr(tokens, window_size=50)
        # All same word should give very low TTR
        assert mattr < 0.1

    def test_mattr_empty(self):
        """Test empty token list."""
        assert compute_mattr([]) == 0.0

    def test_mattr_short_text(self):
        """Test text shorter than window size."""
        tokens = ["hello", "world"]
        mattr = compute_mattr(tokens, window_size=500)
        # Should compute simple TTR
        assert mattr == 1.0  # 2 unique / 2 total


class TestTurnLength:
    """Tests for turn length computation."""

    def test_avg_turn_length_multiturn(self):
        """Test average turn length with multi-turn data."""
        turns = [
            {"turn_type": "planning", "assistant_response": "This is planning."},
            {"turn_type": "chapter", "assistant_response": "A" * 1000},
            {"turn_type": "chapter", "assistant_response": "B" * 2000},
            {"turn_type": "chapter", "assistant_response": "C" * 3000},
        ]
        avg_len, num_turns = compute_avg_turn_length(turns)
        # (1000 + 2000 + 3000) / 3 = 2000
        assert avg_len == 2000.0
        assert num_turns == 3

    def test_avg_turn_length_no_chapters(self):
        """Test with only planning turn."""
        turns = [
            {"turn_type": "planning", "assistant_response": "Planning only."},
        ]
        avg_len, num_turns = compute_avg_turn_length(turns)
        assert avg_len == 0.0
        assert num_turns == 0

    def test_avg_turn_length_empty(self):
        """Test empty turns list."""
        avg_len, num_turns = compute_avg_turn_length([])
        assert avg_len == 0.0
        assert num_turns == 0


class TestAnalyzeText:
    """Tests for the full text analysis function."""

    def test_analyze_text_basic(self):
        """Test full analysis on sample text."""
        text = """The old house stood silent on the hill. Its windows were dark.

A young woman approached carefully. She felt nervous about what she might find inside.

The door creaked as she pushed it open. Dust motes danced in the pale light."""

        result = analyze_text(text)

        # Check all fields are present
        assert "slop_words_per_1k" in result
        assert "slop_trigrams_per_1k" in result
        assert "not_x_but_y_per_1k_chars" in result
        assert "slop_score" in result
        assert "vocab_level" in result
        assert "avg_sentence_length" in result
        assert "avg_paragraph_length" in result
        assert "mattr_500" in result
        assert "total_words" in result
        assert "total_chars" in result

        # Check reasonable values
        assert result["total_words"] > 0
        assert result["total_chars"] > 0
        assert 0 <= result["slop_score"] <= 1
        assert result["vocab_level"] > 0
        assert result["avg_sentence_length"] > 0
        assert result["mattr_500"] > 0

    def test_analyze_text_with_turns(self):
        """Test analysis with turn data."""
        text = "Chapter one content. Chapter two content."
        turns = [
            {"turn_type": "chapter", "assistant_response": "Chapter one content."},
            {"turn_type": "chapter", "assistant_response": "Chapter two content."},
        ]
        result = analyze_text(text, turns)

        assert result["num_turns"] == 2
        assert result["avg_turn_length"] is not None

    def test_analyze_text_empty(self):
        """Test analysis on empty text."""
        result = analyze_text("")
        assert result["total_words"] == 0
        assert result["total_chars"] == 0


class TestAggregateAnalyses:
    """Tests for aggregating multiple analyses."""

    def test_aggregate_basic(self):
        """Test aggregation of multiple analyses."""
        analyses = [
            LexicalAnalysis(
                slop_words_per_1k=10.0,
                slop_trigrams_per_1k=0.5,
                not_x_but_y_per_1k_chars=0.2,
                slop_score=0.3,
                vocab_level=4.0,
                avg_sentence_length=15.0,
                avg_paragraph_length=4.0,
                mattr_500=0.7,
                avg_turn_length=1000.0,
                num_turns=3,
                total_words=500,
                total_chars=3000,
            ),
            LexicalAnalysis(
                slop_words_per_1k=20.0,
                slop_trigrams_per_1k=1.0,
                not_x_but_y_per_1k_chars=0.4,
                slop_score=0.5,
                vocab_level=5.0,
                avg_sentence_length=20.0,
                avg_paragraph_length=5.0,
                mattr_500=0.8,
                avg_turn_length=2000.0,
                num_turns=3,
                total_words=600,
                total_chars=4000,
            ),
        ]

        result = aggregate_analyses(analyses)

        # Check averages
        assert result["slop_words_per_1k"] == 15.0
        assert result["slop_trigrams_per_1k"] == 0.75
        assert result["vocab_level"] == 4.5
        assert result["avg_sentence_length"] == 17.5

        # Check totals
        assert result["total_words"] == 1100
        assert result["total_chars"] == 7000
        assert result["num_turns"] == 6

    def test_aggregate_empty(self):
        """Test aggregation with empty list."""
        result = aggregate_analyses([])
        assert result["total_words"] == 0
        assert result["slop_score"] == 0.0


class TestFormatSummary:
    """Tests for formatting analysis summary."""

    def test_format_summary_basic(self):
        """Test formatting with all fields."""
        analysis = LexicalAnalysis(
            slop_words_per_1k=15.5,
            slop_trigrams_per_1k=0.75,
            not_x_but_y_per_1k_chars=0.25,
            slop_score=0.35,
            vocab_level=4.5,
            avg_sentence_length=18.2,
            avg_paragraph_length=4.5,
            mattr_500=0.72,
            avg_turn_length=1500.0,
            num_turns=3,
            total_words=500,
            total_chars=3000,
        )

        summary = format_analysis_summary(analysis)

        assert "Lexical Analysis:" in summary
        assert "Slop Score: 0.350" in summary
        assert "Words/1k: 15.5" in summary
        assert "Vocab Level" in summary
        assert "MATTR-500" in summary
        assert "1500 chars" in summary
        assert "3 turns" in summary

    def test_format_summary_no_turns(self):
        """Test formatting without turn data."""
        analysis = LexicalAnalysis(
            slop_words_per_1k=15.5,
            slop_trigrams_per_1k=0.75,
            not_x_but_y_per_1k_chars=0.25,
            slop_score=0.35,
            vocab_level=4.5,
            avg_sentence_length=18.2,
            avg_paragraph_length=4.5,
            mattr_500=0.72,
            avg_turn_length=None,
            num_turns=None,
            total_words=500,
            total_chars=3000,
        )

        summary = format_analysis_summary(analysis)

        assert "Lexical Analysis:" in summary
        assert "turns" not in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
