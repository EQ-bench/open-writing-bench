"""POS tagging utilities using spaCy.

Ported from not-x-but-y-bench/src/pos_tagger.py
"""
import re
from functools import lru_cache

import spacy

# Load once, keep only POS + sents for speed.
try:
    _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' is not installed.\n"
        "This model is REQUIRED for lexical analysis.\n\n"
        "To install it, run:\n"
        "  uv run python -m spacy download en_core_web_sm\n"
    )

if "senter" not in _NLP.pipe_names and "sentencizer" not in _NLP.pipe_names:
    _NLP.add_pipe("sentencizer")

def get_nlp():
    return _NLP

@lru_cache(maxsize=1024)
def _doc(text: str):
    return _NLP(text)



def tag_with_pos(text: str, pos_type: str = 'verb') -> str:
    if get_nlp() is None:
        return text
    doc = _doc(text)
    result = []
    for token in doc:
        if pos_type == 'verb' and token.pos_ == 'VERB':
            out = 'VERB'
        elif pos_type == 'noun' and token.pos_ in ('NOUN', 'PROPN'):
            out = 'NOUN'
        elif pos_type == 'adj' and token.pos_ == 'ADJ':
            out = 'ADJ'
        elif pos_type == 'adv' and token.pos_ == 'ADV':
            out = 'ADV'
        elif pos_type == 'all':
            if token.pos_ == 'VERB':
                out = 'VERB'
            elif token.pos_ in ('NOUN', 'PROPN'):
                out = 'NOUN'
            elif token.pos_ == 'ADJ':
                out = 'ADJ'
            elif token.pos_ == 'ADV':
                out = 'ADV'
            else:
                out = token.text
        else:
            out = token.text
        result.append(out)
        if token.whitespace_:
            result.append(token.whitespace_)
    return ''.join(result)




def tag_with_lemma(text: str, pos_types: list = ['VERB']) -> str:
    if get_nlp() is None:
        return text
    doc = _doc(text)
    result = []
    for token in doc:
        if token.pos_ in pos_types and token.lemma_:
            result.append(token.lemma_.upper())
        else:
            result.append(token.text)
        if token.whitespace_:
            result.append(token.whitespace_)
    return ''.join(result)




def create_pos_variants(text: str) -> dict:
    """
    Create multiple POS-tagged variants of text for different matching strategies.

    Returns dict with keys:
        - 'original': normalized text
        - 'verb_tagged': verbs replaced with VERB
        - 'verb_lemma': verbs replaced with their lemma
        - 'noun_adj_tagged': nouns and adjectives tagged
        - 'content_words': all content words (nouns, verbs, adjs, advs) tagged
    """
    variants = {
        'original': text,
        'verb_tagged': tag_with_pos(text, 'verb'),
        'verb_lemma': tag_with_lemma(text, ['VERB']),
        'noun_adj_tagged': text,  # Will implement if needed
        'content_words': text,     # Will implement if needed
    }

    return variants

def tag_doc(text: str):
    """Return cached spaCy Doc or None."""
    return _doc(text)

def sentences_pos_inline(text: str) -> list[str]:
    """
    Return per-sentence strings with tokens replaced by TOKEN/POS.
    Keeps whitespace between tokens inside each sentence.
    """
    doc = _doc(text)
    if doc is None:
        return [text]
    sents = []
    for sent in doc.sents:
        parts = []
        for tok in sent:
            parts.append(f"{tok.text}/{tok.pos_}")
            if tok.whitespace_:
                parts.append(tok.whitespace_)
        sents.append(''.join(parts))
    return sents


def tag_stream_with_offsets(text: str, pos_type: str = 'verb'):
    """
    Build a POS-tagged stream that preserves raw-character alignment via a piece map.
    Returns:
        stream: str
        pieces: list of tuples (stream_start, stream_end, raw_start, raw_end)
                for both tokens and their following whitespace.
    """
    doc = _doc(text)
    if doc is None:
        return text, [(0, len(text), 0, len(text))]

    parts = []
    pieces = []
    cur = 0

    def _emit(s, raw_start, raw_end):
        nonlocal cur
        parts.append(s)
        s_len = len(s)
        pieces.append((cur, cur + s_len, raw_start, raw_end))
        cur += s_len

    for tok in doc:
        # token piece
        if pos_type == 'verb' and tok.pos_ == 'VERB':
            out = 'VERB'
        elif pos_type == 'noun' and tok.pos_ in ('NOUN', 'PROPN'):
            out = 'NOUN'
        elif pos_type == 'adj' and tok.pos_ == 'ADJ':
            out = 'ADJ'
        elif pos_type == 'adv' and tok.pos_ == 'ADV':
            out = 'ADV'
        elif pos_type == 'all':
            if tok.pos_ == 'VERB':
                out = 'VERB'
            elif tok.pos_ in ('NOUN', 'PROPN'):
                out = 'NOUN'
            elif tok.pos_ == 'ADJ':
                out = 'ADJ'
            elif tok.pos_ == 'ADV':
                out = 'ADV'
            else:
                out = tok.text
        else:
            out = tok.text
        _emit(out, tok.idx, tok.idx + len(tok.text))

        # whitespace piece (keeps sentence boundaries and spacing)
        if tok.whitespace_:
            ws_start = tok.idx + len(tok.text)
            ws_end = ws_start + len(tok.whitespace_)
            _emit(tok.whitespace_, ws_start, ws_end)

    return ''.join(parts), pieces