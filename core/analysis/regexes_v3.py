# -*- coding: utf-8 -*-
"""Stage 1 surface-level contrast pattern regexes.

Ported from not-x-but-y-bench/src/regexes_v3.py
"""
import re

MAXG = 160

# ── helpers
PRON = r"(?:it|they|this|that)"
BE   = r"(?:is|are|was|were)"
BE_NEG = r"(?:is\s+not|are\s+not|was\s+not|were\s+not|isn't|aren't|wasn't|weren't|ain't)"

# ── 1) "not X, but Y"
RE_NOT_BUT = re.compile(rf"""
    \b(?:(?:{BE_NEG})|not(?!\s+(?:that|only)\b))\s+
    (?:(?!\bbut\b|[.?!]).){{1,100}}?
    [,;:]\s*but\s+
    (?!when\b|while\b|which\b|who\b|whom\b|whose\b|where\b|if\b|that\b|as\b|because\b|although\b|though\b|till\b|until\b|unless\b|
       here\b|there\b|then\b|my\b|we\b|I\b|you\b|it\s+seems\b|it\s+appears\b|it\s+felt\b|it\s+looks?\b|anything\b)
""", re.I | re.X)

# ── 2) Dash form "… not/n't … — pron + (BE or lexical) …"
RE_NOT_DASH = re.compile(rf"""
    \b(?:\w+n't|not)\s+(?:just|only|merely)?\s+
    (?:(?![.?!]).){{1,160}}?
    (?:-|\s-\s|[\u2014\u2013])\s*
    {PRON}\s+(?:(?:'re|are|'s|is|were|was)\b|(?!'re|are|'s|is|were|was)[*_~]*[a-z]\w*)
""", re.I | re.X)

# ── 3) Pronoun-led "It/They … not … . It/They BE …"
RE_PRON_BE_NOT_SEP_BE = re.compile(rf"""
    (?:(?<=^)|(?<=[.?!]\s))\s*[""']?
    (?:(?:{PRON}\s+{BE}\s+not)|(?:{PRON}\s+{BE}n't)|(?:it's|they're|that's)\s+not)\b
    [^.?!]{{0,{MAXG}}}[.;:?!]\s*[""']?
    {PRON}\s+(?:{BE}|(?:'s|'re))\b(?!\s+not\b)
""", re.I | re.X)

# ── 4) NP-led "… was/weren't not … . It/They BE …" with reporter-frame + "not put" guards
RE_NP_BE_NOT_SEP_THEY_BE = re.compile(rf"""
    (?:(?<=^)|(?<=[.?!]\s))\s*
    (?![^.?!]{{0,80}}\b(?:knew|know|thought|think|said|says|told|heard|learned)\b[^.?!]{{0,40}}?\bthat\b)
    (?!\s*not\s+without\b)
    (?![^.?!]{{0,50}}\bnot\s+put\b)
    [^.?!]{{0,{MAXG}}}?\b(?:{BE_NEG})\b[^.?!]{{0,{MAXG}}}[.;:?!]\s*
    [""']?{PRON}\b(?:'re|\s+(?:are|were|is|was))\b(?!\s+not\b)
""", re.I | re.X)

# ── 5) "no longer … ; it/they was …"
RE_NO_LONGER = re.compile(rf"""
    (?:(?<=^)|(?<=[.?!]\s))\s*[^.?!]{{0,{MAXG}}}\bno\s+longer\b[^.;:?!]{{0,{MAXG}}}
    [.;:?!]\s*(?:it|they|this|that)\s+(?:is|are|was|were)\b(?!\s+not\b)
""", re.I | re.X)

# ── 6) "not just … . It/They …"
RE_NOT_JUST_SEP = re.compile(rf"""
    (?:(?<=^)|(?<=[.?!]\s))\s*[""']?
    {PRON}\b(?:'s|'re|\s+(?:is|are|was|were))?\s+not\s+just\b[^.?!]{{0,{MAXG}}}[.?!]\s*[""']?
    {PRON}\b(?:'s|'re|\s+(?:is|are|was|were))\b(?!\s+not\b)
""", re.I | re.X)

# ── 7) Cross-sentence same-verb: "didn't V. It/They V…"
RE_NOT_PERIOD_SAMEVERB = re.compile(rf"""
    (?:(?<=^)|(?<=[.?!]\s))[^.?!]*?\b(?:do|does|did)n't\b\s+
    (?:(?:\w+\s+){{0,2}})([a-z]{{3,}})\b[^.?!]*[.?!]\s*
    {PRON}\s+\1(?:ed|es|s|ing)?\b
""", re.I | re.X)

# ── 8) Simple BE: "… isn't/wasn't … . It's/It is …" (+ reporter-frame guard)
RE_SIMPLE_BE_NOT_IT_BE = re.compile(rf"""
    (?:(?<=^)|(?<=[.?!]\s))\s*[""']?
    (?!he\b|she\b|i\b|you\b|we\b)
    (?![^.?!]{{0,80}}\b(?:knew|know|thought|think|said|says|told|heard|learned)\b[^.?!]{{0,40}}?\bthat\b)
    [^.?!]{{0,{MAXG}}}?\b{BE_NEG}\b[^.?!]{{0,{MAXG}}}[.;:?!]\s*
    [""']?it(?:'s|\s+(?:is|are|was|were))\b
""", re.I | re.X)

# ── 9) Embedded "not just … ; It/They …" (allows a lead-in like "That means …")
RE_EMBEDDED_NOT_JUST_SEP = re.compile(rf"""
    (?:(?<=^)|(?<=[.?!]\s))
    [^.?!]{{0,80}}?\b(?:(?:it|they)\s+(?:is|are)|(?:it's|they're))\s+not\s+just\b
    [^.?!]{{0,{MAXG}}}[.?!]\s*
    (?:(?:it|they)\s+(?:is|are)|(?:it's|they're))\b
""", re.I | re.X)

# ── 10) Dialogue-aware: "You're not just X," <said Y>. "You're Z."
RE_DIALOGUE_NOT_JUST = re.compile(rf"""
    [""']?{PRON}(?:'re|'s|\s+(?:are|is|was|were))\s+not\s+just\b[^""']{{0,{MAXG}}}[""']?\s*
    (?:[^.?!]{{0,80}}\b(?:said|asked|whispered|muttered|replied|added|shouted|cried)\b[^.?!]{{0,80}}[.?!]\s*)?
    [""']?{PRON}(?:'re|'s|\s+(?:are|is|was|were))\s+[*_~]?[a-z]\w*
""", re.I | re.X)

compiled = {
    "RE_NOT_BUT": RE_NOT_BUT,
    "RE_NOT_DASH": RE_NOT_DASH,
    "RE_PRON_BE_NOT_SEP_BE": RE_PRON_BE_NOT_SEP_BE,
    "RE_NP_BE_NOT_SEP_THEY_BE": RE_NP_BE_NOT_SEP_THEY_BE,
    "RE_NO_LONGER": RE_NO_LONGER,
    "RE_NOT_JUST_SEP": RE_NOT_JUST_SEP,
    "RE_NOT_PERIOD_SAMEVERB": RE_NOT_PERIOD_SAMEVERB,
    "RE_SIMPLE_BE_NOT_IT_BE": RE_SIMPLE_BE_NOT_IT_BE,
    "RE_EMBEDDED_NOT_JUST_SEP": RE_EMBEDDED_NOT_JUST_SEP,
    "RE_DIALOGUE_NOT_JUST": RE_DIALOGUE_NOT_JUST,
}