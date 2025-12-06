#!/usr/bin/env python3
"""
POS-based regex patterns for catching abstract contrast structures.
These run on POS-tagged text where verbs are replaced with VERB, etc.

Ported from not-x-but-y-bench/src/regexes_pos.py
"""

import re

###############################################################################
# POS-BASED PATTERNS (run on tagged text)
###############################################################################

###############################################################################
# 1. "doesn't VERB. It VERB" - catches verb contrasts regardless of specific verb
###############################################################################
RE_POS_DOESNT_VERB = re.compile(r"""
    ["'"]\s*
    (?:[Tt]he\s+\w+|[Ii]t|[Tt]hey|[Yy]ou)\s+
    doesn't\s+VERB
    [^.!?]*?
    [.!?]\s*
    (?:it|they|you|that)\s+
    [*_~]?(?:VERB|whispers?|reminds?|signals?|tests?|speaks?)  # POS or specific verb
""", re.I | re.X)

###############################################################################
# 2. "don't just VERB. They VERB" - gerund or verb contrast
###############################################################################
RE_POS_DONT_JUST_VERB = re.compile(r"""
    ["'"]\s*
    (?:[Tt]hey|[Yy]ou|[Ii]t)\s+
    don't\s+just\s+VERB
    [^.!?]*?
    [—-]\s*
    they\s+[*_~]?VERB
""", re.I | re.X)

###############################################################################
# 3. "not just VERBing. VERBing" - gerund fragment
###############################################################################
RE_POS_GERUND_FRAGMENT = re.compile(r"""
    ["'"]\s*
    Not\s+just\s+VERB
    [.!?]\s+
    [*_~]?VERB
    [.!?]
""", re.X)

###############################################################################
# 4. "not ADJ. They were ADJ" (requires adj tagging)
###############################################################################
RE_POS_NOT_ADJ = re.compile(r"""
    \bnot\s+
    (random|passive|simple|normal)                     # Common adjectives
    [.!?;]\s+
    (?:[Tt]hey|[Ii]t)\s+(?:were|was|are|is|'re|'s)\s+
    [*_~]?(intentional|active|complex|different|\w{8,})  # Contrasting adj
""", re.I | re.X)

###############################################################################
# 5. Broader dash pattern with VERB
###############################################################################
RE_POS_DASH_VERB = re.compile(r"""
    \b(?:wasn't|weren't|isn't|aren't)\s+
    just\s+
    (?:VERB|a\s+\w+)
    [^-]{0,30}?
    -\s*
    (?:it|they)\s+
    (?:was|were|is|are|'s|'re)\s+
    [*_~]?(?:VERB|a\s+[*_~]?\w+)
""", re.I | re.X)

###############################################################################
# 6. "not just VERB. It was VERB" - past tense verb contrast
###############################################################################
RE_POS_NOT_JUST_VERB_PAST = re.compile(r"""
    \b(?:was|were)\s+
    not\s+just\s+
    (?:VERB|a\s+\w+)
    [.!?]\s+
    (?:[Ii]t|[Tt]hey)\s+
    (?:was|were)\s+
    [*_~]?(?:VERB|a\s+[*_~]?\w+)
""", re.I | re.X)

###############################################################################
# 7. Colon separator pattern
###############################################################################
RE_POS_COLON_VERB = re.compile(r"""
    :\s+
    (?:the\s+\w+|it|they)\s+
    (?:was|were)\s+
    not\s+just\s+
    VERB
    [.!?]\s+
    (?:[Ii]t|[Tt]hey)\s+
    (?:was|were)\s+
    [*_~]?VERB
""", re.I | re.X)

###############################################################################
# LEMMA-BASED PATTERNS (for same-verb contrasts)
###############################################################################

###############################################################################
# 8. Same verb lemma in different forms
###############################################################################
RE_LEMMA_SAME_VERB = re.compile(r"""
    \b(REACT|SPEAK|LISTEN|LEARN|SIGNAL|WARN|DIE|LIVE|TEST|TEACH|AMPLIFY|INTERPRET|TRANSLATE|DECODE|EMIT)\b
    [^.!?]{5,80}?
    [.!?;—-]\s*
    [^.!?]{0,40}?
    \1                                                 # Same lemma appears again
""", re.I | re.X)

###############################################################################
# 9. Broader "isn't just VERB" within quotes
###############################################################################
RE_POS_ISNT_JUST_VERB = re.compile(r"""
    ["'"]\s*
    (?:[^"'"]{0,100}?\b)?                             # Optional prefix
    (?:The\s+\w+|It|They|You)\s+
    (?:isn't|aren't|wasn't|weren't)\s+just\s+
    VERB
    [^"'".!?]{0,40}?
    [—-]\s*
    (?:it's|they're)\s+
    [*_~]?VERB
""", re.I | re.X)

###############################################################################
# 10. Complex multi-sentence in quotes with VERB
###############################################################################
RE_POS_QUOTE_MULTI_VERB = re.compile(r"""
    ["'"]\s*
    [^"'"]{0,150}?\b
    (?:not\s+just|isn't|aren't)\s+
    (?:VERB|a\s+\w+)
    [^"'".!?]{0,60}?
    [.!?]\s+
    (?:[^"'"]{0,40}?\b)?                               # Optional mid-sentence text
    (?:It's|They're|You're|That's)\s+
    [*_~]?(?:VERB|a\s+[*_~]?\w+)
""", re.I | re.X)

###############################################################################
# 11. Ellipsis with VERB
###############################################################################
RE_POS_ELLIPSIS_VERB = re.compile(r"""
    ["'"]\s*
    [^"'"]{0,100}?\b
    (?:not\s+just|isn't)\s+
    VERB
    [^"'"]{0,30}?
    [.…]\s*[.…]\s*
    (?:they're|it's|you're)\s+
    [*_~]?VERB
""", re.I | re.X)

###############################################################################
# 12. "not NOUN. It's/That's a NOUN" pattern
###############################################################################
RE_POS_NOT_NOUN = re.compile(r"""
    ["'"]\s*
    (?:That's|It's)\s+
    not\s+
    (?:a\s+)?(sign|message|warning|pattern|test|phenomenon|one\s+\w+)
    [.!?]\s+
    (?:That's|It's)\s+
    (?:a\s+|\*?all\s+)?[*_~]?(warning|question|language|symbol|test|presence|story|challenge|\w+)
""", re.I | re.X)

###############################################################################
# 13. "doesn't VERB. It *VERB" with emphasis markers
###############################################################################
RE_POS_DOESNT_VERB_EMPHASIS = re.compile(r"""
    ["'"]\s*
    (?:The\s+\w+|It|They)\s+
    doesn't\s+
    (?:VERB|react|warn|speak)
    [.!?]\s+
    It\s+
    \*(?:VERB|whispers?|reminds?|signals?)
""", re.I | re.X)

###############################################################################
# 14. Better dash patterns with VERB
###############################################################################
RE_POS_DASH_VERB_BROAD = re.compile(r"""
    \b(?:wasn't|weren't|isn't|aren't|don't|doesn't)\s+
    just\s+
    (?:VERB|(?:the|a)\s+\w+)
    [^-]{0,40}?
    -\s*
    (?:it|they)\s+
    (?:was|were|is|are|'s|'re)?\s*
    [*_~]?(?:VERB|(?:the|a)\s+[*_~]?\w+)
""", re.I | re.X)

###############################################################################
# 15. Ellipsis patterns - broader
###############################################################################
RE_POS_ELLIPSIS_BROAD = re.compile(r"""
    ["'"]\s*
    (?:[^"'"]{0,100}?\b)?
    (?:They're|You're|This)\s+
    (?:not\s+just|isn't)\s+
    (?:VERB|a\s+\w+)
    [^"'"]{0,40}?
    [.…]\s*[.…]\s*
    (?:they're|it's|you're|this)\s+
    (?:VERB|a\s+\w+)
""", re.I | re.X)

###############################################################################
# 16. "not because X. It's because Y"
###############################################################################
RE_POS_NOT_BECAUSE = re.compile(r"""
    \bit's\s+not\s+because\s+
    [^.!?]{5,60}?
    [.!?]\s+
    (?:It's|That's)\s+because\s+
    [^.!?]{5,60}
""", re.I | re.X)

###############################################################################
# 17. Fragment with gerunds "Not just VERBing. *VERBing"
###############################################################################
RE_POS_GERUND_BROAD = re.compile(r"""
    ["'"]\s*
    Not\s+just\s+
    VERB
    [.!?]\s+
    \*VERB
    [.!?]?
""", re.X)

###############################################################################
# 18. Complex quoted multi-sentence with VERBing
###############################################################################
RE_POS_QUOTE_VERBING = re.compile(r"""
    ["'"]\s*
    (?:You're|They're|It's)\s+
    not\s+
    (?:just\s+)?
    VERB
    [^"'".!?]{0,30}?
    [.,]\s+
    [^"'"]{0,50}?
    (?:You're|They're|It's)\s+
    (?:VERB|waiting)
""", re.I | re.X)

###############################################################################
# 19. "doesn't verb. It *verb*" where verb isn't tagged (emphasis keeps it literal)
###############################################################################
RE_POS_DOESNT_LITERAL = re.compile(r"""
    ["'"]\s*
    (?:The\s+\w+|It|They)\s+
    doesn't\s+
    (?:VERB|react|warn|speak|listen)\s*
    [.!?]\s+
    It\s+
    \*\w+\*
""", re.I | re.X)

###############################################################################
# 20. "not just a NOUN—it was a *NOUN*" (dash with noun swap)
###############################################################################
RE_POS_DASH_NOUN_SWAP = re.compile(r"""
    \b(?:was|were|is|are)\s+
    not\s+just\s+
    a\s+\w+
    [^-]{0,10}?
    -\s*
    (?:it|they)\s+
    (?:was|were|is|are)\s+
    (?:a\s+)?\*\w+\*
""", re.I | re.X)

###############################################################################
# 21. "isn't just VERB/noun—it's VERBing/noun" (broader dash with emphasis)
###############################################################################
RE_POS_ISNT_DASH_EMPHASIS = re.compile(r"""
    ["'"]\s*
    (?:The\s+\w+|It|They)\s+
    (?:isn't|aren't|wasn't|weren't)\s+
    just\s+
    (?:VERB|a\s+\w+)
    [^-]{0,40}?
    -\s*
    (?:it's|they're)\s+
    \*\w+\*
""", re.I | re.X)

###############################################################################
# 22. "That's not a NOUN. That's a *NOUN*" (simple noun swap)
###############################################################################
RE_POS_THATS_NOT_NOUN = re.compile(r"""
    ["'"]\s*
    That's\s+not\s+
    (?:a\s+)?(?:sign|message|pattern|phenomenon|test|one\s+\w+|\w+)
    [.!?]\s+
    (?:That's|It's)\s+
    (?:a\s+)?\*\w+\*
""", re.I | re.X)

###############################################################################
# 23. "not just VERB. *VERBing*" (gerund fragment with emphasis)
###############################################################################
RE_POS_GERUND_EMPHASIS = re.compile(r"""
    ["'"]\s*
    Not\s+just\s+
    (?:VERB|reacting|dying|\w+ing)
    [.!?]\s+
    \*[A-Z]\w+\*
""", re.X)

###############################################################################
# 24. "are not just VERBing. They're VERBing" (with dialogue attribution)
###############################################################################
RE_POS_QUOTE_ATTRIBUTION_VERB = re.compile(r"""
    ["'"]\s*
    (?:The\s+\w+|They)\s+
    (?:are|were|'re)\s+
    not\s+just\s+
    VERB
    ,"\s+
    [^"'"]{0,30}?
    \.\s+
    "They're\s+
    \*?VERB
""", re.I | re.X)

###############################################################################
# 25. "isn't just a NOUN. It's a *NOUN*" (quoted noun swap)
###############################################################################
RE_POS_ISNT_NOUN = re.compile(r"""
    ["'"]\s*
    (?:This|That|It)\s+
    isn't\s+just\s+
    a\s+\w+
    [.!?]\s+
    It's\s+
    (?:a\s+)?\*\w+\*
""", re.I | re.X)

###############################################################################
# 26. "It's not just NOUN. It's *NOUN*" (quoted, simple structure)
###############################################################################
RE_POS_ITS_NOT_JUST = re.compile(r"""
    ["'"]\s*
    It's\s+not\s+just\s+
    (?:one\s+)?(\w+)
    [.!?]\s+
    It's\s+
    \*(?:all|every|each|\w+)\*
""", re.I | re.X)

###############################################################################
# 27. "They're not just VERBing X—they're *VERBing*" (dash, gerund with object)
###############################################################################
RE_POS_DASH_GERUND_OBJ = re.compile(r"""
    ["'"]\s*
    (?:They're|You're|It's)\s+
    not\s+just\s+
    (?:VERB|emitting|dying|\w+ing)\s+
    (?:a|an|the)\s+\w+
    [^-]{0,10}?
    -\s*
    (?:they're|you're|it's)\s+
    \*\w+\*
""", re.I | re.X)

###############################################################################
# 28. Ellipsis with dialogue attribution "not just VERBing," X said. "VERBing"
###############################################################################
RE_POS_ELLIPSIS_DIALOGUE = re.compile(r"""
    ["'"]\s*
    (?:They're|You're|It's)\s+
    not\s+just\s+
    VERB
    ,"\s+
    [^"'"]{5,40}?\.\s+
    "(?:They're|You're|It's)[…\s]+
    (?:VERB|\w+ing)
""", re.I | re.X)

###############################################################################
# 29. "were not just NOUN; they were NOUN" (semicolon with noun)
###############################################################################
RE_POS_SEMI_NOUN = re.compile(r"""
    \b(?:were|was|are|is)\s+
    not\s+just\s+
    (?:folklore|\w+)
    ;\s+
    (?:they|it)\s+
    (?:were|was|are|is)\s+
    a\s+\w+
""", re.I | re.X)

###############################################################################
# 30. "isn't just a NOUN. It's a *NOUN*" (with "natural" or other adj)
###############################################################################
RE_POS_ISNT_ADJ_NOUN = re.compile(r"""
    ["'"]\s*
    (?:[^"'"]{0,30}?\b)?
    (?:this|that|it)\s+
    isn't\s+just\s+
    a\s+
    (?:natural\s+)?\w+
    [.!?]\s+
    It's\s+
    (?:a\s+)?\*\w+\*
""", re.I | re.X)

###############################################################################
# 31. Dialogue attribution: "not just X," he said. "Y" (various forms)
###############################################################################
RE_POS_DIALOGUE_ATTR = re.compile(r"""
    ["'"]\s*
    (?:You're|They're|It's|The\s+\w+)\s+
    (?:(?:are|is|'re|'s)\s+)?
    not\s+just\s+
    (?:VERB(?:\s+\w+)?|a\s+\w+)  # Allow VERB with optional object
    ,"\s+
    [^"'"]{3,50}?\.\s+
    "(?:You're|They're|It's)\s+
    (?:a\s+)?\*\w+\*  # Allow optional "a" before emphasized word
""", re.I | re.X)

###############################################################################
# 32. "To VERB that X isn't just Y. It's *Z*"
###############################################################################
RE_POS_TO_VERB_ISNT = re.compile(r"""
    ["'"]\s*
    To\s+VERB\s+
    (?:that\s+)?
    [^"'"]{5,50}?
    isn't\s+just\s+
    a\s+\w+
    [.!?]\s+
    It's\s+
    (?:a\s+)?\*\w+\*
""", re.I | re.X)

###############################################################################
# 33. "I am not VERBing X; it is Y" - first-person contrast with semicolon
###############################################################################
RE_POS_I_AM_NOT_SEMI = re.compile(r"""
    \bI\s+am\s+not\s+
    VERB
    [^;]{5,80}?
    ;\s*
    it\s+is\b
""", re.I | re.X)

###############################################################################
# 34. "It's not NAME anymore. It's NAME" - name/identity swap with anymore
###############################################################################
RE_POS_NOT_ANYMORE_ITS = re.compile(r"""
    \bIt's\s+not\s+
    [A-Z]\w+
    \s+anymore
    [.!?]\s+
    It's\s+
    [A-Z]\w+
""", re.X)

###############################################################################
# 35. "That/This ain't X. They/It Y" - simple ain't contrast
###############################################################################
RE_POS_AINT_SIMPLE = re.compile(r"""
    \b(?:That|This)\s+
    ain't\s+
    [^.!?]{3,40}?
    [.!?]\s+
    (?:They|It)\s+
    \w+
""", re.I | re.X)