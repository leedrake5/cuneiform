"""
text_pipeline.py - Text Processing Pipeline for Cuneiform Machine Translation

This module provides comprehensive text preprocessing functions for training
machine translation models on ancient cuneiform languages (Akkadian, Sumerian,
Elamite, Hittite, Linear B/Mycenaean Greek).

Main Components:
================

1. TEXT NORMALIZATION FUNCTIONS (lines ~1-700)
   - Unicode normalization, accent stripping, bracket removal
   - Gap/lacuna handling for damaged tablet text
   - Language-specific normalizations (CDLI, ORACC formats)
   - Vowel diacritic conversions (a2 -> á, u3 -> ù)

2. STRING NORMALIZATION FOR ML (lines ~700-1400)
   - normalizeString_* functions prepare text for model input
   - Handle different transliteration styles (simple, complex, grouped)
   - Add task prefixes for T5-style models
   - Support multiple target languages (English, German)

3. DATA LOADING FUNCTIONS (lines ~1400-5000)
   - readLangs* functions load parallel corpora from data files
   - Support for different model architectures (T5, NLLB)
   - Train/validation/test splitting
   - Multiple ancient language support

4. UTILITY FUNCTIONS (lines ~5000+)
   - CSV I/O for training data
   - Length filtering and trimming
   - Batch processing helpers
   - Character-level segmentation for ByT5

Supported Languages:
   - akk: Akkadian (Babylonian/Assyrian cuneiform)
   - sux: Sumerian
   - elx: Elamite
   - hit: Hittite
   - gmy: Mycenaean Greek (Linear B)

Data Sources:
   - CDLI (Cuneiform Digital Library Initiative)
   - ORACC (Open Richly Annotated Cuneiform Corpus)
   - OARE (Old Assyrian Research Environment)

Transliteration Styles:
   - simple: Basic Latin characters only (a-z), no diacritics
   - complex/original: Preserves scholarly diacritics (š, ṣ, ṭ, ḫ)
   - grouped: Syllables joined into word-like units
"""

import re, sys, os, random, glob, typing, numpy, gc, requests, unicodedata, logging, csv, regex
import pandas as pd
import numpy as np

# =============================================================================
# SECTION 1: BASIC TEXT NORMALIZATION UTILITIES
# =============================================================================

def unicodeToAscii(s):
    """
    Convert a Unicode string to plain ASCII by removing diacritical marks.

    Uses NFD normalization to decompose characters, then filters out
    combining marks (category 'Mn').

    Args:
        s: Input Unicode string

    Returns:
        ASCII-only string with accents/diacritics removed

    Example:
        >>> unicodeToAscii("šarrum")
        'sarrum'
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def strip_accents(text: str) -> str:
    """
    Remove diacritics and accents from Unicode text,
    reducing characters to simple Latin equivalents.
    """
    # Normalize to decomposed form (NFD)
    normalized = unicodedata.normalize("NFD", text)
    
    # Remove combining diacritical marks
    stripped = "".join(
        ch for ch in normalized
        if unicodedata.category(ch) != "Mn"
    )
    
    return stripped

def remove_doc_refs(text: str) -> str:
    """
    1) Remove any p<digits> tokens
    2) Remove any standalone runs of 6 or more digits (e.g. '014209')
    3) Collapse leftover whitespace
    """
    # 1) strip p<digits>
    cleaned = re.sub(r'\bp\d+\b', '', text)
    # 2) strip bare doc-IDs (6+ digits)
    cleaned = re.sub(r'\b\d{6,}\b', '', cleaned)
    # 3) collapse spaces
    return re.sub(r'\s{2,}', ' ', cleaned).strip()

REMOVE_BRACKETS_TRANS = str.maketrans({
    '(': ' ',
    ')': ' ',
    '〈': ' ',
    '〉': ' ',
    '˹': ' ',
    '˺': ' ',
    '⸢': ' ',
    '⸣': ' ',
    '⌞': ' ',
    '⌟': ' ',
    '˻': ' ',
    '˼': ' ',
    '˽': ' ',
    '𐄇': ' ',
    '[': ' ',
    ']': ' ',
    '|': ' ',
    '|': ' ',
    '{': ' ',
    '}': ' ',
    '»': ' ',
    '«': ' ',
    '<': ' ',
    '>': ' ',
    '-': ' ',
    '—': ' ',
    '~': ' ',
    '+': ' ',
    '.': ' ',
    '_': ' ',
    '/': ' ',
    ',': ' ',
    ':': ' ',
    '–': ' ',
    '&': ' ',
    '!': '',
    '?': '',
    '#': '',
    '=': '',
    '%': '',
    '$': '',
    '𐄼': ''
})

REMOVE_SOME_BRACKETS = str.maketrans({
    '(': ' ',
    ')': ' ',
    '〈': ' ',
    '〉': ' ',
    '˹': ' ',
    '˺': ' ',
    '⸢': ' ',
    '⸣': ' ',
    '⌞': ' ',
    '⌟': ' ',
    '˻': ' ',
    '˼': ' ',
    '˽': ' ',
    #'𐄇': ' ',
    '[': ' ',
    ']': ' ',
    '|': ' ',
    '|': ' ',
    '{': ' ',
    '}': ' ',
    '»': ' ',
    '«': ' ',
    '<': ' ',
    '>': ' ',
    '~': ' ',
    '+': ' ',
    '.': ' ',
    '_': ' ',
    '/': ' ',
    ',': ' ',
    ':': ' ',
    '&': ' ',
    '!': '',
    '?': '',
    '#': '',
    '=': '',
    '%': '',
    '$': '',
    '𐄼': ''
})

def remove_brackets(s: str) -> str:
    """
    Replace various bracket-like characters (and '-') with spaces.
    """
    return s.translate(REMOVE_BRACKETS_TRANS)

def remove_some_brackets(s: str) -> str:
    """
    Replace various bracket-like characters with spaces.
    """
    return s.translate(REMOVE_SOME_BRACKETS)


# =============================================================================
# Editorial Bracket Removal (preserves semantic determinatives)
# =============================================================================
# Editorial brackets indicate textual damage/uncertainty - remove for ML training
# BUT preserve curly braces {d}, {DUMU}, {ki} etc. which are semantic determinatives

REMOVE_EDITORIAL_BRACKETS = str.maketrans({
    # Square brackets - damaged/missing text
    '[': ' ',
    ']': ' ',
    # Parentheses - uncertain readings, glosses
    '(': ' ',
    ')': ' ',
    # Half brackets - partially preserved signs
    '⸢': ' ',
    '⸣': ' ',
    '˹': ' ',
    '˺': ' ',
    '⌞': ' ',
    '⌟': ' ',
    '˻': ' ',
    '˼': ' ',
    '˽': ' ',
    # Angle brackets - editorial additions/omissions
    '<': ' ',
    '>': ' ',
    '〈': ' ',
    '〉': ' ',
    '«': ' ',
    '»': ' ',
    # Other editorial marks
    '|': ' ',
    '#': '',   # damage indicator
    '!': '',   # collation mark
    '?': '',   # uncertainty
    '*': '',   # collation/correction
})

def remove_editorial_brackets(s: str) -> str:
    """
    Remove editorial markup brackets while preserving semantic determinatives.

    Removes:
        - [...] square brackets (damaged/missing text)
        - (...) parentheses (uncertain readings)
        - ⸢...⸣ half brackets (partially preserved)
        - <...> angle brackets (editorial additions)
        - Editorial marks: #, !, ?, *

    Preserves:
        - {...} curly braces (semantic determinatives like {d}, {DUMU}, {ki})

    Args:
        s: Input transliteration string

    Returns:
        String with editorial brackets removed, determinatives preserved
    """
    return s.translate(REMOVE_EDITORIAL_BRACKETS)


def normalize_determinatives(s: str) -> str:
    """
    Normalize determinative formatting for consistency.

    Ensures determinatives are formatted consistently:
        - Lowercase determinative content: {DUMU} -> {dumu}
        - Unify male determinatives: {1}, {m} -> {disz}
        - Preserve curly braces and original sign names

    The male determinative appears in different notations across data sources:
        - CDLI uses {disz} (the sign name)
        - Some sources use {1} (the numeric value)
        - Some sources use {m} (abbreviation for "male")
    We unify these to {disz} for consistency.

    Args:
        s: Input string with determinatives

    Returns:
        String with normalized determinatives (lowercased, male det unified)
    """
    def _normalize_det(m: re.Match) -> str:
        content = m.group(1).lower()
        # Unify male determinative variants to {disz}
        if content in ('1', 'm'):
            return '{disz}'
        return '{' + content + '}'

    return re.sub(r'\{([^}]+)\}', _normalize_det, s)


DASH_NORMALIZE = str.maketrans({
    '‐': '-',  # U+2010 hyphen
    '-': '-',  # U+2011 non-breaking hyphen
    '‒': '-',  # U+2012 figure dash
    '–': '-',  # U+2013 en dash
    '—': '-',  # U+2014 em dash
    '−': '-',  # U+2212 minus
})

def normalize_dashes(s: str) -> str:
    """
    Normalize various Unicode dash/hyphen characters to standard ASCII hyphen.

    Converts: hyphen (U+2010), non-breaking hyphen (U+2011), figure dash (U+2012),
    en dash (U+2013), em dash (U+2014), and minus sign (U+2212) all to '-'.

    Args:
        s: Input string containing various dash characters

    Returns:
        String with all dash variants normalized to ASCII hyphen '-'
    """
    return s.translate(DASH_NORMALIZE)



def normalize_digits(s: str) -> str:
    """
    Replace any subscript or superscript digits in the input string with their 
    regular ASCII equivalents. This will convert things like "SI₂₂" or "SI²²" 
    to "SI22".
    """
    # Automatically generate mapping for subscript digits (U+2080 to U+2089).
    subscript_map = {ord(chr(0x2080 + i)): str(i) for i in range(10)}
    # Manually specify the mapping for superscript digits.
    superscript_map = {
        ord('⁰'): '0',
        ord('¹'): '1',
        ord('²'): '2',
        ord('³'): '3',
        ord('⁴'): '4',
        ord('⁵'): '5',
        ord('⁶'): '6',
        ord('⁷'): '7',
        ord('⁸'): '8',
        ord('⁹'): '9',
    }
    # Combine both mappings into one translation table.
    translation_map = {**subscript_map, **superscript_map}
    # Translate the string.
    return s.translate(translation_map)

def strip_sign_indices(s: str) -> str:
    """
    Remove trailing numeric indices from transliterated syllables.

    In CDLI/ORACC notation, numbers after syllables indicate sign variants:
        - ba2, ba3, ba4 all represent the sound "ba" written with different signs
        - gu4 = "gu", he2 = "he", du3 = "du"
        - Single consonants also: b1, b2, b5 -> b

    This reduces the transliteration to pure phonetic sounds.

    Examples:
        strip_sign_indices("ba2-ri-gu4") -> "ba-ri-gu"
        strip_sign_indices("du3 he2 me3") -> "du he me"
        strip_sign_indices("b5 sze3") -> "b sze"
    """
    # Match syllables followed by digits (1+ letters followed by 1+ digits)
    # Handle both hyphenated and space-separated
    return re.sub(r'([a-zA-Z]+)\d+', r'\1', s)


def diacritics_to_ascii(s: str) -> str:
    """
    Convert Assyriological diacritics to ASCII digraph equivalents.

    Used for 'simple' transliteration where we want pure ASCII but need
    to preserve sound distinctions that diacritics represent.

    Conversions:
        - š/Š -> sz/SZ (shin)
        - ṣ/Ṣ -> s/S (emphatic s - simplified)
        - ṭ/Ṭ -> t/T (emphatic t - simplified)
        - ḫ/Ḫ -> h/H (or could use x - simplified)
        - ʾ/ʿ -> ' (glottal stops - removed or simplified)
        - Vowels with accents (ú, ù, á, à, etc.) -> base vowel

    After this, strip_accents() can safely remove any remaining marks.

    Examples:
        diacritics_to_ascii("šu-ma") -> "szu-ma"
        diacritics_to_ascii("ṣa-lim") -> "sa-lim"
        diacritics_to_ascii("be-lí-ia") -> "be-li-ia"
    """
    # Shin: š -> sz
    s = s.replace('š', 'sz').replace('Š', 'SZ')
    # Emphatic consonants -> simple (these are phonemic distinctions
    # that don't change the basic sound for simple transliteration)
    s = s.replace('ṣ', 's').replace('Ṣ', 'S')
    s = s.replace('ṭ', 't').replace('Ṭ', 'T')
    s = s.replace('ḫ', 'h').replace('Ḫ', 'H')
    # Glottal stops - remove or simplify
    s = s.replace('ʾ', '').replace('ʿ', '')
    # Accented vowels will be handled by strip_accents() after this
    return s


def reduce_doubled_vowels(s: str) -> str:
    """
    Reduce exactly-doubled vowels (aa->a) but keep 3+ consecutive (aaa->aaa).

    In cuneiform transliteration:
    - Doubled vowels (aa) are usually meaningless script duplication
    - Tripled+ vowels (aaa) indicate real phonetic lengthening
    - Doubled consonants (bb, mm, nn) are meaningful and kept unchanged

    Examples:
        reduce_doubled_vowels("dinaan") -> "dinan"
        reduce_doubled_vowels("aaa") -> "aaa"  (kept - real lengthening)
        reduce_doubled_vowels("anumma") -> "anumma"  (mm kept - consonant)
    """
    def replace_match(m):
        if len(m.group(0)) == 2:
            return m.group(1)  # aa -> a (exactly doubled)
        return m.group(0)  # aaa+ stays unchanged (real lengthening)
    return re.sub(r'([aeiou])\1+', replace_match, s)


def normalize_brackets(s: str) -> str:
    """
    Replace all types of brackets in the input string with curly brackets.

    The following conversions are performed:
      - Any opening bracket (e.g. '(', '[', '{', '⸢', '<', '«', '⌞') becomes '{'
      - Any closing bracket (e.g. ')', ']', '}', '⸣', '>', '»', '⌟') becomes '}'

    Example:
        normalize_brackets("2(u) 5(disz) i3 ak unu{ki}")
        # returns: "2{u} 5{disz} i3 ak unu{ki}"
    """
    mapping = {
        # Parentheses
        ord('('): '{', ord(')'): '}',
        # Square brackets
        ord('['): '{', ord(']'): '}',
        # Curly brackets (they might already be the desired ones, but we enforce them)
        ord('{'): '{', ord('}'): '}',
        # Alternative bracket types
        ord('⸢'): '{', ord('⸣'): '}',
        ord('<'): '{', ord('>'): '}',
        ord('«'): '{', ord('»'): '}',
        ord('⌞'): '{', ord('⌟'): '}',
    }
    return s.translate(mapping)

def gap_filler(s, source="cuneiform"):
    """
    Replace lacuna/gap markers with a standardized '*' placeholder.

    Cuneiform tablets often have damaged or illegible sections. Scholars mark these
    using various conventions: [...], vac., x x x, etc. This function normalizes
    all such markers to a single '*' character for ML processing.

    Common markers replaced:
        - [...], [ ], ... - Square bracket lacunae
        - vac, vac., vacat - Vacant space markers
        - vest, vestigia - Traces of text
        - fragmentum, infmut - Fragment markers
        - xxx, x x x - Illegible signs
        - $blank space$ - Explicit blank notation

    Args:
        s: Input string containing gap markers
        source: Source type ('cuneiform' applies the replacements)

    Returns:
        String with gap markers normalized to '*'
    """
    if source=="cuneiform":
        s = s.replace('*', ' * ')
        s = s.replace('[...]', '*')
        s = s.replace('[ ]', '*')
        s = s.replace('vac.', '*')
        s = s.replace('vac', '*')
        s = s.replace('vacat', '*')
        s = s.replace('fragmentum', '*')
        s = s.replace('fragmentum', '*')
        s = s.replace('infmut', '*')
        s = s.replace('gup', '*')
        s = s.replace('qs', '*')
        s = s.replace('vest.', '*')
        s = s.replace('vest', '*')
        s = s.replace('vestigia', '*')
        s = s.replace('...', '*')
        s = s.replace('…', '*')
        s = s.replace('. . .', '*')
        s = s.replace('xxx', '*')
        s = s.replace('x x x', '*')
        s = s.replace(' x ', ' * ')
        s = s.replace('x ', ' * ')
        s = s.replace(' x', ' * ')
        s = s.replace('($blank space$)', '*')
        s = s.replace('$blank space$', '*')
        s = s.replace('blank space', '*')
        #s = re.sub(r'x+', '<cuneiform_gap>', s)
        #s = remove_brackets(s)
        s = re.sub(r'\s+', ' ', s).strip()
    return s

# --- helpers for "x-run" gap normalization -----------------------------------

# things that strongly imply a multi-sign lacuna
_BIG_LACUNA_PATTERNS = [
    r"\[\s*\.\.\.\s*\]",        # [...]
    r"\[\s*\]",                 # [ ]
    r"\.\s*\.\s*\.",            # . . .
    r"…",                       # ellipsis char
    r"\bvac\.?\b",              # vac / vac.
    r"\bvacat\b",
    r"\bvest\.?\b",             # vest / vest.
    r"\bvestigia\b",
    r"\bfragmentum\b",
    r"\bblank\s+space\b",
    r"\$\s*blank\s+space\s*\$",
    r"\(\s*\$blank\s+space\$\s*\)",
    r"\bxxx+\b",                # xxx, xxxx, etc (as a token)
]

_BIG_LACUNA_RE = re.compile("|".join(f"(?:{p})" for p in _BIG_LACUNA_PATTERNS), flags=re.IGNORECASE)

# match a run of x characters as a standalone token (e.g., xxxx-kam or xxxx)
# We'll handle both plain runs and hyphen-attached ones separately.
_XRUN_TOKEN_RE = re.compile(r"\b[xX]{2,}\b")  # "xx", "xxx", "xxxx" as a token

# match hyphen-attached x runs: xxxx-kam, someword-xxxx, xxxx-someword
_XRUN_HYPHEN_RE = re.compile(r"(?i)(?<!\w)(x{2,})(?=-)|(?<=-)(x{2,})(?!\w)")

# match sequences of standalone x tokens (x x x ...)
_XTOK_RUN_RE = re.compile(r"(?i)(?:\bx\b(?:\s+\bx\b)+)")

def gap_filler_complex(
    s: str,
    *,
    emit: str = "x",          # "x" -> emit x x x ; "<big_gap>" -> emit <big_gap>
    preserve_angle_gaps: bool = True,  # keep <gap> and <big_gap> as-is
) -> str:
    """
    Kaggle-oriented gap filler.

    - Maps lacuna markers like vac, [...], xxx, …, . . . to a multi-gap.
      By default emits "x x x" (3 tokens) to represent a larger lacuna.
      Set emit="<big_gap>" if you'd rather emit <big_gap> directly.

    - Preserves <gap>-A-šùr-like attachments:
        <gap>-A-šùr remains unchanged
        <big_gap>-kam remains unchanged
      (i.e., we don't split or rewrite angle-bracket gap tokens).

    - Converts x-runs like "xxxx-kam" or "someword-xxxx" into the chosen gap
      representation, preserving hyphen attachments where possible.

    Notes:
      * This function does NOT collapse x-runs to <gap>/<big_gap> by itself
        unless you set emit="<big_gap>". If you keep emit="x", you can later
        run your normalize_gaps_x_tokens() (with an attachment-aware variant
        if you want) to produce <gap>/<big_gap>.
    """
    if not s:
        return ""

    out = s

    # 0) Optionally protect existing <gap> / <big_gap> tokens (including attachments)
    protected = {}
    if preserve_angle_gaps:
        # protect things like <gap>, <big_gap>, <gap>-A-šùr, <big_gap>-kam
        prot_re = re.compile(r"<\s*(?:gap|big_gap)\s*>\s*(?:-\S+)?", flags=re.IGNORECASE)

        def _protect(m: re.Match) -> str:
            key = f"__GAPPROT{len(protected)}__"
            protected[key] = m.group(0)
            return key

        out = prot_re.sub(_protect, out)

    # 1) Normalize strong lacuna markers to BIG
    BIG = "<big_gap>" if emit == "<big_gap>" else "x x x"
    out = _BIG_LACUNA_RE.sub(BIG, out)

    # 2) Handle "xxxx" as token -> BIG
    out = _XRUN_TOKEN_RE.sub(BIG, out)

    # 3) Handle hyphen-attached x runs like "xxxx-kam" or "someword-xxxx"
    # Replace the x-run part with BIG while preserving the hyphen attachment.
    # If emitting "x x x", keep it hyphen-attached as "x-x-x" so it stays attached.
    if emit == "<big_gap>":
        repl_attached = "<big_gap>"
    else:
        # Keep attachment: xxxx-kam -> x-x-x-kam (still obviously a gap but attached)
        repl_attached = "x-x-x"

    def _xhy_repl(m: re.Match) -> str:
        # m can match either group 1 or group 2 depending on side
        return repl_attached

    out = _XRUN_HYPHEN_RE.sub(_xhy_repl, out)

    # 4) Collapse whitespace
    out = re.sub(r"\s+", " ", out).strip()

    # 5) If emitting x tokens, clean up common variants:
    # - Convert lone "*" to x (optional; comment out if you never use "*")
    if emit == "x":
        out = out.replace("*", "x")
        # collapse "x x x x" (4+) down to 3 (still "big" but bounded)
        def _cap_xtok_run(m: re.Match) -> str:
            return "x x x"
        out = _XTOK_RUN_RE.sub(_cap_xtok_run, out)
        out = re.sub(r"\s+", " ", out).strip()

    # 6) Restore protected <gap> / <big_gap> tokens
    if preserve_angle_gaps and protected:
        for k, v in protected.items():
            out = out.replace(k, v)

    return out


def fix_cuneiform_gap(s: str) -> str:
    """
    Replaces any spaced-out version of:
      < c u n e i f o r m _ g a p >
    or
      c u n e i f o r m gap
    or
      cuneiform gap
    with <cuneiform_gap>.
    """
    pattern = re.compile(
        r'(?:<\s*c\s*u\s*n\s*e\s*i\s*f\s*o\s*r\s*m\s*_\s*g\s*a\s*p\s*>)'  # spaced-out <cuneiform_gap> with underscore
        r'|(?:c\s*u\s*n\s*e\s*i\s*f\s*o\s*r\s*m\s+gap)'                  # spaced-out c u n e i f o r m gap
        r'|(?:cuneiform\s+gap)',                                        # plain "cuneiform gap"
        flags=re.IGNORECASE
    )
    return pattern.sub("<cuneiform_gap>", s)


def postprocess_gaps_for_display(
    s: str,
    *,
    short_gap: str = "...",
    long_gap: str = "[...]",
    collapse_consecutive: bool = True,
) -> str:
    """
    Convert internal gap markers to human-readable format for public display.

    During training we use '*' for short gaps and '***' or '* * *' for longer
    gaps. This function converts those back to conventional scholarly notation
    for displaying translations to end users.

    Args:
        s: Model output string with internal gap markers (* and ***)
        short_gap: Replacement for single '*' (default: '...')
        long_gap: Replacement for '***' or '* * *' (default: '[...]')
        collapse_consecutive: If True, merge adjacent gaps into one (default: True)

    Returns:
        String with gap markers converted to display format

    Examples:
        >>> postprocess_gaps_for_display("the king * went to * * * the temple")
        'the king ... went to [...] the temple'

        >>> postprocess_gaps_for_display("he said * * that", short_gap="…", long_gap="[gap]")
        'he said … … that'

        >>> postprocess_gaps_for_display("text * * * more * * * end", collapse_consecutive=True)
        'text [...] more [...] end'
    """
    if not s:
        return s

    # First, handle explicit long gap markers (*** or * * *)
    # Match '* * *' (3+ spaced asterisks) or '***' (3+ consecutive)
    s = re.sub(r'\*\s*\*\s*\*+', long_gap, s)
    s = re.sub(r'\*{3,}', long_gap, s)

    # Handle pairs of asterisks (* *) - treat as short gaps
    s = re.sub(r'\*\s+\*(?!\s*\*)', f'{short_gap} {short_gap}', s)

    # Handle single asterisks
    s = re.sub(r'\*', short_gap, s)

    # Collapse consecutive gaps if requested
    if collapse_consecutive:
        # Collapse multiple short gaps with only whitespace between them
        short_escaped = re.escape(short_gap)
        long_escaped = re.escape(long_gap)

        # Multiple short gaps -> single short gap
        s = re.sub(rf'({short_escaped}\s*)+{short_escaped}', short_gap, s)

        # Multiple long gaps -> single long gap
        s = re.sub(rf'({long_escaped}\s*)+{long_escaped}', long_gap, s)

        # Short gap adjacent to long gap -> just long gap
        s = re.sub(rf'{short_escaped}\s*{long_escaped}', long_gap, s)
        s = re.sub(rf'{long_escaped}\s*{short_escaped}', long_gap, s)

    # Clean up whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    return s


def postprocess_for_display(
    s: str,
    *,
    convert_gaps: bool = True,
    short_gap: str = "...",
    long_gap: str = "[...]",
    capitalize_first: bool = True,
    ensure_final_punctuation: bool = False,
    final_punctuation: str = ".",
) -> str:
    """
    Full post-processing pipeline for converting model output to display format.

    Combines gap conversion with other common post-processing steps needed
    for presenting translations to end users.

    Args:
        s: Raw model output string
        convert_gaps: Whether to convert gap markers (default: True)
        short_gap: Replacement for single '*' (default: '...')
        long_gap: Replacement for '***' (default: '[...]')
        capitalize_first: Capitalize first letter (default: True)
        ensure_final_punctuation: Add period if no ending punctuation (default: False)
        final_punctuation: Punctuation to add if missing (default: '.')

    Returns:
        Cleaned string ready for display

    Example:
        >>> postprocess_for_display("the king * said * * * to his servant")
        'The king ... said [...] to his servant'
    """
    if not s:
        return s

    # Convert gap markers
    if convert_gaps:
        s = postprocess_gaps_for_display(
            s,
            short_gap=short_gap,
            long_gap=long_gap,
            collapse_consecutive=True,
        )

    # Clean up whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    # Capitalize first letter
    if capitalize_first and s:
        s = s[0].upper() + s[1:]

    # Ensure final punctuation
    if ensure_final_punctuation and s and s[-1] not in '.!?':
        s = s + final_punctuation

    return s


_SUBSCRIPT_TO_ASCII = str.maketrans({"₂": "2", "₃": "3"})

_VOWEL23_MAP = {
    ("a", "2"): "á", ("a", "3"): "à",
    ("e", "2"): "é", ("e", "3"): "è",
    ("i", "2"): "í", ("i", "3"): "ì",
    ("u", "2"): "ú", ("u", "3"): "ù",
}
_VOWEL23_MAP_UP = {k: v.upper() for k, v in _VOWEL23_MAP.items()}

_SZ_IN_WORD = re.compile(r"(?i)(?<=\w)sz(?=\w)|\bsz(?=\w)|(?<=\w)sz\b")
def normalize_sz_to_shin(s: str) -> str:
    """
    Convert CDLI ASCII 'sz' digraph to proper shin character (š/Š).

    CDLI uses 'sz' to represent the Akkadian/Sumerian shin sound in ASCII.
    This function converts those digraphs to the proper Unicode character,
    preserving case (sz -> š, SZ -> Š).

    Only converts 'sz' when it appears within words (not standalone),
    to avoid false positives.

    Args:
        s: Input string with CDLI-style 'sz' digraphs

    Returns:
        String with 'sz' converted to 'š' (case-preserved)

    Example:
        >>> normalize_sz_to_shin("szarrum")
        'šarrum'
    """
    def repl(m: re.Match) -> str:
        frag = m.group(0)
        return "š" if frag.islower() else "Š"
    return _SZ_IN_WORD.sub(repl, s)

# subscript digits (₀..₉) -> ASCII digits
_SUB_TO_ASCII_ALL = str.maketrans({chr(0x2080 + i): str(i) for i in range(10)})

# superscript digits (⁰..⁹) -> ASCII digits
_SUP_TO_ASCII_ALL = str.maketrans({
    "⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9"
})

# CDLI ATF: "sz" used for š  :contentReference[oaicite:4]{index=4}
_SZ_ANYWHERE = re.compile(r"sz", flags=re.IGNORECASE)

# C-ATF index style: gesz_{2}, ban_{2}, szar_{2}, etc. :contentReference[oaicite:5]{index=5}
# This targets: _{digits}  (optionally with whitespace inside braces)
_ATF_BRACED_INDEX = re.compile(r"_\{\s*(\d+)\s*\}")

# Some people write _2 without braces; normalize that too (conservative)
_ATF_PLAIN_INDEX = re.compile(r"_(\d+)\b")

def normalize_sumerian_cdli(line: str) -> str:
    """
    Normalize CDLI Sumerian transliteration (C-ATF-ish) without destroying sign indices.

    Keeps: gu4, SIG7, he2, du3, etc.
    Normalizes:
      - Unicode stability (NFC)
      - sub/sup digits -> ASCII digits
      - CDLI ATF sz/SZ -> š/Š
      - C-ATF underscore indices: ban_{2} -> ban2, szar_{2} -> szar2
      - whitespace
    """
    if not line:
        return ""

    s = unicodedata.normalize("NFC", line)

    # if you already have these helpers, keep using them
    try:
        s = remove_control_characters(s)
    except NameError:
        pass
    try:
        s = normalize_dashes(s)
    except NameError:
        pass

    # sub/sup digits -> ASCII digits
    s = s.translate(_SUB_TO_ASCII_ALL).translate(_SUP_TO_ASCII_ALL)

    # C-ATF index normalization: _{2} -> 2, _2 -> 2
    # IMPORTANT: this preserves sign indices like gu4 (no underscore)
    s = _ATF_BRACED_INDEX.sub(r"\1", s)
    s = _ATF_PLAIN_INDEX.sub(r"\1", s)

    # sz -> š (case-aware-ish)
    def _sz_repl(m: re.Match) -> str:
        frag = m.group(0)
        return "Š" if frag.isupper() else "š"
    s = _SZ_ANYWHERE.sub(_sz_repl, s)

    # final whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Convert all subscript digits to ASCII digits (₀..₉)
_SUB_TO_ASCII_ALL = str.maketrans({chr(0x2080 + i): str(i) for i in range(10)})

# Replace CDLI ASCII "sz"/"SZ" with š/Š inside tokens
_SZ_ANYWHERE = re.compile(r"sz", flags=re.IGNORECASE)

# Optional: normalize some common ḫ encodings people run into
# (covers decomposed forms too)
def _normalize_heth(s: str) -> str:
    # Stabilize first
    s = unicodedata.normalize("NFC", s)

    # Some corpora have 'h' + combining breve below / dot below variants
    # We'll normalize a few plausible ones to ḫ/Ḫ.
    # NOTE: This is conservative-ish; if your data uses plain 'h' for real /h/,
    # do NOT enable this without checking your corpus.
    replacements = {
        "ḫ": "ḫ",  # h + combining mark variant (common in some exports)
        "Ḫ": "Ḫ",
        "ḥ": "ḫ",  # h + dot below (sometimes used)
        "Ḥ": "Ḫ",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    return s

def normalize_hittite_cdli(line: str, *, normalize_ḫ: bool = False) -> str:
    """
    Normalize CDLI/ORACC-ish Hittite transliteration without destroying sign indices.

    Keeps: ka4, pir2, DU10, {d} determinatives, etc.
    Converts: subscript digits -> digits, sz/SZ -> š/Š (inside tokens).
    Optionally normalizes various ḫ encodings to ḫ/Ḫ.
    """
    if not line:
        return ""

    s = unicodedata.normalize("NFC", line)

    # 1) subscripts -> digits
    s = s.translate(_SUB_TO_ASCII_ALL)

    # 2) sz -> š (works inside tokens: rasz2 -> raš2)
    def _sz_repl(m: re.Match) -> str:
        frag = m.group(0)
        return "Š" if frag.isupper() else "š"
    s = _SZ_ANYWHERE.sub(_sz_repl, s)

    # 3) optional ḫ normalization
    if normalize_ḫ:
        s = _normalize_heth(s)

    # 4) whitespace cleanup
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 0) unify subscript digits (₀..₉) + common ₂/₃ specifically
_SUB_TO_ASCII_ALL = str.maketrans({chr(0x2080 + i): str(i) for i in range(10)})
_SUP_TO_ASCII = str.maketrans({
    '⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9'
})

_VOWEL23 = {
    ("a","2"):"á", ("a","3"):"à",
    ("e","2"):"é", ("e","3"):"è",
    ("i","2"):"í", ("i","3"):"ì",
    ("u","2"):"ú", ("u","3"):"ù",
}
_VOWEL23_UP = {k: v.upper() for k,v in _VOWEL23.items()}

# convert sz anywhere inside a token
_SZ_ANYWHERE = re.compile(r"sz", flags=re.IGNORECASE)

# vowel+2/3 conversion inside *any* token, BUT ONLY for a/e/i/u (not DU10 etc.)
_VOWEL23_INNER = re.compile(r"([AEIUaeiu])([23])")

# braces helper
_BRACED = re.compile(r"\{([^}]*)\}")

def akkadian_vowel23_to_diacritics_hyphenaware(text: str) -> str:
    """
    Convert Akkadian-style vowel+{2,3} inside syllables:
      be-li2-ia  -> be-lí-ia  (hyphenated)
      be li2 ia  -> be lí ia  (space-separated)
      a-na       -> unchanged
      i3-...     -> í-...

    Rule: convert [aeiu][23] when the digit is immediately followed by
    hyphen, space, or end-of-string. This targets Akkadian vowel quality
    indices, not Sumerian-style sign indices like gu4/du3/etc.
    """
    if not text:
        return text

    s = unicodedata.normalize("NFC", text)
    s = s.translate(_SUBSCRIPT_TO_ASCII)  # handle i₂ etc.

    # Match vowel+2/3 followed by hyphen, space, or end-of-string
    pat = re.compile(r"(?i)([AEIUaeiu])([23])(?=(?:-|\s|$))")

    def repl(m: re.Match) -> str:
        v, d = m.group(1), m.group(2)
        return _VOWEL23_MAP_UP[(v.lower(), d)] if v.isupper() else _VOWEL23_MAP[(v, d)]

    return pat.sub(repl, s)


def normalize_akkadian_cdli_oracc(line: str, *, convert_braced: bool = True) -> str:
    if not line:
        return ""

    s = unicodedata.normalize("NFC", line)

    # 1) normalize dash variants early (your normalize_dashes is fine)
    s = normalize_dashes(s)

    # 2) normalize sub/superscripts to digits
    s = s.translate(_SUB_TO_ASCII_ALL).translate(_SUP_TO_ASCII)

    # 3) sz -> š (inside tokens)
    def _sz_repl(m: re.Match) -> str:
        frag = m.group(0)
        return "Š" if frag.isupper() else "š"
    s = _SZ_ANYWHERE.sub(_sz_repl, s)

    # 4) convert vowel+2/3 -> diacritics
    def _v23(m: re.Match) -> str:
        v, d = m.group(1), m.group(2)
        if v.isupper():
            return _VOWEL23_UP[(v.lower(), d)]
        return _VOWEL23[(v, d)]

    if convert_braced:
        # convert inside {...} determinatives too (e.g., {tug2} -> {túg})
        def _brace_repl(m: re.Match) -> str:
            inner = m.group(1)
            inner = _VOWEL23_INNER.sub(_v23, inner)
            return "{" + inner + "}"
        s = _BRACED.sub(_brace_repl, s)

    # convert everywhere else (this will convert u2 in e.g. be-li2-ia -> be-lí-ia)
    s = _VOWEL23_INNER.sub(_v23, s)

    # 5) whitespace cleanup + doc refs (optional)
    s = re.sub(r"\s+", " ", s).strip()
    s = remove_doc_refs(s)

    return s

def cdli_oracc_vowel_numbers_to_diacritics(text: str) -> str:
    """
    Convert vowel readings a/e/i/u + (2|3) (including ORACC subscripts) to diacritics.
    Leaves DU10, GIŠ2, etc. untouched. Idempotent when text already uses diacritics.
    """
    if not text:
        return text

    # Normalize Unicode composition so diacritics are stable
    s = unicodedata.normalize("NFC", text)

    # Convert ORACC subscripts to ASCII digits for unified handling
    s = s.translate(_SUBSCRIPT_TO_ASCII)

    # Replace only when the vowel+digit is "standalone-ish":
    # not preceded or followed by a letter/digit.
    pat = re.compile(r"(?<![A-Za-z0-9])([AEIUaeiu])([23])(?![A-Za-z0-9])")

    def repl(m: re.Match) -> str:
        v, d = m.group(1), m.group(2)
        return _VOWEL23_MAP_UP[(v.lower(), d)] if v.isupper() else _VOWEL23_MAP[(v, d)]

    return pat.sub(repl, s)

_BRACED = re.compile(r"\{([^}]*)\}")

def _convert_vowel23_inside_token(token: str) -> str:
    # Convert occurrences like "tug2" -> "túg" and "a3" -> "à" *within* a token.
    # Still only a/e/i/u + 2/3.
    def _inner(m: re.Match) -> str:
        v, d = m.group(1), m.group(2)
        return _VOWEL23_MAP_UP[(v.lower(), d)] if v.isupper() else _VOWEL23_MAP[(v, d)]
    # also handle subscripts
    token = token.translate(_SUBSCRIPT_TO_ASCII)
    return re.sub(r"([AEIUaeiu])([23])", _inner, token)

def cdli_oracc_convert_braced_determinatives(text: str) -> str:
    """
    Convert vowel+2/3 inside {...} tokens (determinatives/sign readouts),
    e.g. {tug2} -> {túg}.
    """
    if not text:
        return text

    text = unicodedata.normalize("NFC", text)

    def repl(m: re.Match) -> str:
        inner = m.group(1)
        inner2 = _convert_vowel23_inside_token(inner)
        return "{" + inner2 + "}"

    return _BRACED.sub(repl, text)

def normalize_line_for_source(line: str, *, source: str | None, kind: str | None) -> str:
    """
    source: 'cdli', 'oracc', 'kaggle', or None
    kind: 'transliteration' or 'translation' (or None)
    """
    if line is None:
        return line

    # Always keep Unicode stable
    s = unicodedata.normalize("NFC", line)

    # Your existing "... -> *" behavior (if you still want it)
    s = re.sub(r'\s*\.\s*\.\s*\.\s*', '*', s)

    src = (source or "").lower().strip()
    kd = (kind or "").lower().strip()

    # Apply only when the source is known to use vowel numbers/subscripts
    if src in {"cdli", "oracc"} and kd == "transliteration":
        # First fix determinatives like {tug2}
        s = cdli_oracc_convert_braced_determinatives(s)
        # Then convert standalone vowel tokens a2/u3 etc.
        s = cdli_oracc_vowel_numbers_to_diacritics(s)

    # You can add other conversions here later:
    # - sz -> š (CDLI ASCII)
    # - etc.

    return s

LANG_CODE = {
    "Akkadian": "akk",
    "Sumerian": "sux",
    "Elamite": "elx",
    "Mycenean Greek": "gmy",
    "Hittite": "hit",
    # accept already-coded values too
    "akk": "akk", "sux": "sux", "elx": "elx", "gmy": "gmy", "hit": "hit",
}

MODERN_CODE = {
    "English": "en",
    "en": "en",
    "German": "de",
    "de": "de",
}

TYPE_CODE = {
    "simple": "simp",
    "group": "grp",
    "grouped": "grp",
    "original": "orig",
    "complex": "orig",
    "raw": "raw",
}

TASK_CODE = {
    "Translate": "tr",
    "Transliterate": "trlit",
    "Convert": "cv",
    "Group": "grp",
}

def make_prefix(
    task="Translate",
    src_lang="akk",
    tgt_lang=None,          # for language targets like English
    src_script=None,        # "cunei" or "latn"
    tgt_script=None,        # "cunei" or "latn"
    variant=None,           # simp/grp/orig
):
    """
    Compact, fixed-format prefix.
    Examples:
      tr|src=akk|srcsc=latn|var=grp|tgt=en:
      tr|src=akk|srcsc=cunei|tgt=en:
      trlit|src=akk|srcsc=cunei|tgtsc=latn|var=simp:
      cv|src=latn|var=orig|tgtsc=cunei|lang=akk:
      grp|src=akk|srcsc=latn|var=simp:
    """
    t = TASK_CODE.get(task, str(task).lower())
    parts = [t]

    if src_lang:
        parts.append(f"src={src_lang}")
    #if src_script:
    #    parts.append(f"srcsc={src_script}")
    if tgt_lang:
        parts.append(f"tgt={tgt_lang}")
   # if tgt_script:
   #     parts.append(f"tgtsc={tgt_script}")
    #if variant:
    #    parts.append(f"var={variant}")

    return "|".join(parts) + ": "

# From the host's "Translations Characters" list (as you pasted)
# IMPORTANT: Must include ALL diacritic vowels used in Akkadian names in English translations
TRANSLATION_ALLOWED = set(list(
    "'?eEaAiItTnNsSoOrRlLhHuUmMdDfFšŠ-"
    "pPwWbBgGyY.Kk,Ccvvā1)(<>zZ_Qq2īṭṬ0:3½5;xē4ū6ṣṢ⅓8'!7jJ⅔""9–⅚¼⅙\"'ı—[]ğâàş+"
))
# Add space (and optionally tab if you want)
TRANSLATION_ALLOWED |= {" "}

# CRITICAL: Add missing circumflex and other diacritic vowels for Akkadian names
# These appear in English translations: Sîn, nabûm, etc.
# Circumflex vowels (used in long vowels in names)
TRANSLATION_ALLOWED |= set("îÎûÛêÊôÔâÂ")
# Grave vowels (used in some transliteration conventions)
TRANSLATION_ALLOWED |= set("ìÌùÙèÈòÒàÀ")
# Acute vowels (used in some transliteration conventions)
TRANSLATION_ALLOWED |= set("íÍúÚéÉóÓáÁ")
# Macron vowels (should already be present, but ensure completeness)
TRANSLATION_ALLOWED |= set("āĀīĪūŪēĒ")
# Breve vowels (rare but may appear)
TRANSLATION_ALLOWED |= set("ăĂĭĬŭŬĕĔ")
# Hacek/caron consonants (š is present, add others if needed)
TRANSLATION_ALLOWED |= set("ḫḪ")  # h with underdot (common in Akkadian)

# Optional: remove common parenthetical annotations without killing normal punctuation
_ANNOT_PARENS = re.compile(
    r"""
    \(\s*
        (?:\?|\!|plur\.?|pl\.?|sing\.?|fem\.?|masc\.?|dual\.?|lit\.?|var\.?|unclear|sic)
    \s*\)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Optional: normalize a few quote variants to the ones Kaggle allows
_QUOTE_TRANSLATE = str.maketrans({
    "\u2018": "‘",  # left single curly -> ‘
    "\u2019": "’",  # right single curly -> ’
    "\u201C": "“",  # left double curly -> “
    "\u201D": "”",  # right double curly -> ”
    "\u2032": "’",  # prime -> treat as apostrophe-ish
    "\u2033": "”",  # double prime -> treat as quote-ish
})

_DASH_TRANSLATE = str.maketrans({
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # nb hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "–",  # en dash (Kaggle allows –)
    "\u2014": "—",  # em dash (Kaggle allows —)
    "\u2212": "-",  # minus
})

GERMAN_EXTRA = set("äöüßÄÖÜ")
TRANSLATION_ALLOWED_DE = TRANSLATION_ALLOWED | GERMAN_EXTRA


# =============================================================================
# ENGLISH TARGET STANDARDIZATION
# =============================================================================
# These functions normalize the English side of training data to match the
# diacritical conventions used in validation/evaluation data (e.g., OARE valid
# uses Aššur while OARE train uses Assur).

# Akkadian proper name components: ASCII -> diacritics
# Applied with word-boundary awareness to avoid corrupting English words.
# Ordered: longest compound names first, then shorter components.
_AKK_NAME_REPLACEMENTS = [
    # --- Full compound names (apply first) ---
    # Assur-X compounds
    ('Assur-bel-awatim', 'Aššur-bēl-awātim'),
    ('Assur-taklaku', 'Aššur-taklāku'),
    ('Assur-lamassi', 'Aššur-lamassī'),
    ('Assur-imitti', 'Aššur-imittī'),
    ('Assur-sululi', 'Aššur-ṣululī'),
    ('Assur-nada', 'Aššur-nādā'),
    ('Assur-bani', 'Aššur-bāni'),
    ('Assur-Samsi', 'Aššur-Šamši'),
    ('Assur-idi', 'Aššur-idī'),
    ("Assur-re'i", "Aššur-rē'ī"),
    # X-Assur compounds
    ('Ennam-Assur', 'Ennam-Aššur'),
    ('Puzur-Assur', 'Puzur-Aššur'),
    ('Iddin-Assur', 'Iddin-Aššur'),
    ('Mannum-ki-Assur', 'Mannum-kī-Aššur'),
    ('Mannum-balum-Assur', 'Mannum-bālum-Aššur'),
    ('Damiq-pi-Assur', 'Damiq-pī-Aššur'),
    ('Tab-sill-Assur', 'Ṭāb-ṣill-Aššur'),
    ('Tab-pi-Assur', 'Ṭāb-pī-Aššur'),
    ('Dan-Assur', 'Dān-Aššur'),
    ('Salim-Assur', 'Šalim-Aššur'),
    ('Su-Assur', 'Šu-Aššur'),
    ('Usur-sa-Assur', 'Uṣur-ša-Aššur'),
    ('Usur-sa-Istar', 'Uṣur-ša-Ištar'),
    # Š- prefix names
    ('Su-Belum', 'Šu-Bēlum'),
    ('Su-Kubum', 'Šu-Kūbum'),
    ('Su-Illil', 'Šu-Illil'),
    ('Su-Istar', 'Šu-Ištar'),
    ('Su-Ishara', 'Šu-Ishara'),
    ('Su-Anum', 'Šu-Anum'),
    ('Su-Suen', 'Šu-Suen'),
    ('Samas-bani', 'Šamaš-bāni'),
    ('Amur-Samas', 'Amur-Šamaš'),
    # -bani compounds
    ('Belum-bani', 'Bēlum-bāni'),
    ('Amurrum-bani', 'Amurrum-bāni'),
    ('Ili-bani', 'Ilī-bāni'),
    ('Illil-bani', 'Illil-bāni'),
    ('Adad-bani', 'Adad-bāni'),
    # -ili compounds
    ('Anah-ili', 'Anah-ilī'),
    ('Ennam-ili', 'Ennam-ilī'),
    ('Iddin-ili', 'Iddin-ilī'),
    # Other compound names
    ('Ili-pi-usur', 'Ilī-pī-uṣur'),
    ('Ili-asranni', 'Ilī-asrannī'),
    ('Ili-nada', 'Ilī-nādā'),
    ('Ili-mutabbil', 'Ilī-mutabbil'),
    ('Ikun-piya', 'Ikūn-pīya'),
    ('Suen-nada', 'Suen-nādā'),
    ('La-qepum', 'Lā-qēpum'),
    # ṭ- prefix names
    ('Tab-Assur', 'Ṭāb-Aššur'),
    # --- Shorter name parts (apply after compounds) ---
    ('Istar', 'Ištar'),
    ('Assur', 'Aššur'),
    ('Samas', 'Šamaš'),
    # Name suffixes/patterns
    ('Usuranum', 'Uṣurānum'),
    ('Ikunum', 'Ikūnum'),
    ('Ennanum', 'Ennānum'),
    ('Belanum', 'Bēlānum'),
    ('Kannutum', 'Kannūtum'),
    ('Turaya', 'Tūraya'),
    ('Innaya', 'Innāya'),
    ('Asanum', 'Asānum'),
    ('Laqep', 'Lāqēp'),
    ('La-qep', 'Lā-qēp'),
    # Common Akkadian words in English translations
    ('karum', 'kārum'),
]

# Compile to regex for word-boundary aware replacement
_AKK_NAME_PATTERNS = [
    (re.compile(r'(?<!\w)' + re.escape(ascii_name) + r'(?!\w)'), diac_name)
    for ascii_name, diac_name in _AKK_NAME_REPLACEMENTS
]

# Decimal -> Unicode fraction mapping
_DECIMAL_TO_FRACTION = {
    '0.5':    '½',
    '0.3333': '⅓',
    '0.33':   '⅓',
    '0.6666': '⅔',
    '0.67':   '⅔',
    '0.25':   '¼',
    '0.1666': '⅙',
    '0.17':   '⅙',
    '0.8333': '⅚',
    '0.83':   '⅚',
}

# Pattern: number followed by ".5" at word boundary (e.g. "13.5" -> "13 ½")
_DECIMAL_HALF_PATTERN = re.compile(r'\b(\d+)\.5\b')


def standardize_akkadian_names(s: str) -> str:
    """
    Convert ASCII Akkadian proper names to standardized diacritical forms.

    Training data often uses ASCII (Assur, Istar) while evaluation data uses
    scholarly diacritics (Aššur, Ištar). This function bridges that gap so the
    model learns to produce the expected diacritical output.

    Uses word-boundary matching to avoid corrupting English words.

    Args:
        s: English text containing Akkadian names

    Returns:
        Text with Akkadian names in standardized diacritical form
    """
    for pattern, replacement in _AKK_NAME_PATTERNS:
        s = pattern.sub(replacement, s)
    return s


def standardize_fractions(s: str) -> str:
    """
    Normalize decimal fractions to Unicode fraction characters.

    Converts patterns like "13.5" to "13 ½" and "0.3333" to "⅓"
    to match the fraction format used in evaluation references.

    Args:
        s: Text potentially containing decimal fractions

    Returns:
        Text with Unicode fractions where applicable
    """
    # First handle "N.5" patterns -> "N ½"
    s = _DECIMAL_HALF_PATTERN.sub(r'\1 ½', s)

    # Then handle standalone decimal fractions
    for decimal, fraction in _DECIMAL_TO_FRACTION.items():
        s = s.replace(decimal, fraction)

    return s


def standardize_english_target(s: str) -> str:
    """
    Full standardization pipeline for English translation targets.

    Applies in order:
        1. Akkadian name diacritics
        2. Fraction normalization

    Call this BEFORE normalizeString_en / normalizeString_diacritic
    so the allowed-character filter can then do its final cleanup.

    Args:
        s: Raw English translation text

    Returns:
        Standardized text ready for normalizeString_en
    """
    s = standardize_akkadian_names(s)
    s = standardize_fractions(s)
    return s


# =============================================================================
# TRANSLITERATION SOURCE STANDARDIZATION (OARE → CDLI conventions)
# =============================================================================
# OARE data uses different conventions from CDLI/ORACC for transliteration:
#   - Parenthetical determinatives: (d)IŠKUR, (TÚG)šu-ru-tum vs {d}iškur, {tug2}šu-ru-tum
#   - Decimal fractions: 0.66666, 13.5 vs ⅔, 13 ½
# These normalizations are safe to apply universally: they're no-ops on CDLI data
# which already uses the target conventions.

# Regex for parenthetical determinatives: (X) immediately followed by a word char
# Matches (d), (m), (f), (TÚG), (ki), etc. but NOT standalone parenthetical text
# The determinative content is typically 1-5 chars (letters, digits, diacritics, dots)
_PAREN_DET_RE = re.compile(
    r'\(([a-zA-ZÀ-ÿ\d.]{1,6})\)(?=\S)'
)

# Decimal fractions used in OARE transliterations
# More permissive than the English-side _DECIMAL_TO_FRACTION: handles variable
# digit counts (0.3333, 0.33333, 0.333333) and compound numbers (5.33333)
_TRANSLIT_FRACTION_MAP = {
    # standalone fractions
    '0.5':      '½',
    '0.33':     '⅓',
    '0.3333':   '⅓',
    '0.33333':  '⅓',
    '0.333333': '⅓',
    '0.66':     '⅔',
    '0.6666':   '⅔',
    '0.66666':  '⅔',
    '0.666666': '⅔',
    '0.83':     '⅚',
    '0.8333':   '⅚',
    '0.83333':  '⅚',
    '0.833333': '⅚',
    '0.25':     '¼',
    '0.1666':   '⅙',
    '0.16666':  '⅙',
    '0.166666': '⅙',
}

# Pattern for "N.5" -> "N ½", "N.33333" -> "N ⅓", "N.66666" -> "N ⅔"
# Uses [1-9]\d* to exclude "0.X" standalone fractions (handled by the map)
_TRANSLIT_COMPOUND_FRAC_PATTERNS = [
    (re.compile(r'\b([1-9]\d*)\.5\b'),                          r'\1 ½'),
    (re.compile(r'\b([1-9]\d*)\.3333?3?3?\b'),                  r'\1 ⅓'),
    (re.compile(r'\b([1-9]\d*)\.6666?6?6?\b'),                  r'\1 ⅔'),
    (re.compile(r'\b([1-9]\d*)\.8333?3?3?\b'),                  r'\1 ⅚'),
]

# Pattern for orphaned number-period: "1. ma-na" -> "1 ma-na"
# This handles corrupted fractions where the decimal part was lost (e.g., "1.3333" -> "1.")
# The period followed by space before non-digit text looks like sentence-end to the model
# and causes early truncation. Remove the orphaned period.
_ORPHANED_NUMBER_PERIOD = re.compile(r'\b(\d+)\.\s+(?=[a-zA-Zḫšṣṭʾʿ{])')


def fix_orphaned_number_periods(s: str) -> str:
    """
    Remove orphaned periods after numbers in transliteration.

    Handles corrupted fraction patterns where the decimal portion was lost,
    leaving patterns like "1. ma-na" instead of "1.3333 ma-na".

    The "N. " pattern followed by text looks like sentence-end to the model,
    causing early truncation. This fix converts "1. ma-na" to "1 ma-na".

    Args:
        s: Transliteration text that may contain orphaned number-periods

    Returns:
        Text with orphaned number-periods converted to just number-space
    """
    return _ORPHANED_NUMBER_PERIOD.sub(r'\1 ', s)


def normalize_translit_determinatives(s: str) -> str:
    """
    Convert parenthetical determinatives to curly-brace notation.

    OARE uses (d)IŠKUR, (TÚG)šu-ru-tum while CDLI/ORACC uses {d}IŠKUR,
    {tug2}šu-ru-tum. After lowering + normalize_translit, both converge
    to {d}iškur, {túg}šu-ru-tum.

    Safe for all data: CDLI already uses curly braces (no-op).
    """
    return _PAREN_DET_RE.sub(r'{\1}', s)


def normalize_translit_fractions(s: str) -> str:
    """
    Convert decimal fractions in transliteration source to Unicode fractions.

    OARE data uses decimals (0.66666, 13.5) where CDLI/evaluation data uses
    Unicode fractions (⅔, 13 ½). Handles both standalone (0.66666 → ⅔) and
    compound (13.5 → 13 ½) patterns.

    Safe for all data: CDLI transliterations don't contain decimal fractions (no-op).
    """
    # Handle compound N.X patterns first (e.g., 13.5 → 13 ½)
    for pattern, replacement in _TRANSLIT_COMPOUND_FRAC_PATTERNS:
        s = pattern.sub(replacement, s)

    # Handle remaining standalone decimals (e.g., 0.66666 → ⅔)
    # Sort by longest key first to avoid partial matches
    for decimal, fraction in sorted(
        _TRANSLIT_FRACTION_MAP.items(), key=lambda x: -len(x[0])
    ):
        s = s.replace(decimal, fraction)

    return s


def standardize_translit_source(s: str) -> str:
    """
    Full standardization pipeline for transliteration source text.

    Normalizes OARE-specific conventions to CDLI/ORACC standard format
    so the model sees consistent input regardless of data source.

    Call BEFORE s.lower() and normalize_translit() in the processing chain.

    Steps:
        1. Fix orphaned number-periods: 1. ma-na → 1 ma-na (prevents early truncation)
        2. Determinatives: (d)IŠKUR → {d}IŠKUR
        3. Fractions: 0.66666 → ⅔, 13.5 → 13 ½
    """
    s = fix_orphaned_number_periods(s)
    s = normalize_translit_determinatives(s)
    s = normalize_translit_fractions(s)
    return s


def normalizeString_diacritic(
    s: str,
    *,
    modern: str = "English",          # <-- add this
    lowercase: bool = False,
    remove_annotations: bool = True,
    allowed: set | None = None,
) -> str:
    """
    Normalize modern-language text (English/German) while keeping allowed punctuation.

    - NFC normalize
    - normalize quotes/dashes to allowed variants
    - optionally strip parenthetical annotations like (pl.), (sic), etc.
    - remove doc refs
    - enforce allowed character set by replacing disallowed chars with spaces
    """
    if not s:
        return ""

    s = unicodedata.normalize("NFC", s.strip())
    if lowercase:
        s = s.lower()

    # choose allowed set if not provided
    if allowed is None:
        mod = MODERN_CODE.get(modern, modern)  # "German" -> "de"
        allowed = TRANSLATION_ALLOWED_DE if mod == "de" else TRANSLATION_ALLOWED

    # normalize a few unicode variants to the ones you allow
    s = s.translate(_QUOTE_TRANSLATE).translate(_DASH_TRANSLATE)

    if remove_annotations:
        s = _ANNOT_PARENS.sub("", s)

    s = remove_doc_refs(s)
    s = re.sub(r"\s+", " ", s).strip()

    # enforce allowed chars
    s = "".join(ch if (ch in allowed) else " " for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    return s



def normalizeString_en(s, use_prefix=False, task="Translate", target="cuneiform", type="simple", language="Akkadian", modern="English", style="T5", method="diacritic"):
    """
    Normalize modern language (English/German) text for ML model input.

    This is the primary normalizer for translation TARGET text (what the model
    produces when translating from cuneiform). Prepares text by cleaning,
    normalizing punctuation, and optionally adding task prefixes.

    Args:
        s: Input string in modern language (English or German)
        use_prefix: If True, prepend a task prefix for T5-style models
        task: Task type ('Translate')
        target: Target format ('cuneiform' or 'transliteration')
        type: Transliteration style ('simple', 'group', 'original')
        language: Ancient language ('Akkadian', 'Sumerian', etc.)
        modern: Modern language ('English', 'German')
        style: Prefix style ('T5' for verbose, 'simple'/'compact' for short)
        method: Normalization method:
            - 'legacy': ASCII-only, strips all diacritics
            - 'diacritic': Preserves scholarly diacritics (recommended)

    Returns:
        Normalized string, optionally with task prefix

    Example:
        >>> normalizeString_en("The king went to the temple.", use_prefix=True)
        'Translate English to Akkadian cuneiform: the king went to the temple'
    """
    # Standardize Akkadian names and fractions BEFORE method-specific processing.
    # This converts ASCII names (Assur) to diacritical forms (Aššur) so training
    # and validation data use a consistent convention.
    if language == "Akkadian":
        s = standardize_english_target(s)

    if method=="legacy":
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?*]+", r" ", s)
        s = s.strip()
        s = remove_brackets(s)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = strip_accents(s)
    elif method == "diacritic":
        mod = MODERN_CODE.get(modern, modern)  # "German" -> "de"
        allowed = TRANSLATION_ALLOWED_DE if mod == "de" else TRANSLATION_ALLOWED
        s = normalizeString_diacritic(s, modern=modern, allowed=allowed)
        
    normalized_string = s  # <-- define it

    if not use_prefix:
        return normalized_string
    
    if str(style).lower() in ("simple", "tags", "compact"):
        src = LANG_CODE.get(language, language)
        tgt = MODERN_CODE.get(modern, modern)
        # this function is specifically: transliteration -> modern language
        pref = make_prefix(
            task="Translate",
            src_lang=tgt,
            tgt_lang=src,
            src_script="latn",
            tgt_script="latn",
            variant=TYPE_CODE[type],   # transliteration
        )
        return pref + normalized_string
    
    if task=="Translate":
        if target=="cuneiform":
            return 'Translate ' + modern + ' to ' + language + ' cuneiform: ' + s
        elif target=="transliteration":
            if type == "simple":
                return 'Translate ' + modern + ' to simple ' + language + ' transliteration: ' + s
            elif type == "group":
                return 'Translate ' + modern + ' to ' + language + ' transliteration: ' + s
            elif type == "original":
                return 'Translate ' + modern + ' to complex ' + language + ' transliteration: ' + s


def normalizeString_cuneiform_transliterate(s, use_prefix=True, type="simple", language="Akkadian", modern="English", style="T5"):
    """
    Normalize cuneiform SOURCE text for cuneiform-to-transliteration tasks.

    Prepares cuneiform input for training models that convert cuneiform glyphs
    to Latin transliteration. Handles gap markers, bracket removal, and
    digit normalization.

    Args:
        s: Input string (cuneiform or transliteration to normalize)
        use_prefix: If True, prepend a task prefix for T5-style models
        type: Output transliteration style:
            - 'simple': ASCII-only, no diacritics (a-z only)
            - 'original': Preserves complex scholarly notation
            - 'group': Syllables joined with hyphens into word units
        language: Ancient language ('Akkadian', 'Sumerian', etc.)
        modern: Target modern language ('English', 'German')
        style: Prefix style ('T5' for verbose, 'simple'/'compact' for short)

    Returns:
        Normalized string, optionally with task prefix like:
        'Transliterate Akkadian cuneiform to simple Latin characters: ...'
    """
    if type == "simple":
        # Simple: pure sounds only - merge hyphenated syllables, no brackets, no sign indices
        # a-na di-na-an lugal be-li2-ia -> ana dinaan lugal beliia -> ana dinan lugal belia
        s = s.lower().strip()
        s = remove_some_brackets(s)       # remove brackets but NOT hyphens (preserve for merging)
        s = normalize_digits(s)           # subscript/superscript -> ASCII digits
        s = strip_sign_indices(s)         # ba2 -> ba, gu4 -> gu
        s = diacritics_to_ascii(s)        # š -> sz, ṣ -> s, etc.
        s = strip_accents(s)              # í -> i, ù -> u, etc. (MUST precede [^a-zA-Z] filter)
        s = re.sub(r'-+', '', s)          # MERGE hyphens (a-na -> ana)
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?*\s]+", r"", s)  # remove non-alpha except spaces
        s = reduce_doubled_vowels(s)  # aa->a but aaa stays (real lengthening)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = re.sub(r'\s+', ' ', s).strip()
    elif type == "original":
        # Complex/original: diacritic form with hyphens AND brackets preserved
        # Uses normalize_translit() to convert CDLI notation -> diacritics (li2 -> lí)
        # Note: vowel2/3->diacritic only applies to Akkadian; other languages keep indices
        s = standardize_translit_source(s)    # (d)X → {d}X, 0.66666 → ⅔
        s = s.lower().strip()
        # Keep brackets - don't call remove_brackets()
        s = normalize_translit(s, language=language, aggressive_sz=True, allow_vowel23=True)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = re.sub(r'\s+', ' ', s).strip()
        # Note: NO strip_accents() - we WANT diacritics (š, ú, ù, lí, etc.)
    elif type == "group":
        s = unicodeToAscii(s.lower().strip())
        s = normalize_dashes(s)
        s = normalize_digits(s)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = strip_accents(s)
        #s = remove_some_brackets(s)
    normalized_string = s.strip()
    # 3) Fix spaced-out cuneiform gap tokens
    normalized_string = fix_cuneiform_gap(normalized_string)

    if not use_prefix:
        return normalized_string

    if str(style).lower() in ("simple", "tags", "compact"):
        src = LANG_CODE.get(language, language)
        tgt = MODERN_CODE.get(modern, modern)
        # this function is specifically: transliteration -> modern language
        pref = make_prefix(
            task="Translate",
            src_lang=src,
            tgt_lang=tgt,
            src_script="cunei",
            tgt_script="latn",
            variant=TYPE_CODE[type],   # transliteration
        )
        return pref + normalized_string

    if type == "simple":
        return 'Transliterate ' + language + ' cuneiform to simple Latin characters: ' + normalized_string
    elif type == "original":
        return 'Transliterate ' + language + ' cuneiform to complex Latin characters: ' + normalized_string


def normalizeString_cuneiform_rev_transliterate(s, use_prefix=True, type="simple", language="Akkadian", modern="English", style="T5"):
    """
    Normalize transliteration SOURCE text for transliteration-to-cuneiform tasks.

    Prepares Latin transliteration input for training models that convert
    transliterated text back to cuneiform glyphs (reverse transliteration).

    Args:
        s: Input transliterated string
        use_prefix: If True, prepend a task prefix for T5-style models
        type: Input transliteration style:
            - 'simple': ASCII-only transliteration
            - 'original': Complex scholarly notation with diacritics
            - 'group': Word-grouped syllables (e.g., 'a-na di-na-an')
        language: Ancient language ('Akkadian', 'Sumerian', etc.)
        modern: Modern language for prefix ('English', 'German') - used in compact style
        style: Prefix style ('T5' for verbose, 'simple'/'compact' for short)

    Returns:
        Normalized string, optionally with task prefix like:
        'Convert simple transliterated Latin characters to Akkadian cuneiform: ...'
    """
    if type == "simple":
        # Simple: pure sounds only - merge hyphenated syllables, no brackets, no sign indices
        # a-na di-na-an lugal be-li2-ia -> ana dinaan lugal beliia -> ana dinan lugal belia
        s = s.lower().strip()
        s = remove_some_brackets(s)       # remove brackets but NOT hyphens (preserve for merging)
        s = normalize_digits(s)           # subscript/superscript -> ASCII digits
        s = strip_sign_indices(s)         # ba2 -> ba, gu4 -> gu
        s = diacritics_to_ascii(s)        # š -> sz, ṣ -> s, etc.
        s = strip_accents(s)              # í -> i, ù -> u, etc. (MUST precede [^a-zA-Z] filter)
        s = re.sub(r'-+', '', s)          # MERGE hyphens (a-na -> ana)
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?*\s]+", r"", s)  # remove non-alpha except spaces
        s = reduce_doubled_vowels(s)  # aa->a but aaa stays (real lengthening)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = re.sub(r'\s+', ' ', s).strip()
    elif type == "original":
        # Complex/original: diacritic form with hyphens AND brackets preserved
        # Uses normalize_translit() to convert CDLI notation -> diacritics (li2 -> lí)
        # Note: vowel2/3->diacritic only applies to Akkadian; other languages keep indices
        s = standardize_translit_source(s)    # (d)X → {d}X, 0.66666 → ⅔
        s = s.lower().strip()
        # Keep brackets - don't call remove_brackets()
        s = normalize_translit(s, language=language, aggressive_sz=True, allow_vowel23=True)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = re.sub(r'\s+', ' ', s).strip()
        # Note: NO strip_accents() - we WANT diacritics (š, ú, ù, lí, etc.)
    elif type == "group":
        s = unicodeToAscii(s.lower().strip())
        s = normalize_dashes(s)
        s = remove_some_brackets(s)
        s = normalize_digits(s)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = strip_accents(s)
    normalized_string = s.strip()
    # 3) Fix spaced-out cuneiform gap tokens
    normalized_string = fix_cuneiform_gap(normalized_string)

    if not use_prefix:
        return normalized_string

    if str(style).lower() in ("simple", "tags", "compact"):
        src = LANG_CODE.get(language, language)
        tgt = MODERN_CODE.get(modern, modern)
        # this function is specifically: transliteration -> modern language
        pref = make_prefix(
            task="Translate",
            src_lang=src,
            tgt_lang=tgt,
            src_script="latn",
            tgt_script="cunei",
            variant=TYPE_CODE[type],   # transliteration
        )
        return pref + normalized_string

    if type == "simple" :
        return 'Convert simple transliterated Latin characters to ' + language + ' cuneiform: ' + normalized_string
    elif type == "group":
        return 'Convert transliterated Latin characters to ' + language + ' cuneiform: ' + normalized_string
    elif type == "original" :
        return 'Convert complex transliterated Latin characters to ' + language + ' cuneiform: ' + normalized_string


def normalizeString_cuneiform_transliterate_translate(s, use_prefix=True, task="Translate", type="simple", language="Akkadian", modern="English", style="T5"):
    """
    Normalize transliteration SOURCE text for transliteration-to-translation tasks.

    Prepares transliterated cuneiform input for training models that translate
    from Latin transliteration directly to a modern language (English/German).

    Supports two task types:
        - Translate: Convert transliteration to modern language
        - Group: Combine syllables into likely word units

    Args:
        s: Input transliterated string
        use_prefix: If True, prepend a task prefix for T5-style models
        task: Task type ('Translate' or 'Group')
        type: Transliteration style:
            - 'simple': ASCII-only (a-z)
            - 'original': Complex with scholarly diacritics
            - 'group': Word-grouped syllables
        language: Ancient language ('Akkadian', 'Sumerian', etc.)
        modern: Target modern language ('English', 'German')
        style: Prefix style ('T5' for verbose, 'simple'/'compact' for short)

    Returns:
        Normalized string, optionally with task prefix like:
        'Translate simple Akkadian transliteration to English: ...'
    """
    if type=="simple":
        # Simple: pure sounds only - merge hyphenated syllables, no brackets, no sign indices
        # a-na di-na-an lugal be-li2-ia -> ana dinaan lugal beliia -> ana dinan lugal belia
        s = s.lower().strip()
        s = remove_some_brackets(s)       # remove brackets but NOT hyphens (preserve for merging)
        s = normalize_digits(s)           # subscript/superscript -> ASCII digits
        s = strip_sign_indices(s)         # ba2 -> ba, gu4 -> gu
        s = diacritics_to_ascii(s)        # š -> sz, ṣ -> s, etc.
        s = strip_accents(s)              # í -> i, ù -> u, etc. (MUST precede [^a-zA-Z] filter)
        s = re.sub(r'-+', '', s)          # MERGE hyphens (a-na -> ana)
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?*\s]+", r"", s)  # remove non-alpha except spaces
        s = reduce_doubled_vowels(s)  # aa->a but aaa stays (real lengthening)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = re.sub(r'\s+', ' ', s).strip()
    elif type=="original":
        # Complex/original: diacritic form with hyphens AND brackets preserved
        # Uses normalize_translit() to convert CDLI notation -> diacritics (li2 -> lí)
        # Note: vowel2/3->diacritic only applies to Akkadian; other languages keep indices
        s = standardize_translit_source(s)    # (d)X → {d}X, 0.66666 → ⅔
        s = s.lower().strip()
        # Keep brackets - don't call remove_brackets()
        s = normalize_translit(s, language=language, aggressive_sz=True, allow_vowel23=True)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = re.sub(r'\s+', ' ', s).strip()
        # Note: NO strip_accents() - we WANT diacritics (š, ú, ù, lí, etc.)
    elif type == "group":
        s = unicodeToAscii(s.lower().strip())
        s = normalize_dashes(s)
        #s = remove_some_brackets(s)
        s = normalize_digits(s)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = strip_accents(s)
    normalized_string = s.strip()
    # 3) Fix spaced-out cuneiform gap tokens
    normalized_string = fix_cuneiform_gap(normalized_string)
    
    if not use_prefix:
        return normalized_string

    if str(style).lower() in ("simple", "tags", "compact"):
        src = LANG_CODE.get(language, language)
        tgt = MODERN_CODE.get(modern, modern)
        # this function is specifically: transliteration -> modern language
        pref = make_prefix(
            task="Translate",
            src_lang=src,
            tgt_lang=tgt,
            src_script="latn",
            tgt_script="latn",
            variant=TYPE_CODE[type],   # transliteration
        )
        return pref + normalized_string
    
    if task == "Translate":
        if type == "simple":
            return 'Translate simple ' + language + ' transliteration to ' + modern + ': ' + normalized_string
        elif type == "original":
            return 'Translate complex ' + language + ' transliteration to ' + modern + ': ' + normalized_string
        elif type == "group":
            return 'Translate ' + language + ' transliteration to ' + modern + ': ' + normalized_string
    elif task == "Group":
        if type == "simple":
            return 'Group simple ' + language + ' transliteration into likely words: ' + normalized_string
        elif type == "original":
            return 'Group complex ' + language + ' transliteration into likely words: ' + normalized_string


ASCII_TO_DIACRITIC = [
    (re.compile(r'\bsz\b', flags=re.IGNORECASE), lambda m: 'š' if m.group(0).islower() else 'Š'),
    # common “sh” encoding sometimes appears too
    (re.compile(r'\bsh\b', flags=re.IGNORECASE), lambda m: 'š' if m.group(0).islower() else 'Š'),
]

def normalize_translit_alphabet(s: str) -> str:
    """
    Convert common ASCII digraphs to proper diacritic characters.

    Handles standalone tokens where ASCII digraphs represent diacritics:
        - 'sz' / 'SZ' -> 'š' / 'Š' (shin)
        - 'sh' / 'SH' -> 'š' / 'Š' (alternate shin encoding)

    Works in NFC normalization to keep diacritics stable.

    Args:
        s: Input string with ASCII digraph encodings

    Returns:
        String with digraphs converted to proper Unicode diacritics
    """
    s = unicodedata.normalize("NFC", s)
    for pat, repl in ASCII_TO_DIACRITIC:
        s = pat.sub(repl, s)
    return s

def normalize_gaps_x_tokens(s: str) -> str:
    """
    Convert x-runs into asterisk sequences using token logic.
    Assumes whitespace tokenization (so normalize whitespace first).

    Single 'x' -> '*'
    Multiple 'x x x' or longer -> '* * *' (always 3 asterisks for longer gaps)

    We use asterisks rather than special tokens like <gap>/<big_gap> to:
    1. Avoid confusion with ancient languages that may use 'x' deliberately
    2. Keep a simple, consistent representation for lacunae/gaps
    3. Asterisk is unlikely to appear in cuneiform transliterations
    """
    toks = s.split()
    out = []
    i = 0
    while i < len(toks):
        if toks[i].lower() == 'x':
            j = i
            while j < len(toks) and toks[j].lower() == 'x':
                j += 1
            run_len = j - i
            # Single x -> single asterisk; multiple x's -> three asterisks
            out.append("*" if run_len == 1 else "* * *")
            i = j
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)

def convert_vowel23_everywhere(s: str) -> str:
    """
    Convert ALL vowel+2/3 patterns to diacritics (aggressive conversion).

    CDLI/ORACC use number suffixes to indicate vowel quality:
        - a2, e2, i2, u2 -> á, é, í, ú (acute accent)
        - a3, e3, i3, u3 -> à, è, ì, ù (grave accent)

    WARNING: This is aggressive and converts everywhere in the string,
    which may incorrectly convert Sumerian sign indices like 'gu4', 'du3'.
    For safer conversion, use akkadian_vowel23_to_diacritics_hyphenaware().

    Args:
        s: Input string with vowel+number notation

    Returns:
        String with vowel+2/3 patterns converted to accented vowels
    """
    s = unicodedata.normalize("NFC", s).translate(_SUBSCRIPT_TO_ASCII)

    def _inner(m: re.Match) -> str:
        v, d = m.group(1), m.group(2)
        if v.isupper():
            return _VOWEL23_MAP_UP[(v.lower(), d)]
        return _VOWEL23_MAP[(v, d)]

    return re.sub(r"([AEIUaeiu])([23])", _inner, s)

# Detect already-diacritic Akkadian vowels used by Kaggle style
_HAS_DIACRITIC_VOWELS = re.compile(r"[áàéèíìúùÁÀÉÈÍÌÚÙ]")

# Detect CDLI/ORACC vowel-number encoding:
# - token-level: u2, i3, a2, e3 as whole tokens or after hyphen boundaries
# - braced: {tug2}, {d}...
_V23_TOKEN = re.compile(r"(?i)(?<![A-Za-z0-9])([aeiu])([23])(?![A-Za-z0-9])")
_V23_IN_BRACES = re.compile(r"\{[^}]*[AEIUaeiu][23][^}]*\}")

# Sumerian-ish heuristics: lots of index digits beyond just 2/3 vowel readings,
# and common sign-index patterns like "gu4", "du3", "he2", "me3", "lu2", "e2", etc.
# (Not perfect, but works well in practice.)
_SUX_INDEXY = re.compile(
    r"(?i)\b(?:gu4|du3|he2|me3|lu2|e2|u3|i3|a2|ka2|ki2|ga2|bi2|ce3|de3|ti3|sig7)\b"
)
# Also detect a lot of digit-bearing tokens (Sumerian tends to have many)
_MANY_DIGIT_TOKENS = re.compile(r"(?:\b\w*\d+\w*\b.*){6,}")  # 6+ digit-bearing tokens

def looks_sumerian_indexy(s: str) -> bool:
    """
    Heuristically detect if text looks like Sumerian with sign indices.

    Sumerian transliterations use numeric indices to disambiguate homophonic
    signs (e.g., gu4, du3, he2, me3, lu2). This differs from Akkadian vowel
    quality markers (u2 = ú, u3 = ù).

    Used to prevent incorrectly converting Sumerian sign indices to
    Akkadian diacritics.

    Args:
        s: Input transliterated text

    Returns:
        True if text appears to contain Sumerian-style sign indices
    """
    if _SUX_INDEXY.search(s):
        return True
    if _MANY_DIGIT_TOKENS.search(s):
        return True
    return False

def normalize_translit(
    s: str,
    *,
    language: str = "Akkadian",
    aggressive_sz: bool = True,
    allow_vowel23: bool = True,
) -> str:
    """
    Source-agnostic transliteration normalizer.

    - Always safe steps: NFC, remove controls, dash normalize, sub/sup digit normalize, whitespace.
    - Always normalizes sz/sh -> š if aggressive_sz True.
    - Applies vowel2/3->diacritic (u2->ú, u3->ù, etc.) ONLY when:
        * language is Akkadian (akk), AND
        * it looks like CDLI/ORACC-style vowel-number encoding, AND
        * it does NOT look like Sumerian index-heavy text.
    """
    if not s:
        return ""

    # --- universal safe normalization ---
    s = unicodedata.normalize("NFC", s)
    s = remove_control_characters(s)
    s = normalize_dashes(s)
    s = normalize_digits(s)             # subscripts/superscripts -> ASCII digits
    s = re.sub(r"\s+", " ", s).strip()

    # --- safe-ish alphabet normalization across corpora ---
    if aggressive_sz:
        # handles rasz2 -> raš2, {asz} -> {aš}, etc.
        s = normalize_sz_to_shin(s)
        s = normalize_translit_alphabet(s)  # if you also want sh->š etc.

    lang = LANG_CODE.get(language, language)

    # If already has diacritic vowels, don't try to infer/convert vowel23 again
    already_diacritic = bool(_HAS_DIACRITIC_VOWELS.search(s))

    # IMPORTANT: Only apply vowel2/3->diacritic conversion for Akkadian.
    # Sumerian, Elamite, Hittite, etc. use numeric indices for SIGN DISAMBIGUATION
    # (e.g., gu4, du3, he2), NOT vowel quality. Converting these would corrupt the data.
    # Languages that should NEVER have vowel conversion:
    #   - sux (Sumerian): gu4, du3, he2, me3 are sign indices
    #   - elx (Elamite): similar sign index system
    #   - hit (Hittite): similar sign index system
    #   - gmy (Mycenaean Greek): Linear B, different system entirely
    is_akkadian = lang == "akk"

    # Decide whether to apply vowel2/3->diacritic conversion
    if (
        allow_vowel23
        and is_akkadian  # Only Akkadian uses vowel numbers for quality
        and not already_diacritic
        and (bool(_V23_TOKEN.search(s)) or bool(_V23_IN_BRACES.search(s)) or ("2" in s or "3" in s))
        and not looks_sumerian_indexy(s)  # Extra safety: skip if text looks Sumerian-ish
    ):
        # determinatives like {tug2} -> {túg}
        s = cdli_oracc_convert_braced_determinatives(s)

        # syllables like be-li2-ia -> be-lí-ia
        s = akkadian_vowel23_to_diacritics_hyphenaware(s)

        # If you STILL want standalone tokens a2/u3 as well, keep this:
        s = cdli_oracc_vowel_numbers_to_diacritics(s)


        # IMPORTANT: do not run convert_vowel23_everywhere() after this
        # (that one will happily convert inside other tokens and can be too aggressive)

    # Final whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =============================================================================
# CDLI ASCII Notation Conversion (diacritics → subscript numbers + ASCII)
# =============================================================================
# CDLI/ORACC conventions:
#   - Vowel diacritics: á/à → a2/a3, ú/ù → u2/u3, etc.
#   - š → sz, ṣ → s,, ṭ → t,, ḫ → h,
# Reference: https://oracc.museum.upenn.edu/doc/help/editinginatf/

# Diacritic vowels to CDLI subscript numbers (reverse of _VOWEL23_MAP)
_DIACRITIC_TO_CDLI = {
    # acute accent → 2
    'á': 'a2', 'Á': 'A2',
    'é': 'e2', 'É': 'E2',
    'í': 'i2', 'Í': 'I2',
    'ú': 'u2', 'Ú': 'U2',
    # grave accent → 3
    'à': 'a3', 'À': 'A3',
    'è': 'e3', 'È': 'E3',
    'ì': 'i3', 'Ì': 'I3',
    'ù': 'u3', 'Ù': 'U3',
    # circumflex → 4 (less common but used in some traditions)
    'â': 'a4', 'Â': 'A4',
    'ê': 'e4', 'Ê': 'E4',
    'î': 'i4', 'Î': 'I4',
    'û': 'u4', 'Û': 'U4',
}

# Semitic consonant diacritics to CDLI ASCII
_SEMITIC_DIACRITIC_TO_CDLI = {
    'š': 'sz', 'Š': 'SZ',
    'ṣ': 's,', 'Ṣ': 'S,',
    'ṭ': 't,', 'Ṭ': 'T,',
    'ḫ': 'h,', 'Ḫ': 'H,',
    # Some sources use ĥ for ḫ
    'ĥ': 'h,', 'Ĥ': 'H,',
    # Aleph (glottal stop) - convert to ASCII apostrophe
    'ʾ': "'",  # U+02BE MODIFIER LETTER RIGHT HALF RING
    'ʿ': "'",  # U+02BF MODIFIER LETTER LEFT HALF RING (ayin, sometimes used)
    'ˀ': "'",  # U+02C0 MODIFIER LETTER GLOTTAL STOP
    'ˁ': "'",  # U+02C1 MODIFIER LETTER REVERSED GLOTTAL STOP
}


def diacritics_to_cdli_vowels(s: str) -> str:
    """
    Convert diacritic vowels to CDLI subscript number notation.

    Converts: á→a2, à→a3, ú→u2, ù→u3, etc.
    This is the reverse of CDLI/ORACC vowel-number to diacritic conversion.

    Args:
        s: Input string with diacritic vowels (OARE/Kaggle style)

    Returns:
        String with vowel diacritics converted to CDLI number notation
    """
    for diac, cdli in _DIACRITIC_TO_CDLI.items():
        s = s.replace(diac, cdli)
    return s


def diacritics_to_cdli_consonants(s: str) -> str:
    """
    Convert Semitic consonant diacritics to CDLI ASCII notation.

    Converts: š→sz, ṣ→s,, ṭ→t,, ḫ→h,
    CDLI uses digraphs/comma notation for these sounds.

    Args:
        s: Input string with Semitic diacritics

    Returns:
        String with consonant diacritics converted to CDLI ASCII
    """
    for diac, cdli in _SEMITIC_DIACRITIC_TO_CDLI.items():
        s = s.replace(diac, cdli)
    return s


def normalize_translit_cdli(
    s: str,
    *,
    language: str = "Akkadian",
    convert_vowels: bool = True,
    convert_consonants: bool = True,
    lowercase: bool = True,
    remove_brackets: bool = True,
    normalize_determinative: bool = True,
) -> str:
    """
    Normalize transliteration to CDLI ASCII notation.

    Converts diacritics to CDLI-style ASCII representation:
        - Vowel diacritics → subscript numbers (á→a2, ù→u3, etc.)
        - Semitic consonants → ASCII digraphs (š→sz, ḫ→h,, etc.)
        - Subscript/superscript digits → ASCII digits (₂→2, ²→2)

    This is the inverse of normalize_translit() which converts TO diacritics.
    Use this for outputting in CDLI notation or when training with CDLI-style data.

    Args:
        s: Input string (may have diacritics from OARE or other sources)
        language: Source language (currently unused, for future compatibility)
        convert_vowels: If True, convert vowel diacritics to numbers
        convert_consonants: If True, convert consonant diacritics to ASCII
        lowercase: If True, convert to lowercase for consistency
        remove_brackets: If True, remove editorial brackets (preserves {determinatives})
        normalize_determinative: If True, normalize determinative content ({DUMU} → {dumu})

    Returns:
        String in CDLI ASCII notation
    """
    if not s:
        return ""

    # --- universal safe normalization ---
    s = unicodedata.normalize("NFC", s)
    s = remove_control_characters(s)
    s = normalize_dashes(s)

    # Convert parenthetical determinatives to curly braces BEFORE bracket removal,
    # otherwise (d)IŠKUR becomes "d IŠKUR" instead of "{d}IŠKUR".
    # Safe no-op for CDLI data which already uses curly braces.
    s = normalize_translit_determinatives(s)

    # Remove editorial brackets (preserves curly braces for determinatives)
    if remove_brackets:
        s = remove_editorial_brackets(s)

    # Convert subscript/superscript digits to ASCII FIRST
    # (before vowel conversion, so we don't double-convert)
    s = normalize_digits(s)

    # Convert vowel diacritics to CDLI subscript notation
    if convert_vowels:
        s = diacritics_to_cdli_vowels(s)

    # Convert Semitic consonant diacritics to CDLI ASCII
    if convert_consonants:
        s = diacritics_to_cdli_consonants(s)

    # Normalize determinative content for consistency
    if normalize_determinative:
        s = normalize_determinatives(s)

    # Lowercase for consistency across data sources
    if lowercase:
        s = s.lower()

    # Final whitespace normalization
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalizeString_cuneiform_transliterate_minimal_cdli(
    s,
    use_prefix=True,
    language="Akkadian",
    modern="English",
    style="T5",
    source: str | None = None,
    lowercase: bool = True,
    remove_brackets: bool = True,
    normalize_determinative: bool = True,
):
    """
    Normalize transliteration to CDLI ASCII notation for ML training.

    Converts all diacritics to CDLI-style ASCII representation:
        - Vowel diacritics → subscript numbers (á→a2, ù→u3)
        - Semitic consonants → digraphs (š→sz, ḫ→h,)
        - Unicode subscripts → ASCII digits (₂→2)

    Applies consistent normalization for ML training:
        - Removes editorial brackets [...], (...), ⸢...⸣, <...>
        - Preserves semantic determinatives {...} like {d}, {dumu}, {ki}
        - Lowercases for consistency across data sources
        - Normalizes determinative content ({DUMU} → {dumu})

    This provides a balance of accurately relaying the cuneiform transliteration
    while minimizing the need to resize embeddings for special characters.

    Args:
        s: Input transliteration string
        use_prefix: If True, prepend task prefix
        language: Source ancient language
        modern: Target modern language
        style: Prefix style ('T5' for verbose, 'compact' for short)
        source: Data source hint (unused, for compatibility)
        lowercase: If True, convert to lowercase for consistency (default: True)
        remove_brackets: If True, remove editorial brackets (default: True)
        normalize_determinative: If True, normalize {DUMU} → {dumu} (default: True)

    Returns:
        Normalized string in CDLI ASCII notation, optionally prefixed
    """
    # 1) Convert to CDLI notation with normalization options
    s = normalize_translit_cdli(
        s,
        language=language,
        lowercase=lowercase,
        remove_brackets=remove_brackets,
        normalize_determinative=normalize_determinative,
    )

    # 2) Gap handling - unified asterisk approach
    # gap_filler(): explicit markers (vac, ...) -> '*'
    # normalize_gaps_x_tokens(): x-runs -> '*' or '* * *'
    s = gap_filler(s)
    s = re.sub(r"\s+", " ", s).strip()
    s = normalize_gaps_x_tokens(s)

    # 3) Doc refs
    s = remove_doc_refs(s)
    s = re.sub(r"\s+", " ", s).strip()

    normalized_string = s

    if not use_prefix:
        return normalized_string

    if str(style).lower() in ("simple", "tags", "compact"):
        src = LANG_CODE.get(language, language)
        tgt = MODERN_CODE.get(modern, modern)
        pref = make_prefix(
            task="Translate",
            src_lang=src,
            tgt_lang=tgt,
            src_script="latn",
            tgt_script="latn",
            variant=TYPE_CODE["group"],
        )
        return pref + normalized_string

    return f"Translate {language} transliteration to {modern}: {normalized_string}"


# Lowercase, trim, and remove non-letter characters
def normalizeString_cuneiform_transliterate_minimal_diacritic(
    s,
    use_prefix=True,
    language="Akkadian",
    modern="English",
    style="T5",
    source: str | None = None,   # <-- new
):
    # 1) transliteration normalization routed by language + source
    s = normalize_translit_determinatives(s)  # (d)X → {d}X before translit normalization
    s = normalize_translit(s, language=language)

    # 2) gap handling - unified asterisk approach
    # gap_filler() handles explicit gap markers (vac, ..., [...]  etc.) -> '*'
    # normalize_gaps_x_tokens() handles x-runs:
    #   - single 'x' -> '*'
    #   - multiple 'x x x' -> '* * *'
    # Using asterisks avoids confusion with ancient languages that may use 'x'
    s = gap_filler(s)
    s = re.sub(r"\s+", " ", s).strip()
    s = normalize_gaps_x_tokens(s)

    # 3) doc refs
    s = remove_doc_refs(s)
    s = re.sub(r"\s+", " ", s).strip()

    normalized_string = s

    if not use_prefix:
        return normalized_string

    if str(style).lower() in ("simple", "tags", "compact"):
        src = LANG_CODE.get(language, language)
        tgt = MODERN_CODE.get(modern, modern)
        pref = make_prefix(
            task="Translate",
            src_lang=src,
            tgt_lang=tgt,
            src_script="latn",
            tgt_script="latn",
            variant=TYPE_CODE["group"],
        )
        return pref + normalized_string

    return f"Translate {language} transliteration to {modern}: {normalized_string}"

# Lowercase, trim, and remove non-letter characters
def normalizeString_cuneiform_transliterate_minimal_legacy(s, use_prefix=True, language="Akkadian", modern="English", style="T5"):
    s = unicodeToAscii(s.lower().strip())
    s = normalize_dashes(s)
    #s = remove_some_brackets(s)
    s = normalize_digits(s)
    s = gap_filler(s)
    s = remove_doc_refs(s)
    s = strip_accents(s)
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    #s = remove_brackets(s)
    normalized_string = s.strip()
    
    if not use_prefix:
        return normalized_string

    if str(style).lower() in ("simple", "tags", "compact"):
        src = LANG_CODE.get(language, language)
        tgt = MODERN_CODE.get(modern, modern)
        # this function is specifically: transliteration -> modern language
        pref = make_prefix(
            task="Translate",
            src_lang=src,
            tgt_lang=tgt,
            src_script="latn",
            tgt_script="latn",
            variant=TYPE_CODE["group"],   # transliteration
        )
        return pref + normalized_string
    # 3) Fix spaced-out cuneiform gap tokens
    #normalized_string = fix_cuneiform_gap(normalized_string)
    return 'Translate ' + language + ' transliteration to ' + modern + ': ' + normalized_string

def normalizeString_cuneiform_transliterate_minimal(
    s,
    use_prefix=True,
    language="Akkadian",
    modern="English",
    style="T5",
    method="cdli",              # <-- changed default to "cdli"
    source: str | None = None,
    lowercase: bool = True,
    remove_brackets: bool = True,
    normalize_determinative: bool = True,
):
    """
    Normalize cuneiform transliteration for ML training.

    Routes to the appropriate normalization function based on method:
        - "cdli": CDLI ASCII notation (á→a2, š→sz) - DEFAULT
                  Recommended for training: uses only ASCII chars + numbers,
                  minimizes embedding vocabulary requirements.
        - "diacritic": Unicode diacritics (a2→á, sz→š)
                       Traditional Assyriological notation.
        - "legacy": Older normalization approach.

    Args:
        s: Input transliteration string
        use_prefix: If True, prepend task prefix for T5-style models
        language: Source ancient language ('Akkadian', 'Sumerian', etc.)
        modern: Target modern language ('English', 'German')
        style: Prefix style ('T5' for verbose, 'compact' for short)
        method: Normalization method ('cdli', 'diacritic', or 'legacy')
        source: Data source hint for source-specific handling
        lowercase: If True, convert to lowercase (CDLI method only)
        remove_brackets: If True, remove editorial brackets (CDLI method only)
        normalize_determinative: If True, normalize determinative content (CDLI method only)

    Returns:
        Normalized transliteration string, optionally prefixed
    """
    if method == "cdli":
        return normalizeString_cuneiform_transliterate_minimal_cdli(
            s=s,
            use_prefix=use_prefix,
            language=language,
            modern=modern,
            style=style,
            source=source,
            lowercase=lowercase,
            remove_brackets=remove_brackets,
            normalize_determinative=normalize_determinative,
        )
    elif method == "diacritic":
        return normalizeString_cuneiform_transliterate_minimal_diacritic(
            s=s,
            use_prefix=use_prefix,
            language=language,
            modern=modern,
            style=style,
            source=source,
        )
    elif method == "legacy":
        return normalizeString_cuneiform_transliterate_minimal_legacy(
            s=s,
            use_prefix=use_prefix,
            language=language,
            modern=modern,
            style=style,
        )
    else:
        raise ValueError(f"Unknown method={method!r}. Valid options: 'cdli', 'diacritic', 'legacy'")


def fix_suprasigillum(text):
    # This pattern means:
    #  - 's' followed by 1+ non-word characters (spaces, etc.)
    #  - 'u' followed by 1+ non-word chars
    #  - 'p' ...
    #  - and so on until 'm'
    pattern = r"s\W+u\W+p\W+r\W+a\W+s\W+i\W+g\W+i\W+l\W+l\W+u\W+m"
    return re.sub(pattern, "suprasigillum", text)


def normalize_cuneiform_spacing(s: str) -> str:
    """
    Ensure each cuneiform sign in `s` is separated by exactly one space.
    - Strips leading/trailing whitespace
    - Replaces any sequence of 2+ whitespace chars with a single space
    """
    # Trim ends
    s = s.strip()
    # Collapse runs of whitespace to one space
    return re.sub(r'\s{2,}', ' ', s)

def normalizeString_cuneiform(s, use_prefix=True, task="Translate", type="simple", language="Akkadian", modern="English", style="T5"):
    """
    Normalize cuneiform Unicode glyph text for ML model input.

    Prepares actual cuneiform Unicode characters (U+12000-U+123FF) for model
    training. Splits each glyph into a separate token (space-separated) for
    character-level processing suitable for ByT5 or similar models.

    Processing steps:
        1. Remove brackets and editorial markup
        2. Normalize gap/lacuna markers
        3. Remove document reference numbers
        4. Space-separate each cuneiform sign
        5. Fix suprasigillum spacing issues
        6. Optionally add task prefix

    Args:
        s: Input string containing cuneiform Unicode characters
        use_prefix: If True, prepend a task prefix for T5-style models
        task: Task type ('Translate' or 'Transliterate')
        type: Output style ('simple', 'group', 'original') for transliteration
        language: Source ancient language ('Akkadian', 'Sumerian', etc.)
        modern: Target modern language ('English', 'German')
        style: Prefix style ('T5' for verbose, 'simple'/'compact' for short)

    Returns:
        Normalized string with space-separated glyphs, optionally prefixed

    Example:
        >>> normalizeString_cuneiform("𒀀𒈾", use_prefix=True)
        'Translate Akkadian cuneiform to English: 𒀀 𒈾'
    """
    s = remove_brackets(s)
    s = gap_filler(s)
    s = remove_doc_refs(s)
    normalized_string = ' '.join(s)  # This joins every character with a space, treating each as a separate token
    normalized_string = fix_suprasigillum(normalized_string)
    normalized_string = normalize_cuneiform_spacing(normalized_string)
    # Add the prefix if use_prefix is True

    if not use_prefix:
        return normalized_string

    if str(style).lower() in ("simple", "tags", "compact"):
        src = LANG_CODE.get(language, language)
        tgt = MODERN_CODE.get(modern, modern)
        # this function is specifically: transliteration -> modern language
        pref = make_prefix(
            task="Translate",
            src_lang=src,
            tgt_lang=tgt,
            src_script="cunei",
            tgt_script="latn",
            variant=TYPE_CODE["group"],   # transliteration
        )
        return pref + normalized_string

    if task == "Translate":
        return 'Translate ' + language + ' cuneiform to '  + modern + ': ' + normalized_string
    elif task == "Transliterate":
        if type == "simple":
            return 'Transliterate ' + language + ' cuneiform to simple Latin characters: ' + normalized_string
        elif type == "group":
            return 'Transliterate ' + language + ' cuneiform to Latin characters: ' + normalized_string
        elif type == "original":
            return 'Transliterate ' + language + ' cuneiform to complex Latin characters: ' + normalized_string

def read_and_process_file_legacy(file_path):
    # Check if the file_path is a URL
    if file_path.startswith('http://') or file_path.startswith('https://'):
        # Fetch the content from the URL
        response = requests.get(file_path)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        lines = response.text.strip().split('\n')
    else:
        # Open the local file and read the lines
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().strip().split('\n')
    # Replace ". . . " with "*" in each line
    processed_lines = [re.sub(r'\s*\.\s*\.\s*\.\s*', '*', line) for line in lines]
    return processed_lines


def read_and_process_file(
    file_path: str,
    *,
    source: str | None = None,
    kind: str | None = None,
    postprocess=None,
    encoding: str = "utf-8",
) -> list[str]:
    if file_path.startswith(("http://", "https://")):
        resp = requests.get(file_path, timeout=60)
        resp.raise_for_status()
        raw_lines = resp.text.splitlines()
    else:
        with open(file_path, "r", encoding=encoding) as f:
            raw_lines = f.read().splitlines()

    processed = [normalize_line_for_source(line, source=source, kind=kind) for line in raw_lines]

    if postprocess is not None:
        processed = [postprocess(line) for line in processed]

    return processed


def validate_parallel_files(*file_lists, names=None):
    """
    Validate that parallel file lists have the same number of lines.

    Parallel corpora (e.g., .tr, .en, .cu files) must have matching line counts
    for proper source/target alignment. This function checks alignment and raises
    an error if mismatched.

    Args:
        *file_lists: Variable number of lists to compare lengths
        names: Optional list of names for better error messages (e.g., ["tr", "en", "cu"])

    Raises:
        ValueError: If any list has a different length than the others

    Example:
        tr_lines = read_and_process_file("akk_train.tr")
        en_lines = read_and_process_file("akk_train.en")
        cu_lines = read_and_process_file("akk_train.cu")
        validate_parallel_files(tr_lines, en_lines, cu_lines, names=["tr", "en", "cu"])
    """
    if len(file_lists) < 2:
        return  # Nothing to compare

    lengths = [len(lst) for lst in file_lists]
    if len(set(lengths)) > 1:  # Not all same length
        if names and len(names) == len(file_lists):
            details = ", ".join(f"{n}={l}" for n, l in zip(names, lengths))
        else:
            details = ", ".join(f"list{i}={l}" for i, l in enumerate(lengths))
        raise ValueError(f"Parallel file line count mismatch: {details}")


def convert(lst):
   res_dict = {}
   for i in range(0, len(lst), 2):
       res_dict[lst[i]] = lst[i + 1]
   return res_dict

def collapse_spaces(obj):
    # This function now does no printing.
    # Just processes the string(s) and returns them silently.
    def _collapse_spaces_in_string(s):
        return re.sub(r'\s+', ' ', s).strip()
    if isinstance(obj, str):
        return _collapse_spaces_in_string(obj)
    elif isinstance(obj, (tuple, list)) and len(obj) == 2 and all(isinstance(x, str) for x in obj):
        return (
            _collapse_spaces_in_string(obj[0]),
            _collapse_spaces_in_string(obj[1])
        )
    else:
        raise ValueError(f"Expected a single string or a 2-string pair, got: {obj}")


def remove_control_characters(s):
    """
    Remove all Cc, Cf, Cs, Co, Cn categories — i.e. non-printable/control chars.
    """
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

def normalize(text):
    """
    Final cleanup pass for any text before use in training.

    Performs three essential cleanup operations:
        1. Remove all control/non-printable characters
        2. Collapse multiple whitespace to single space
        3. Strip leading/trailing whitespace

    This should be called as the last normalization step.

    Args:
        text: Any string to clean up

    Returns:
        Clean string ready for model input
    """
    text = remove_control_characters(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def trim_singles(pairs, max_length1, max_length2, max_length_threshold, min_length_threshold):
    """
    Filter and trim single-element sequences by length constraints.

    Used to filter training data by word count, removing sequences that
    are too short (noise) or too long (memory issues).

    Args:
        pairs: List of single-element tuples/lists containing text
        max_length1: Maximum character length for truncation
        max_length2: Unused (kept for interface compatibility)
        max_length_threshold: Max word count (above = dropped)
        min_length_threshold: Min word count (below = dropped)

    Returns:
        List of filtered, normalized, and truncated strings
    """
    valid_pairs = []
    for pair in pairs:
        # Make sure the pair itself is not None AND has at least one element
        if pair and pair[0]:
            valid_pairs.append(pair)
    # 2. Filter out pairs by word count threshold
    max_filtered_pairs = [
        p for p in valid_pairs
        if len(p[0].split()) <= max_length_threshold
    ]
    min_filtered_pairs = [
        p for p in max_filtered_pairs
        if len(p[0].split()) >= min_length_threshold
    ]
    # 3. Normalize and trim
    trimmed_pairs = []
    for p in min_filtered_pairs:
        s1 = p[0]  # p is presumably a 1-element tuple or a 1-element list
        # Normalize
        s1_norm = normalize(s1)
        # Truncate
        s1_trunc = s1_norm[:max_length1]
        # Normalize again just to be safe
        s1_final = normalize(s1_trunc)
        trimmed_pairs.append(s1_final)
    return trimmed_pairs

def trim_pairs(pairs, max_length1, max_length2, max_length_threshold, min_length_threshold):
    """
    Filter and trim source-target pairs by length constraints.

    Used to filter parallel training data, keeping only pairs where both
    source and target meet word count requirements.

    Args:
        pairs: List of (source, target) tuples
        max_length1: Maximum character length for source truncation
        max_length2: Maximum character length for target truncation
        max_length_threshold: Max word count for source (target gets -5)
        min_length_threshold: Min word count (both sides)

    Returns:
        List of filtered, normalized, and truncated (source, target) tuples
    """
    valid_pairs = []
    for pair in pairs:
        # Ensure the pair has 2 elements and neither is None
        if pair and len(pair) == 2 and pair[0] and pair[1]:
            valid_pairs.append(pair)
    # Filter out pairs by word count threshold
    max_filtered_pairs = [
        (s1, s2) for s1, s2 in valid_pairs
        if len(s1.split()) <= max_length_threshold and len(s2.split()) <= (max_length_threshold - 5)
    ]
    min_filtered_pairs = [
        (s1, s2) for s1, s2 in max_filtered_pairs
        if len(s1.split()) >= min_length_threshold and len(s2.split()) >= min_length_threshold
    ]
    trimmed_pairs = []
    for s1, s2 in min_filtered_pairs:
        # Normalize
        s1_norm = normalize(s1)
        s2_norm = normalize(s2)
        # Truncate
        s1_trunc = s1_norm[:max_length1]
        s2_trunc = s2_norm[:max_length2]
        # Normalize again
        s1_final = normalize(s1_trunc)
        s2_final = normalize(s2_trunc)
        trimmed_pairs.append((s1_final, s2_final))
    return trimmed_pairs


def trim_examples(examples,
                  max_length1, max_length2,
                  max_length_threshold, min_length_threshold):
    """
    examples: list of (source_text, target_text, lang_code)
    returns:   list of (source_text, target_text, lang_code),
               filtered by length, normalized & truncated
    """
    valid = []
    # 1) Drop any bad records
    for src, tgt, lang in examples:
        if not (src and tgt):
            continue
        valid.append((src, tgt, lang))
    # 2) Filter by token‐count thresholds
    filtered = [
        (s, t, lang)
        for s, t, lang in valid
        if min_length_threshold <= len(s.split()) <= max_length_threshold
        and min_length_threshold <= len(t.split()) <= (max_length_threshold)
    ]
    # 3) Normalize, truncate, normalize again
    trimmed = []
    for s, t, lang in filtered:
        # your existing normalize() function just works on strings
        s1 = normalize(s)
        t1 = normalize(t)
        # truncate character‑wise (or adjust to your token count logic)
        s2 = s1[:max_length1]
        t2 = t1[:max_length2]
        # final cleanup
        s3 = normalize(s2)
        t3 = normalize(t2)
        trimmed.append((s3, t3, lang))
    return trimmed

def nllb_filter_pairs_by_length(pairs, min_length=3):
    """
    Keep only those parallel examples whose source_text and target_text
    each have at least `min_length` tokens (words).

    Args:
      pairs (list of dict): each dict must contain at least
        - "source_text": str
        - "target_text": str
      min_length (int): minimum number of tokens required on each side

    Returns:
      list of dict: the filtered list of pairs
    """
    filtered = []
    for ex in pairs:
        src_len = len(ex["source_text"].split())
        tgt_len = len(ex["target_text"].split())
        if src_len >= min_length and tgt_len >= min_length:
            filtered.append(ex)
    return filtered

def nllb_filter_singles_by_length(pairs, min_length=3):
    """
    Keep only those parallel examples whose source_text and target_text
    each have at least `min_length` tokens (words).

    Args:
      pairs (list of dict): each dict must contain at least
        - "source_text": str
      min_length (int): minimum number of tokens required on each side

    Returns:
      list of dict: the filtered list of pairs
    """
    filtered = []
    for ex in pairs:
        src_len = len(ex["source_text"].split())
        if src_len >= min_length:
            filtered.append(ex)
    return filtered

def reverse_nllb_pairs(pairs):
    reverse_pairs = [
        {
          "source_text":   ex["target_text"],
          "target_text":   ex["source_text"],
          "src_lang":      ex["tgt_lang"],
          "tgt_lang":      ex["src_lang"],
        }
        for ex in pairs
    ]
    return reverse_pairs


def readLangsCDLINLLB(user_directory="/", max_length_akk=5000, max_length_en=5000, max_length_threshold=100, min_length_threshold=50, debug=True, group=False):
    """
    Load CDLI Akkadian parallel corpus for NLLB model training.

    Reads CDLI (Cuneiform Digital Library Initiative) test/validation data
    and creates source-target pairs for translation tasks.

    Args:
        user_directory: Base path for data files (e.g., '/home/user')
        max_length_akk: Maximum length for Akkadian text (unused, for interface compat)
        max_length_en: Maximum length for English text (unused, for interface compat)
        max_length_threshold: Upper length filter threshold
        min_length_threshold: Minimum length filter (pairs below this are dropped)
        debug: If True, print sample pairs during loading
        group: If True, include word-grouped transliteration tasks

    Returns:
        Tuple of (val_pairs, pairs) where pairs are dicts with:
            - source_text: Normalized source string
            - target_text: Normalized target string
            - src_lang: BCP-47 source language code
            - tgt_lang: BCP-47 target language code
    """
    akk_cdli_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'akk_cdli_test.tr'))
    akk_cdli_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'akk_cdli_test.en'))
    akk_cdli_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'akk_cdli_valid.tr'))
    akk_cdli_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'akk_cdli_valid.en'))
    akk_cdli_transliteration_val = akk_cdli_transliteration_val + akk_cdli_en_test
    akk_cdli_en_val = akk_cdli_en_val = akk_cdli_en_test
    ###Create list
    val_pairs = []
    ###Translate from Akkadian cuneiform to English
    for src, tgt in zip(akk_cdli_transliteration_val, akk_cdli_en_val):
            val_pairs.append({
            "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Akkadian")),
            "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
            "src_lang": LANG_MAP["akk_tr"],
            "tgt_lang": "eng_Latn",
        })
    if debug == True:
        print(val_pairs[len(val_pairs)-1])
    if group:
        ###Translate from transliterated Akkadian to English
        for src, tgt in zip(akk_cdli_transliteration_val, akk_cdli_en_val):
                val_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Akkadian")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["akk_gp"],
                "tgt_lang": "eng_Latn",
            })
        if debug == True:
            print(val_pairs[len(val_pairs)-1])
    if group:
        ###Group Akkadian transliterated signs to words
        for src in akk_cdli_transliteration_val:
                val_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Group", type="original", language="Akkadian")),
                "target_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, type="group", language="Akkadian")),
                "src_lang": LANG_MAP["akk_tr"],
                "tgt_lang": LANG_MAP["akk_gp"],
            })
        if debug == True:
            print(val_pairs[len(val_pairs)-1])
    ###Augment with reversed sequences
    val_rev_pairs = reverse_nllb_pairs(val_pairs)
    val_pairs = val_pairs + val_rev_pairs
    ###Merge all data sets
    pairs = val_pairs
    print(f"Total pairs imported: {len(pairs)}")
    val_pairs = nllb_filter_pairs_by_length(val_pairs, min_length=min_length_threshold)
    pairs = val_pairs
    print(f"Total pairs available: {len(pairs)}")
    return val_pairs, pairs

def readLangsTrainNLLB(user_directory="/", max_length_akk=5000, max_length_en=5000, max_length_threshold=100, min_length_threshold=50, debug=True, akk=True, sux=True, elx=True, gmy=True, hit=True, group=False, transliterate=False, de=True, cuneiform=True, rev=True, ld=True, data="both"):
    print("Reading lines...")
    ###Create list
    train_pairs = []
    test_pairs = []
    val_pairs = []
    # map of ISO codes (akk, hit, gmy, sux, elx) to BCP‑47 style with a script tag:
    if group:
        LANG_MAP = {
            "akk_cu": "akk_Cune",    # Akkadian cuneiform
            "akk_tr": "akk_Tran",    # Akkadian transliteration
            "akk_gp": "akk_Latn",    # Akkadian transliteration
            "elx_tr": "elx_Tran",    # Elamite transliteration
            "elx_gp": "elx_Latn",    # Elamite transliteration
            "gmy_lb": "gmy_Linb",    # Mycenean Greek script
            "gmy_tr": "gmy_Tran",    # Mycenean Greek transliteration
            "gmy_gp": "gmy_Latn",    # Mycenean Greek transliteration
            "hit_tr": "hit_Tran",    # Hittite transliteration
            "hit_gp": "hit_Latn",    # Hittite transliteration
            "sux_cu": "sux_Cune",    # Sumerian transliteration
            "sux_tr": "sux_Tran",    # Sumerian transliteration
            "sux_gp": "sux_Latn",    # Sumerian transliteration
        }
    else:
        LANG_MAP = {
            "akk_cu": "akk_Cune",    # Akkadian cuneiform
            "akk_tr": "akk_Latn",    # Akkadian transliteration
            "elx_tr": "elx_Latn",    # Elamite transliteration
            "gmy_lb": "gmy_Linb",    # Mycenean Greek script
            "gmy_tr": "gmy_Latn",    # Mycenean Greek transliteration
            "hit_tr": "hit_Latn",    # Hittite transliteration
            "sux_cu": "sux_Cune",    # Sumerian transliteration
            "sux_tr": "sux_Latn",    # Sumerian transliteration
        }
    if akk:
        ##############
        ###Akkadian###
        ##############
        # Read the file and split into lines
        if data=="line":
            akk_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_line.cu'))
            akk_gs_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_line.tr'))
            akk_gs_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_line.en'))
            akk_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_line.cu'))
            akk_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_line.tr'))
            akk_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_line.en'))
            akk_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_line.cu'))
            akk_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_line.tr'))
            akk_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_line.en'))
            akk_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_line.tr'))
            akk_transliteration_train = akk_gs_transliteration_train + akk_cdli_transliteration_train
            akk_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_line.en'))
            akk_en_train = akk_gs_en_train + akk_cdli_en_train
        elif data=="document":
            akk_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_document.cu'))
            akk_gs_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_document.tr'))
            akk_gs_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_document.en'))
            akk_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_document.cu'))
            akk_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_document.tr'))
            akk_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_document.en'))
            akk_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_document.cu'))
            akk_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_document.tr'))
            akk_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_document.en'))
            akk_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_document.tr'))
            akk_transliteration_train = akk_gs_transliteration_train + akk_cdli_transliteration_train
            akk_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_document.en'))
            akk_en_train = akk_gs_en_train + akk_cdli_en_train
        elif data=="both":
            akk_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train.cu'))
            akk_gs_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train.tr'))
            akk_gs_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train.en'))
            akk_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test.cu'))
            akk_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test.tr'))
            akk_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test.en'))
            akk_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid.cu'))
            akk_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid.tr'))
            akk_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid.en'))
            akk_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train.tr'))
            akk_transliteration_train = akk_gs_transliteration_train + akk_cdli_transliteration_train
            akk_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train.en'))
            akk_en_train = akk_gs_en_train + akk_cdli_en_train
        # Split every line into pairs and normalize
        if cuneiform:
            ###Translate from Akkadian cuneiform to English
            for src, tgt in zip(akk_cuneiform_train, akk_gs_en_train):
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Akkadian")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["akk_cu"],
                    "tgt_lang": "eng_Latn",
                })
            for src, tgt in zip(akk_cuneiform_test, akk_en_test):
                    test_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Akkadian")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["akk_cu"],
                    "tgt_lang": "eng_Latn",
                })
            for src, tgt in zip(akk_cuneiform_val, akk_en_val):
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Akkadian")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["akk_cu"],
                    "tgt_lang": "eng_Latn",
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        ###Translate from original transliterated Akkadian to English
        for src, tgt in zip(akk_transliteration_train, akk_en_train):
                train_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Akkadian")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["akk_tr"],
                "tgt_lang": "eng_Latn",
            })
        for src, tgt in zip(akk_transliteration_test, akk_en_test):
                test_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Akkadian")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["akk_tr"],
                "tgt_lang": "eng_Latn",
            })
        for src, tgt in zip(akk_transliteration_val, akk_en_val):
                val_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Akkadian")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["akk_tr"],
                "tgt_lang": "eng_Latn",
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
        if group:
            ###Translate from transliterated Akkadian to English
            for src, tgt in zip(akk_transliteration_train, akk_en_train):
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Akkadian")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["akk_gp"],
                    "tgt_lang": "eng_Latn",
                })
            for src, tgt in zip(akk_transliteration_test, akk_en_test):
                    test_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Akkadian")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["akk_gp"],
                    "tgt_lang": "eng_Latn",
                })
            for src, tgt in zip(akk_transliteration_val, akk_en_val):
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Akkadian")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["akk_gp"],
                    "tgt_lang": "eng_Latn",
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        if group:
            ###Group Akkadian transliterated signs to words
            for src in akk_transliteration_train:
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Group", type="original", language="Akkadian")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, type="group", language="Akkadian")),
                    "src_lang": LANG_MAP["akk_tr"],
                    "tgt_lang": LANG_MAP["akk_gp"],
                })
            for src in akk_transliteration_test:
                    test_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Group", type="original", language="Akkadian")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, type="group", language="Akkadian")),
                    "src_lang": LANG_MAP["akk_tr"],
                    "tgt_lang": LANG_MAP["akk_gp"],
                })
            for src in akk_transliteration_val:
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Group", type="original", language="Akkadian")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, type="group", language="Akkadian")),
                    "src_lang": LANG_MAP["akk_tr"],
                    "tgt_lang": LANG_MAP["akk_gp"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        if transliterate:
            ###Transliterate from Akkadian cuneiform to single signs
            for src, tgt in zip(akk_cuneiform_train, akk_gs_transliteration_train):
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Akkadian")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="original")),
                    "src_lang": LANG_MAP["akk_cu"],
                    "tgt_lang": LANG_MAP["akk_tr"],
                })
            for src, tgt in zip(akk_cuneiform_test, akk_transliteration_test):
                    test_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Akkadian")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="original")),
                    "src_lang": LANG_MAP["akk_cu"],
                    "tgt_lang": LANG_MAP["akk_tr"],
                })
            for src, tgt in zip(akk_cuneiform_val, akk_transliteration_val):
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Akkadian")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="original")),
                    "src_lang": LANG_MAP["akk_cu"],
                    "tgt_lang": LANG_MAP["akk_tr"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
            if group:
                ###Transliterate from Akkadian cuneiform to words
                for src, tgt in zip(akk_cuneiform_train, akk_transliteration_train):
                        train_pairs.append({
                        "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Akkadian")),
                        "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="group")),
                        "src_lang": LANG_MAP["akk_cu"],
                        "tgt_lang": LANG_MAP["akk_gp"],
                    })
                for src, tgt in zip(akk_cuneiform_test, akk_transliteration_test):
                        test_pairs.append({
                        "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Akkadian")),
                        "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="group")),
                        "src_lang": LANG_MAP["akk_cu"],
                        "tgt_lang": LANG_MAP["akk_gp"],
                    })
                for src, tgt in zip(akk_cuneiform_val, akk_transliteration_val):
                        val_pairs.append({
                        "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Akkadian")),
                        "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="group")),
                        "src_lang": LANG_MAP["akk_cu"],
                        "tgt_lang": LANG_MAP["akk_gp"],
                    })
                if debug == True:
                    print(train_pairs[len(train_pairs)-1])
    if sux:
        ##############
        ###Sumerian###
        ##############
        # Read the file and split into lines
        if data=="line":
            sux_ox_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.cu'))
            sux_ox_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.tr'))
            sux_ox_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.en'))
            sux_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_line.cu'))
            sux_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_line.tr'))
            sux_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_line.en'))
            sux_cuneiform_train = sux_ox_cuneiform_train + sux_cdli_cuneiform_train
            sux_transliteration_train = sux_ox_transliteration_train + sux_cdli_transliteration_train
            sux_en_train = sux_ox_en_train + sux_cdli_en_train
            sux_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_line.cu'))
            sux_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_line.tr'))
            sux_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_line.en'))
            sux_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_line.cu'))
            sux_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_line.tr'))
            sux_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_line.en'))
        elif data=="document":
            sux_ox_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.cu'))
            sux_ox_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.tr'))
            sux_ox_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.en'))
            sux_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_document.cu'))
            sux_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_document.tr'))
            sux_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_document.en'))
            sux_cuneiform_train = sux_ox_cuneiform_train + sux_cdli_cuneiform_train
            sux_transliteration_train = sux_ox_transliteration_train + sux_cdli_transliteration_train
            sux_en_train = sux_ox_en_train + sux_cdli_en_train
            sux_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_document.cu'))
            sux_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_document.tr'))
            sux_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_document.en'))
            sux_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_document.cu'))
            sux_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_document.tr'))
            sux_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_document.en'))
        elif data=="both":
            sux_ox_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.cu'))
            sux_ox_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.tr'))
            sux_ox_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.en'))
            sux_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'sux_cdli_train.cu'))
            sux_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train.tr'))
            sux_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train.en'))
            sux_cuneiform_train = sux_ox_cuneiform_train + sux_cdli_cuneiform_train
            sux_transliteration_train = sux_ox_transliteration_train + sux_cdli_transliteration_train
            sux_en_train = sux_ox_en_train + sux_cdli_en_train
            sux_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test.cu'))
            sux_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test.tr'))
            sux_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test.en'))
            sux_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid.cu'))
            sux_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid.tr'))
            sux_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid.en'))
        # Split every line into pairs and normalize
        if cuneiform:
            ###Translate from Sumerian cuneiform to English
            for src, tgt in zip(sux_cuneiform_train, sux_en_train):
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Sumerian")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["sux_cu"],
                    "tgt_lang": "eng_Latn",
                })
            for src, tgt in zip(sux_cuneiform_test, sux_en_test):
                test_pairs.append({
                "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Sumerian")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["sux_cu"],
                "tgt_lang": "eng_Latn",
            })
            for src, tgt in zip(sux_cuneiform_val, sux_en_val):
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Sumerian")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["sux_cu"],
                    "tgt_lang": "eng_Latn",
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        ###Translate from original transliterated Sumerian to English
        for src, tgt in zip(sux_transliteration_train, sux_en_train):
                train_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Sumerian")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["sux_tr"],
                "tgt_lang": "eng_Latn",
            })
        for src, tgt in zip(sux_transliteration_test, sux_en_test):
                test_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Sumerian")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["sux_tr"],
                "tgt_lang": "eng_Latn",
            })
        for src, tgt in zip(sux_transliteration_val, sux_en_val):
                val_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Sumerian")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["sux_tr"],
                "tgt_lang": "eng_Latn",
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
        if group:
            ###Translate from transliterated Sumerian to English
            for src, tgt in zip(sux_transliteration_train, sux_en_train):
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Sumerian")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["sux_gp"],
                    "tgt_lang": "eng_Latn",
                })
            for src, tgt in zip(sux_transliteration_test, sux_en_test):
                    test_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Sumerian")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["sux_gp"],
                    "tgt_lang": "eng_Latn",
                })
            for src, tgt in zip(sux_transliteration_val, sux_en_val):
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Sumerian")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["sux_gp"],
                    "tgt_lang": "eng_Latn",
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        if group:
            ###Group Sumerian transliterated signs to words
            for src in sux_transliteration_train:
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Group", type="original", language="Sumerian")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, type="group", language="Sumerian")),
                    "src_lang": LANG_MAP["sux_tr"],
                    "tgt_lang": LANG_MAP["sux_gp"],
                })
            for src in sux_transliteration_test:
                    test_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Group", type="original", language="Sumerian")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, type="group", language="Sumerian")),
                    "src_lang": LANG_MAP["sux_tr"],
                    "tgt_lang": LANG_MAP["sux_gp"],
                })
            for src in sux_transliteration_val:
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Group", type="original", language="Sumerian")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, type="group", language="Sumerian")),
                    "src_lang": LANG_MAP["sux_tr"],
                    "tgt_lang": LANG_MAP["sux_gp"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        if transliterate:
            ###Transliterate from Sumerian cuneiform to single signs
            for src, tgt in zip(sux_cuneiform_train, sux_transliteration_train):
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Sumerian")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="original")),
                    "src_lang": LANG_MAP["sux_cu"],
                    "tgt_lang": LANG_MAP["sux_tr"],
                })
            for src, tgt in zip(sux_cuneiform_test, sux_transliteration_test):
                    test_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Sumerian")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="original")),
                    "src_lang": LANG_MAP["sux_cu"],
                    "tgt_lang": LANG_MAP["sux_tr"],
                })
            for src, tgt in zip(sux_cuneiform_val, sux_transliteration_val):
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Sumerian")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="original")),
                    "src_lang": LANG_MAP["sux_cu"],
                    "tgt_lang": LANG_MAP["sux_tr"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
            if group:
                ###Transliterate from Sumerian cuneiform to words
                for src, tgt in zip(sux_cuneiform_train, sux_transliteration_train):
                        train_pairs.append({
                        "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Sumerian")),
                        "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="group")),
                        "src_lang": LANG_MAP["sux_cu"],
                        "tgt_lang": LANG_MAP["sux_gp"],
                    })
                for src, tgt in zip(sux_cuneiform_test, sux_transliteration_test):
                        test_pairs.append({
                        "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Sumerian")),
                        "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="group")),
                        "src_lang": LANG_MAP["sux_cu"],
                        "tgt_lang": LANG_MAP["sux_gp"],
                    })
                for src, tgt in zip(sux_cuneiform_val, sux_transliteration_val):
                        val_pairs.append({
                        "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Sumerian")),
                        "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="group")),
                        "src_lang": LANG_MAP["sux_cu"],
                        "tgt_lang": LANG_MAP["sux_gp"],
                    })
                if debug == True:
                    print(train_pairs[len(train_pairs)-1])
    if elx:
        ###############
        ###Elamite###
        ###############
        # Read the file and split into lines
        if data=="line":
            elx_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_line.tr'))
            elx_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_line.en'))
        elif data=="document":
            elx_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_document.tr'))
            elx_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_document.en'))
        elif data=="both":
            elx_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train.tr'))
            elx_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train.en'))
        # Split every line into pairs and normalize
        ###Translate from original transliterated Elamite to English
        for src, tgt in zip(elx_transliteration_train, elx_en_train):
                train_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Elamite")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["elx_tr"],
                "tgt_lang": "eng_Latn",
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
        if group:
            ###Translate from transliterated Elamite to English
            for src, tgt in zip(elx_transliteration_train, elx_en_train):
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Elamite")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["elx_gp"],
                    "tgt_lang": "eng_Latn",
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        if group:
            ###Group Elamite transliterated signs to words
            for src in elx_transliteration_train:
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Group", type="original", language="Elamite")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, type="group", language="Elamite")),
                    "src_lang": LANG_MAP["elx_tr"],
                    "tgt_lang": LANG_MAP["elx_gp"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
    if gmy:
        ##############
        ###Linear B###
        ##############
        # Read the file and split into lines
        gmy_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_train.cu'))
        gmy_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_train.tr'))
        gmy_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_train.en'))
        gmy_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_test.cu'))
        gmy_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_test.tr'))
        gmy_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_test.en'))
        gmy_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_valid.cu'))
        gmy_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_valid.tr'))
        gmy_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_valid.en'))
        # Split every line into pairs and normalize
        if cuneiform:
            ###Translate from Linear B cuneiform to English
            for src, tgt in zip(gmy_cuneiform_train, gmy_en_train):
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Linear B")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["gmy_lb"],
                    "tgt_lang": "eng_Latn",
                })
            for src, tgt in zip(gmy_cuneiform_test, gmy_en_test):
                    test_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Linear B")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["gmy_lb"],
                    "tgt_lang": "eng_Latn",
                })
            for src, tgt in zip(gmy_cuneiform_val, gmy_en_val):
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Linear B")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["gmy_lb"],
                    "tgt_lang": "eng_Latn",
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        ###Translate from original transliterated Linear B to English
        for src, tgt in zip(gmy_transliteration_train, gmy_en_train):
                train_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Linear B")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["gmy_tr"],
                "tgt_lang": "eng_Latn",
            })
        for src, tgt in zip(gmy_transliteration_test, gmy_en_test):
                test_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Linear B")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["gmy_tr"],
                "tgt_lang": "eng_Latn",
            })
        for src, tgt in zip(gmy_transliteration_val, gmy_en_val):
                val_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Linear B")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["gmy_tr"],
                "tgt_lang": "eng_Latn",
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
        if group:
            ###Translate from transliterated Linear B to English
            for src, tgt in zip(gmy_transliteration_train, gmy_en_train):
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Linear B")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["gmy_gp"],
                    "tgt_lang": "eng_Latn",
                })
            for src, tgt in zip(gmy_transliteration_test, gmy_en_test):
                    test_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Linear B")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["gmy_gp"],
                    "tgt_lang": "eng_Latn",
                })
            for src, tgt in zip(gmy_transliteration_val, gmy_en_val):
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Linear B")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["gmy_gp"],
                    "tgt_lang": "eng_Latn",
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        if group:
            ###Group Linear B transliterated signs to words
            for src in gmy_transliteration_train:
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Group", type="original", language="Linear B")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, type="group", language="Linear B")),
                    "src_lang": LANG_MAP["gmy_tr"],
                    "tgt_lang": LANG_MAP["gmy_gp"],
                })
            for src in gmy_transliteration_test:
                    test_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Group", type="original", language="Linear B")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, type="group", language="Linear B")),
                    "src_lang": LANG_MAP["gmy_tr"],
                    "tgt_lang": LANG_MAP["gmy_gp"],
                })
            for src in gmy_transliteration_val:
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Group", type="original", language="Linear B")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, type="group", language="Linear B")),
                    "src_lang": LANG_MAP["gmy_tr"],
                    "tgt_lang": LANG_MAP["gmy_gp"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        if transliterate:
            ###Transliterate from Linear B to single sounds
            for src, tgt in zip(gmy_cuneiform_train, gmy_transliteration_train):
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Linear B")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="original")),
                    "src_lang": LANG_MAP["gmy_lb"],
                    "tgt_lang": LANG_MAP["gmy_tr"],
                })
            for src, tgt in zip(gmy_cuneiform_test, gmy_transliteration_test):
                    test_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Linear B")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="original")),
                    "src_lang": LANG_MAP["gmy_lb"],
                    "tgt_lang": LANG_MAP["gmy_tr"],
                })
            for src, tgt in zip(gmy_cuneiform_val, gmy_transliteration_val):
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Linear B")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="original")),
                    "src_lang": LANG_MAP["gmy_lb"],
                    "tgt_lang": LANG_MAP["gmy_tr"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
            if group:
                ###Transliterate from Linear B to grouped
                for src, tgt in zip(gmy_cuneiform_train, gmy_transliteration_train):
                        train_pairs.append({
                        "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Linear B")),
                        "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="group")),
                        "src_lang": LANG_MAP["gmy_lb"],
                        "tgt_lang": LANG_MAP["gmy_gp"],
                    })
                for src, tgt in zip(gmy_cuneiform_test, gmy_transliteration_test):
                        test_pairs.append({
                        "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Linear B")),
                        "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="group")),
                        "src_lang": LANG_MAP["gmy_lb"],
                        "tgt_lang": LANG_MAP["gmy_gp"],
                    })
                for src, tgt in zip(gmy_cuneiform_val, gmy_transliteration_val):
                        val_pairs.append({
                        "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Linear B")),
                        "target_text": normalize(normalizeString_cuneiform_transliterate(tgt, use_prefix=False, type="group")),
                        "src_lang": LANG_MAP["gmy_lb"],
                        "tgt_lang": LANG_MAP["gmy_gp"],
                    })
                if debug == True:
                    print(train_pairs[len(train_pairs)-1])
    if hit:
        #############
        ###Hittite###
        #############
        # Read the file and split into lines
        if data=="line":
            hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_line.tr'))
            hit_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_line.en'))
            hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_line.tr'))
            hit_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_line.en'))
            hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_line.tr'))
            hit_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_line.en'))
        elif data=="document":
            hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_document.tr'))
            hit_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_document.en'))
            hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_document.tr'))
            hit_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_document.en'))
            hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_document.tr'))
            hit_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_document.en'))
        elif data=="both":
            hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train.tr'))
            hit_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train.en'))
            hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test.tr'))
            hit_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test.en'))
            hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid.tr'))
            hit_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid.en'))
        if ld:
            hit_ld_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_train.tr'))
            hit_ld_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_train.en'))
            hit_ld_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_test.tr'))
            hit_ld_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_test.en'))
            hit_ld_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_valid.tr'))
            hit_ld_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_valid.en'))
            # Add ALL data_line data to training (including ld test/val for max coverage)
            # Val/test remain main-data-only to avoid evaluating on data_line examples
            hit_transliteration_train = hit_transliteration_train + hit_ld_transliteration_train + hit_ld_transliteration_test + hit_ld_transliteration_val
            hit_en_train = hit_en_train + hit_ld_en_train + hit_ld_en_test + hit_ld_en_val
        # Split every line into pairs and normalize
        ###Translate from original transliterated Hittite to English
        for src, tgt in zip(hit_transliteration_train, hit_en_train):
                train_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Hittite")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["hit_tr"],
                "tgt_lang": "eng_Latn",
            })
        for src, tgt in zip(hit_transliteration_test, hit_en_test):
                test_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Hittite")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["hit_tr"],
                "tgt_lang": "eng_Latn",
            })
        for src, tgt in zip(hit_transliteration_val, hit_en_val):
                val_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Hittite")),
                "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                "src_lang": LANG_MAP["hit_tr"],
                "tgt_lang": "eng_Latn",
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
        if group:
            ###Translate from transliterated Hittite to English
            for src, tgt in zip(hit_transliteration_train, hit_en_train):
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Hittite")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["hit_gp"],
                    "tgt_lang": "eng_Latn",
                })
            for src, tgt in zip(hit_transliteration_test, hit_en_test):
                    test_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Hittite")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["hit_gp"],
                    "tgt_lang": "eng_Latn",
                })
            for src, tgt in zip(hit_transliteration_val, hit_en_val):
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Hittite")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["hit_gp"],
                    "tgt_lang": "eng_Latn",
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        if group:
            ###Group Hittite transliterated signs to words
            for src in hit_transliteration_train:
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Group", type="original", language="Hittite")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, type="group", language="Hittite")),
                    "src_lang": LANG_MAP["hit_tr"],
                    "tgt_lang": LANG_MAP["hit_gp"],
                })
            for src in hit_transliteration_test:
                    test_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Group", type="original", language="Hittite")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, type="group", language="Hittite")),
                    "src_lang": LANG_MAP["hit_tr"],
                    "tgt_lang": LANG_MAP["hit_gp"],
                })
            for src in hit_transliteration_val:
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Group", type="original", language="Hittite")),
                    "target_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, type="group", language="Hittite")),
                    "src_lang": LANG_MAP["hit_tr"],
                    "tgt_lang": LANG_MAP["hit_gp"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        if de:
            # Read the file and split into lines
            if data=="line":
                hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_line.tr'))
                hit_de_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_line.de'))
                hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_line.tr'))
                hit_de_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_line.de'))
                hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_line.tr'))
                hit_de_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_line.de'))
            elif data=="document":
                hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_document.tr'))
                hit_de_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_document.de'))
                hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_document.tr'))
                hit_de_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_document.de'))
                hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_document.tr'))
                hit_de_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_document.de'))
            elif data=="both":
                hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train.tr'))
                hit_de_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train.de'))
                hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test.tr'))
                hit_de_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test.de'))
                hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid.tr'))
                hit_de_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid.de'))
            # Split every line into pairs and normalize
            ###Translate from original transliterated Hittite to German
            for src, tgt in zip(hit_transliteration_train, hit_de_train):
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Hittite")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["hit_tr"],
                    "tgt_lang": "deu_Latn",
                })
            for src, tgt in zip(hit_transliteration_test, hit_de_test):
                    test_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Hittite")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["hit_tr"],
                    "tgt_lang": "deu_Latn",
                })
            for src, tgt in zip(hit_transliteration_val, hit_de_val):
                    val_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Hittite")),
                    "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                    "src_lang": LANG_MAP["hit_tr"],
                    "tgt_lang": "deu_Latn",
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
            if group:
                ###Translate from transliterated Hittite to German
                for src, tgt in zip(hit_transliteration_train, hit_de_train):
                        train_pairs.append({
                        "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Hittite")),
                        "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                        "src_lang": LANG_MAP["hit_gp"],
                        "tgt_lang": "deu_Latn",
                    })
                for src, tgt in zip(hit_transliteration_test, hit_de_test):
                        test_pairs.append({
                        "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Hittite")),
                        "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                        "src_lang": LANG_MAP["hit_gp"],
                        "tgt_lang": "deu_Latn",
                    })
                for src, tgt in zip(hit_transliteration_val, hit_de_val):
                        val_pairs.append({
                        "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Hittite")),
                        "target_text": normalize(normalizeString_en(tgt, use_prefix=False)),
                        "src_lang": LANG_MAP["hit_gp"],
                        "tgt_lang": "deu_Latn",
                    })
                if debug == True:
                    print(train_pairs[len(train_pairs)-1])
    ###Augment with reversed sequences
    if rev:
        pairs_og = train_pairs + test_pairs + val_pairs
        train_rev_pairs = reverse_nllb_pairs(train_pairs)
        train_pairs_og = train_pairs
        train_pairs = train_pairs + train_rev_pairs
        test_rev_pairs = reverse_nllb_pairs(test_pairs)
        test_pairs = test_pairs + test_rev_pairs
        val_rev_pairs = reverse_nllb_pairs(val_pairs)
        val_pairs = val_pairs + val_rev_pairs
    ###Merge all data sets
    pairs = train_pairs + test_pairs + val_pairs
    print(f"Total pairs imported: {len(pairs)}")
    train_pairs = nllb_filter_pairs_by_length(train_pairs, min_length=min_length_threshold)
    val_pairs = nllb_filter_pairs_by_length(val_pairs, min_length=min_length_threshold)
    test_pairs = nllb_filter_pairs_by_length(test_pairs, min_length=min_length_threshold)
    pairs = train_pairs + test_pairs + val_pairs
    print(f"Total pairs available: {len(pairs)}")
    return train_pairs, val_pairs, test_pairs, pairs_og, LANG_MAP

def readLangsPreTrainNLLB(user_directory="/", max_length_akk=5000, max_length_en=5000, max_length_threshold=100, min_length_threshold=50, debug=True, akk=True, sux=True, elx=True, gmy=True, hit=True, group=False, de=True, cuneiform=True, ld=True, data="both"):
    print("Reading lines...")
    ###Create list
    train_pairs = []
    # map of ISO codes (akk, hit, gmy, sux, elx) to BCP‑47 style with a script tag:
    if group:
        LANG_MAP = {
            "akk_cu": "akk_Cune",    # Akkadian cuneiform
            "akk_tr": "akk_Tran",    # Akkadian transliteration
            "akk_gp": "akk_Latn",    # Akkadian transliteration
            "elx_tr": "elx_Tran",    # Elamite transliteration
            "elx_gp": "elx_Latn",    # Elamite transliteration
            "gmy_lb": "gmy_Linb",    # Mycenean Greek script
            "gmy_tr": "gmy_Tran",    # Mycenean Greek transliteration
            "gmy_gp": "gmy_Latn",    # Mycenean Greek transliteration
            "hit_tr": "hit_Tran",    # Hittite transliteration
            "hit_gp": "hit_Latn",    # Hittite transliteration
            "sux_cu": "sux_Cune",    # Sumerian transliteration
            "sux_tr": "sux_Tran",    # Sumerian transliteration
            "sux_gp": "sux_Latn",    # Sumerian transliteration
        }
    else:
        LANG_MAP = {
            "akk_cu": "akk_Cune",    # Akkadian cuneiform
            "akk_tr": "akk_Latn",    # Akkadian transliteration
            "elx_tr": "elx_Latn",    # Elamite transliteration
            "gmy_lb": "gmy_Linb",    # Mycenean Greek script
            "gmy_tr": "gmy_Latn",    # Mycenean Greek transliteration
            "hit_tr": "hit_Latn",    # Hittite transliteration
            "sux_cu": "sux_Cune",    # Sumerian transliteration
            "sux_tr": "sux_Latn",    # Sumerian transliteration
        }
    if akk:
        ##############
        ###Akkadian###
        ##############
        # Read the file and split into lines
        if data=="line":
            akk_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_line.cu'))
            akk_gs_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_line.tr'))
            akk_gs_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_line.en'))
            akk_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_line.cu'))
            akk_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_line.tr'))
            akk_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_line.en'))
            akk_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_line.cu'))
            akk_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_line.tr'))
            akk_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_line.en'))
            akk_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_line.tr'))
            akk_transliteration_train = akk_gs_transliteration_train + akk_cdli_transliteration_train
            akk_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_line.en'))
            akk_en_train = akk_gs_en_train + akk_cdli_en_train
            akk_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_u_train_line.tr'))
            akk_transliteration_train = akk_transliteration_train + akk_transliteration_u_train
        elif data=="document":
            akk_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_document.cu'))
            akk_gs_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_document.tr'))
            akk_gs_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_document.en'))
            akk_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_document.cu'))
            akk_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_document.tr'))
            akk_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_document.en'))
            akk_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_document.cu'))
            akk_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_document.tr'))
            akk_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_document.en'))
            akk_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_document.tr'))
            akk_transliteration_train = akk_gs_transliteration_train + akk_cdli_transliteration_train
            akk_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_document.en'))
            akk_en_train = akk_gs_en_train + akk_cdli_en_train
            akk_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_u_train_document.tr'))
            akk_transliteration_train = akk_transliteration_train + akk_transliteration_u_train
        elif data=="both":
            akk_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train.cu'))
            akk_gs_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train.tr'))
            akk_gs_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train.en'))
            akk_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test.cu'))
            akk_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test.tr'))
            akk_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test.en'))
            akk_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid.cu'))
            akk_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid.tr'))
            akk_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid.en'))
            akk_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train.tr'))
            akk_transliteration_train = akk_gs_transliteration_train + akk_cdli_transliteration_train
            akk_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train.en'))
            akk_en_train = akk_gs_en_train + akk_cdli_en_train
            akk_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_u_train.tr'))
            akk_transliteration_train = akk_transliteration_train + akk_transliteration_u_train
        # Split every line into pairs and normalize
        ###Akkadian cuneiform
        if cuneiform:
            for src in akk_cuneiform_train:
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Akkadian")),
                    "src_lang": LANG_MAP["akk_cu"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        ###Transliterated Akkadian
        for src in akk_transliteration_train:
                train_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Akkadian")),
                "src_lang": LANG_MAP["akk_tr"],
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
        if group:
            ###Grouped Akkadian
            for src in akk_transliteration_train:
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Akkadian")),
                    "src_lang": LANG_MAP["akk_gp"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        ###Translated Akkadian
        for src in akk_en_train:
                train_pairs.append({
                "source_text": normalize(normalizeString_en(src, use_prefix=False)),
                "src_lang": "eng_Latn",
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
    if sux:
        ##############
        ###Sumerian###
        ##############
        # Read the file and split into lines
        if data=="line":
            sux_ox_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.cu'))
            sux_ox_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.tr'))
            sux_ox_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.en'))
            sux_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_line.cu'))
            sux_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_line.tr'))
            sux_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_line.en'))
            sux_cuneiform_train = sux_ox_cuneiform_train + sux_cdli_cuneiform_train
            sux_transliteration_train = sux_ox_transliteration_train + sux_cdli_transliteration_train
            sux_en_train = sux_ox_en_train + sux_cdli_en_train
            sux_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_line.cu'))
            sux_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_line.tr'))
            sux_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_line.en'))
            sux_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_line.cu'))
            sux_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_line.tr'))
            sux_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_line.en'))
            sux_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_u_train_line.tr'))
            sux_transliteration_train = sux_transliteration_train + sux_transliteration_u_train
        elif data=="document":
            sux_ox_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.cu'))
            sux_ox_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.tr'))
            sux_ox_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.en'))
            sux_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_document.cu'))
            sux_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_document.tr'))
            sux_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_document.en'))
            sux_cuneiform_train = sux_ox_cuneiform_train + sux_cdli_cuneiform_train
            sux_transliteration_train = sux_ox_transliteration_train + sux_cdli_transliteration_train
            sux_en_train = sux_ox_en_train + sux_cdli_en_train
            sux_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_document.cu'))
            sux_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_document.tr'))
            sux_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_document.en'))
            sux_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_document.cu'))
            sux_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_document.tr'))
            sux_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_document.en'))
            sux_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_u_train_document.tr'))
            sux_transliteration_train = sux_transliteration_train + sux_transliteration_u_train
        elif data=="both":
            sux_ox_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.cu'))
            sux_ox_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.tr'))
            sux_ox_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.en'))
            sux_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'sux_cdli_train.cu'))
            sux_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train.tr'))
            sux_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train.en'))
            sux_cuneiform_train = sux_ox_cuneiform_train + sux_cdli_cuneiform_train
            sux_transliteration_train = sux_ox_transliteration_train + sux_cdli_transliteration_train
            sux_en_train = sux_ox_en_train + sux_cdli_en_train
            sux_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test.cu'))
            sux_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test.tr'))
            sux_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test.en'))
            sux_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid.cu'))
            sux_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid.tr'))
            sux_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid.en'))
            sux_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_u_train.tr'))
            sux_transliteration_train = sux_transliteration_train + sux_transliteration_u_train
        # Split every line into pairs and normalize
        if cuneiform:
            for src in sux_cuneiform_train:
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Sumerian")),
                    "src_lang": LANG_MAP["sux_cu"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        ###Transliterated Sumerian
        for src in sux_transliteration_train:
                train_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Sumerian")),
                "src_lang": LANG_MAP["sux_tr"],
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
        if group:
            ###Grouped Sumerian
            for src in sux_transliteration_train:
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Sumerian")),
                    "src_lang": LANG_MAP["sux_gp"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        ###Translated Sumerian
        for src in sux_en_train:
                train_pairs.append({
                "source_text": normalize(normalizeString_en(src, use_prefix=False)),
                "src_lang": "eng_Latn",
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
    if elx:
        ###############
        ###Elamite###
        ###############
        # Read the file and split into lines
        if data=="line":
            elx_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_line.tr'))
            elx_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_line.en'))
            elx_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_u_train_line.tr'))
            elx_transliteration_train = elx_transliteration_train + elx_transliteration_u_train
        elif data=="document":
            elx_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_document.tr'))
            elx_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_document.en'))
            elx_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_u_train_document.tr'))
            elx_transliteration_train = elx_transliteration_train + elx_transliteration_u_train
        elif data=="both":
            elx_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train.tr'))
            elx_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train.en'))
            elx_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_u_train.tr'))
            elx_transliteration_train = elx_transliteration_train + elx_transliteration_u_train
        # Split every line into pairs and normalize
        ###Transliterated Elamite
        for src in elx_transliteration_train:
                train_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Elamite")),
                "src_lang": LANG_MAP["elx_tr"],
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
        if group:
            ###Grouped Elamite
            for src in elx_transliteration_train:
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Elamite")),
                    "src_lang": LANG_MAP["elx_gp"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        ###Translated Elamite
        for src in elx_en_train:
                train_pairs.append({
                "source_text": normalize(normalizeString_en(src, use_prefix=False)),
                "src_lang": "eng_Latn",
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
    if gmy:
        ##############
        ###Linear B###
        ##############
        # Read the file and split into lines
        gmy_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_train.cu'))
        gmy_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_train.tr'))
        gmy_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_train.en'))
        # Split every line into pairs and normalize
        if cuneiform:
            ###Linear B signs
            for src in gmy_cuneiform_train:
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform(src, use_prefix=False, task="Translate", language="Linear B")),
                    "src_lang": LANG_MAP["gmy_lb"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        ###Transliterated Linear B
        for src in gmy_transliteration_train:
                train_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Linear B")),
                "src_lang": LANG_MAP["gmy_tr"],
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
        if group:
            ###Grouped Linear B
            for src in gmy_transliteration_train:
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Linear B")),
                    "src_lang": LANG_MAP["gmy_gp"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        ###Translated Linear B
        for src in gmy_en_train:
                train_pairs.append({
                "source_text": normalize(normalizeString_en(src, use_prefix=False)),
                "src_lang": "eng_Latn",
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
    if hit:
        #############
        ###Hittite###
        #############
        # Read the file and split into lines
        if data=="line":
            hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_line.tr'))
            hit_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_line.en'))
            hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_line.tr'))
            hit_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_line.en'))
            hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_line.tr'))
            hit_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_line.en'))
            hit_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_u_train_line.tr'))
            hit_transliteration_train = hit_transliteration_train + hit_transliteration_u_train
        elif data=="document":
            hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_document.tr'))
            hit_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_document.en'))
            hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_document.tr'))
            hit_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_document.en'))
            hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_document.tr'))
            hit_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_document.en'))
            hit_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_u_train_document.tr'))
            hit_transliteration_train = hit_transliteration_train + hit_transliteration_u_train
        elif data=="both":
            hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train.tr'))
            hit_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train.en'))
            hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test.tr'))
            hit_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test.en'))
            hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid.tr'))
            hit_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid.en'))
            hit_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_u_train.tr'))
            hit_transliteration_train = hit_transliteration_train + hit_transliteration_u_train
        if ld:
            hit_ld_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_train.tr'))
            hit_ld_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_train.en'))
            hit_ld_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_test.tr'))
            hit_ld_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_test.en'))
            hit_ld_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_valid.tr'))
            hit_ld_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_valid.en'))
            # Add ALL data_line data to training (including ld test/val for max coverage)
            # Val/test remain main-data-only to avoid evaluating on data_line examples
            hit_transliteration_train = hit_transliteration_train + hit_ld_transliteration_train + hit_ld_transliteration_test + hit_ld_transliteration_val
            hit_en_train = hit_en_train + hit_ld_en_train + hit_ld_en_test + hit_ld_en_val
        if de:
            if data=="line":
                hit_de_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_line.de'))
                hit_de_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_line.de'))
                hit_de_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_line.de'))
            elif data=="document":
                hit_de_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_document.de'))
                hit_de_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_document.de'))
                hit_de_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_document.de'))
            elif data=="both":
                hit_de_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train.de'))
                hit_de_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test.de'))
                hit_de_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid.de'))
        # Split every line into pairs and normalize
        ###Transliterated Hittite
        for src in hit_transliteration_train:
                train_pairs.append({
                "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="original", language="Hittite")),
                "src_lang": LANG_MAP["hit_tr"],
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
        if group:
            ###Grouped Hittite
            for src in hit_transliteration_train:
                    train_pairs.append({
                    "source_text": normalize(normalizeString_cuneiform_transliterate_translate(src, use_prefix=False, task="Translate", type="group", language="Hittite")),
                    "src_lang": LANG_MAP["hit_gp"],
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
        ###Translated Hittite
        for src in hit_en_train:
                train_pairs.append({
                "source_text": normalize(normalizeString_en(src, use_prefix=False)),
                "src_lang": "eng_Latn",
            })
        if debug == True:
            print(train_pairs[len(train_pairs)-1])
        if de:
            ###Translated Hittite
            for src in hit_de_train:
                    train_pairs.append({
                    "source_text": normalize(normalizeString_en(src, use_prefix=False)),
                    "src_lang": "deu_Latn",
                })
            if debug == True:
                print(train_pairs[len(train_pairs)-1])
    ###Merge all data sets
    pairs = train_pairs
    print(f"Total pairs imported: {len(pairs)}")
    train_pairs = nllb_filter_singles_by_length(train_pairs, min_length=min_length_threshold)
    pairs = train_pairs
    print(f"Total pairs available: {len(pairs)}")
    return train_pairs,  pairs, LANG_MAP

def extract_unique_source_tokens(pairs, split_on=' '):
    tokens = set()
    for pair in pairs:
        tokens.update(pair[0].split(split_on))
    return tokens

def unique_token_grab(*pair_lists, split_on=' '):
    tokens = set()
    for pair_list in pair_lists:
        for pair in pair_list:
            tokens.update(pair[0].split(split_on))
    return sorted(tokens)

def add_split(pairs_dict, name, train, test, val, token_fn):
    pairs_dict["train"][name] = train
    pairs_dict["test"] [name] = test
    pairs_dict["val"]  [name] = val
    pairs_dict["token"][name] = token_fn(train, test, val)


def readLangsCDLIT5(user_directory="/", max_length_akk=5000, max_length_en=5000, max_length_threshold=100, min_length_threshold=3, debug=True, simple=True, complex=True, group=True, cuneiform=True, transliterate=True, rev=True, data="both"):
    val_pairs = []
    # Read the file and split into lines
    if data=="line":
        akk_cdli_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test_line.cu'))
        akk_cdli_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test_line.tr'))
        akk_cdli_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test_line.en'))
        akk_cdli_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid_line.cu'))
        akk_cdli_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid_line.tr'))
        akk_cuneiform_val = akk_cdli_cuneiform_val + akk_cdli_cuneiform_test
        akk_transliteration_val = akk_cdli_transliteration_val + akk_cdli_transliteration_test
        akk_cdli_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid_line.en'))
        akk_en_val = akk_cdli_en_val + akk_cdli_en_test
    elif data=="document":
        akk_cdli_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test_document.cu'))
        akk_cdli_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test_document.tr'))
        akk_cdli_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test_document.en'))
        akk_cdli_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid_document.cu'))
        akk_cdli_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid_document.tr'))
        akk_cuneiform_val = akk_cdli_cuneiform_val + akk_cdli_cuneiform_test
        akk_transliteration_val = akk_cdli_transliteration_val + akk_cdli_transliteration_test
        akk_cdli_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid_document.en'))
        akk_en_val = akk_cdli_en_val + akk_cdli_en_test
    elif data=="both":
        akk_cdli_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test.cu'))
        akk_cdli_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test.tr'))
        akk_cdli_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test.en'))
        akk_cdli_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid.cu'))
        akk_cdli_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid.tr'))
        akk_cuneiform_val = akk_cdli_cuneiform_val + akk_cdli_cuneiform_test
        akk_transliteration_val = akk_cdli_transliteration_val + akk_cdli_transliteration_test
        akk_cdli_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid.en'))
        akk_en_val = akk_cdli_en_val + akk_cdli_en_test
    if cuneiform:
        #Complex Transliterated
        akk_val_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Translate", language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
        val_pairs.append(akk_val_pairs_cuneiform_translate)
        if debug == True:
            print(f"", {akk_val_pairs_cuneiform_translate[1][0]}, " -> ", {akk_val_pairs_cuneiform_translate[1][1]})
    if complex:
        #Complex Transliterated
        akk_val_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_val[l], use_prefix=True, task="Translate", type="original", language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
        val_pairs.append(akk_val_pairs_original_transliterated_translate)
        if debug == True:
            print(f"", {akk_val_pairs_original_transliterated_translate[1][0]}, " -> ", {akk_val_pairs_original_transliterated_translate[1][1]})
    if simple:
        ####Translate simple unknown langauge transliteration to English
        akk_val_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_val[l], use_prefix=True, task="Translate", type="simple", language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
        val_pairs.append(akk_val_pairs_simple_transliterated_translate)
        if debug:
            print(f"", {akk_val_pairs_simple_transliterated_translate[1][0]}, " -> ", {akk_val_pairs_simple_transliterated_translate[1][1]})
    if group:
        #Translate Grouped Akkadian transliteration into English
        akk_val_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transliteration_val[l], use_prefix=True, language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
        val_pairs.append(akk_val_pairs_group_transliterated_translate)
        if debug:
            print(f"", {akk_val_pairs_group_transliterated_translate[1][0]}, " -> ", {akk_val_pairs_group_transliterated_translate[1][1]})
    if transliterate:
        if complex:
            ###Transliterate from Akkadian Cuenfirom to complex Latin characters
            akk_val_pairs_transliterated_original = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="original")] for l in range(len(akk_cuneiform_val))]
            val_pairs.append(akk_val_pairs_transliterated_original)
            if debug == True:
                print(f"", {akk_val_pairs_transliterated_original[1][0]}, " -> ", {akk_val_pairs_transliterated_original[1][1]})
        if simple:
            ###Transliterate from Akkadian Cuenfirom to simple Latin characters
            akk_val_pairs_transliterated_simple = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
            val_pairs.append(akk_val_pairs_transliterated_simple)
            if debug == True:
                print(f"", {akk_val_pairs_transliterated_simple[1][0]}, " -> ", {akk_val_pairs_transliterated_simple[1][1]})
        if group:
            ###Transliterate from Akkadian Cuenfirom to Latin characters
            akk_val_pairs_transliterated_group = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="group")] for l in range(len(akk_cuneiform_val))]
            val_pairs.append(akk_val_pairs_transliterated_group)
            if debug == True:
                print(f"", {akk_val_pairs_transliterated_group[1][0]}, " -> ", {akk_val_pairs_transliterated_group[1][1]})
    if rev:
        if complex:
            #Translate English to complex transliterated Akkadian
            akk_val_rev_pairs_original_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="original")] for l in range(len(akk_en_val))]
            val_pairs.append(akk_val_rev_pairs_original_transliterated_translate)
            if debug:
                print(f"", {akk_val_rev_pairs_original_transliterated_translate[1][0]}, " -> ", {akk_val_rev_pairs_original_transliterated_translate[1][1]})
        if simple:
            #Translate English to simple Akkadian transliteration
            akk_val_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="simple")] for l in range(len(akk_en_val))]
            val_pairs.append(akk_val_rev_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {akk_val_rev_pairs_simple_transliterated_translate[1][0]}, " -> ", {akk_val_rev_pairs_simple_transliterated_translate[1][1]})
        if group:
            #Translate English into Akkadian transliteration
            akk_val_rev_pairs_group_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_val[l], use_prefix=False)] for l in range(len(akk_en_val))]
            val_pairs.append(akk_val_rev_pairs_group_transliterated_translate)
            if debug == True:
                print(f"", {akk_val_rev_pairs_group_transliterated_translate[1][0]}, " -> ", {akk_val_rev_pairs_group_transliterated_translate[1][1]})
        if transliterate:
            if simple:
                ###Convert from simple transliterated Akkadian to cuneiform
                akk_val_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_val[l], use_prefix=True, type="simple", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
                val_pairs.append(akk_val_rev_pairs_transliterate_simple)
                if debug == True:
                    print(f"", {akk_val_rev_pairs_transliterate_simple[1][0]}, " -> ", {akk_val_rev_pairs_transliterate_simple[1][1]})
            if group:
                ###Convert from transliterated Akkadian to cuneiform
                akk_val_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_val[l], use_prefix=True, type="group", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
                val_pairs.append(akk_val_rev_pairs_transliterate_group)
                if debug == True:
                    print(f"", {akk_val_rev_pairs_transliterate_group[1][0]}, " -> ", {akk_val_rev_pairs_transliterate_group[1][1]})
            if complex:
                ###Convert from complex transliterated Akkadian to cuneiform
                akk_val_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_val[l], use_prefix=True, type="original", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
                val_pairs.append(akk_val_rev_pairs_transliterate_original)
                if debug == True:
                    print(f"", {akk_val_rev_pairs_transliterate_original[1][0]}, " -> ", {akk_val_rev_pairs_transliterate_original[1][1]})
    val_pairs = [pair for subset in val_pairs for pair in subset]
    pairs = val_pairs
    print(f"Total pairs imported: {len(pairs)}")
    val_pairs = trim_pairs(val_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    if debug == True:
        print("val set trimmed")
    pairs = val_pairs
    print(f"Total pairs filtered: {len(pairs)}")
    return val_pairs

def readLangsOARET5(user_directory="/", max_length_akk=5000, max_length_en=5000, max_length_threshold=100, min_length_threshold=3, debug=True, cuneiform=True, simple=False, complex=False, group=True, rev=True, transliterate=True, data="both"):
    train_pairs = []
    val_pairs = []
    # Read the file and split into lines
    akk_oare_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_document.cu'))
    akk_oare_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_document.tr'))
    akk_oare_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_valid_document.cu'))
    akk_oare_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_valid_document.tr'))
    akk_oare_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_document.en'))
    akk_oare_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_valid_document.en'))
    akk_cuneiform_train = akk_oare_cuneiform_train
    akk_transliteration_train = akk_oare_transliteration_train
    akk_cuneiform_val = akk_oare_cuneiform_val
    akk_transliteration_val = akk_oare_transliteration_val
    akk_en_train = akk_oare_en_train
    akk_en_val = akk_oare_en_val
    if cuneiform:
        #Complex Transliterated
        akk_train_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Translate", language="Akkadian"), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
        train_pairs.append(akk_train_pairs_cuneiform_translate)
        akk_val_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Translate", language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
        val_pairs.append(akk_val_pairs_cuneiform_translate)
        if debug == True:
            print(f"", {akk_val_pairs_cuneiform_translate[1][0]}, " -> ", {akk_val_pairs_cuneiform_translate[1][1]})
    if complex:
        #Complex Transliterated
        akk_train_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_train[l], use_prefix=True, task="Translate", type="original", language="Akkadian"), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_transliteration_train))]
        train_pairs.append(akk_train_pairs_original_transliterated_translate)
        akk_val_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_val[l], use_prefix=True, task="Translate", type="original", language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
        val_pairs.append(akk_val_pairs_original_transliterated_translate)
        if debug == True:
            print(f"", {akk_val_pairs_original_transliterated_translate[1][0]}, " -> ", {akk_val_pairs_original_transliterated_translate[1][1]})
    if simple:
        ####Translate simple unknown langauge transliteration to English
        akk_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_train[l], use_prefix=True, task="Translate", type="simple", language="Akkadian"), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_transliteration_train))]
        train_pairs.append(akk_train_pairs_simple_transliterated_translate)
        akk_val_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_val[l], use_prefix=True, task="Translate", type="simple", language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
        val_pairs.append(akk_val_pairs_simple_transliterated_translate)
        if debug:
            print(f"", {akk_val_pairs_simple_transliterated_translate[1][0]}, " -> ", {akk_val_pairs_simple_transliterated_translate[1][1]})
    if group:
        #Translate Grouped Akkadian transliteration into English
        akk_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transliteration_train[l], use_prefix=True, language="Akkadian"), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_transliteration_train))]
        train_pairs.append(akk_train_pairs_group_transliterated_translate)
        akk_val_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transliteration_val[l], use_prefix=True, language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
        val_pairs.append(akk_val_pairs_group_transliterated_translate)
        if debug:
            print(f"", {akk_val_pairs_group_transliterated_translate[1][0]}, " -> ", {akk_val_pairs_group_transliterated_translate[1][1]})
    if transliterate:
        if complex:
            ###Transliterate from Akkadian Cuenfirom to complex Latin characters
            akk_train_pairs_transliterated_original = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Transliterate", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_train[l], use_prefix=False, type="original")] for l in range(len(akk_cuneiform_train))]
            train_pairs.append(akk_train_pairs_transliterated_original)
            akk_val_pairs_transliterated_original = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="original")] for l in range(len(akk_cuneiform_val))]
            val_pairs.append(akk_val_pairs_transliterated_original)
            if debug == True:
                print(f"", {akk_val_pairs_transliterated_original[1][0]}, " -> ", {akk_val_pairs_transliterated_original[1][1]})
        if simple:
            ###Transliterate from Akkadian Cuenfirom to simple Latin characters
            akk_train_pairs_transliterated_simple = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Transliterate", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
            train_pairs.append(akk_train_pairs_transliterated_simple)
            akk_val_pairs_transliterated_simple = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
            val_pairs.append(akk_val_pairs_transliterated_simple)
            if debug == True:
                print(f"", {akk_val_pairs_transliterated_simple[1][0]}, " -> ", {akk_val_pairs_transliterated_simple[1][1]})
        if group:
            ###Transliterate from Akkadian Cuenfirom to Latin characters
            akk_train_pairs_transliterated_group = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Transliterate", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_train[l], use_prefix=False, type="group")] for l in range(len(akk_cuneiform_train))]
            train_pairs.append(akk_train_pairs_transliterated_group)
            akk_val_pairs_transliterated_group = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="group")] for l in range(len(akk_cuneiform_val))]
            val_pairs.append(akk_val_pairs_transliterated_group)
            if debug == True:
                print(f"", {akk_val_pairs_transliterated_group[1][0]}, " -> ", {akk_val_pairs_transliterated_group[1][1]})
    if rev:
        if complex:
            #Translate English to complex transliterated Akkadian
            akk_train_rev_pairs_original_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_train[l], use_prefix=False, type="original")] for l in range(len(akk_en_train))]
            train_pairs.append(akk_train_rev_pairs_original_transliterated_translate)
            akk_val_rev_pairs_original_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="original")] for l in range(len(akk_en_val))]
            val_pairs.append(akk_val_rev_pairs_original_transliterated_translate)
            if debug:
                print(f"", {akk_val_rev_pairs_original_transliterated_translate[1][0]}, " -> ", {akk_val_rev_pairs_original_transliterated_translate[1][1]})
        if simple:
            #Translate English to simple Akkadian transliteration
            akk_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_train[l], use_prefix=False, type="simple")] for l in range(len(akk_en_train))]
            train_pairs.append(akk_train_rev_pairs_simple_transliterated_translate)
            akk_val_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="simple")] for l in range(len(akk_en_val))]
            val_pairs.append(akk_val_rev_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {akk_val_rev_pairs_simple_transliterated_translate[1][0]}, " -> ", {akk_val_rev_pairs_simple_transliterated_translate[1][1]})
        if group:
            #Translate English into Akkadian transliteration
            akk_train_rev_pairs_group_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_train[l], use_prefix=False)] for l in range(len(akk_en_train))]
            train_pairs.append(akk_train_rev_pairs_group_transliterated_translate)
            akk_val_rev_pairs_group_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_val[l], use_prefix=False)] for l in range(len(akk_en_val))]
            val_pairs.append(akk_val_rev_pairs_group_transliterated_translate)
            if debug == True:
                print(f"", {akk_val_rev_pairs_group_transliterated_translate[1][0]}, " -> ", {akk_val_rev_pairs_group_transliterated_translate[1][1]})
        if transliterate:
            if simple:
                ###Convert from simple transliterated Akkadian to cuneiform
                akk_train_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_train[l], use_prefix=True, type="simple", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_transliteration_train))]
                train_pairs.append(akk_train_rev_pairs_transliterate_simple)
                akk_val_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_val[l], use_prefix=True, type="simple", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
                val_pairs.append(akk_val_rev_pairs_transliterate_simple)
                if debug == True:
                    print(f"", {akk_val_rev_pairs_transliterate_simple[1][0]}, " -> ", {akk_val_rev_pairs_transliterate_simple[1][1]})
            if group:
                ###Convert from transliterated Akkadian to cuneiform
                akk_train_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_train[l], use_prefix=True, type="group", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_transliteration_train))]
                train_pairs.append(akk_train_rev_pairs_transliterate_group)
                akk_val_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_val[l], use_prefix=True, type="group", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
                val_pairs.append(akk_val_rev_pairs_transliterate_group)
                if debug == True:
                    print(f"", {akk_val_rev_pairs_transliterate_group[1][0]}, " -> ", {akk_val_rev_pairs_transliterate_group[1][1]})
            if complex:
                ###Convert from complex transliterated Akkadian to cuneiform
                akk_train_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_train[l], use_prefix=True, type="original", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_transliteration_train))]
                train_pairs.append(akk_train_rev_pairs_transliterate_original)
                akk_val_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_val[l], use_prefix=True, type="original", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
                val_pairs.append(akk_val_rev_pairs_transliterate_original)
                if debug == True:
                    print(f"", {akk_train_rev_pairs_transliterate_original[1][0]}, " -> ", {akk_train_rev_pairs_transliterate_original[1][1]})
    train_pairs = [pair for subset in train_pairs for pair in subset]
    val_pairs = [pair for subset in val_pairs for pair in subset]
    pairs = train_pairs + val_pairs
    print(f"Total pairs imported: {len(pairs)}")
    train_pairs = trim_pairs(train_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    val_pairs = trim_pairs(val_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    if debug == True:
        print("val set trimmed")
    pairs = train_pairs + val_pairs
    print(f"Total pairs filtered: {len(pairs)}")
    return train_pairs, val_pairs


def readLangsTrainT5Akkademia(user_directory="/", max_length_akk=5000, max_length_en=5000, max_length_threshold=100, min_length_threshold=50, debug=True, cuneiform=True,complex=True, group=False, active_group=False, simple=False, transliterate=False, rev=True, de=False, ld=True, paragraph=False):
    print("Reading lines...")
    ###Create lists
    train_pairs = []
    test_pairs = []
    val_pairs = []
    token_pairs = []
    ##############
    ###Akkadian###
    ##############
    # Read the file and split into lines
    akk_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'Akkademia', 'NMT_input', 'train.ak'))
    akk_gs_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'Akkademia', 'NMT_input', 'train.tr'))
    akk_gs_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'Akkademia', 'NMT_input', 'train.en'))
    akk_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'Akkademia', 'NMT_input', 'test.ak'))
    akk_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'Akkademia', 'NMT_input', 'test.tr'))
    akk_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'Akkademia', 'NMT_input', 'test.en'))
    akk_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'Akkademia', 'NMT_input', 'valid.ak'))
    akk_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'Akkademia', 'NMT_input', 'valid.tr'))
    akk_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'Akkademia', 'NMT_input', 'valid.en'))
    akk_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'akk_cdli_train.tr'))
    akk_transliteration_train = akk_gs_transliteration_train + akk_cdli_transliteration_train
    akk_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'akk_cdli_train.en'))
    akk_en_train = akk_gs_en_train + akk_cdli_en_train
    # Split every line into pairs and normalize
    if cuneiform:
        ###Translate from Akkadian cuneiform to English
        akk_train_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Translate", language="Akkadian"), normalizeString_en(akk_gs_en_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
        akk_test_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=True, task="Translate", language="Akkadian"), normalizeString_en(akk_en_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
        akk_val_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Translate", language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
        train_pairs.append(akk_train_pairs_cuneiform_translate)
        test_pairs.append(akk_test_pairs_cuneiform_translate)
        val_pairs.append(akk_val_pairs_cuneiform_translate)
        token_pairs.append(akk_train_pairs_cuneiform_translate)
        token_pairs.append(akk_test_pairs_cuneiform_translate)
        token_pairs.append(akk_val_pairs_cuneiform_translate)
        if debug == True:
            print(f"", {akk_train_pairs_cuneiform_translate[1][0]}, " -> ", {akk_train_pairs_cuneiform_translate[1][1]})
    if simple:
        ###Translate from simple transliterated Akkadian to English
        akk_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_train[l], use_prefix=True, task="Translate", type="simple", language="Akkadian"), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_transliteration_train))]
        akk_test_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_test[l], use_prefix=True, task="Translate", type="simple", language="Akkadian"), normalizeString_en(akk_en_test[l], use_prefix=False)] for l in range(len(akk_transliteration_test))]
        akk_val_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_val[l], use_prefix=True, task="Translate", type="simple", language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
        train_pairs.append(akk_train_pairs_simple_transliterated_translate)
        test_pairs.append(akk_test_pairs_simple_transliterated_translate)
        val_pairs.append(akk_val_pairs_simple_transliterated_translate)
        token_pairs.append(akk_train_pairs_simple_transliterated_translate)
        token_pairs.append(akk_test_pairs_simple_transliterated_translate)
        token_pairs.append(akk_val_pairs_simple_transliterated_translate)
        if debug == True:
            print(f"", {akk_train_pairs_simple_transliterated_translate[1][0]}, " -> ", {akk_train_pairs_simple_transliterated_translate[1][1]})
    if complex:
        ###Translate from original transliterated Akkadian to English
        akk_train_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_train[l], use_prefix=True, task="Translate", type="original", language="Akkadian"), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_transliteration_train))]
        akk_test_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_test[l], use_prefix=True, task="Translate", type="original", language="Akkadian"), normalizeString_en(akk_en_test[l], use_prefix=False)] for l in range(len(akk_transliteration_test))]
        akk_val_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_val[l], use_prefix=True, task="Translate", type="original", language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
        train_pairs.append(akk_train_pairs_original_transliterated_translate)
        test_pairs.append(akk_test_pairs_original_transliterated_translate)
        val_pairs.append(akk_val_pairs_original_transliterated_translate)
        token_pairs.append(akk_train_pairs_original_transliterated_translate)
        token_pairs.append(akk_test_pairs_original_transliterated_translate)
        token_pairs.append(akk_val_pairs_original_transliterated_translate)
        if debug == True:
            print(f"", {akk_train_pairs_original_transliterated_translate[1][0]}, " -> ", {akk_train_pairs_original_transliterated_translate[1][1]})
    if group:
        ###Translate from transliterated Akkadian to English
        akk_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transliteration_train[l], use_prefix=True, language="Akkadian"), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_transliteration_train))]
        akk_test_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transliteration_test[l], use_prefix=True, language="Akkadian"), normalizeString_en(akk_en_test[l], use_prefix=False)] for l in range(len(akk_transliteration_test))]
        akk_val_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transliteration_val[l], use_prefix=True, language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
        train_pairs.append(akk_train_pairs_group_transliterated_translate)
        test_pairs.append(akk_test_pairs_group_transliterated_translate)
        val_pairs.append(akk_val_pairs_group_transliterated_translate)
        if debug == True:
            print(f"", {akk_train_pairs_group_transliterated_translate[1][0]}, " -> ", {akk_train_pairs_group_transliterated_translate[1][1]})
    if transliterate:
        if complex:
            ###Transliterate from Akkadian Cuenfirom to complex Latin characters
            akk_train_pairs_transliterate_original = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Transliterate", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_gs_transliteration_train[l], use_prefix=False, type="original")] for l in range(len(akk_cuneiform_train))]
            akk_test_pairs_transliterate_original = [[normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=True, task="Transliterate", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_test[l], use_prefix=False, type="original")] for l in range(len(akk_cuneiform_test))]
            akk_val_pairs_transliterated_original = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="original")] for l in range(len(akk_cuneiform_val))]
            train_pairs.append(akk_train_pairs_transliterate_original)
            test_pairs.append(akk_test_pairs_transliterate_original)
            val_pairs.append(akk_val_pairs_transliterated_original)
            if debug == True:
                print(f"", {akk_train_pairs_transliterate_original[1][0]}, " -> ", {akk_train_pairs_transliterate_original[1][1]})
        if simple:
            ###Transliterate from Akkadian Cuenfirom to simple Latin characters
            akk_train_pairs_transliterate_simple = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Transliterate", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_gs_transliteration_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
            akk_test_pairs_transliterate_simple = [[normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=True, task="Transliterate", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
            akk_val_pairs_transliterated_simple = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
            train_pairs.append(akk_train_pairs_transliterate_simple)
            test_pairs.append(akk_test_pairs_transliterate_simple)
            val_pairs.append(akk_val_pairs_transliterated_simple)
            if debug == True:
                print(f"", {akk_train_pairs_transliterate_simple[1][0]}, " -> ", {akk_train_pairs_transliterate_simple[1][1]})
        if group:
            ###Transliterate from Akkadian Cuenfirom to Latin characters
            akk_train_pairs_transliterate_group = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Transliterate", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_gs_transliteration_train[l], use_prefix=False, type="group")] for l in range(len(akk_cuneiform_train))]
            akk_test_pairs_transliterate_group = [[normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=True, task="Transliterate", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_test[l], use_prefix=False, type="group")] for l in range(len(akk_cuneiform_test))]
            akk_val_pairs_transliterated_group = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="group")] for l in range(len(akk_cuneiform_val))]
            train_pairs.append(akk_train_pairs_transliterate_group)
            test_pairs.append(akk_test_pairs_transliterate_group)
            val_pairs.append(akk_val_pairs_transliterated_group)
            if debug == True:
                print(f"", {akk_train_pairs_transliterate_group[1][0]}, " -> ", {akk_train_pairs_transliterate_group[1][1]})
            if group:
                if simple:
                    if active_group:
                        ###Group simple transliterated Akkadian into words
                        akk_train_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_train[l], use_prefix=True, task="Group", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
                        akk_test_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_test[l], use_prefix=True, task="Group", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
                        akk_val_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_val[l], use_prefix=True, task="Group", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
                        train_pairs.append(akk_train_pairs_group_simple_transliterate)
                        test_pairs.append(akk_test_pairs_group_simple_transliterate)
                        val_pairs.append(akk_val_pairs_group_simple_transliterate)
                        if debug == True:
                            print(f"", {akk_train_pairs_group_simple_transliterate[1][0]}, " -> ", {akk_train_pairs_group_simple_transliterate[1][1]})
            if complex:
                ###Group complex transliterated Akkadian into words
                akk_train_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_train[l], use_prefix=True, task="Group", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
                akk_test_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_test[l], use_prefix=True, task="Group", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
                akk_val_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_val[l], use_prefix=True, task="Group", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
                train_pairs.append(akk_train_pairs_group_original_transliterate)
                test_pairs.append(akk_test_pairs_group_original_transliterate)
                val_pairs.append(akk_val_pairs_group_original_transliterate)
                if debug == True:
                    print(f"", {akk_train_pairs_group_original_transliterate[1][0]}, " -> ", {akk_train_pairs_group_original_transliterate[1][1]})
    if rev:
        if cuneiform:
            ###Translate from English to cuneiform Akkadian
            akk_train_rev_pairs_cuneiform_translate = [[normalizeString_en(akk_gs_en_train[l], use_prefix=True, task="Translate", target="cuneiform", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
            akk_test_rev_pairs_cuneiform_translate = [[normalizeString_en(akk_en_test[l], use_prefix=True, task="Translate", target="cuneiform", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
            akk_val_rev_pairs_cuneiform_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="cuneiform", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False), ] for l in range(len(akk_cuneiform_val))]
            train_pairs.append(akk_train_rev_pairs_cuneiform_translate)
            test_pairs.append(akk_test_rev_pairs_cuneiform_translate)
            val_pairs.append(akk_val_rev_pairs_cuneiform_translate)
            if debug == True:
                print(f"", {akk_train_rev_pairs_cuneiform_translate[1][0]}, " -> ", {akk_train_rev_pairs_cuneiform_translate[1][1]})
        if simple:
            ###Translate from English to simple transliterated Akkadian
            akk_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_train[l], use_prefix=False, type="simple")] for l in range(len(akk_en_train))]
            akk_test_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_test[l], use_prefix=False, type="simple")] for l in range(len(akk_en_test))]
            akk_val_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="simple")] for l in range(len(akk_en_val))]
            if debug == True:
                print(f"", {akk_train_rev_pairs_simple_transliterated_translate[1][0]}, " -> ", {akk_train_rev_pairs_simple_transliterated_translate[1][1]})
        if complex:
            ###Translate from English to complex transliterated Akkadian
            akk_train_rev_pairs_original_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_train[l], use_prefix=False, type="original")] for l in range(len(akk_en_train))]
            akk_test_rev_pairs_original_transliterated_translate = [[normalizeString_en(akk_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_test[l], use_prefix=False, type="original")] for l in range(len(akk_en_test))]
            akk_val_rev_pairs_original_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="original")] for l in range(len(akk_en_val))]
            train_pairs.append(akk_train_rev_pairs_original_transliterated_translate)
            test_pairs.append(akk_test_rev_pairs_original_transliterated_translate)
            val_pairs.append(akk_val_rev_pairs_original_transliterated_translate)
            token_pairs.append(akk_train_rev_pairs_original_transliterated_translate)
            token_pairs.append(akk_test_rev_pairs_original_transliterated_translate)
            token_pairs.append(akk_val_rev_pairs_original_transliterated_translate)
            if debug == True:
                print(f"", {akk_train_rev_pairs_original_transliterated_translate[1][0]}, " -> ", {akk_train_rev_pairs_original_transliterated_translate[1][1]})
        if group:
            ###Translate from English to transliterated Akkadian
            akk_train_rev_pairs_group_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_train[l], use_prefix=False)] for l in range(len(akk_en_train))]
            akk_test_rev_pairs_group_transliterated_translate = [[normalizeString_en(akk_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_test[l], use_prefix=False)] for l in range(len(akk_en_test))]
            akk_val_rev_pairs_group_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_val[l], use_prefix=False)] for l in range(len(akk_en_val))]
            train_pairs.append(akk_train_rev_pairs_group_transliterated_translate)
            test_pairs.append(akk_test_rev_pairs_group_transliterated_translate)
            val_pairs.append(akk_val_rev_pairs_group_transliterated_translate)
            if debug == True:
                print(f"", {akk_train_rev_pairs_group_transliterated_translate[1][0]}, " -> ", {akk_train_rev_pairs_group_transliterated_translate[1][1]})
        if transliterate:
            if simple:
                ###Convert from simple transliterated Akkadian to cuneiform
                akk_train_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_gs_transliteration_train[l], use_prefix=True, type="simple", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
                akk_test_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_test[l], use_prefix=True, type="simple", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=False)] for l in range(len(akk_transliteration_test))]
                akk_val_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_val[l], use_prefix=True, type="simple", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
                train_pairs.append(akk_train_rev_pairs_transliterate_simple)
                test_pairs.append(akk_test_rev_pairs_transliterate_simple)
                val_pairs.append(akk_val_rev_pairs_transliterate_simple)
                if debug == True:
                    print(f"", {akk_train_rev_pairs_transliterate_simple[1][0]}, " -> ", {akk_train_rev_pairs_transliterate_simple[1][1]})
            if group:
                ###Convert from transliterated Akkadian to cuneiform
                akk_train_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(akk_gs_transliteration_train[l], use_prefix=True, type="group", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
                akk_test_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_test[l], use_prefix=True, type="group", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=False)] for l in range(len(akk_transliteration_test))]
                akk_val_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_val[l], use_prefix=True, type="group", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
                train_pairs.append(akk_train_rev_pairs_transliterate_group)
                test_pairs.append(akk_test_rev_pairs_transliterate_group)
                val_pairs.append(akk_val_rev_pairs_transliterate_group)
                if debug == True:
                    print(f"", {akk_train_rev_pairs_transliterate_group[1][0]}, " -> ", {akk_train_rev_pairs_transliterate_group[1][1]})
            if complex:
                ###Convert from complex transliterated Akkadian to cuneiform
                akk_train_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(akk_gs_transliteration_train[l], use_prefix=True, type="original", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
                akk_test_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_test[l], use_prefix=True, type="original", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=False)] for l in range(len(akk_transliteration_test))]
                akk_val_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_val[l], use_prefix=True, type="original", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
                train_pairs.append(akk_train_rev_pairs_transliterate_original)
                test_pairs.append(akk_test_rev_pairs_transliterate_original)
                val_pairs.append(akk_val_rev_pairs_transliterate_original)
                if debug == True:
                    print(f"", {akk_train_rev_pairs_transliterate_original[1][0]}, " -> ", {akk_train_rev_pairs_transliterate_original[1][1]})
    train_pairs = [pair for subset in train_pairs for pair in subset]
    test_pairs = [pair for subset in test_pairs for pair in subset]
    val_pairs = [pair for subset in val_pairs for pair in subset]
    token_pairs = [pair for subset in token_pairs for pair in subset]
    pairs = train_pairs + test_pairs + val_pairs
    #print("Examples:")
    #print(f"", {train_pairs_cuneiform_translate[1][0]}, " -> ", {train_pairs_cuneiform_translate[1][1]})
    #print(f"", {train_pairs_transliterate_original[1][0]}, " -> ", {train_pairs_transliterate_original[1][1]})
    #print(f"", {train_pairs_transliterate_group[1][0]}, " -> ", {train_pairs_transliterate_group[1][1]})
    #print(f"", {train_pairs_group_transliterated_translate[1][0]}, " -> ", {train_pairs_group_transliterated_translate[1][1]})
    print(f"Total pairs imported: {len(pairs)}")
    train_pairs = trim_pairs(train_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    if debug == True:
        print("training set trimmed")
    val_pairs = trim_pairs(val_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    if debug == True:
        print("val set trimmed")
    test_pairs = trim_pairs(test_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    if debug == True:
        print("test set trimmed")
    pairs = train_pairs + test_pairs + val_pairs
    print(f"Total pairs filtered: {len(pairs)}")
    max_length_pair_0 = max(pairs, key=lambda pair: len(pair[0].split()))
    max_length_pair_1 = max(pairs, key=lambda pair: len(pair[1].split()))
    print("Largest number of words in pair[0]:")
    print(f"Word Count: {len(max_length_pair_0[0].split())}, Content: {max_length_pair_0[0]}")
    print("Largest number of words in pair[1]:")
    print(f"Word Count: {len(max_length_pair_1[1].split())}, Content: {max_length_pair_1[1]}")
    mean_length_pair_0 = sum(len(pair[0].split()) for pair in pairs) / len(pairs)
    mean_length_pair_1 = sum(len(pair[1].split()) for pair in pairs) / len(pairs)
    print(f"Mean number of tokens in source langauge: {mean_length_pair_0:.2f}")
    print(f"Mean number of tokens in target language: {mean_length_pair_1:.2f}")
    #train_pairs = collapse_spaces(train_pairs)
    #test_pairs = collapse_spaces(test_pairs)
    #val_pairs = collapse_spaces(val_pairs)
    #pairs = collapse_spaces(pairs)
    return train_pairs, val_pairs, test_pairs, pairs, token_pairs

def readLangsPreTrainT5Akkademia(user_directory="/", max_length_akk=5000, max_length_en=5000, max_length_threshold=100, min_length_threshold=50, debug=False, cuneiform=True, akk=True, sux=True, elx=True, gmy=True, hit=True, complex=True, group=False, simple=False, rev=True, de=False, ld=True):
    print("Reading lines...")
    ###Create list
    train_pairs = []
    token_pairs = []
    if akk:
        ##############
        ###Akkadian###
        ##############
        # Read the file and split into lines
        akk_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'Akkademia', 'NMT_input', 'train.ak'))
        akk_gs_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'Akkademia', 'NMT_input', 'train.tr'))
        akk_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'akk_u_train.tr'))
        akk_transliteration_train = akk_gs_transliteration_train + akk_transliteration_u_train
        akk_en_gs_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'Akkademia', 'NMT_input', 'train.en'))
        akk_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'akk_cdli_train.tr'))
        akk_transliteration_train = akk_transliteration_train + akk_cdli_transliteration_train
        akk_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'akk_cdli_train.en'))
        akk_en_train = akk_en_gs_train + akk_cdli_en_train
        # Split every line into pairs and normalize
        if cuneiform:
            ###Translate from Akkadian cuneiform to English
            akk_train_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False, task="Translate", language="Akkadian")] for l in range(len(akk_cuneiform_train))]
            train_pairs.append(akk_train_pairs_cuneiform_translate)
            token_pairs.append(akk_train_pairs_cuneiform_translate)
            if debug == True:
                print(f"", {akk_train_pairs_cuneiform_translate[1][0]})
        if simple:
            ###Translate from simple transliterated Akkadian to English
            akk_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_train[l], use_prefix=False, task="Translate", type="simple", language="Akkadian")] for l in range(len(akk_transliteration_train))]
            train_pairs.append(akk_train_pairs_simple_transliterated_translate)
            token_pairs.append(akk_train_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {akk_train_pairs_simple_transliterated_translate[1][0]})
        if complex:
            ###Translate from original transliterated Akkadian to English
            akk_train_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_train[l], use_prefix=False, task="Translate", type="original", language="Akkadian")] for l in range(len(akk_transliteration_train))]
            train_pairs.append(akk_train_pairs_original_transliterated_translate)
            token_pairs.append(akk_train_pairs_original_transliterated_translate)
            if debug == True:
                print(f"", {akk_train_pairs_original_transliterated_translate[1][0]})
        if group:
            ###Translate from transliterated Akkadian to English
            akk_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transliteration_train[l], use_prefix=False, language="Akkadian")] for l in range(len(akk_transliteration_train))]
            train_pairs.append(akk_train_pairs_group_transliterated_translate)
            if debug == True:
                print(f"", {akk_train_pairs_group_transliterated_translate[1][0]})
        if rev:
            ###Translate from English
            akk_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=False, task="Translate", target="transliteration", type="simple", language="Akkadian")] for l in range(len(akk_en_train))]
            train_pairs.append(akk_train_rev_pairs_simple_transliterated_translate)
            token_pairs.append(akk_train_rev_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {akk_train_rev_pairs_simple_transliterated_translate[1][0]})

    train_pairs = [pair for subset in train_pairs for pair in subset]
    token_pairs = [pair for subset in token_pairs for pair in subset]
    pairs = train_pairs
    #print("Examples:")
    #print(f"", {train_pairs_cuneiform_translate[1][0]}, " -> ", {train_pairs_cuneiform_translate[1][1]})
    #print(f"", {train_pairs_transliterate_original[1][0]}, " -> ", {train_pairs_transliterate_original[1][1]})
    #print(f"", {train_pairs_transliterate_group[1][0]}, " -> ", {train_pairs_transliterate_group[1][1]})
    #print(f"", {train_pairs_group_transliterated_translate[1][0]}, " -> ", {train_pairs_group_transliterated_translate[1][1]})
    print(f"Total pairs imported: {len(pairs)}")
    train_pairs = trim_singles(train_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    if debug == True:
        print("training set trimmed")
    pairs = train_pairs
    print(f"Total pairs filtered: {len(pairs)}")
    max_length_pair_0 = max(pairs, key=lambda pair: len(pair[0].split()))
    print("Largest number of words in pair:")
    print(f"Word Count: {len(max_length_pair_0.split())}, Content: {max_length_pair_0}")
    mean_length_pair_0 = sum(len(pair[0].split()) for pair in pairs) / len(pairs)
    print(f"Mean number of tokens in source langauge: {mean_length_pair_0:.2f}")
    #train_pairs = collapse_spaces(train_pairs)
    #pairs = collapse_spaces(pairs)
    return train_pairs, pairs, token_pairs
            
def readLangsTrainT5(user_directory="/", max_length_akk=5000, max_length_en=5000, max_length_threshold=100, min_length_threshold=50, debug=True, cuneiform=True, akk=True, sux=True, elx=True, gmy=True, hit=True, complex=True, group=False, active_group=False, simple=False, transliterate=False, rev=True, de=False, ld=True, data="both", rev_simp_trans=False, rev_complex_trans=False, use_oracc=True, use_cdli=True, use_oare=True, prompt_style="T5"):
    """
    Load multi-language cuneiform parallel corpus for T5 model training.

    This is the primary data loader for training T5-style seq2seq models on
    cuneiform translation tasks. Loads data from CDLI, ORACC, and OARE sources
    for multiple ancient languages.

    Supported Languages:
        - akk: Akkadian (Babylonian/Assyrian)
        - sux: Sumerian
        - elx: Elamite
        - gmy: Mycenaean Greek (Linear B)
        - hit: Hittite

    Task Types Created:
        - cuneiform -> English: Direct glyph to translation
        - transliteration -> English: Latin script to translation
        - cuneiform <-> transliteration: Bidirectional if transliterate=True
        - transliteration: Word-grouped if group=True

    Args:
        user_directory: Base path for data files
        max_length_akk: Max source length for filtering
        max_length_en: Max target length for filtering
        max_length_threshold: Upper bound for pair lengths
        min_length_threshold: Minimum pair length (below = dropped)
        debug: Print progress and sample pairs
        cuneiform: Include cuneiform glyph tasks
        akk/sux/elx/gmy/hit: Enable specific languages
        complex: Use diacritic-preserving transliteration
        group: Include word-grouped transliteration tasks
        active_group: Include syllable grouping tasks
        simple: Include ASCII-only transliteration
        transliterate: Include transliteration conversion tasks
        rev: Include reverse direction pairs
        de: Include German translations
        ld: Include long-document data
        data: Data granularity ('line', 'document', or 'both')
        use_oracc: Include ORACC corpus data (akk_* files)
        use_cdli: Include CDLI corpus data (akk_cdli_* files)
        use_oare: Include OARE corpus data (akk_oare_* files)
        prompt_style: Prefix style ('T5' verbose or 'simple'/'compact')

    Returns:
        Tuple of (train_pairs, pairs, token_pairs) where:
            - train_pairs: List of (source, target) tuples with prefixes
            - pairs: Same as train_pairs (for compatibility)
            - token_pairs: Token-level pairs for analysis
    """
    print("Reading lines...")
    ###Create lists
    train_pairs = []
    test_pairs = []
    val_pairs = []
    token_pairs = []
    if akk:
        ##############
        ###Akkadian###
        ##############
        # Read the file and split into lines
        # Initialize empty lists for accumulating data from different sources
        if data=="line":
            akk_cuneiform_train = []
            akk_transliteration_train = []
            akk_en_train = []
            akk_cuneiform_test = []
            akk_transliteration_test = []
            akk_en_test = []
            akk_cuneiform_val = []
            akk_transliteration_val = []
            akk_en_val = []
            # ORACC data (akk_* files)
            if use_oracc:
                akk_oracc_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_line.cu'))
                akk_oracc_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_line.tr'))
                akk_oracc_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_line.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oracc_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_oracc_transliteration_train
                akk_en_train = akk_en_train + akk_oracc_en_train
                # ORACC test/val
                akk_oracc_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_line.cu'))
                akk_oracc_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_line.tr'))
                akk_oracc_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_line.en'))
                akk_cuneiform_test = akk_cuneiform_test + akk_oracc_cuneiform_test
                akk_transliteration_test = akk_transliteration_test + akk_oracc_transliteration_test
                akk_en_test = akk_en_test + akk_oracc_en_test
                akk_oracc_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_line.cu'))
                akk_oracc_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_line.tr'))
                akk_oracc_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_line.en'))
                akk_cuneiform_val = akk_cuneiform_val + akk_oracc_cuneiform_val
                akk_transliteration_val = akk_transliteration_val + akk_oracc_transliteration_val
                akk_en_val = akk_en_val + akk_oracc_en_val
            # CDLI data (akk_cdli_* files)
            if use_cdli:
                akk_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_line.cu'))
                akk_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_line.tr'))
                akk_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_line.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_cdli_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_cdli_transliteration_train
                akk_en_train = akk_en_train + akk_cdli_en_train
                # CDLI test/val
                akk_cdli_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test_line.cu'))
                akk_cdli_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test_line.tr'))
                akk_cdli_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test_line.en'))
                akk_cuneiform_test = akk_cuneiform_test + akk_cdli_cuneiform_test
                akk_transliteration_test = akk_transliteration_test + akk_cdli_transliteration_test
                akk_en_test = akk_en_test + akk_cdli_en_test
                akk_cdli_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid_line.cu'))
                akk_cdli_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid_line.tr'))
                akk_cdli_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid_line.en'))
                akk_cuneiform_val = akk_cuneiform_val + akk_cdli_cuneiform_val
                akk_transliteration_val = akk_transliteration_val + akk_cdli_transliteration_val
                akk_en_val = akk_en_val + akk_cdli_en_val
            # OARE data (akk_oare_* files) - has cuneiform for train, no test files, no line-level valid
            if use_oare:
                akk_oare_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_line.cu'))
                akk_oare_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_line.tr'))
                akk_oare_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_line.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oare_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_oare_transliteration_train
                akk_en_train = akk_en_train + akk_oare_en_train
                # OARE val only (no test files exist, no line-level valid .cu exists)
        elif data=="document":
            akk_cuneiform_train = []
            akk_transliteration_train = []
            akk_en_train = []
            akk_cuneiform_test = []
            akk_transliteration_test = []
            akk_en_test = []
            akk_cuneiform_val = []
            akk_transliteration_val = []
            akk_en_val = []
            # ORACC data (akk_* files)
            if use_oracc:
                akk_oracc_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_document.cu'))
                akk_oracc_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_document.tr'))
                akk_oracc_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_document.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oracc_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_oracc_transliteration_train
                akk_en_train = akk_en_train + akk_oracc_en_train
                # ORACC test/val
                akk_oracc_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_document.cu'))
                akk_oracc_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_document.tr'))
                akk_oracc_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test_document.en'))
                akk_cuneiform_test = akk_cuneiform_test + akk_oracc_cuneiform_test
                akk_transliteration_test = akk_transliteration_test + akk_oracc_transliteration_test
                akk_en_test = akk_en_test + akk_oracc_en_test
                akk_oracc_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_document.cu'))
                akk_oracc_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_document.tr'))
                akk_oracc_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid_document.en'))
                akk_cuneiform_val = akk_cuneiform_val + akk_oracc_cuneiform_val
                akk_transliteration_val = akk_transliteration_val + akk_oracc_transliteration_val
                akk_en_val = akk_en_val + akk_oracc_en_val
            # CDLI data (akk_cdli_* files)
            if use_cdli:
                akk_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_document.cu'))
                akk_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_document.tr'))
                akk_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_document.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_cdli_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_cdli_transliteration_train
                akk_en_train = akk_en_train + akk_cdli_en_train
                # CDLI test/val
                akk_cdli_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test_document.cu'))
                akk_cdli_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test_document.tr'))
                akk_cdli_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test_document.en'))
                akk_cuneiform_test = akk_cuneiform_test + akk_cdli_cuneiform_test
                akk_transliteration_test = akk_transliteration_test + akk_cdli_transliteration_test
                akk_en_test = akk_en_test + akk_cdli_en_test
                akk_cdli_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid_document.cu'))
                akk_cdli_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid_document.tr'))
                akk_cdli_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid_document.en'))
                akk_cuneiform_val = akk_cuneiform_val + akk_cdli_cuneiform_val
                akk_transliteration_val = akk_transliteration_val + akk_cdli_transliteration_val
                akk_en_val = akk_en_val + akk_cdli_en_val
            # OARE data (akk_oare_* files) - has cuneiform, no test files
            if use_oare:
                akk_oare_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_document.cu'))
                akk_oare_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_document.tr'))
                akk_oare_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_document.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oare_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_oare_transliteration_train
                akk_en_train = akk_en_train + akk_oare_en_train
                # OARE val only (no test files exist)
                akk_oare_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_valid_document.cu'))
                akk_oare_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_valid_document.tr'))
                akk_oare_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_valid_document.en'))
                akk_cuneiform_val = akk_cuneiform_val + akk_oare_cuneiform_val
                akk_transliteration_val = akk_transliteration_val + akk_oare_transliteration_val
                akk_en_val = akk_en_val + akk_oare_en_val
        elif data=="both":
            akk_cuneiform_train = []
            akk_transliteration_train = []
            akk_en_train = []
            akk_cuneiform_test = []
            akk_transliteration_test = []
            akk_en_test = []
            akk_cuneiform_val = []
            akk_transliteration_val = []
            akk_en_val = []
            # ORACC data (akk_* files)
            if use_oracc:
                akk_oracc_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train.cu'))
                akk_oracc_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train.tr'))
                akk_oracc_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oracc_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_oracc_transliteration_train
                akk_en_train = akk_en_train + akk_oracc_en_train
                # ORACC test/val
                akk_oracc_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test.cu'))
                akk_oracc_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test.tr'))
                akk_oracc_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_test.en'))
                akk_cuneiform_test = akk_cuneiform_test + akk_oracc_cuneiform_test
                akk_transliteration_test = akk_transliteration_test + akk_oracc_transliteration_test
                akk_en_test = akk_en_test + akk_oracc_en_test
                akk_oracc_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid.cu'))
                akk_oracc_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid.tr'))
                akk_oracc_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_valid.en'))
                akk_cuneiform_val = akk_cuneiform_val + akk_oracc_cuneiform_val
                akk_transliteration_val = akk_transliteration_val + akk_oracc_transliteration_val
                akk_en_val = akk_en_val + akk_oracc_en_val
            # CDLI data (akk_cdli_* files)
            if use_cdli:
                akk_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train.cu'))
                akk_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train.tr'))
                akk_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_cdli_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_cdli_transliteration_train
                akk_en_train = akk_en_train + akk_cdli_en_train
                # CDLI test/val
                akk_cdli_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test.cu'))
                akk_cdli_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test.tr'))
                akk_cdli_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_test.en'))
                akk_cuneiform_test = akk_cuneiform_test + akk_cdli_cuneiform_test
                akk_transliteration_test = akk_transliteration_test + akk_cdli_transliteration_test
                akk_en_test = akk_en_test + akk_cdli_en_test
                akk_cdli_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid.cu'))
                akk_cdli_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid.tr'))
                akk_cdli_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_valid.en'))
                akk_cuneiform_val = akk_cuneiform_val + akk_cdli_cuneiform_val
                akk_transliteration_val = akk_transliteration_val + akk_cdli_transliteration_val
                akk_en_val = akk_en_val + akk_cdli_en_val
            # OARE data (akk_oare_* files) - has cuneiform, no test files
            if use_oare:
                akk_oare_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train.cu'))
                akk_oare_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train.tr'))
                akk_oare_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oare_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_oare_transliteration_train
                akk_en_train = akk_en_train + akk_oare_en_train
                # OARE val only (no test files exist, val .cu file is named akk_oare_val.cu)
                akk_oare_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_val.cu'))
                akk_oare_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_valid.tr'))
                akk_oare_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_valid.en'))
                akk_cuneiform_val = akk_cuneiform_val + akk_oare_cuneiform_val
                akk_transliteration_val = akk_transliteration_val + akk_oare_transliteration_val
                akk_en_val = akk_en_val + akk_oare_en_val
        # Split every line into pairs and normalize
        if cuneiform:
            ###Translate from Akkadian cuneiform to English
            akk_train_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Translate", language="Akkadian", style=prompt_style), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
            akk_test_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=True, task="Translate", language="Akkadian", style=prompt_style), normalizeString_en(akk_en_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
            akk_val_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Translate", language="Akkadian", style=prompt_style), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
            train_pairs.append(akk_train_pairs_cuneiform_translate)
            test_pairs.append(akk_test_pairs_cuneiform_translate)
            val_pairs.append(akk_val_pairs_cuneiform_translate)
            token_pairs.append(akk_train_pairs_cuneiform_translate)
            token_pairs.append(akk_test_pairs_cuneiform_translate)
            token_pairs.append(akk_val_pairs_cuneiform_translate)
            if debug == True:
                print(f"", {akk_train_pairs_cuneiform_translate[1][0]}, " -> ", {akk_train_pairs_cuneiform_translate[1][1]})
        if simple:
            ###Translate from simple transliterated Akkadian to English
            akk_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_train[l], use_prefix=True, task="Translate", type="simple", language="Akkadian", style=prompt_style), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_transliteration_train))]
            akk_test_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_test[l], use_prefix=True, task="Translate", type="simple", language="Akkadian", style=prompt_style), normalizeString_en(akk_en_test[l], use_prefix=False)] for l in range(len(akk_transliteration_test))]
            akk_val_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_val[l], use_prefix=True, task="Translate", type="simple", language="Akkadian", style=prompt_style), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
            train_pairs.append(akk_train_pairs_simple_transliterated_translate)
            test_pairs.append(akk_test_pairs_simple_transliterated_translate)
            val_pairs.append(akk_val_pairs_simple_transliterated_translate)
            token_pairs.append(akk_train_pairs_simple_transliterated_translate)
            token_pairs.append(akk_test_pairs_simple_transliterated_translate)
            token_pairs.append(akk_val_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {akk_train_pairs_simple_transliterated_translate[1][0]}, " -> ", {akk_train_pairs_simple_transliterated_translate[1][1]})
        if complex:
            ###Translate from original transliterated Akkadian to English
            akk_train_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_train[l], use_prefix=True, task="Translate", type="original", language="Akkadian", style=prompt_style), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_transliteration_train))]
            akk_test_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_test[l], use_prefix=True, task="Translate", type="original", language="Akkadian", style=prompt_style), normalizeString_en(akk_en_test[l], use_prefix=False)] for l in range(len(akk_transliteration_test))]
            akk_val_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_val[l], use_prefix=True, task="Translate", type="original", language="Akkadian", style=prompt_style), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
            train_pairs.append(akk_train_pairs_original_transliterated_translate)
            test_pairs.append(akk_test_pairs_original_transliterated_translate)
            val_pairs.append(akk_val_pairs_original_transliterated_translate)
            token_pairs.append(akk_train_pairs_original_transliterated_translate)
            token_pairs.append(akk_test_pairs_original_transliterated_translate)
            token_pairs.append(akk_val_pairs_original_transliterated_translate)
            if debug == True:
                print(f"", {akk_train_pairs_original_transliterated_translate[1][0]}, " -> ", {akk_train_pairs_original_transliterated_translate[1][1]})
        if group:
            ###Translate from transliterated Akkadian to English
            akk_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transliteration_train[l], use_prefix=True, language="Akkadian", style=prompt_style), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_transliteration_train))]
            akk_test_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transliteration_test[l], use_prefix=True, language="Akkadian", style=prompt_style), normalizeString_en(akk_en_test[l], use_prefix=False)] for l in range(len(akk_transliteration_test))]
            akk_val_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transliteration_val[l], use_prefix=True, language="Akkadian", style=prompt_style), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transliteration_val))]
            train_pairs.append(akk_train_pairs_group_transliterated_translate)
            test_pairs.append(akk_test_pairs_group_transliterated_translate)
            val_pairs.append(akk_val_pairs_group_transliterated_translate)
            if debug == True:
                print(f"", {akk_train_pairs_group_transliterated_translate[1][0]}, " -> ", {akk_train_pairs_group_transliterated_translate[1][1]})
        if transliterate:
            if rev_complex_trans:
                ###Transliterate from Akkadian Cuenfirom to complex Latin characters
                akk_train_pairs_transliterate_original = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Transliterate", type="original", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_train[l], use_prefix=False, type="original")] for l in range(len(akk_cuneiform_train))]
                akk_test_pairs_transliterate_original = [[normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=True, task="Transliterate", type="original", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_test[l], use_prefix=False, type="original")] for l in range(len(akk_cuneiform_test))]
                akk_val_pairs_transliterated_original = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="original", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="original")] for l in range(len(akk_cuneiform_val))]
                train_pairs.append(akk_train_pairs_transliterate_original)
                test_pairs.append(akk_test_pairs_transliterate_original)
                val_pairs.append(akk_val_pairs_transliterated_original)
                if debug == True:
                    print(f"", {akk_train_pairs_transliterate_original[1][0]}, " -> ", {akk_train_pairs_transliterate_original[1][1]})
            if rev_simp_trans:
                ###Transliterate from Akkadian Cuenfirom to simple Latin characters
                akk_train_pairs_transliterate_simple = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Transliterate", type="simple", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
                akk_test_pairs_transliterate_simple = [[normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=True, task="Transliterate", type="simple", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
                akk_val_pairs_transliterated_simple = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="simple", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
                train_pairs.append(akk_train_pairs_transliterate_simple)
                test_pairs.append(akk_test_pairs_transliterate_simple)
                val_pairs.append(akk_val_pairs_transliterated_simple)
                if debug == True:
                    print(f"", {akk_train_pairs_transliterate_simple[1][0]}, " -> ", {akk_train_pairs_transliterate_simple[1][1]})
            if group:
                ###Transliterate from Akkadian Cuenfirom to Latin characters
                akk_train_pairs_transliterate_group = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Transliterate", type="group", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_train[l], use_prefix=False, type="group")] for l in range(len(akk_cuneiform_train))]
                akk_test_pairs_transliterate_group = [[normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=True, task="Transliterate", type="group", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_test[l], use_prefix=False, type="group")] for l in range(len(akk_cuneiform_test))]
                akk_val_pairs_transliterated_group = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="group", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="group")] for l in range(len(akk_cuneiform_val))]
                train_pairs.append(akk_train_pairs_transliterate_group)
                test_pairs.append(akk_test_pairs_transliterate_group)
                val_pairs.append(akk_val_pairs_transliterated_group)
                if debug == True:
                    print(f"", {akk_train_pairs_transliterate_group[1][0]}, " -> ", {akk_train_pairs_transliterate_group[1][1]})
                if group:
                    if active_group:
                        if simple:
                            ###Group simple transliterated Akkadian into words
                            akk_train_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_train[l], use_prefix=True, task="Group", type="simple", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
                            akk_test_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_test[l], use_prefix=True, task="Group", type="simple", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
                            akk_val_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_val[l], use_prefix=True, task="Group", type="simple", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
                            train_pairs.append(akk_train_pairs_group_simple_transliterate)
                            test_pairs.append(akk_test_pairs_group_simple_transliterate)
                            val_pairs.append(akk_val_pairs_group_simple_transliterate)
                            if debug == True:
                                print(f"", {akk_train_pairs_group_simple_transliterate[1][0]}, " -> ", {akk_train_pairs_group_simple_transliterate[1][1]})
                        if complex:
                            ###Group complex transliterated Akkadian into words
                            akk_train_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_train[l], use_prefix=True, task="Group", type="original", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
                            akk_test_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_test[l], use_prefix=True, task="Group", type="original", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
                            akk_val_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_val[l], use_prefix=True, task="Group", type="original", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
                            train_pairs.append(akk_train_pairs_group_original_transliterate)
                            test_pairs.append(akk_test_pairs_group_original_transliterate)
                            val_pairs.append(akk_val_pairs_group_original_transliterate)
                            if debug == True:
                                print(f"", {akk_train_pairs_group_original_transliterate[1][0]}, " -> ", {akk_train_pairs_group_original_transliterate[1][1]})
        if rev:
            if cuneiform:
                ###Translate from English to cuneiform Akkadian
                akk_train_rev_pairs_cuneiform_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="cuneiform", language="Akkadian", style=prompt_style), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
                akk_test_rev_pairs_cuneiform_translate = [[normalizeString_en(akk_en_test[l], use_prefix=True, task="Translate", target="cuneiform", language="Akkadian", style=prompt_style), normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
                akk_val_rev_pairs_cuneiform_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="cuneiform", language="Akkadian", style=prompt_style), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False), ] for l in range(len(akk_cuneiform_val))]
                train_pairs.append(akk_train_rev_pairs_cuneiform_translate)
                test_pairs.append(akk_test_rev_pairs_cuneiform_translate)
                val_pairs.append(akk_val_rev_pairs_cuneiform_translate)
                if debug == True:
                    print(f"", {akk_train_rev_pairs_cuneiform_translate[1][0]}, " -> ", {akk_train_rev_pairs_cuneiform_translate[1][1]})
            if simple:
                ###Translate from English to simple transliterated Akkadian
                akk_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_train[l], use_prefix=False, type="simple")] for l in range(len(akk_en_train))]
                akk_test_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_test[l], use_prefix=False, type="simple")] for l in range(len(akk_en_test))]
                akk_val_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="simple")] for l in range(len(akk_en_val))]
                if debug == True:
                    print(f"", {akk_train_rev_pairs_simple_transliterated_translate[1][0]}, " -> ", {akk_train_rev_pairs_simple_transliterated_translate[1][1]})
            if complex:
                ###Translate from English to complex transliterated Akkadian
                akk_train_rev_pairs_original_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_train[l], use_prefix=False, type="original")] for l in range(len(akk_en_train))]
                akk_test_rev_pairs_original_transliterated_translate = [[normalizeString_en(akk_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_test[l], use_prefix=False, type="original")] for l in range(len(akk_en_test))]
                akk_val_rev_pairs_original_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate(akk_transliteration_val[l], use_prefix=False, type="original")] for l in range(len(akk_en_val))]
                train_pairs.append(akk_train_rev_pairs_original_transliterated_translate)
                test_pairs.append(akk_test_rev_pairs_original_transliterated_translate)
                val_pairs.append(akk_val_rev_pairs_original_transliterated_translate)
                token_pairs.append(akk_train_rev_pairs_original_transliterated_translate)
                token_pairs.append(akk_test_rev_pairs_original_transliterated_translate)
                token_pairs.append(akk_val_rev_pairs_original_transliterated_translate)
                if debug == True:
                    print(f"", {akk_train_rev_pairs_original_transliterated_translate[1][0]}, " -> ", {akk_train_rev_pairs_original_transliterated_translate[1][1]})
            if group:
                ###Translate from English to transliterated Akkadian
                akk_train_rev_pairs_group_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_train[l], use_prefix=False)] for l in range(len(akk_en_train))]
                akk_test_rev_pairs_group_transliterated_translate = [[normalizeString_en(akk_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_test[l], use_prefix=False)] for l in range(len(akk_en_test))]
                akk_val_rev_pairs_group_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Akkadian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(akk_transliteration_val[l], use_prefix=False)] for l in range(len(akk_en_val))]
                train_pairs.append(akk_train_rev_pairs_group_transliterated_translate)
                test_pairs.append(akk_test_rev_pairs_group_transliterated_translate)
                val_pairs.append(akk_val_rev_pairs_group_transliterated_translate)
                if debug == True:
                    print(f"", {akk_train_rev_pairs_group_transliterated_translate[1][0]}, " -> ", {akk_train_rev_pairs_group_transliterated_translate[1][1]})
            if transliterate:
                if rev_simp_trans:
                    ###Convert from simple transliterated Akkadian to cuneiform
                    akk_train_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_train[l], use_prefix=True, type="simple", language="Akkadian", style=prompt_style), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
                    akk_test_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_test[l], use_prefix=True, type="simple", language="Akkadian", style=prompt_style), normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
                    akk_val_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_val[l], use_prefix=True, type="simple", language="Akkadian", style=prompt_style), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
                    train_pairs.append(akk_train_rev_pairs_transliterate_simple)
                    test_pairs.append(akk_test_rev_pairs_transliterate_simple)
                    val_pairs.append(akk_val_rev_pairs_transliterate_simple)
                    if debug == True:
                        print(f"", {akk_train_rev_pairs_transliterate_simple[1][0]}, " -> ", {akk_train_rev_pairs_transliterate_simple[1][1]})
                if group:
                    ###Convert from transliterated Akkadian to cuneiform
                    akk_train_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_train[l], use_prefix=True, type="group", language="Akkadian", style=prompt_style), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
                    akk_test_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_test[l], use_prefix=True, type="group", language="Akkadian", style=prompt_style), normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
                    akk_val_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_val[l], use_prefix=True, type="group", language="Akkadian", style=prompt_style), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
                    train_pairs.append(akk_train_rev_pairs_transliterate_group)
                    test_pairs.append(akk_test_rev_pairs_transliterate_group)
                    val_pairs.append(akk_val_rev_pairs_transliterate_group)
                    if debug == True:
                        print(f"", {akk_train_rev_pairs_transliterate_group[1][0]}, " -> ", {akk_train_rev_pairs_transliterate_group[1][1]})
                if rev_simp_trans:
                    ###Convert from complex transliterated Akkadian to cuneiform
                    akk_train_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_train[l], use_prefix=True, type="original", language="Akkadian", style=prompt_style), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
                    akk_test_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_test[l], use_prefix=True, type="original", language="Akkadian", style=prompt_style), normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
                    akk_val_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(akk_transliteration_val[l], use_prefix=True, type="original", language="Akkadian", style=prompt_style), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
                    train_pairs.append(akk_train_rev_pairs_transliterate_original)
                    test_pairs.append(akk_test_rev_pairs_transliterate_original)
                    val_pairs.append(akk_val_rev_pairs_transliterate_original)
                    if debug == True:
                        print(f"", {akk_train_rev_pairs_transliterate_original[1][0]}, " -> ", {akk_train_rev_pairs_transliterate_original[1][1]})
    if sux:
        ##############
        ###Sumerian###
        ##############
        # Read the file and split into lines
        if data=="line":
            sux_ox_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.cu'))
            sux_ox_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.tr'))
            sux_ox_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.en'))
            sux_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_line.cu'))
            sux_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_line.tr'))
            sux_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_line.en'))
            sux_cuneiform_train = sux_ox_cuneiform_train + sux_cdli_cuneiform_train
            sux_transliteration_train = sux_ox_transliteration_train + sux_cdli_transliteration_train
            sux_en_train = sux_ox_en_train + sux_cdli_en_train
            sux_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_line.cu'))
            sux_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_line.tr'))
            sux_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_line.en'))
            sux_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_line.cu'))
            sux_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_line.tr'))
            sux_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_line.en'))
        elif data=="document":
            sux_ox_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.cu'))
            sux_ox_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.tr'))
            sux_ox_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.en'))
            sux_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_document.cu'))
            sux_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_document.tr'))
            sux_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_document.en'))
            sux_cuneiform_train = sux_ox_cuneiform_train + sux_cdli_cuneiform_train
            sux_transliteration_train = sux_ox_transliteration_train + sux_cdli_transliteration_train
            sux_en_train = sux_ox_en_train + sux_cdli_en_train
            sux_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_document.cu'))
            sux_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_document.tr'))
            sux_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_document.en'))
            sux_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_document.cu'))
            sux_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_document.tr'))
            sux_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_document.en'))
        elif data=="both":
            sux_ox_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.cu'))
            sux_ox_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.tr'))
            sux_ox_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.en'))
            sux_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'sux_cdli_train.cu'))
            sux_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train.tr'))
            sux_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train.en'))
            sux_cuneiform_train = sux_ox_cuneiform_train + sux_cdli_cuneiform_train
            sux_transliteration_train = sux_ox_transliteration_train + sux_cdli_transliteration_train
            sux_en_train = sux_ox_en_train + sux_cdli_en_train
            sux_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test.cu'))
            sux_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test.tr'))
            sux_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test.en'))
            sux_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid.cu'))
            sux_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid.tr'))
            sux_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid.en'))
        # Split every line into pairs and normalize
        if cuneiform:
            ###Translate from Sumerian cuneiform to English
            sux_train_pairs_cuneiform_translate = [[normalizeString_cuneiform(sux_cuneiform_train[l], use_prefix=True, task="Translate", language="Sumerian", style=prompt_style), normalizeString_en(sux_en_train[l], use_prefix=False)] for l in range(len(sux_cuneiform_train))]
            sux_test_pairs_cuneiform_translate = [[normalizeString_cuneiform(sux_cuneiform_test[l], use_prefix=True, task="Translate", language="Sumerian", style=prompt_style), normalizeString_en(sux_en_test[l], use_prefix=False)] for l in range(len(sux_cuneiform_test))]
            sux_val_pairs_cuneiform_translate = [[normalizeString_cuneiform(sux_cuneiform_val[l], use_prefix=True, task="Translate", language="Sumerian", style=prompt_style), normalizeString_en(sux_en_val[l], use_prefix=False)] for l in range(len(sux_cuneiform_val))]
            train_pairs.append(sux_train_pairs_cuneiform_translate)
            test_pairs.append(sux_test_pairs_cuneiform_translate)
            val_pairs.append(sux_val_pairs_cuneiform_translate)
            token_pairs.append(sux_train_pairs_cuneiform_translate)
            token_pairs.append(sux_test_pairs_cuneiform_translate)
            token_pairs.append(sux_val_pairs_cuneiform_translate)
            if debug == True:
                print(f"", {sux_train_pairs_cuneiform_translate[1][0]}, " -> ", {sux_train_pairs_cuneiform_translate[1][1]})
        if simple:
            ##Translate from simple transliterated Sumerian to English
            sux_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transliteration_train[l], use_prefix=True, task="Translate", type="simple", language="Sumerian", style=prompt_style), normalizeString_en(sux_en_train[l], use_prefix=False)] for l in range(len(sux_transliteration_train))]
            sux_test_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transliteration_test[l], use_prefix=True, task="Translate", type="simple", language="Sumerian", style=prompt_style), normalizeString_en(sux_en_test[l], use_prefix=False)] for l in range(len(sux_transliteration_test))]
            sux_val_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transliteration_val[l], use_prefix=True, task="Translate", type="simple", language="Sumerian", style=prompt_style), normalizeString_en(sux_en_val[l], use_prefix=False)] for l in range(len(sux_transliteration_val))]
            train_pairs.append(sux_train_pairs_simple_transliterated_translate)
            test_pairs.append(sux_test_pairs_simple_transliterated_translate)
            val_pairs.append(sux_val_pairs_simple_transliterated_translate)
            token_pairs.append(sux_train_pairs_simple_transliterated_translate)
            token_pairs.append(sux_test_pairs_simple_transliterated_translate)
            token_pairs.append(sux_val_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {sux_train_pairs_simple_transliterated_translate[1][0]}, " -> ", {sux_train_pairs_simple_transliterated_translate[1][1]})
        if complex:
            ###Translate from original transliterated Sumerian to English
            sux_train_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transliteration_train[l], use_prefix=True, task="Translate", type="original", language="Sumerian", style=prompt_style), normalizeString_en(sux_en_train[l], use_prefix=False)] for l in range(len(sux_transliteration_train))]
            sux_test_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transliteration_test[l], use_prefix=True, task="Translate", type="original", language="Sumerian", style=prompt_style), normalizeString_en(sux_en_test[l], use_prefix=False)] for l in range(len(sux_transliteration_test))]
            sux_val_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transliteration_val[l], use_prefix=True, task="Translate", type="original", language="Sumerian", style=prompt_style), normalizeString_en(sux_en_val[l], use_prefix=False)] for l in range(len(sux_transliteration_val))]
            train_pairs.append(sux_train_pairs_original_transliterated_translate)
            test_pairs.append(sux_test_pairs_original_transliterated_translate)
            val_pairs.append(sux_val_pairs_original_transliterated_translate)
            token_pairs.append(sux_train_pairs_original_transliterated_translate)
            token_pairs.append(sux_test_pairs_original_transliterated_translate)
            token_pairs.append(sux_val_pairs_original_transliterated_translate)
            if debug == True:
                print(f"", {sux_train_pairs_original_transliterated_translate[1][0]}, " -> ", {sux_train_pairs_original_transliterated_translate[1][1]})
        if group:
            ###Translate from transliterated Sumerian to English
            sux_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(sux_transliteration_train[l], use_prefix=True, language="Sumerian", style=prompt_style), normalizeString_en(sux_en_train[l], use_prefix=False)] for l in range(len(sux_transliteration_train))]
            sux_test_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(sux_transliteration_test[l], use_prefix=True, language="Sumerian", style=prompt_style), normalizeString_en(sux_en_test[l], use_prefix=False)] for l in range(len(sux_transliteration_test))]
            sux_val_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(sux_transliteration_val[l], use_prefix=True, language="Sumerian", style=prompt_style), normalizeString_en(sux_en_val[l], use_prefix=False)] for l in range(len(sux_transliteration_val))]
            train_pairs.append(sux_train_pairs_group_transliterated_translate)
            test_pairs.append(sux_test_pairs_group_transliterated_translate)
            val_pairs.append(sux_val_pairs_group_transliterated_translate)
            if debug == True:
                print(f"", {sux_train_pairs_group_transliterated_translate[1][0]}, " -> ", {sux_train_pairs_group_transliterated_translate[1][1]})
            if transliterate:
                if complex:
                    ###Transliterate from Sumerian Cuenfirom to complex Latin characters
                    sux_train_pairs_transliterate_original = [[normalizeString_cuneiform(sux_cuneiform_train[l], use_prefix=True, task="Transliterate", type="original", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_train[l], use_prefix=False, type="original")] for l in range(len(sux_cuneiform_train))]
                    sux_test_pairs_transliterate_original = [[normalizeString_cuneiform(sux_cuneiform_test[l], use_prefix=True, task="Transliterate", type="original", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_test[l], use_prefix=False, type="original")] for l in range(len(sux_cuneiform_test))]
                    sux_val_pairs_transliterated_original = [[normalizeString_cuneiform(sux_cuneiform_val[l], use_prefix=True, task="Transliterate", type="original", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_val[l], use_prefix=False, type="original")] for l in range(len(sux_cuneiform_val))]
                    train_pairs.append(sux_train_pairs_transliterate_original)
                    test_pairs.append(sux_test_pairs_transliterate_original)
                    val_pairs.append(sux_val_pairs_transliterated_original)
                    if debug == True:
                        print(f"", {sux_train_pairs_transliterate_original[1][0]}, " -> ", {sux_train_pairs_transliterate_original[1][1]})
                if rev_simp_trans:
                    ###Transliterate from Sumerian Cuenfirom to simple Latin characters
                    sux_train_pairs_transliterate_simple = [[normalizeString_cuneiform(sux_cuneiform_train[l], use_prefix=True, task="Transliterate", type="simple", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_train[l], use_prefix=False)] for l in range(len(sux_cuneiform_train))]
                    sux_test_pairs_transliterate_simple = [[normalizeString_cuneiform(sux_cuneiform_test[l], use_prefix=True, task="Transliterate", type="simple", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_test[l], use_prefix=False)] for l in range(len(sux_cuneiform_test))]
                    sux_val_pairs_transliterated_simple = [[normalizeString_cuneiform(sux_cuneiform_val[l], use_prefix=True, task="Transliterate", type="simple", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_val[l], use_prefix=False)] for l in range(len(sux_cuneiform_val))]
                    train_pairs.append(sux_train_pairs_transliterate_simple)
                    test_pairs.append(sux_test_pairs_transliterate_simple)
                    val_pairs.append(sux_val_pairs_transliterated_simple)
                    if debug == True:
                        print(f"", {sux_train_pairs_transliterate_simple[1][0]}, " -> ", {sux_train_pairs_transliterate_simple[1][1]})
                if group:
                    ###Transliterate from Sumerian Cuenfirom to Latin characters
                    sux_train_pairs_transliterate_group = [[normalizeString_cuneiform(sux_cuneiform_train[l], use_prefix=True, task="Transliterate", type="group", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_train[l], use_prefix=False, type="group")] for l in range(len(sux_cuneiform_train))]
                    sux_test_pairs_transliterate_group = [[normalizeString_cuneiform(sux_cuneiform_test[l], use_prefix=True, task="Transliterate", type="group", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_test[l], use_prefix=False, type="group")] for l in range(len(sux_cuneiform_test))]
                    sux_val_pairs_transliterated_group = [[normalizeString_cuneiform(sux_cuneiform_val[l], use_prefix=True, task="Transliterate", type="group", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_val[l], use_prefix=False, type="group")] for l in range(len(sux_cuneiform_val))]
                    train_pairs.append(sux_train_pairs_transliterate_group)
                    test_pairs.append(sux_test_pairs_transliterate_group)
                    val_pairs.append(sux_val_pairs_transliterated_group)
                    if debug == True:
                        print(f"", {sux_train_pairs_transliterate_group[1][0]}, " -> ", {sux_train_pairs_transliterate_group[1][1]})
        if active_group:
            if simple:
                ###Group simple transliterated Sumerian into words
                sux_train_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(sux_transliteration_train[l], use_prefix=True, task="Group", type="simple", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(sux_transliteration_train[l], use_prefix=False)] for l in range(len(sux_transliteration_train))]
                sux_test_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(sux_transliteration_test[l], use_prefix=True, task="Group", type="simple", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(sux_transliteration_test[l], use_prefix=False)] for l in range(len(sux_transliteration_test))]
                sux_val_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(sux_transliteration_val[l], use_prefix=True, task="Group", type="simple", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(sux_transliteration_val[l], use_prefix=False)] for l in range(len(sux_transliteration_val))]
                train_pairs.append(sux_train_pairs_group_simple_transliterate)
                test_pairs.append(sux_test_pairs_group_simple_transliterate)
                val_pairs.append(sux_val_pairs_group_simple_transliterate)
                if debug == True:
                    print(f"", {sux_train_pairs_group_simple_transliterate[1][0]}, " -> ", {sux_train_pairs_group_simple_transliterate[1][1]})
            if complex:
                ###Group complex transliterated Sumerian into words
                sux_train_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(sux_transliteration_train[l], use_prefix=True, task="Group", type="original", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(sux_transliteration_train[l], use_prefix=False)] for l in range(len(sux_transliteration_train))]
                sux_test_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(sux_transliteration_test[l], use_prefix=True, task="Group", type="original", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(sux_transliteration_test[l], use_prefix=False)] for l in range(len(sux_transliteration_test))]
                sux_val_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(sux_transliteration_val[l], use_prefix=True, task="Group", type="original", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(sux_transliteration_val[l], use_prefix=False)] for l in range(len(sux_transliteration_val))]
                train_pairs.append(sux_train_pairs_group_original_transliterate)
                test_pairs.append(sux_test_pairs_group_original_transliterate)
                val_pairs.append(sux_val_pairs_group_original_transliterate)
                if debug == True:
                    print(f"", {sux_train_pairs_group_original_transliterate[1][0]}, " -> ", {sux_train_pairs_group_original_transliterate[1][1]})
        if rev:
            if cuneiform:
                ###Translate from English to cuneiform Sumerian
                sux_train_rev_pairs_cuneiform_translate = [[normalizeString_en(sux_en_train[l], use_prefix=True, task="Translate", target="cuneiform", language="Sumerian", style=prompt_style), normalizeString_cuneiform(sux_cuneiform_train[l], use_prefix=False)] for l in range(len(sux_cuneiform_train))]
                sux_test_rev_pairs_cuneiform_translate = [[normalizeString_en(sux_en_test[l], use_prefix=True, task="Translate", target="cuneiform", language="Sumerian", style=prompt_style), normalizeString_cuneiform(sux_cuneiform_test[l], use_prefix=False)] for l in range(len(sux_cuneiform_test))]
                sux_val_rev_pairs_cuneiform_translate = [[normalizeString_en(sux_en_val[l], use_prefix=True, task="Translate", target="cuneiform", language="Sumerian", style=prompt_style), normalizeString_cuneiform(sux_cuneiform_val[l], use_prefix=False), ] for l in range(len(sux_cuneiform_val))]
                train_pairs.append(sux_train_rev_pairs_cuneiform_translate)
                test_pairs.append(sux_test_rev_pairs_cuneiform_translate)
                val_pairs.append(sux_val_rev_pairs_cuneiform_translate)
                if debug == True:
                    print(f"", {sux_train_rev_pairs_cuneiform_translate[1][0]}, " -> ", {sux_train_rev_pairs_cuneiform_translate[1][1]})
            if simple:
                ###Translate from English to simple transliterated Sumerian
                sux_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(sux_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_train[l], use_prefix=False, type="simple")] for l in range(len(sux_en_train))]
                sux_test_rev_pairs_simple_transliterated_translate = [[normalizeString_en(sux_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_test[l], use_prefix=False, type="simple")] for l in range(len(sux_en_test))]
                sux_val_rev_pairs_simple_transliterated_translate = [[normalizeString_en(sux_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_val[l], use_prefix=False, type="simple")] for l in range(len(sux_en_val))]
                train_pairs.append(sux_train_rev_pairs_simple_transliterated_translate)
                test_pairs.append(sux_test_rev_pairs_simple_transliterated_translate)
                val_pairs.append(sux_val_rev_pairs_simple_transliterated_translate)
                if debug == True:
                    print(f"", {sux_train_rev_pairs_simple_transliterated_translate[1][0]}, " -> ", {sux_train_rev_pairs_simple_transliterated_translate[1][1]})
            if complex:
                ###Translate from English to complex transliterated Sumerian
                sux_train_rev_pairs_original_transliterated_translate = [[normalizeString_en(sux_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_train[l], use_prefix=False, type="original")] for l in range(len(sux_en_train))]
                sux_test_rev_pairs_original_transliterated_translate = [[normalizeString_en(sux_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_test[l], use_prefix=False, type="original")] for l in range(len(sux_en_test))]
                sux_val_rev_pairs_original_transliterated_translate = [[normalizeString_en(sux_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate(sux_transliteration_val[l], use_prefix=False, type="original")] for l in range(len(sux_en_val))]
                train_pairs.append(sux_train_rev_pairs_original_transliterated_translate)
                test_pairs.append(sux_test_rev_pairs_original_transliterated_translate)
                val_pairs.append(sux_val_rev_pairs_original_transliterated_translate)
                token_pairs.append(sux_train_rev_pairs_original_transliterated_translate)
                token_pairs.append(sux_test_rev_pairs_original_transliterated_translate)
                token_pairs.append(sux_val_rev_pairs_original_transliterated_translate)
                if debug == True:
                    print(f"", {sux_train_rev_pairs_original_transliterated_translate[1][0]}, " -> ", {sux_train_rev_pairs_original_transliterated_translate[1][1]})
            if group:
                ###Translate from English to transliterated Sumerian
                sux_train_rev_pairs_group_transliterated_translate = [[normalizeString_en(sux_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(sux_transliteration_train[l], use_prefix=False)] for l in range(len(sux_en_train))]
                sux_test_rev_pairs_group_transliterated_translate = [[normalizeString_en(sux_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(sux_transliteration_test[l], use_prefix=False)] for l in range(len(sux_en_test))]
                sux_val_rev_pairs_group_transliterated_translate = [[normalizeString_en(sux_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Sumerian", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(sux_transliteration_val[l], use_prefix=False)] for l in range(len(sux_en_val))]
                train_pairs.append(sux_train_rev_pairs_group_transliterated_translate)
                test_pairs.append(sux_test_rev_pairs_group_transliterated_translate)
                val_pairs.append(sux_val_rev_pairs_group_transliterated_translate)
                if debug == True:
                    print(f"", {sux_train_rev_pairs_group_transliterated_translate[1][0]}, " -> ", {sux_train_rev_pairs_group_transliterated_translate[1][1]})
            if transliterate:
                if rev_simp_trans:
                    ###Convert from simple transliterated Sumerian to cuneiform
                    sux_train_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(sux_transliteration_train[l], use_prefix=True, type="simple", language="Sumerian", style=prompt_style), normalizeString_cuneiform(sux_cuneiform_train[l], use_prefix=False)] for l in range(len(sux_cuneiform_train))]
                    sux_test_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(sux_transliteration_test[l], use_prefix=True, type="simple", language="Sumerian", style=prompt_style), normalizeString_cuneiform(sux_cuneiform_test[l], use_prefix=False)] for l in range(len(sux_transliteration_test))]
                    sux_val_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(sux_transliteration_val[l], use_prefix=True, type="simple", language="Sumerian", style=prompt_style), normalizeString_cuneiform(sux_cuneiform_val[l], use_prefix=False)] for l in range(len(sux_transliteration_val))]
                    train_pairs.append(sux_train_rev_pairs_transliterate_simple)
                    test_pairs.append(sux_test_rev_pairs_transliterate_simple)
                    val_pairs.append(sux_val_rev_pairs_transliterate_simple)
                    if debug == True:
                        print(f"", {sux_train_rev_pairs_transliterate_simple[1][0]}, " -> ", {sux_train_rev_pairs_transliterate_simple[1][1]})
                if group:
                    ###Convert from transliterated Sumerian to cuneiform
                    sux_train_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(sux_transliteration_train[l], use_prefix=True, type="group", language="Sumerian", style=prompt_style), normalizeString_cuneiform(sux_cuneiform_train[l], use_prefix=False)] for l in range(len(sux_cuneiform_train))]
                    sux_test_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(sux_transliteration_test[l], use_prefix=True, type="group", language="Sumerian", style=prompt_style), normalizeString_cuneiform(sux_cuneiform_test[l], use_prefix=False)] for l in range(len(sux_transliteration_test))]
                    sux_val_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(sux_transliteration_val[l], use_prefix=True, type="group", language="Sumerian", style=prompt_style), normalizeString_cuneiform(sux_cuneiform_val[l], use_prefix=False)] for l in range(len(sux_transliteration_val))]
                    train_pairs.append(sux_train_rev_pairs_transliterate_group)
                    test_pairs.append(sux_test_rev_pairs_transliterate_group)
                    val_pairs.append(sux_val_rev_pairs_transliterate_group)
                    if debug == True:
                        print(f"", {sux_train_rev_pairs_transliterate_group[1][0]}, " -> ", {sux_train_rev_pairs_transliterate_group[1][1]})
                if rev_complex_trans:
                    ###Convert from complex transliterated Sumerian to cuneiform
                    sux_train_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(sux_transliteration_train[l], use_prefix=True, type="original", language="Sumerian", style=prompt_style), normalizeString_cuneiform(sux_cuneiform_train[l], use_prefix=False)] for l in range(len(sux_cuneiform_train))]
                    sux_test_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(sux_transliteration_test[l], use_prefix=True, type="original", language="Sumerian", style=prompt_style), normalizeString_cuneiform(sux_cuneiform_test[l], use_prefix=False)] for l in range(len(sux_transliteration_test))]
                    sux_val_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(sux_transliteration_val[l], use_prefix=True, type="original", language="Sumerian", style=prompt_style), normalizeString_cuneiform(sux_cuneiform_val[l], use_prefix=False)] for l in range(len(sux_transliteration_val))]
                    train_pairs.append(sux_train_rev_pairs_transliterate_original)
                    test_pairs.append(sux_test_rev_pairs_transliterate_original)
                    val_pairs.append(sux_val_rev_pairs_transliterate_original)
                    if debug == True:
                        print(f"", {sux_train_rev_pairs_transliterate_original[1][0]}, " -> ", {sux_train_rev_pairs_transliterate_original[1][1]})
    if elx:
        ###############
        ###Elamite###
        ###############
        # Read the file and split into lines
        if data=="line":
            elx_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_line.tr'))
            elx_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_line.en'))
        elif data=="document":
            elx_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_document.tr'))
            elx_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_document.en'))
        elif data=="both":
            elx_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train.tr'))
            elx_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train.en'))
        # Split every line into pairs and normalize
        if simple:
            ##Translate from simple transliterated Elamite to English
            elx_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(elx_transliteration_train[l], use_prefix=True, task="Translate", type="simple", language="Linear B", style=prompt_style), normalizeString_en(elx_en_train[l], use_prefix=False)] for l in range(len(elx_transliteration_train))]
            train_pairs.append(elx_train_pairs_simple_transliterated_translate)
            token_pairs.append(elx_train_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {elx_train_pairs_simple_transliterated_translate[1][0]}, " -> ", {elx_train_pairs_simple_transliterated_translate[1][1]})
        if complex:
            ###Translate from original transliterated Elamite to English
            elx_train_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(elx_transliteration_train[l], use_prefix=True, task="Translate", type="original", language="Elamite", style=prompt_style), normalizeString_en(elx_en_train[l], use_prefix=False)] for l in range(len(elx_transliteration_train))]
            train_pairs.append(elx_train_pairs_original_transliterated_translate)
            token_pairs.append(elx_train_pairs_original_transliterated_translate)
            if debug == True:
                print(f"", {elx_train_pairs_original_transliterated_translate[1][0]}, " -> ", {elx_train_pairs_original_transliterated_translate[1][1]})
        if group:
            ###Translate from transliterated Elamite to English
            elx_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(elx_transliteration_train[l], use_prefix=True, language="Elamite", style=prompt_style), normalizeString_en(elx_en_train[l], use_prefix=False)] for l in range(len(elx_transliteration_train))]
            train_pairs.append(elx_train_pairs_group_transliterated_translate)
            if debug == True:
               print(f"", {elx_train_pairs_group_transliterated_translate[1][0]}, " -> ", {elx_train_pairs_group_transliterated_translate[1][1]})
            if simple:
                ###Group simple transliterated Elamite into words
                elx_train_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(elx_transliteration_train[l], use_prefix=True, task="Group", type="simple", language="Elamite", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(elx_transliteration_train[l], use_prefix=False)] for l in range(len(elx_transliteration_train))]
                train_pairs.append(elx_train_pairs_group_simple_transliterate)
                if debug == True:
                    print(f"", {elx_train_pairs_group_simple_transliterate[1][0]}, " -> ", {elx_train_pairs_group_simple_transliterate[1][1]})
            if active_group:
                ###Group complex transliterated Elamite into words
                elx_train_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(elx_transliteration_train[l], use_prefix=True, task="Group", type="original", language="Elamite", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(elx_transliteration_train[l], use_prefix=False)] for l in range(len(elx_transliteration_train))]
                train_pairs.append(elx_train_pairs_group_original_transliterate)
                if debug == True:
                    print(f"", {elx_train_pairs_group_original_transliterate[1][0]}, " -> ", {elx_train_pairs_group_original_transliterate[1][1]})
        if rev:
            if simple:
                ###Translate from English to simple transliterated Elamite
                elx_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(elx_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(elx_transliteration_train[l], use_prefix=False, type="simple")] for l in range(len(elx_en_train))]
                train_pairs.append(elx_train_rev_pairs_simple_transliterated_translate)
                if debug == True:
                    print(f"", {elx_train_rev_pairs_simple_transliterated_translate[1][0]}, " -> ", {elx_train_rev_pairs_simple_transliterated_translate[1][1]})
            if complex:
                ###Translate from English to complex transliterated Elamite
                elx_train_rev_pairs_original_transliterated_translate = [[normalizeString_en(elx_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Elamite", style=prompt_style), normalizeString_cuneiform_transliterate(elx_transliteration_train[l], use_prefix=False, type="original")] for l in range(len(elx_en_train))]
                train_pairs.append(elx_train_rev_pairs_original_transliterated_translate)
                if debug == True:
                    print(f"", {elx_train_rev_pairs_original_transliterated_translate[1][0]}, " -> ", {elx_train_rev_pairs_original_transliterated_translate[1][1]})
            if group:
                ###Translate from English to transliterated Elamite
                elx_train_rev_pairs_group_transliterated_translate = [[normalizeString_en(elx_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Elamite", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(elx_transliteration_train[l], use_prefix=False)] for l in range(len(elx_en_train))]
                train_pairs.append(elx_train_rev_pairs_group_transliterated_translate)
                if debug == True:
                    print(f"", {elx_train_rev_pairs_group_transliterated_translate[1][0]}, " -> ", {elx_train_rev_pairs_group_transliterated_translate[1][1]})
    if gmy:
        ##############
        ###Linear B###
        ##############
        # Read the file and split into lines
        gmy_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'gmy_train.cu'))
        gmy_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'gmy_train.tr'))
        gmy_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'gmy_train.en'))
        gmy_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'gmy_test.cu'))
        gmy_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'gmy_test.tr'))
        gmy_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'gmy_test.en'))
        gmy_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'gmy_valid.cu'))
        gmy_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'gmy_valid.tr'))
        gmy_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'gmy_valid.en'))
        # Split every line into pairs and normalize
        if cuneiform:
            ###Translate from Linear B cuneiform to English
            gmy_train_pairs_cuneiform_translate = [[normalizeString_cuneiform(gmy_cuneiform_train[l], use_prefix=True, task="Translate", language="Linear B", style=prompt_style), normalizeString_en(gmy_en_train[l], use_prefix=False)] for l in range(len(gmy_cuneiform_train))]
            gmy_test_pairs_cuneiform_translate = [[normalizeString_cuneiform(gmy_cuneiform_test[l], use_prefix=True, task="Translate", language="Linear B", style=prompt_style), normalizeString_en(gmy_en_test[l], use_prefix=False)] for l in range(len(gmy_cuneiform_test))]
            gmy_val_pairs_cuneiform_translate = [[normalizeString_cuneiform(gmy_cuneiform_val[l], use_prefix=True, task="Translate", language="Linear B", style=prompt_style), normalizeString_en(gmy_en_val[l], use_prefix=False)] for l in range(len(gmy_cuneiform_val))]
            train_pairs.append(gmy_train_pairs_cuneiform_translate)
            test_pairs.append(gmy_test_pairs_cuneiform_translate)
            val_pairs.append(gmy_val_pairs_cuneiform_translate)
            token_pairs.append(gmy_train_pairs_cuneiform_translate)
            token_pairs.append(gmy_test_pairs_cuneiform_translate)
            token_pairs.append(gmy_val_pairs_cuneiform_translate)
            if debug == True:
                print(f"", {gmy_train_pairs_cuneiform_translate[1][0]}, " -> ", {gmy_train_pairs_cuneiform_translate[1][1]})
        if simple:
            ###Translate from simple transliterated Linear B to English
            gmy_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(gmy_transliteration_train[l], use_prefix=True, task="Translate", type="simple", language="Linear B", style=prompt_style), normalizeString_en(gmy_en_train[l], use_prefix=False)] for l in range(len(gmy_transliteration_train))]
            gmy_test_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(gmy_transliteration_test[l], use_prefix=True, task="Translate", type="simple", language="Linear B", style=prompt_style), normalizeString_en(gmy_en_test[l], use_prefix=False)] for l in range(len(gmy_transliteration_test))]
            gmy_val_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(gmy_transliteration_val[l], use_prefix=True, task="Translate", type="simple", language="Linear B", style=prompt_style), normalizeString_en(gmy_en_val[l], use_prefix=False)] for l in range(len(gmy_transliteration_val))]
            train_pairs.append(gmy_train_pairs_simple_transliterated_translate)
            test_pairs.append(gmy_test_pairs_simple_transliterated_translate)
            val_pairs.append(gmy_val_pairs_simple_transliterated_translate)
            token_pairs.append(gmy_train_pairs_simple_transliterated_translate)
            token_pairs.append(gmy_test_pairs_simple_transliterated_translate)
            token_pairs.append(gmy_val_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {gmy_train_pairs_simple_transliterated_translate[1][0]}, " -> ", {gmy_train_pairs_simple_transliterated_translate[1][1]})
        if complex:
            ###Translate from original transliterated Linear B to English
            gmy_train_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(gmy_transliteration_train[l], use_prefix=True, task="Translate", type="original", language="Linear B", style=prompt_style), normalizeString_en(gmy_en_train[l], use_prefix=False)] for l in range(len(gmy_transliteration_train))]
            gmy_test_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(gmy_transliteration_test[l], use_prefix=True, task="Translate", type="original", language="Linear B", style=prompt_style), normalizeString_en(gmy_en_test[l], use_prefix=False)] for l in range(len(gmy_transliteration_test))]
            gmy_val_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(gmy_transliteration_val[l], use_prefix=True, task="Translate", type="original", language="Linear B", style=prompt_style), normalizeString_en(gmy_en_val[l], use_prefix=False)] for l in range(len(gmy_transliteration_val))]
            train_pairs.append(gmy_train_pairs_original_transliterated_translate)
            test_pairs.append(gmy_test_pairs_original_transliterated_translate)
            val_pairs.append(gmy_val_pairs_original_transliterated_translate)
            token_pairs.append(gmy_train_pairs_original_transliterated_translate)
            token_pairs.append(gmy_test_pairs_original_transliterated_translate)
            token_pairs.append(gmy_val_pairs_original_transliterated_translate)
            if debug == True:
                print(f"", {gmy_train_pairs_original_transliterated_translate[1][0]}, " -> ", {gmy_train_pairs_original_transliterated_translate[1][1]})
        if group:
            ###Translate from transliterated Linear B to English
            gmy_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(gmy_transliteration_train[l], use_prefix=True, language="Linear B", style=prompt_style), normalizeString_en(gmy_en_train[l], use_prefix=False)] for l in range(len(gmy_transliteration_train))]
            gmy_test_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(gmy_transliteration_test[l], use_prefix=True, language="Linear B", style=prompt_style), normalizeString_en(gmy_en_test[l], use_prefix=False)] for l in range(len(gmy_transliteration_test))]
            gmy_val_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(gmy_transliteration_val[l], use_prefix=True, language="Linear B", style=prompt_style), normalizeString_en(gmy_en_val[l], use_prefix=False)] for l in range(len(gmy_transliteration_val))]
            train_pairs.append(gmy_train_pairs_group_transliterated_translate)
            test_pairs.append(gmy_test_pairs_group_transliterated_translate)
            val_pairs.append(gmy_val_pairs_group_transliterated_translate)
            if debug == True:
                print(f"", {gmy_train_pairs_group_transliterated_translate[1][0]}, " -> ", {gmy_train_pairs_group_transliterated_translate[1][1]})
        if transliterate:
            if rev_simp_trans:
                ###Transliterate from Linear B Cuenfirom to simple Latin characters
                gmy_train_pairs_transliterate_simple = [[normalizeString_cuneiform(gmy_cuneiform_train[l], use_prefix=True, task="Transliterate", type="simple", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_train[l], use_prefix=False)] for l in range(len(gmy_cuneiform_train))]
                gmy_test_pairs_transliterate_simple = [[normalizeString_cuneiform(gmy_cuneiform_test[l], use_prefix=True, task="Transliterate", type="simple", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_test[l], use_prefix=False)] for l in range(len(gmy_cuneiform_test))]
                gmy_val_pairs_transliterated_simple = [[normalizeString_cuneiform(gmy_cuneiform_val[l], use_prefix=True, task="Transliterate", type="simple", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_val[l], use_prefix=False)] for l in range(len(gmy_cuneiform_val))]
                train_pairs.append(gmy_train_pairs_transliterate_simple)
                test_pairs.append(gmy_test_pairs_transliterate_simple)
                val_pairs.append(gmy_val_pairs_transliterated_simple)
                if debug == True:
                    print(f"", {gmy_train_pairs_transliterate_simple[1][0]}, " -> ", {gmy_train_pairs_transliterate_simple[1][1]})
            if rev_complex_trans:
                ###Transliterate from Linear B Cuenfirom to complex Latin characters
                gmy_train_pairs_transliterate_original = [[normalizeString_cuneiform(gmy_cuneiform_train[l], use_prefix=True, task="Transliterate", type="original", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_train[l], use_prefix=False, type="original")] for l in range(len(gmy_cuneiform_train))]
                gmy_test_pairs_transliterate_original = [[normalizeString_cuneiform(gmy_cuneiform_test[l], use_prefix=True, task="Transliterate", type="original", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_test[l], use_prefix=False, type="original")] for l in range(len(gmy_cuneiform_test))]
                gmy_val_pairs_transliterated_original = [[normalizeString_cuneiform(gmy_cuneiform_val[l], use_prefix=True, task="Transliterate", type="original", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_val[l], use_prefix=False, type="original")] for l in range(len(gmy_cuneiform_val))]
                train_pairs.append(gmy_train_pairs_transliterate_original)
                test_pairs.append(gmy_test_pairs_transliterate_original)
                val_pairs.append(gmy_val_pairs_transliterated_original)
                if debug == True:
                    print(f"", {gmy_train_pairs_transliterate_original[1][0]}, " -> ", {gmy_train_pairs_transliterate_original[1][1]})
            if group:
                ###Transliterate from Linear B Cuenfirom to Latin characters
                gmy_train_pairs_transliterate_group = [[normalizeString_cuneiform(gmy_cuneiform_train[l], use_prefix=True, task="Transliterate", type="group", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_train[l], use_prefix=False, type="group")] for l in range(len(gmy_cuneiform_train))]
                gmy_test_pairs_transliterate_group = [[normalizeString_cuneiform(gmy_cuneiform_test[l], use_prefix=True, task="Transliterate", type="group", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_test[l], use_prefix=False, type="group")] for l in range(len(gmy_cuneiform_test))]
                gmy_val_pairs_transliterated_group = [[normalizeString_cuneiform(gmy_cuneiform_val[l], use_prefix=True, task="Transliterate", type="group", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_val[l], use_prefix=False, type="group")] for l in range(len(gmy_cuneiform_val))]
                train_pairs.append(gmy_train_pairs_transliterate_group)
                test_pairs.append(gmy_test_pairs_transliterate_group)
                val_pairs.append(gmy_val_pairs_transliterated_group)
                if debug == True:
                    print(f"", {gmy_train_pairs_transliterate_group[1][0]}, " -> ", {gmy_train_pairs_transliterate_group[1][1]})
        if active_group:
            if simple:
                ###Group simple transliterated Linear B into words
                gmy_train_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(gmy_transliteration_train[l], use_prefix=True, task="Group", type="simple", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(gmy_transliteration_train[l], use_prefix=False)] for l in range(len(gmy_cuneiform_train))]
                gmy_test_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(gmy_transliteration_test[l], use_prefix=True, task="Group", type="simple", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(gmy_transliteration_test[l], use_prefix=False)] for l in range(len(gmy_cuneiform_test))]
                gmy_val_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(gmy_transliteration_val[l], use_prefix=True, task="Group", type="simple", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(gmy_transliteration_val[l], use_prefix=False)] for l in range(len(gmy_cuneiform_val))]
                train_pairs.append(gmy_train_pairs_group_simple_transliterate)
                test_pairs.append(gmy_test_pairs_group_simple_transliterate)
                val_pairs.append(gmy_val_pairs_group_simple_transliterate)
                if debug == True:
                    print(f"", {gmy_train_pairs_group_simple_transliterate[1][0]}, " -> ", {gmy_train_pairs_group_simple_transliterate[1][1]})
            if complex:
                ###Group complex transliterated Linear B into words
                gmy_train_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(gmy_transliteration_train[l], use_prefix=True, task="Group", type="original", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(gmy_transliteration_train[l], use_prefix=False)] for l in range(len(gmy_cuneiform_train))]
                gmy_test_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(gmy_transliteration_test[l], use_prefix=True, task="Group", type="original", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(gmy_transliteration_test[l], use_prefix=False)] for l in range(len(gmy_cuneiform_test))]
                gmy_val_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(gmy_transliteration_val[l], use_prefix=True, task="Group", type="original", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(gmy_transliteration_val[l], use_prefix=False)] for l in range(len(gmy_cuneiform_val))]
                train_pairs.append(gmy_train_pairs_group_original_transliterate)
                test_pairs.append(gmy_test_pairs_group_original_transliterate)
                val_pairs.append(gmy_val_pairs_group_original_transliterate)
                if debug == True:
                    print(f"", {gmy_train_pairs_group_original_transliterate[1][0]}, " -> ", {gmy_train_pairs_group_original_transliterate[1][1]})
        if rev:
            if cuneiform:
                ###Translate from English to cuneiform Linear B
                gmy_train_rev_pairs_cuneiform_translate = [[normalizeString_en(gmy_en_train[l], use_prefix=True, task="Translate", target="cuneiform", language="Linear B", style=prompt_style), normalizeString_cuneiform(gmy_cuneiform_train[l], use_prefix=False)] for l in range(len(gmy_cuneiform_train))]
                gmy_test_rev_pairs_cuneiform_translate = [[normalizeString_en(gmy_en_test[l], use_prefix=True, task="Translate", target="cuneiform", language="Linear B", style=prompt_style), normalizeString_cuneiform(gmy_cuneiform_test[l], use_prefix=False)] for l in range(len(gmy_cuneiform_test))]
                gmy_val_rev_pairs_cuneiform_translate = [[normalizeString_en(gmy_en_val[l], use_prefix=True, task="Translate", target="cuneiform", language="Linear B", style=prompt_style), normalizeString_cuneiform(gmy_cuneiform_val[l], use_prefix=False), ] for l in range(len(gmy_cuneiform_val))]
                train_pairs.append(gmy_train_rev_pairs_cuneiform_translate)
                test_pairs.append(gmy_test_rev_pairs_cuneiform_translate)
                val_pairs.append(gmy_val_rev_pairs_cuneiform_translate)
                if debug == True:
                    print(f"", {gmy_train_rev_pairs_cuneiform_translate[1][0]}, " -> ", {gmy_train_rev_pairs_cuneiform_translate[1][1]})
            if simple:
                ###Translate from English to simple transliterated Linear B
                gmy_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(gmy_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_train[l], use_prefix=False, type="simple")] for l in range(len(gmy_en_train))]
                gmy_test_rev_pairs_simple_transliterated_translate = [[normalizeString_en(gmy_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_test[l], use_prefix=False, type="simple")] for l in range(len(gmy_en_test))]
                gmy_val_rev_pairs_simple_transliterated_translate = [[normalizeString_en(gmy_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_val[l], use_prefix=False, type="simple")] for l in range(len(gmy_en_val))]
                train_pairs.append(gmy_train_rev_pairs_simple_transliterated_translate)
                test_pairs.append(gmy_test_rev_pairs_simple_transliterated_translate)
                val_pairs.append(gmy_val_rev_pairs_simple_transliterated_translate)
                if debug == True:
                    print(f"", {gmy_train_rev_pairs_simple_transliterated_translate[1][0]}, " -> ", {gmy_train_rev_pairs_simple_transliterated_translate[1][1]})
            if complex:
                ###Translate from English to complex transliterated Linear B
                gmy_train_rev_pairs_original_transliterated_translate = [[normalizeString_en(gmy_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_train[l], use_prefix=False, type="original")] for l in range(len(gmy_en_train))]
                gmy_test_rev_pairs_original_transliterated_translate = [[normalizeString_en(gmy_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_test[l], use_prefix=False, type="original")] for l in range(len(gmy_en_test))]
                gmy_val_rev_pairs_original_transliterated_translate = [[normalizeString_en(gmy_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate(gmy_transliteration_val[l], use_prefix=False, type="original")] for l in range(len(gmy_en_val))]
                train_pairs.append(gmy_train_rev_pairs_original_transliterated_translate)
                test_pairs.append(gmy_test_rev_pairs_original_transliterated_translate)
                val_pairs.append(gmy_val_rev_pairs_original_transliterated_translate)
                token_pairs.append(gmy_train_rev_pairs_original_transliterated_translate)
                token_pairs.append(gmy_test_rev_pairs_original_transliterated_translate)
                token_pairs.append(gmy_val_rev_pairs_original_transliterated_translate)
                if debug == True:
                    print(f"", {gmy_train_rev_pairs_original_transliterated_translate[1][0]}, " -> ", {gmy_train_rev_pairs_original_transliterated_translate[1][1]})
            if group:
                ###Translate from English to transliterated Linear B
                gmy_train_rev_pairs_group_transliterated_translate = [[normalizeString_en(gmy_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(gmy_transliteration_train[l], use_prefix=False)] for l in range(len(gmy_en_train))]
                gmy_test_rev_pairs_group_transliterated_translate = [[normalizeString_en(gmy_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(gmy_transliteration_test[l], use_prefix=False)] for l in range(len(gmy_en_test))]
                gmy_val_rev_pairs_group_transliterated_translate = [[normalizeString_en(gmy_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Linear B", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(gmy_transliteration_val[l], use_prefix=False)] for l in range(len(gmy_en_val))]
                train_pairs.append(gmy_train_rev_pairs_group_transliterated_translate)
                test_pairs.append(gmy_test_rev_pairs_group_transliterated_translate)
                val_pairs.append(gmy_val_rev_pairs_group_transliterated_translate)
                if debug == True:
                    print(f"", {gmy_train_rev_pairs_group_transliterated_translate[1][0]}, " -> ", {gmy_train_rev_pairs_group_transliterated_translate[1][1]})
            if rev_simp_trans:
                ###Convert from simple transliterated Linear B to cuneiform
                gmy_train_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(gmy_transliteration_train[l], use_prefix=True, type="simple", language="Linear B", style=prompt_style), normalizeString_cuneiform(gmy_cuneiform_train[l], use_prefix=False)] for l in range(len(gmy_transliteration_train))]
                gmy_test_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(gmy_transliteration_test[l], use_prefix=True, type="simple", language="Linear B", style=prompt_style), normalizeString_cuneiform(gmy_cuneiform_test[l], use_prefix=False)] for l in range(len(gmy_transliteration_test))]
                gmy_val_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(gmy_transliteration_val[l], use_prefix=True, type="simple", language="Linear B", style=prompt_style), normalizeString_cuneiform(gmy_cuneiform_val[l], use_prefix=False)] for l in range(len(gmy_transliteration_val))]
                train_pairs.append(gmy_train_rev_pairs_transliterate_simple)
                test_pairs.append(gmy_test_rev_pairs_transliterate_simple)
                val_pairs.append(gmy_val_rev_pairs_transliterate_simple)
                if debug == True:
                    print(f"", {gmy_train_rev_pairs_transliterate_simple[1][0]}, " -> ", {gmy_train_rev_pairs_transliterate_simple[1][1]})
            if transliterate:
                if group:
                    ###Convert from transliterated Linear B to cuneiform
                    gmy_train_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(gmy_transliteration_train[l], use_prefix=True, type="group", language="Linear B", style=prompt_style), normalizeString_cuneiform(gmy_cuneiform_train[l], use_prefix=False)] for l in range(len(gmy_transliteration_train))]
                    gmy_test_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(gmy_transliteration_test[l], use_prefix=True, type="group", language="Linear B", style=prompt_style), normalizeString_cuneiform(gmy_cuneiform_test[l], use_prefix=False)] for l in range(len(gmy_transliteration_test))]
                    gmy_val_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(gmy_transliteration_val[l], use_prefix=True, type="group", language="Linear B", style=prompt_style), normalizeString_cuneiform(gmy_cuneiform_val[l], use_prefix=False)] for l in range(len(gmy_transliteration_val))]
                    train_pairs.append(gmy_train_rev_pairs_transliterate_group)
                    test_pairs.append(gmy_test_rev_pairs_transliterate_group)
                    val_pairs.append(gmy_val_rev_pairs_transliterate_group)
                    if debug == True:
                        print(f"", {gmy_train_rev_pairs_transliterate_group[1][0]}, " -> ", {gmy_train_rev_pairs_transliterate_group[1][1]})
                if rev_complex_trans:
                    ###Convert from complex transliterated Linear B to cuneiform
                    gmy_train_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(gmy_transliteration_train[l], use_prefix=True, type="original", language="Linear B", style=prompt_style), normalizeString_cuneiform(gmy_cuneiform_train[l], use_prefix=False)] for l in range(len(gmy_transliteration_train))]
                    gmy_test_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(gmy_transliteration_test[l], use_prefix=True, type="original", language="Linear B", style=prompt_style), normalizeString_cuneiform(gmy_cuneiform_test[l], use_prefix=False)] for l in range(len(gmy_transliteration_test))]
                    gmy_val_rev_pairs_transliterate_original = [[normalizeString_cuneiform_rev_transliterate(gmy_transliteration_val[l], use_prefix=True, type="original", language="Linear B", style=prompt_style), normalizeString_cuneiform(gmy_cuneiform_val[l], use_prefix=False)] for l in range(len(gmy_transliteration_val))]
                    train_pairs.append(gmy_train_rev_pairs_transliterate_original)
                    test_pairs.append(gmy_test_rev_pairs_transliterate_original)
                    val_pairs.append(gmy_val_rev_pairs_transliterate_original)
                    if debug == True:
                        print(f"", {gmy_train_rev_pairs_transliterate_original[1][0]}, " -> ", {gmy_train_rev_pairs_transliterate_original[1][1]})
    if hit:
        #############
        ###Hittite###
        #############
        # Read the file and split into lines
        if data=="line":
            hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_line.tr'))
            hit_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_line.en'))
            hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_line.tr'))
            hit_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_line.en'))
            hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_line.tr'))
            hit_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_line.en'))
        elif data=="document":
            hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_document.tr'))
            hit_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_document.en'))
            hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_document.tr'))
            hit_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_document.en'))
            hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_document.tr'))
            hit_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_document.en'))
        elif data=="both":
            hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train.tr'))
            hit_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train.en'))
            hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test.tr'))
            hit_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test.en'))
            hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid.tr'))
            hit_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid.en'))
        if ld:
            hit_ld_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_train.tr'))
            hit_ld_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_train.en'))
            hit_ld_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_test.tr'))
            hit_ld_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_test.en'))
            hit_ld_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_valid.tr'))
            hit_ld_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_valid.en'))
            # Add ALL data_line data to training (including ld test/val for max coverage)
            # Val/test remain main-data-only to avoid evaluating on data_line examples
            hit_transliteration_train = hit_transliteration_train + hit_ld_transliteration_train + hit_ld_transliteration_test + hit_ld_transliteration_val
            hit_en_train = hit_en_train + hit_ld_en_train + hit_ld_en_test + hit_ld_en_val
        # Split every line into pairs and normalize
        if simple:
            ##Translate from simple transliterated Hittite to English
            hit_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_train[l], use_prefix=True, task="Translate", type="simple", language="Hittite", style=prompt_style), normalizeString_en(hit_en_train[l], use_prefix=False)] for l in range(len(hit_transliteration_train))]
            hit_test_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_test[l], use_prefix=True, task="Translate", type="simple", language="Hittite", style=prompt_style), normalizeString_en(hit_en_test[l], use_prefix=False)] for l in range(len(hit_transliteration_test))]
            hit_val_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_val[l], use_prefix=True, task="Translate", type="simple", language="Hittite", style=prompt_style), normalizeString_en(hit_en_val[l], use_prefix=False)] for l in range(len(hit_transliteration_val))]
            train_pairs.append(hit_train_pairs_simple_transliterated_translate)
            test_pairs.append(hit_test_pairs_simple_transliterated_translate)
            val_pairs.append(hit_val_pairs_simple_transliterated_translate)
            token_pairs.append(hit_train_pairs_simple_transliterated_translate)
            token_pairs.append(hit_test_pairs_simple_transliterated_translate)
            token_pairs.append(hit_val_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {hit_train_pairs_simple_transliterated_translate[1][0]}, " -> ", {hit_train_pairs_simple_transliterated_translate[1][1]})
        if complex:
            ###Translate from original transliterated Hittite to English
            hit_train_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_train[l], use_prefix=True, task="Translate", type="original", language="Hittite", style=prompt_style), normalizeString_en(hit_en_train[l], use_prefix=False)] for l in range(len(hit_transliteration_train))]
            hit_test_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_test[l], use_prefix=True, task="Translate", type="original", language="Hittite", style=prompt_style), normalizeString_en(hit_en_test[l], use_prefix=False)] for l in range(len(hit_transliteration_test))]
            hit_val_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_val[l], use_prefix=True, task="Translate", type="original", language="Hittite", style=prompt_style), normalizeString_en(hit_en_val[l], use_prefix=False)] for l in range(len(hit_transliteration_val))]
            train_pairs.append(hit_train_pairs_original_transliterated_translate)
            test_pairs.append(hit_test_pairs_original_transliterated_translate)
            val_pairs.append(hit_val_pairs_original_transliterated_translate)
            token_pairs.append(hit_train_pairs_original_transliterated_translate)
            token_pairs.append(hit_test_pairs_original_transliterated_translate)
            token_pairs.append(hit_val_pairs_original_transliterated_translate)
            if debug == True:
                print(f"", {hit_train_pairs_original_transliterated_translate[1][0]}, " -> ", {hit_train_pairs_original_transliterated_translate[1][1]})
        if group:
            ###Translate from transliterated Hittite to English
            hit_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(hit_transliteration_train[l], use_prefix=True, language="Hittite", style=prompt_style), normalizeString_en(hit_en_train[l], use_prefix=False)] for l in range(len(hit_transliteration_train))]
            hit_test_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(hit_transliteration_test[l], use_prefix=True, language="Hittite", style=prompt_style), normalizeString_en(hit_en_test[l], use_prefix=False)] for l in range(len(hit_transliteration_test))]
            hit_val_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(hit_transliteration_val[l], use_prefix=True, language="Hittite", style=prompt_style), normalizeString_en(hit_en_val[l], use_prefix=False)] for l in range(len(hit_transliteration_val))]
            train_pairs.append(hit_train_pairs_group_transliterated_translate)
            test_pairs.append(hit_test_pairs_group_transliterated_translate)
            val_pairs.append(hit_val_pairs_group_transliterated_translate)
            if debug == True:
                print(f"", {hit_train_pairs_group_transliterated_translate[1][0]}, " -> ", {hit_train_pairs_group_transliterated_translate[1][1]})
            if active_group:
                if simple:
                    ###Group simple transliterated Hittite into words
                    hit_train_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_train[l], use_prefix=True, task="Group", type="simple", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(hit_transliteration_train[l], use_prefix=False)] for l in range(len(hit_transliteration_train))]
                    hit_test_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_test[l], use_prefix=True, task="Group", type="simple", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(hit_transliteration_test[l], use_prefix=False)] for l in range(len(hit_transliteration_test))]
                    hit_val_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_val[l], use_prefix=True, task="Group", type="simple", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(hit_transliteration_val[l], use_prefix=False)] for l in range(len(hit_transliteration_val))]
                    train_pairs.append(hit_train_pairs_group_simple_transliterate)
                    test_pairs.append(hit_test_pairs_group_simple_transliterate)
                    val_pairs.append(hit_val_pairs_group_simple_transliterate)
                    if debug == True:
                        print(f"", {hit_train_pairs_group_simple_transliterate[1][0]}, " -> ", {hit_train_pairs_group_simple_transliterate[1][1]})
                if complex:
                    ###Group complex transliterated Hittite into words
                    hit_train_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_train[l], use_prefix=True, task="Group", type="original", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(hit_transliteration_train[l], use_prefix=False)] for l in range(len(hit_transliteration_train))]
                    hit_test_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_test[l], use_prefix=True, task="Group", type="original", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(hit_transliteration_test[l], use_prefix=False)] for l in range(len(hit_transliteration_test))]
                    hit_val_pairs_group_original_transliterate = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_val[l], use_prefix=True, task="Group", type="original", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(hit_transliteration_val[l], use_prefix=False)] for l in range(len(hit_transliteration_val))]
                    train_pairs.append(hit_train_pairs_group_original_transliterate)
                    test_pairs.append(hit_test_pairs_group_original_transliterate)
                    val_pairs.append(hit_val_pairs_group_original_transliterate)
                    if debug == True:
                        print(f"", {hit_train_pairs_group_original_transliterate[1][0]}, " -> ", {hit_train_pairs_group_original_transliterate[1][1]})
        if rev:
            if simple:
                ###Translate from English to simple transliterated Hittite
                hit_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(hit_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate(hit_transliteration_train[l], use_prefix=False, type="simple")] for l in range(len(hit_en_train))]
                hit_test_rev_pairs_simple_transliterated_translate = [[normalizeString_en(hit_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate(hit_transliteration_test[l], use_prefix=False, type="simple")] for l in range(len(hit_en_test))]
                hit_val_rev_pairs_simple_transliterated_translate = [[normalizeString_en(hit_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate(hit_transliteration_val[l], use_prefix=False, type="simple")] for l in range(len(hit_en_val))]
                train_pairs.append(hit_train_rev_pairs_simple_transliterated_translate)
                test_pairs.append(hit_test_rev_pairs_simple_transliterated_translate)
                val_pairs.append(hit_val_rev_pairs_simple_transliterated_translate)
                if debug == True:
                    print(f"", {hit_train_rev_pairs_simple_transliterated_translate[1][0]}, " -> ", {hit_train_rev_pairs_simple_transliterated_translate[1][1]})
            if complex:
                ###Translate from English to complex transliterated Hittite
                hit_train_rev_pairs_original_transliterated_translate = [[normalizeString_en(hit_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate(hit_transliteration_train[l], use_prefix=False, type="original")] for l in range(len(hit_en_train))]
                hit_test_rev_pairs_original_transliterated_translate = [[normalizeString_en(hit_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate(hit_transliteration_test[l], use_prefix=False, type="original")] for l in range(len(hit_en_test))]
                hit_val_rev_pairs_original_transliterated_translate = [[normalizeString_en(hit_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate(hit_transliteration_val[l], use_prefix=False, type="original")] for l in range(len(hit_en_val))]
                train_pairs.append(hit_train_rev_pairs_original_transliterated_translate)
                test_pairs.append(hit_test_rev_pairs_original_transliterated_translate)
                val_pairs.append(hit_val_rev_pairs_original_transliterated_translate)
                token_pairs.append(hit_train_rev_pairs_original_transliterated_translate)
                token_pairs.append(hit_test_rev_pairs_original_transliterated_translate)
                token_pairs.append(hit_val_rev_pairs_original_transliterated_translate)
                if debug == True:
                    print(f"", {hit_train_rev_pairs_original_transliterated_translate[1][0]}, " -> ", {hit_train_rev_pairs_original_transliterated_translate[1][1]})
            if group:
                ###Translate from English to transliterated Hittite
                hit_train_rev_pairs_group_transliterated_translate = [[normalizeString_en(hit_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(hit_transliteration_train[l], use_prefix=False)] for l in range(len(hit_en_train))]
                hit_test_rev_pairs_group_transliterated_translate = [[normalizeString_en(hit_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(hit_transliteration_test[l], use_prefix=False)] for l in range(len(hit_en_test))]
                hit_val_rev_pairs_group_transliterated_translate = [[normalizeString_en(hit_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Hittite", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(hit_transliteration_val[l], use_prefix=False)] for l in range(len(hit_en_val))]
                train_pairs.append(hit_train_rev_pairs_group_transliterated_translate)
                test_pairs.append(hit_test_rev_pairs_group_transliterated_translate)
                val_pairs.append(hit_val_rev_pairs_group_transliterated_translate)
                if debug == True:
                    print(f"", {hit_train_rev_pairs_group_transliterated_translate[1][0]}, " -> ", {hit_train_rev_pairs_group_transliterated_translate[1][1]})
        if de:
            #############
            ###Hittite###
            #############
            # Read the file and split into lines
            if data=="line":
                hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_line.tr'))
                hit_de_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_line.de'))
                hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_line.tr'))
                hit_de_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_line.de'))
                hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_line.tr'))
                hit_de_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_line.de'))
            elif data=="document":
                hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_document.tr'))
                hit_de_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_document.de'))
                hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_document.tr'))
                hit_de_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_document.de'))
                hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_document.tr'))
                hit_de_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_document.de'))
            elif data=="both":
                hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train.tr'))
                hit_de_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train.de'))
                hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test.tr'))
                hit_de_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test.de'))
                hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid.tr'))
                hit_de_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid.de'))
            # Split every line into pairs and normalize
            if simple:
                ##Translate from simple transliterated Hittite to German
                hit_train_pairs_simple_transliterated_translate_de = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_train[l], use_prefix=True, task="Translate", type="simple", language="Hittite", modern="German", style=prompt_style), normalizeString_en(hit_de_train[l], use_prefix=False, modern="German")] for l in range(len(hit_transliteration_train))]
                hit_test_pairs_simple_transliterated_translate_de = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_test[l], use_prefix=True, task="Translate", type="simple", language="Hittite", modern="German", style=prompt_style), normalizeString_en(hit_de_test[l], use_prefix=False, modern="German")] for l in range(len(hit_transliteration_test))]
                hit_val_pairs_simple_transliterated_translate_de = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_val[l], use_prefix=True, task="Translate", type="simple", language="Hittite", modern="German", style=prompt_style), normalizeString_en(hit_de_val[l], use_prefix=False, modern="German")] for l in range(len(hit_transliteration_val))]
                train_pairs.append(hit_train_pairs_simple_transliterated_translate_de)
                test_pairs.append(hit_test_pairs_simple_transliterated_translate_de)
                val_pairs.append(hit_val_pairs_simple_transliterated_translate_de)
                if debug == True:
                    print(f"", {hit_train_pairs_simple_transliterated_translate_de[1][0]}, " -> ", {hit_train_pairs_simple_transliterated_translate_de[1][1]})
            if complex:
                ###Translate from original transliterated Hittite to German
                hit_train_pairs_original_transliterated_translate_de = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_train[l], use_prefix=True, task="Translate", type="original", language="Hittite", modern="German", style=prompt_style), normalizeString_en(hit_de_train[l], use_prefix=False, modern="German")] for l in range(len(hit_transliteration_train))]
                hit_test_pairs_original_transliterated_translate_de = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_test[l], use_prefix=True, task="Translate", type="original", language="Hittite", modern="German", style=prompt_style), normalizeString_en(hit_de_test[l], use_prefix=False, modern="German")] for l in range(len(hit_transliteration_test))]
                hit_val_pairs_original_transliterated_translate_de = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_val[l], use_prefix=True, task="Translate", type="original", language="Hittite", modern="German", style=prompt_style), normalizeString_en(hit_de_val[l], use_prefix=False, modern="German")] for l in range(len(hit_transliteration_val))]
                train_pairs.append(hit_train_pairs_original_transliterated_translate_de)
                test_pairs.append(hit_test_pairs_original_transliterated_translate_de)
                val_pairs.append(hit_val_pairs_original_transliterated_translate_de)
                if debug == True:
                    print(f"", {hit_train_pairs_original_transliterated_translate_de[1][0]}, " -> ", {hit_train_pairs_original_transliterated_translate_de[1][1]})
            if group:
                ###Translate from transliterated Hittite to German
                hit_train_pairs_group_transliterated_translate_de = [[normalizeString_cuneiform_transliterate_minimal(hit_transliteration_train[l], use_prefix=True, language="Hittite", modern="German", style=prompt_style), normalizeString_en(hit_de_train[l], use_prefix=False, modern="German")] for l in range(len(hit_transliteration_train))]
                hit_test_pairs_group_transliterated_translate_de = [[normalizeString_cuneiform_transliterate_minimal(hit_transliteration_test[l], use_prefix=True, language="Hittite", modern="German", style=prompt_style), normalizeString_en(hit_de_test[l], use_prefix=False, modern="German")] for l in range(len(hit_transliteration_test))]
                hit_val_pairs_group_transliterated_translate_de = [[normalizeString_cuneiform_transliterate_minimal(hit_transliteration_val[l], use_prefix=True, language="Hittite", modern="German", style=prompt_style), normalizeString_en(hit_de_val[l], use_prefix=False, modern="German")] for l in range(len(hit_transliteration_val))]
                train_pairs.append(hit_train_pairs_group_transliterated_translate_de)
                test_pairs.append(hit_test_pairs_group_transliterated_translate_de)
                val_pairs.append(hit_val_pairs_group_transliterated_translate_de)
                if debug == True:
                    print(f"", {hit_train_pairs_group_transliterated_translate_de[1][0]}, " -> ", {hit_train_pairs_group_transliterated_translate_de[1][1]})
            if rev:
                if simple:
                    ###Translate from German to simple transliterated Hittite
                    hit_train_rev_pairs_simple_transliterated_translate_de = [[normalizeString_en(hit_de_train[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Hittite", modern="German", style=prompt_style), normalizeString_cuneiform_transliterate(hit_transliteration_train[l], use_prefix=False, type="simple")] for l in range(len(hit_de_train))]
                    hit_test_rev_pairs_simple_transliterated_translate_de = [[normalizeString_en(hit_de_test[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Hittite", modern="German", style=prompt_style), normalizeString_cuneiform_transliterate(hit_transliteration_test[l], use_prefix=False, type="simple")] for l in range(len(hit_de_test))]
                    hit_val_rev_pairs_simple_transliterated_translate_de = [[normalizeString_en(hit_de_val[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Hittite", modern="German", style=prompt_style), normalizeString_cuneiform_transliterate(hit_transliteration_val[l], use_prefix=False, type="simple")] for l in range(len(hit_de_val))]
                    train_pairs.append(hit_train_rev_pairs_simple_transliterated_translate_de)
                    test_pairs.append(hit_test_rev_pairs_simple_transliterated_translate_de)
                    val_pairs.append(hit_val_rev_pairs_simple_transliterated_translate_de)
                    if debug == True:
                        print(f"", {hit_train_rev_pairs_simple_transliterated_translate_de[1][0]}, " -> ", {hit_train_rev_pairs_simple_transliterated_translate_de[1][1]})
                if complex:
                    ###Translate from English to complex transliterated Hittite
                    hit_train_rev_pairs_original_transliterated_translate_de = [[normalizeString_en(hit_de_train[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Hittite", modern="German", style=prompt_style), normalizeString_cuneiform_transliterate(hit_transliteration_train[l], use_prefix=False, type="original")] for l in range(len(hit_de_train))]
                    hit_test_rev_pairs_original_transliterated_translate_de = [[normalizeString_en(hit_de_test[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Hittite", modern="German", style=prompt_style), normalizeString_cuneiform_transliterate(hit_transliteration_test[l], use_prefix=False, type="original")] for l in range(len(hit_de_test))]
                    hit_val_rev_pairs_original_transliterated_translate_de = [[normalizeString_en(hit_de_val[l], use_prefix=True, task="Translate", target="transliteration", type="original", language="Hittite", modern="German", style=prompt_style), normalizeString_cuneiform_transliterate(hit_transliteration_val[l], use_prefix=False, type="original")] for l in range(len(hit_de_val))]
                    train_pairs.append(hit_train_rev_pairs_original_transliterated_translate_de)
                    test_pairs.append(hit_test_rev_pairs_original_transliterated_translate_de)
                    val_pairs.append(hit_val_rev_pairs_original_transliterated_translate_de)
                    token_pairs.append(hit_train_rev_pairs_original_transliterated_translate_de)
                    token_pairs.append(hit_test_rev_pairs_original_transliterated_translate_de)
                    token_pairs.append(hit_val_rev_pairs_original_transliterated_translate_de)
                    if debug == True:
                        print(f"", {hit_train_rev_pairs_original_transliterated_translate_de[1][0]}, " -> ", {hit_train_rev_pairs_original_transliterated_translate_de[1][1]})
                if group:
                    ###Translate from English to transliterated Hittite
                    hit_train_rev_pairs_group_transliterated_translate_de = [[normalizeString_en(hit_de_train[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Hittite", modern="German", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(hit_transliteration_train[l], use_prefix=False)] for l in range(len(hit_de_train))]
                    hit_test_rev_pairs_group_transliterated_translate_de = [[normalizeString_en(hit_de_test[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Hittite", modern="German", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(hit_transliteration_test[l], use_prefix=False)] for l in range(len(hit_de_test))]
                    hit_val_rev_pairs_group_transliterated_translate_de = [[normalizeString_en(hit_de_val[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Hittite", modern="German", style=prompt_style), normalizeString_cuneiform_transliterate_minimal(hit_transliteration_val[l], use_prefix=False)] for l in range(len(hit_de_val))]
                    train_pairs.append(hit_train_rev_pairs_group_transliterated_translate_de)
                    test_pairs.append(hit_test_rev_pairs_group_transliterated_translate_de)
                    val_pairs.append(hit_val_rev_pairs_group_transliterated_translate_de)
                    if debug == True:
                        print(f"", {hit_train_rev_pairs_group_transliterated_translate_de[1][0]}, " -> ", {hit_train_rev_pairs_group_transliterated_translate_de[1][1]})
    train_pairs = [pair for subset in train_pairs for pair in subset]
    test_pairs = [pair for subset in test_pairs for pair in subset]
    val_pairs = [pair for subset in val_pairs for pair in subset]
    token_pairs = [pair for subset in token_pairs for pair in subset]
    pairs = train_pairs + test_pairs + val_pairs
    #print("Examples:")
    #print(f"", {train_pairs_cuneiform_translate[1][0]}, " -> ", {train_pairs_cuneiform_translate[1][1]})
    #print(f"", {train_pairs_transliterate_original[1][0]}, " -> ", {train_pairs_transliterate_original[1][1]})
    #print(f"", {train_pairs_transliterate_group[1][0]}, " -> ", {train_pairs_transliterate_group[1][1]})
    #print(f"", {train_pairs_group_transliterated_translate[1][0]}, " -> ", {train_pairs_group_transliterated_translate[1][1]})
    print(f"Total pairs imported: {len(pairs)}")
    train_pairs = trim_pairs(train_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    if debug == True:
        print("training set trimmed")
    val_pairs = trim_pairs(val_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    if debug == True:
        print("val set trimmed")
    test_pairs = trim_pairs(test_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    if debug == True:
        print("test set trimmed")
    pairs = train_pairs + test_pairs + val_pairs
    print(f"Total pairs filtered: {len(pairs)}")
    max_length_pair_0 = max(pairs, key=lambda pair: len(pair[0].split()))
    max_length_pair_1 = max(pairs, key=lambda pair: len(pair[1].split()))
    print("Largest number of words in pair[0]:")
    print(f"Word Count: {len(max_length_pair_0[0].split())}, Content: {max_length_pair_0[0]}")
    print("Largest number of words in pair[1]:")
    print(f"Word Count: {len(max_length_pair_1[1].split())}, Content: {max_length_pair_1[1]}")
    mean_length_pair_0 = sum(len(pair[0].split()) for pair in pairs) / len(pairs)
    mean_length_pair_1 = sum(len(pair[1].split()) for pair in pairs) / len(pairs)
    print(f"Mean number of tokens in source langauge: {mean_length_pair_0:.2f}")
    print(f"Mean number of tokens in target language: {mean_length_pair_1:.2f}")
    #train_pairs = collapse_spaces(train_pairs)
    #test_pairs = collapse_spaces(test_pairs)
    #val_pairs = collapse_spaces(val_pairs)
    #pairs = collapse_spaces(pairs)
    return train_pairs, val_pairs, test_pairs, pairs, token_pairs


def readLangsPreTrainT5(user_directory="/", max_length_akk=5000, max_length_en=5000, max_length_threshold=100, min_length_threshold=50, debug=False, cuneiform=True, akk=True, sux=True, elx=True, gmy=True, hit=True, complex=True, group=False, simple=False, rev=True, de=False, ld=True, data="both", use_oracc=True, use_cdli=True, use_oare=True):
    print("Reading lines...")
    ###Create list
    train_pairs = []
    token_pairs = []
    if akk:
        ##############
        ###Akkadian###
        ##############
        # Read the file and split into lines
        # Initialize empty lists for accumulating data from different sources
        # Note: PreTrainT5 only uses train data for unsupervised pretraining
        if data=="line":
            akk_cuneiform_train = []
            akk_transliteration_train = []
            akk_en_train = []
            # ORACC data (akk_* files)
            if use_oracc:
                akk_oracc_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_line.cu'))
                akk_oracc_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_line.tr'))
                akk_oracc_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_line.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oracc_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_oracc_transliteration_train
                akk_en_train = akk_en_train + akk_oracc_en_train
                # ORACC unsupervised data (akk_u_* files)
                akk_cuneiform_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_u_train_line.cu'))
                akk_cuneiform_train = akk_cuneiform_train + akk_cuneiform_u_train
                akk_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_u_train_line.tr'))
                akk_transliteration_train = akk_transliteration_train + akk_transliteration_u_train
            # CDLI data (akk_cdli_* files)
            if use_cdli:
                akk_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_line.cu'))
                akk_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_line.tr'))
                akk_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_line.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_cdli_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_cdli_transliteration_train
                akk_en_train = akk_en_train + akk_cdli_en_train
            # OARE data (akk_oare_* files) - has cuneiform
            if use_oare:
                akk_oare_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_line.cu'))
                akk_oare_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_line.tr'))
                akk_oare_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_line.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oare_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_oare_transliteration_train
                akk_en_train = akk_en_train + akk_oare_en_train
                # OARE unsupervised data
                akk_oare_cuneiform_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_u_train.cu'))
                akk_oare_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_u_train.tr'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oare_cuneiform_u_train
                akk_transliteration_train = akk_transliteration_train + akk_oare_transliteration_u_train
        elif data=="document":
            akk_cuneiform_train = []
            akk_transliteration_train = []
            akk_en_train = []
            # ORACC data (akk_* files)
            if use_oracc:
                akk_oracc_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_document.cu'))
                akk_oracc_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_document.tr'))
                akk_oracc_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train_document.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oracc_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_oracc_transliteration_train
                akk_en_train = akk_en_train + akk_oracc_en_train
                # ORACC unsupervised data (akk_u_* files)
                akk_cuneiform_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_u_train_document.cu'))
                akk_cuneiform_train = akk_cuneiform_train + akk_cuneiform_u_train
                akk_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_u_train_document.tr'))
                akk_transliteration_train = akk_transliteration_train + akk_transliteration_u_train
            # CDLI data (akk_cdli_* files)
            if use_cdli:
                akk_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_document.cu'))
                akk_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_document.tr'))
                akk_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train_document.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_cdli_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_cdli_transliteration_train
                akk_en_train = akk_en_train + akk_cdli_en_train
            # OARE data (akk_oare_* files) - has cuneiform
            if use_oare:
                akk_oare_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_document.cu'))
                akk_oare_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_document.tr'))
                akk_oare_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train_document.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oare_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_oare_transliteration_train
                akk_en_train = akk_en_train + akk_oare_en_train
                # OARE unsupervised data
                akk_oare_cuneiform_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_u_train.cu'))
                akk_oare_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_u_train.tr'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oare_cuneiform_u_train
                akk_transliteration_train = akk_transliteration_train + akk_oare_transliteration_u_train
        elif data=="both":
            akk_cuneiform_train = []
            akk_transliteration_train = []
            akk_en_train = []
            # ORACC data (akk_* files)
            if use_oracc:
                akk_oracc_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train.cu'))
                akk_oracc_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train.tr'))
                akk_oracc_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_train.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oracc_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_oracc_transliteration_train
                akk_en_train = akk_en_train + akk_oracc_en_train
                # ORACC unsupervised data (akk_u_* files)
                akk_cuneiform_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_u_train.cu'))
                akk_cuneiform_train = akk_cuneiform_train + akk_cuneiform_u_train
                akk_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_u_train.tr'))
                akk_transliteration_train = akk_transliteration_train + akk_transliteration_u_train
            # CDLI data (akk_cdli_* files)
            if use_cdli:
                akk_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train.cu'))
                akk_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train.tr'))
                akk_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_cdli_train.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_cdli_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_cdli_transliteration_train
                akk_en_train = akk_en_train + akk_cdli_en_train
            # OARE data (akk_oare_* files) - has cuneiform
            if use_oare:
                akk_oare_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train.cu'))
                akk_oare_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train.tr'))
                akk_oare_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_train.en'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oare_cuneiform_train
                akk_transliteration_train = akk_transliteration_train + akk_oare_transliteration_train
                akk_en_train = akk_en_train + akk_oare_en_train
                # OARE unsupervised data
                akk_oare_cuneiform_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_u_train.cu'))
                akk_oare_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'akk_oare_u_train.tr'))
                akk_cuneiform_train = akk_cuneiform_train + akk_oare_cuneiform_u_train
                akk_transliteration_train = akk_transliteration_train + akk_oare_transliteration_u_train
        # Split every line into pairs and normalize
        if cuneiform:
            ###Translate from Akkadian cuneiform to English
            akk_train_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False, task="Translate", language="Akkadian")] for l in range(len(akk_cuneiform_train))]
            train_pairs.append(akk_train_pairs_cuneiform_translate)
            token_pairs.append(akk_train_pairs_cuneiform_translate)
            if debug == True:
                print(f"", {akk_train_pairs_cuneiform_translate[1][0]})
        if simple:
            ###Translate from simple transliterated Akkadian to English
            akk_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_train[l], use_prefix=False, task="Translate", type="simple", language="Akkadian")] for l in range(len(akk_transliteration_train))]
            train_pairs.append(akk_train_pairs_simple_transliterated_translate)
            token_pairs.append(akk_train_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {akk_train_pairs_simple_transliterated_translate[1][0]})
        if complex:
            ###Translate from original transliterated Akkadian to English
            akk_train_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transliteration_train[l], use_prefix=False, task="Translate", type="original", language="Akkadian")] for l in range(len(akk_transliteration_train))]
            train_pairs.append(akk_train_pairs_original_transliterated_translate)
            token_pairs.append(akk_train_pairs_original_transliterated_translate)
            if debug == True:
                print(f"", {akk_train_pairs_original_transliterated_translate[1][0]})
        if group:
            ###Translate from transliterated Akkadian to English
            akk_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transliteration_train[l], use_prefix=False, language="Akkadian")] for l in range(len(akk_transliteration_train))]
            train_pairs.append(akk_train_pairs_group_transliterated_translate)
            if debug == True:
                print(f"", {akk_train_pairs_group_transliterated_translate[1][0]})
        if rev:
            ###Translate from English
            akk_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=False, task="Translate", target="transliteration", type="simple", language="Akkadian")] for l in range(len(akk_en_train))]
            train_pairs.append(akk_train_rev_pairs_simple_transliterated_translate)
            token_pairs.append(akk_train_rev_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {akk_train_rev_pairs_simple_transliterated_translate[1][0]})
    if sux:
        ##############
        ###Sumerian###
        ##############
        if data=="line":
            sux_ox_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.cu'))
            sux_ox_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.tr'))
            sux_ox_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.en'))
            sux_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_line.cu'))
            sux_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_line.tr'))
            sux_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_line.en'))
            sux_cuneiform_train = sux_ox_cuneiform_train + sux_cdli_cuneiform_train
            sux_transliteration_train = sux_ox_transliteration_train + sux_cdli_transliteration_train
            sux_en_train = sux_ox_en_train + sux_cdli_en_train
            sux_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_line.cu'))
            sux_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_line.tr'))
            sux_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_line.en'))
            sux_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_line.cu'))
            sux_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_line.tr'))
            sux_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_line.en'))
            sux_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_u_train_line.tr'))
            sux_transliteration_train = sux_transliteration_train + sux_transliteration_u_train
        elif data=="document":
            sux_ox_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.cu'))
            sux_ox_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.tr'))
            sux_ox_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.en'))
            sux_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_document.cu'))
            sux_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_document.tr'))
            sux_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train_document.en'))
            sux_cuneiform_train = sux_ox_cuneiform_train + sux_cdli_cuneiform_train
            sux_transliteration_train = sux_ox_transliteration_train + sux_cdli_transliteration_train
            sux_en_train = sux_ox_en_train + sux_cdli_en_train
            sux_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_document.cu'))
            sux_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_document.tr'))
            sux_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test_document.en'))
            sux_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_document.cu'))
            sux_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_document.tr'))
            sux_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid_document.en'))
            sux_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_u_train_document.tr'))
            sux_transliteration_train = sux_transliteration_train + sux_transliteration_u_train
        elif data=="both":
            sux_ox_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.cu'))
            sux_ox_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.tr'))
            sux_ox_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_ox_train.en'))
            sux_cdli_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'sux_cdli_train.cu'))
            sux_cdli_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train.tr'))
            sux_cdli_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_train.en'))
            sux_cuneiform_train = sux_ox_cuneiform_train + sux_cdli_cuneiform_train
            sux_transliteration_train = sux_ox_transliteration_train + sux_cdli_transliteration_train
            sux_en_train = sux_ox_en_train + sux_cdli_en_train
            sux_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test.cu'))
            sux_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test.tr'))
            sux_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_test.en'))
            sux_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid.cu'))
            sux_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid.tr'))
            sux_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_cdli_valid.en'))
            sux_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'sux_u_train.tr'))
            sux_transliteration_train = sux_transliteration_train + sux_transliteration_u_train
        # Split every line into pairs and normalize
        if cuneiform:
            ###Translate from Sumerian cuneiform to English
            sux_train_pairs_cuneiform_translate = [[normalizeString_cuneiform(sux_cuneiform_train[l], use_prefix=False, task="Translate", language="Sumerian")] for l in range(len(sux_cuneiform_train))]
            train_pairs.append(sux_train_pairs_cuneiform_translate)
            token_pairs.append(sux_train_pairs_cuneiform_translate)
            if debug == True:
                print(f"", {sux_train_pairs_cuneiform_translate[1][0]})
        if simple:
            ##Translate from simple transliterated Sumerian to English
            sux_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transliteration_train[l], use_prefix=False, task="Translate", type="simple", language="Sumerian")] for l in range(len(sux_transliteration_train))]
            train_pairs.append(sux_train_pairs_simple_transliterated_translate)
            token_pairs.append(sux_train_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {sux_train_pairs_simple_transliterated_translate[1][0]})
        if complex:
            ###Translate from original transliterated Sumerian to English
            sux_train_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transliteration_train[l], use_prefix=False, task="Translate", type="original", language="Sumerian")] for l in range(len(sux_transliteration_train))]
            train_pairs.append(sux_train_pairs_original_transliterated_translate)
            token_pairs.append(sux_train_pairs_original_transliterated_translate)
            if debug == True:
                print(f"", {sux_train_pairs_original_transliterated_translate[1][0]})
        if group:
            ###Translate from transliterated Sumerian to English
            sux_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(sux_transliteration_train[l], use_prefix=False, language="Sumerian")] for l in range(len(sux_transliteration_train))]
            train_pairs.append(sux_train_pairs_group_transliterated_translate)
            if debug == True:
                print(f"", {sux_train_pairs_group_transliterated_translate[1][0]})
        if rev:
            ###Translate from English to simple transliterated Sumerian
            sux_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(sux_en_train[l], use_prefix=False, task="Translate", target="transliteration", type="simple", language="Sumerian")] for l in range(len(sux_en_train))]
            train_pairs.append(sux_train_rev_pairs_simple_transliterated_translate)
            token_pairs.append(sux_train_rev_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {sux_train_rev_pairs_simple_transliterated_translate[1][0]})
    if elx:
        ###############
        ###Elamite###
        ###############
        # Read the file and split into lines
        if data=="line":
            elx_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_line.tr'))
            elx_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_line.en'))
            elx_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_u_train_line.tr'))
            elx_transliteration_train = elx_transliteration_train + elx_transliteration_u_train
        elif data=="document":
            elx_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_document.tr'))
            elx_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train_document.en'))
            elx_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_u_train_document.tr'))
            elx_transliteration_train = elx_transliteration_train + elx_transliteration_u_train
        elif data=="both":
            elx_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train.tr'))
            elx_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_cdli_train.en'))
            elx_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'elx_u_train.tr'))
            elx_transliteration_train = elx_transliteration_train + elx_transliteration_u_train
        # Split every line into pairs and normalize
        if simple:
            ##Translate from simple transliterated Elamite to English
            elx_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(elx_transliteration_train[l], use_prefix=False, task="Translate", type="simple", language="Elamite")] for l in range(len(elx_transliteration_train))]
            train_pairs.append(elx_train_pairs_simple_transliterated_translate)
            token_pairs.append(elx_train_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {elx_train_pairs_simple_transliterated_translate[1][0]})
        if complex:
            ###Translate from original transliterated Elamite to English
            elx_train_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(elx_transliteration_train[l], use_prefix=False, task="Translate", type="original", language="Elamite")] for l in range(len(elx_transliteration_train))]
            train_pairs.append(elx_train_pairs_original_transliterated_translate)
            token_pairs.append(elx_train_pairs_original_transliterated_translate)
            if debug == True:
                print(f"", {elx_train_pairs_original_transliterated_translate[1][0]})
        if group:
            ###Translate from transliterated Elamite to English
            elx_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(elx_transliteration_train[l], use_prefix=False, language="Elamite")] for l in range(len(elx_transliteration_train))]
            train_pairs.append(elx_train_pairs_group_transliterated_translate)
            if debug == True:
               print(f"", {elx_train_pairs_group_transliterated_translate[1][0]})
        if rev:
            ###Translate from English to simple transliterated Elamite
            elx_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(elx_en_train[l], use_prefix=False, task="Translate", target="transliteration", type="simple", language="Elamite")] for l in range(len(elx_en_train))]
            train_pairs.append(elx_train_rev_pairs_simple_transliterated_translate)
            token_pairs.append(elx_train_rev_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {elx_train_rev_pairs_simple_transliterated_translate[1][0]})
    if gmy:
        ##############
        ###Linear B###
        ##############
        # Read the file and split into lines
        gmy_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_train.cu'))
        gmy_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_train.tr'))
        gmy_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'gmy_train.en'))
        # Split every line into pairs and normalize
        if cuneiform:
            ###Translate from Linear B cuneiform to English
            gmy_train_pairs_cuneiform_translate = [[normalizeString_cuneiform(gmy_cuneiform_train[l], use_prefix=False, task="Translate", language="Linear B")] for l in range(len(gmy_cuneiform_train))]
            train_pairs.append(gmy_train_pairs_cuneiform_translate)
            token_pairs.append(gmy_train_pairs_cuneiform_translate)
            if debug == True:
                print(f"", {gmy_train_pairs_cuneiform_translate[1][0]})
        if simple:
            ###Translate from simple transliterated Linear B to English
            gmy_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(gmy_transliteration_train[l], use_prefix=False, task="Translate", type="simple", language="Linear B")] for l in range(len(gmy_transliteration_train))]
            train_pairs.append(gmy_train_pairs_simple_transliterated_translate)
            token_pairs.append(gmy_train_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {gmy_train_pairs_simple_transliterated_translate[1][0]})
        if complex:
            ###Translate from original transliterated Linear B to English
            gmy_train_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(gmy_transliteration_train[l], use_prefix=False, task="Translate", type="original", language="Linear B")] for l in range(len(gmy_transliteration_train))]
            train_pairs.append(gmy_train_pairs_original_transliterated_translate)
            token_pairs.append(gmy_train_pairs_original_transliterated_translate)
            if debug == True:
                print(f"", {gmy_train_pairs_original_transliterated_translate[1][0]})
        if group:
            ###Translate from transliterated Linear B to English
            gmy_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(gmy_transliteration_train[l], use_prefix=False, language="Linear B")] for l in range(len(gmy_transliteration_train))]
            train_pairs.append(gmy_train_pairs_group_transliterated_translate)
            if debug == True:
                print(f"", {gmy_train_pairs_group_transliterated_translate[1][0]})
        if rev:
            ###Translate from English to simple transliterated Linear B
            gmy_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(gmy_en_train[l], use_prefix=False, task="Translate", target="transliteration", type="simple", language="Linear B")] for l in range(len(gmy_en_train))]
            train_pairs.append(gmy_train_rev_pairs_simple_transliterated_translate)
            token_pairs.append(gmy_train_rev_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {gmy_train_rev_pairs_simple_transliterated_translate[1][0]})
    if hit:
        ##############
        ###Hittite###
        ##############
        # Read the file and split into lines
        if data=="line":
            hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_line.tr'))
            hit_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_line.en'))
            hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_line.tr'))
            hit_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_line.en'))
            hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_line.tr'))
            hit_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_line.en'))
            hit_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_u_train_line.tr'))
            hit_transliteration_train = hit_transliteration_train + hit_transliteration_u_train
        elif data=="document":
            hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_document.tr'))
            hit_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_document.en'))
            hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_document.tr'))
            hit_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_document.en'))
            hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_document.tr'))
            hit_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_document.en'))
            hit_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_u_train_document.tr'))
            hit_transliteration_train = hit_transliteration_train + hit_transliteration_u_train
        elif data=="both":
            hit_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train.tr'))
            hit_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train.en'))
            hit_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test.tr'))
            hit_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test.en'))
            hit_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid.tr'))
            hit_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid.en'))
            hit_transliteration_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_u_train.tr'))
            hit_transliteration_train = hit_transliteration_train + hit_transliteration_u_train
        if ld:
            hit_ld_transliteration_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_train.tr'))
            hit_ld_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_train.en'))
            hit_ld_transliteration_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_test.tr'))
            hit_ld_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_test.en'))
            hit_ld_transliteration_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_valid.tr'))
            hit_ld_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data_line', 'hit_valid.en'))
            # Add ALL data_line data to training (including ld test/val for max coverage)
            # Val/test remain main-data-only to avoid evaluating on data_line examples
            hit_transliteration_train = hit_transliteration_train + hit_ld_transliteration_train + hit_ld_transliteration_test + hit_ld_transliteration_val
            hit_en_train = hit_en_train + hit_ld_en_train + hit_ld_en_test + hit_ld_en_val
        if de:
            if data=="line":
                hit_de_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_line.de'))
                hit_de_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_line.de'))
                hit_de_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_line.de'))
            elif data=="document":
                hit_de_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train_document.de'))
                hit_de_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test_document.de'))
                hit_de_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid_document.de'))
            elif data=="both":
                hit_de_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_train.de'))
                hit_de_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_test.de'))
                hit_de_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'hit_valid.de'))
        # Split every line into pairs and normalize
        if simple:
            ##Translate from simple transliterated Hittite to English
            hit_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_train[l], use_prefix=False, task="Translate", type="simple", language="Hittite")] for l in range(len(hit_transliteration_train))]
            train_pairs.append(hit_train_pairs_simple_transliterated_translate)
            token_pairs.append(hit_train_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {hit_train_pairs_simple_transliterated_translate[1][0]})
        if complex:
            ###Translate from original transliterated Hittite to English
            hit_train_pairs_original_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(hit_transliteration_train[l], use_prefix=False, task="Translate", type="original", language="Hittite")] for l in range(len(hit_transliteration_train))]
            train_pairs.append(hit_train_pairs_original_transliterated_translate)
            token_pairs.append(hit_train_pairs_original_transliterated_translate)
            if debug == True:
                print(f"", {hit_train_pairs_original_transliterated_translate[1][0]})
        if group:
            ###Translate from transliterated Hittite to English
            hit_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(hit_transliteration_train[l], use_prefix=False, language="Hittite")] for l in range(len(hit_transliteration_train))]
            train_pairs.append(hit_train_pairs_group_transliterated_translate)
            if debug == True:
                print(f"", {hit_train_pairs_group_transliterated_translate[1][0]})
        if rev:
            ###Translate from English to simple transliterated Hittite
            hit_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(hit_en_train[l], use_prefix=False, task="Translate", target="transliteration", type="simple", language="Hittite")] for l in range(len(hit_en_train))]
            train_pairs.append(hit_train_rev_pairs_simple_transliterated_translate)
            token_pairs.append(hit_train_rev_pairs_simple_transliterated_translate)
            if debug == True:
                print(f"", {hit_train_rev_pairs_simple_transliterated_translate[1][0]})
            if de:
                ###Translate from English to simple transliterated Hittite
                hit_train_rev_pairs_simple_transliterated_translate_de = [[normalizeString_en(hit_de_train[l], use_prefix=False, task="Translate", target="transliteration", type="simple", language="Hittite", modern="German")] for l in range(len(hit_de_train))]
                train_pairs.append(hit_train_rev_pairs_simple_transliterated_translate_de)
                token_pairs.append(hit_train_rev_pairs_simple_transliterated_translate_de)
                if debug == True:
                    print(f"", {hit_train_rev_pairs_simple_transliterated_translate_de[1][0]})
    train_pairs = [pair for subset in train_pairs for pair in subset]
    token_pairs = [pair for subset in token_pairs for pair in subset]
    pairs = train_pairs
    #print("Examples:")
    #print(f"", {train_pairs_cuneiform_translate[1][0]}, " -> ", {train_pairs_cuneiform_translate[1][1]})
    #print(f"", {train_pairs_transliterate_original[1][0]}, " -> ", {train_pairs_transliterate_original[1][1]})
    #print(f"", {train_pairs_transliterate_group[1][0]}, " -> ", {train_pairs_transliterate_group[1][1]})
    #print(f"", {train_pairs_group_transliterated_translate[1][0]}, " -> ", {train_pairs_group_transliterated_translate[1][1]})
    print(f"Total pairs imported: {len(pairs)}")
    train_pairs = trim_singles(train_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    if debug == True:
        print("training set trimmed")
    pairs = train_pairs
    print(f"Total pairs filtered: {len(pairs)}")
    max_length_pair_0 = max(pairs, key=lambda pair: len(pair[0].split()))
    print("Largest number of words in pair:")
    print(f"Word Count: {len(max_length_pair_0.split())}, Content: {max_length_pair_0}")
    mean_length_pair_0 = sum(len(pair[0].split()) for pair in pairs) / len(pairs)
    print(f"Mean number of tokens in source langauge: {mean_length_pair_0:.2f}")
    #train_pairs = collapse_spaces(train_pairs)
    #pairs = collapse_spaces(pairs)
    return train_pairs, pairs, token_pairs

def write_t5_pairs_to_csv(pairs, output_csv_path):
    """
    Write a list of (source, target) pairs to a CSV with columns: source, target.
    """
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["source", "target"])
        # Write each pair as a new row
        for src, tgt in pairs:
            writer.writerow([src, tgt])

def write_t5_singles_to_csv(pairs, output_csv_path):
    """
    Write a list of (source, target) pairs to a CSV with columns: source, target.
    """
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["source", "target"])
        # Write each pair as a new row
        for src in pairs:
            writer.writerow([src])
            
import csv

def load_t5_pairs_from_csv(csv_path):
    """Return a list of (source, target) tuples from a CSV with columns: source,target."""
    pairs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("source", "") or ""
            tgt = row.get("target", "") or ""
            pairs.append((src, tgt))
    return pairs
    
def pick_generation_len(pairs, pct=95):
    tgt_bytes = [len(t.encode("utf-8")) for _, t in pairs]
    return int(np.percentile(tgt_bytes, pct))
    
def pick_caps_from_csv(csv_path, pct_src=98, pct_tgt=98):
    src_bytes, tgt_bytes = [], []
    with open(csv_path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            s, t = row["source"] or "", row["target"] or ""
            src_bytes.append(len(s.encode("utf-8")))
            tgt_bytes.append(len(t.encode("utf-8")))
    def round_up_8(x): return int((x + 7) // 8 * 8)
    src_cap = round_up_8(np.percentile(src_bytes, pct_src))
    tgt_cap = round_up_8(np.percentile(tgt_bytes, pct_tgt))
    return int(src_cap), int(tgt_cap)

def write_nllb_pairs_to_csv(pairs, filepath):
    """
    Write a list of parallel‐text examples to CSV.
    Args:
      pairs (list of dict): each dict must have the same keys, e.g.
        {
          'source_text': '…',
          'target_text': '…',
          'src_lang': '…',
          'tgt_lang': '…',
        }
      filepath (str): path to the .csv file to write.
    The output CSV will have one header row with the dict’s keys,
    and one row per example.
    """
    # If pairs is empty, just write an empty file with headers
    if not pairs:
        raise ValueError("No examples to write")
    # Use the keys of the first dict as your column order
    fieldnames = list(pairs[0].keys())
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for ex in pairs:
            # ensure no missing keys
            row = {k: ex.get(k, "") for k in fieldnames}
            writer.writerow(row)

def filter_nllb_pairs_by_lang(pairs, src_lang=None, tgt_lang=None, flip=False):
    """
    Filter a list of parallel‐text dicts by source and/or target language code,
    optionally swapping source<->target in the results.
    Args:
      pairs (list of dict): each dict must have keys
        - "source_text": str
        - "target_text": str
        - "src_lang":     e.g. "akk_Cune"
        - "tgt_lang":     e.g. "eng_Latn"
      src_lang (str or None): if set, only keep pairs with this source code
      tgt_lang (str or None): if set, only keep pairs with this target code
      flip (bool): if True, swap source<->target in each returned example
    Returns:
      list of dict: filtered (and possibly flipped) examples
    """
    # 1) Filter
    filtered = []
    for ex in pairs:
        if src_lang is not None and ex.get("src_lang") != src_lang:
            continue
        if tgt_lang is not None and ex.get("tgt_lang") != tgt_lang:
            continue
        filtered.append(ex)
    # 2) Optionally flip
    if flip:
        flipped = []
        for ex in filtered:
            new_ex = ex.copy()
            # swap the language codes
            new_ex["src_lang"], new_ex["tgt_lang"] = ex["tgt_lang"], ex["src_lang"]
            # swap the texts
            new_ex["source_text"], new_ex["target_text"] = ex["target_text"], ex["source_text"]
            flipped.append(new_ex)
            filtered = flipped
    return filtered

def exclude_nllb_pairs_by_lang(pairs, src_lang=None, tgt_lang=None):
    """
    Remove from `pairs` any examples whose src_lang or tgt_lang
    matches the given code(s).

    Args:
      pairs (list of dict): each dict must have keys
        - "src_lang": source language code
        - "tgt_lang": target language code
      src_lang (str or None): if set, drop any example with this src_lang
      tgt_lang (str or None): if set, drop any example with this tgt_lang

    Returns:
      list of dict: all examples *not* matching the given codes
    """
    result = []
    for ex in pairs:
        if src_lang is not None and ex.get("src_lang") == src_lang:
            continue
        if tgt_lang is not None and ex.get("tgt_lang") == tgt_lang:
            continue
        result.append(ex)
    return result

def dump_source_texts(pairs, column, filepath):
    """
    Write all 'source_text' strings from a list of parallel examples
    to a plaintext file, one sentence per line.

    Args:
      pairs (list of dict): each dict must have a 'source_text' key
      filepath (str): path to the output .txt file

    Example:
      dump_source_texts(train_pairs, "all_sources.txt")
    """
    # ensure output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for ex in pairs:
            src = ex.get(column, "").strip()
            if src:
                f.write(src + "\n")

def read_source_texts(filepath):
    """
    Returns a list of non‐empty lines from the given text file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        # strip out blank lines
        lines = [line.rstrip("\n") for line in f if line.strip()]
    return lines

def remove_grouped_transliteration_src(pairs):
    """
    Remove all examples where the source language code indicates a transliteration
    (i.e. ends with '_Latn').

    Args:
      pairs (list of dict): each dict must have a 'src_lang' key

    Returns:
      list of dict: only those examples whose src_lang does not end with '_Latn'
    """
    return [
        ex for ex in pairs
        if not ex.get("src_lang", "").endswith("_Latn")
    ]

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Read a CSV (no header) where each line is:
      Prompt: source_text,reference_text
    and return a DataFrame with columns: prompt, source, reference.
    """
    # Read as single-column
    df = pd.read_csv(csv_path, header=None, names=["line"], dtype=str)
    # Split at the first colon
    prompt_rest = df["line"].str.split(":", n=1, expand=True)
    df["prompt"] = prompt_rest[0].str.strip() + ":"
    df["rest"]   = prompt_rest[1]
    # Then split rest on the last comma
    src_ref = df["rest"].str.rsplit(",", n=1, expand=True)
    df["source"]    = src_ref[0].str.strip()
    df["reference"] = src_ref[1].str.strip()
    return df[["prompt", "source", "reference"]]

from typing import List, Tuple, Any

def extract_erroneous_tokens(
    token_error_pairs: List[Tuple[str, Any]]
) -> List[str]:
    """
    Given a list of (prompt, target) pairs, pull out every token
    appearing after the first ':' in the prompt, drop the '*' markers,
    and return a sorted list of the unique tokens.

    Example input:
      [
        ("Translate simple ...: pi tiq d nin men na sa a na be lut kur kur * an ti al sid *",
         {...}),
        ("Translate complex ...: pi tiq d nin men na sa2 a na be lut kur kur * an ti al sid *",
         {...})
      ]
    Output:
      ['a','al','an','be','d','kur','lut','men','na','nin','pi','sid','ti','tiq','sa','sa2']
    """
    unique = set()
    for prompt, _ in token_error_pairs:
        # split off everything after the first colon
        if ":" not in prompt:
            continue
        after = prompt.split(":", 1)[1].strip()
        # split on whitespace and filter out '*'
        for syll in after.split():
            if syll != "*":
                unique.add(syll)
    # return a reproducible order (alphabetical)
    return sorted(unique)

def remove_prompt(x):
    if ':' in x:
        return x.split(':', 1)[1].strip()
    return x.strip()

def report_mean_lengths(pairs):
    """
    Given a list of (source_text, target_text) tuples, prints:
      - mean source char length
      - mean target char length
      - mean source byte length (ByT5 tokens)
      - mean target byte length (ByT5 tokens)
    """
    import sys
    
    n = len(pairs)
    if n == 0:
        print("No examples to analyze.", file=sys.stderr)
        return

    total_src_chars = 0
    total_tgt_chars = 0
    total_src_bytes = 0
    total_tgt_bytes = 0

    for src, tgt in pairs:
        # 1) character counts
        total_src_chars += len(src)
        total_tgt_chars += len(tgt)

        # 2) byte counts (UTF-8)
        total_src_bytes += len(src.encode("utf-8"))
        total_tgt_bytes += len(tgt.encode("utf-8"))

    print(f"Examples: {n}")
    print(f"Mean source length (chars): {total_src_chars / n:.1f}")
    print(f"Mean target length (chars): {total_tgt_chars / n:.1f}")
    print(f"Mean source length (bytes): {total_src_bytes / n:.1f}")
    print(f"Mean target length (bytes): {total_tgt_bytes / n:.1f}")

def report_length_distribution(pairs):
    """
    Given a list of (source_text, target_text) tuples, prints:
      • count of examples
      • mean, 75th-percentile, and max for:
        – source length (chars)
        – target length (chars)
        – source length (bytes)
        – target length (bytes)
    """
    import sys
    import numpy as np

    n = len(pairs)
    if n == 0:
        print("No examples to analyze.", file=sys.stderr)
        return

    # build four lists of lengths
    src_chars = [len(src) for src, _ in pairs]
    tgt_chars = [len(tgt) for _, tgt in pairs]
    src_bytes = [len(src.encode("utf-8")) for src, _ in pairs]
    tgt_bytes = [len(tgt.encode("utf-8")) for _, tgt in pairs]

    # helper to compute metrics
    def metrics(arr):
        return {
            "mean":    np.mean(arr),
            "q3":      np.percentile(arr, 75),
            "max":     np.max(arr),
        }

    m_src_c = metrics(src_chars)
    m_tgt_c = metrics(tgt_chars)
    m_src_b = metrics(src_bytes)
    m_tgt_b = metrics(tgt_bytes)

    print(f"Examples: {n}\n")

    print("SOURCE TEXT:")
    print(f"  – Mean length (chars): {m_src_c['mean']:.1f}")
    print(f"  – 75th percentile (chars): {m_src_c['q3']:.1f}")
    print(f"  – Max length (chars): {m_src_c['max']}")
    print(f"  – Mean length (bytes): {m_src_b['mean']:.1f}")
    print(f"  – 75th percentile (bytes): {m_src_b['q3']:.1f}")
    print(f"  – Max length (bytes): {m_src_b['max']}")

    print("\nTARGET TEXT:")
    print(f"  – Mean length (chars): {m_tgt_c['mean']:.1f}")
    print(f"  – 75th percentile (chars): {m_tgt_c['q3']:.1f}")
    print(f"  – Max length (chars): {m_tgt_c['max']}")
    print(f"  – Mean length (bytes): {m_tgt_b['mean']:.1f}")
    print(f"  – 75th percentile (bytes): {m_tgt_b['q3']:.1f}")
    print(f"  – Max length (bytes): {m_tgt_b['max']}")

def normalize_and_segment(batch):
    # Unicode NFC
    src = [unicodedata.normalize("NFC", s) for s in batch["source_text"]]
    tgt = [unicodedata.normalize("NFC", t) for t in batch["target_text"]]
    # Insert spaces between each grapheme cluster
    batch["source_text"] = [" ".join(regex.findall(r"\X", s)) for s in src]
    batch["target_text"] = [" ".join(regex.findall(r"\X", t)) for t in tgt]
    return batch

def normalize_and_segment_pretrain(batch):
    # Unicode NFC
    src = [unicodedata.normalize("NFC", s) for s in batch["source_text"]]
    # Insert spaces between each grapheme cluster
    batch["source_text"] = [" ".join(regex.findall(r"\X", s)) for s in src]
    return batch

def split_graphemes(text):
    return " ".join(regex.findall(r"\X", text))

def segment_batch(batch):
    batch["source_text"] = [split_graphemes(s) for s in batch["source_text"]]
    batch["target_text"] = [split_graphemes(t) for t in batch["target_text"]]
    return batch

def is_non_latin(s, threshold=0.5):
    # if more than `threshold` fraction of chars are outside ASCII, treat as cuneiform/Linear B
    non_ascii = sum(1 for ch in s if ord(ch) > 127)
    return non_ascii / max(len(s), 1) >= threshold

def read_cdli_corpus(filepath):
    transliteration = read_and_process_file(filepath)
    transliterated_pairs = [[normalizeString_cuneiform_transliterate_translate(transliteration[l], use_prefix=True, task="Translate", type="original", language="Akkadian")] for l in range(len(transliteration))]
    return(transliterated_pairs)
