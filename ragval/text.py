"""Deterministic, dependency-free text processing.

Everything in this module is pure and side-effect free, which means the whole
harness is reproducible: same input bytes in, same numbers out, on any machine
and any Python 3.9+ interpreter.

Contents
--------
* ``normalize`` / ``tokenize``   - unicode folding, casing, stopwords, stemming
* ``split_sentences``            - abbreviation- and decimal-aware segmentation
* ``extract_numbers``            - numeric literals with magnitude/percent scaling
* ``extract_entities``           - capitalised spans (proper-noun-ish) + acronyms
* ``NEGATION_CUES`` / ``ANTONYMS`` - lexical signals used by the contradiction check
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Sequence, Set, Tuple

__all__ = [
    "STOPWORDS",
    "NEGATION_CUES",
    "ANTONYMS",
    "OPPOSITE_GROUPS",
    "direction_conflict",
    "normalize",
    "tokenize",
    "stem",
    "split_sentences",
    "extract_numbers",
    "extract_number_units",
    "extract_entities",
    "char_ngrams",
    "jaccard",
]

# --------------------------------------------------------------------------- #
# Lexicons
# --------------------------------------------------------------------------- #

STOPWORDS: Set[str] = frozenset(
    """
a about above after again against all am an and any are as at be because been before being
below between both but by can cannot could did do does doing down during each few for from
further had has have having he her here hers herself him himself his how i if in into is it
its itself just me more most my myself no nor not now of off on once only or other our ours
ourselves out over own same she should so some such than that the their theirs them themselves
then there these they this those through to too under until up very was we were what when
where which while who whom why will with would you your yours yourself yourselves it's don't
also may might must shall since upon within without per via across among
""".split()
)

# Words that flip the polarity of a statement. Used only when a claim otherwise
# aligns strongly with a context sentence -> polarity mismatch = contradiction.
NEGATION_CUES: Set[str] = frozenset(
    """
not no never none nor neither without cannot cant dont doesnt didnt isnt arent wasnt werent
wont hasnt havent hadnt fails failed failing unable lacks lacked absent denied rejected
excluded prohibited disallowed unsupported
""".split()
)

# Opposed word *groups*, not pairs. Pairwise antonym tables are brittle because
# a stemmer maps "increase"/"increased" to different stems; listing surface
# forms in a group and stemming them all sidesteps the problem entirely.
_OPPOSITE_GROUPS_RAW: Sequence[Tuple[str, Sequence[str], Sequence[str]]] = (
    (
        "direction",
        "increase increased increasing increases rise rises rose rising grew grow growth "
        "up higher gained gain climbed climbing surged doubled".split(),
        "decrease decreased decreasing decreases fall falls fell falling declined decline "
        "down lower dropped drop shrank reduced reduction halved plunged".split(),
    ),
    (
        "approval",
        "approved approves approval accepted accepts authorised authorized granted".split(),
        "rejected rejects rejection denied denies refused declined_application".split(),
    ),
    (
        "outcome",
        "succeeded success successful passed complied compliant".split(),
        "failed failure unsuccessful noncompliant".split(),
    ),
)

# name -> (stems of side A, stems of side B)
OPPOSITE_GROUPS: List[Tuple[str, Set[str], Set[str]]] = []
ANTONYMS: Dict[str, Set[str]] = {}


def _build_opposites() -> None:
    for name, side_a, side_b in _OPPOSITE_GROUPS_RAW:
        stems_a = {stem(w) for w in side_a}
        stems_b = {stem(w) for w in side_b}
        overlap = stems_a & stems_b
        stems_a -= overlap
        stems_b -= overlap
        OPPOSITE_GROUPS.append((name, stems_a, stems_b))
        for token in stems_a:
            ANTONYMS.setdefault(token, set()).update(stems_b)
        for token in stems_b:
            ANTONYMS.setdefault(token, set()).update(stems_a)


def direction_conflict(claim_tokens: Set[str], evidence_tokens: Set[str]) -> str:
    """Return a description when two token sets assert opposite directions.

    A conflict requires each side to be *unambiguous*: if a sentence mentions
    both an increase and a decrease (very common in comparative prose) it is
    skipped rather than guessed at.
    """
    for name, side_a, side_b in OPPOSITE_GROUPS:
        claim_a, claim_b = claim_tokens & side_a, claim_tokens & side_b
        ev_a, ev_b = evidence_tokens & side_a, evidence_tokens & side_b
        if claim_a and not claim_b and ev_b and not ev_a:
            return f"{name}: claim says '{sorted(claim_a)[0]}', context says '{sorted(ev_b)[0]}'"
        if claim_b and not claim_a and ev_a and not ev_b:
            return f"{name}: claim says '{sorted(claim_b)[0]}', context says '{sorted(ev_a)[0]}'"
    return ""

# --------------------------------------------------------------------------- #
# Normalisation and tokenisation
# --------------------------------------------------------------------------- #

_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
}
_QUOTE_TABLE = {ord(k): v for k, v in _QUOTES.items()}
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def normalize(text: str) -> str:
    """Fold unicode, unify quotes/dashes, and collapse whitespace.

    Accents are stripped (``café`` -> ``cafe``) so that inconsistently encoded
    corpora still match. The result keeps original casing.
    """
    if not text:
        return ""
    text = text.translate(_QUOTE_TABLE)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _WS_RE.sub(" ", text).strip()


@lru_cache(maxsize=100_000)
def stem(word: str) -> str:
    """A conservative English suffix stripper.

    Deliberately far less aggressive than Porter: it only removes suffixes where
    the risk of collapsing unrelated words is low, and it never shortens a token
    below three characters. This keeps ``gas``/``analysis``/``bus`` intact while
    still merging ``filters``/``filtering``/``filtered``.
    """
    w = word
    n = len(w)
    if n <= 3:
        return w
    if w.endswith("'s"):
        w = w[:-2]
        n = len(w)
        if n <= 3:
            return w
    if w.endswith("ies") and n > 4:
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ses") and n > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is", "as")):
        w = w[:-1]
        n = len(w)
    if n > 4 and w.endswith("ing"):
        base = w[:-3]
        base = _undouble(base)
        if len(base) >= 3:
            return base
    if n > 4 and w.endswith("ed"):
        base = w[:-2]
        base = _undouble(base)
        if len(base) >= 3:
            return base
    if n > 5 and w.endswith("ly"):
        return w[:-2]
    if n > 5 and w.endswith("es") and not w.endswith(("ses", "ies")):
        return w[:-2]
    return w


def _undouble(base: str) -> str:
    """``stopp`` -> ``stop`` but keep ``pass``/``fall`` style real doubles."""
    if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "sfl":
        return base[:-1]
    return base


def tokenize(
    text: str,
    *,
    remove_stopwords: bool = True,
    do_stem: bool = True,
    min_len: int = 1,
) -> List[str]:
    """Normalise then split into comparable terms.

    Numbers are kept (they matter a great deal for factual grounding) and
    thousands separators are stripped so ``1,200`` and ``1200`` unify.
    """
    if not text:
        return []
    lowered = normalize(text).lower()
    # Unify thousands separators before tokenising: 1,200,000 -> 1200000
    lowered = re.sub(r"(?<=\d),(?=\d{3}\b)", "", lowered)
    tokens: List[str] = []
    for match in _TOKEN_RE.finditer(lowered):
        tok = match.group(0)
        if tok.endswith("'s"):
            tok = tok[:-2]
        if len(tok) < min_len:
            continue
        if remove_stopwords and tok in STOPWORDS:
            continue
        tokens.append(stem(tok) if do_stem else tok)
    return tokens


def char_ngrams(text: str, n: int = 4) -> Set[str]:
    """Character n-grams of the normalised string; used as a fuzzy backstop."""
    s = re.sub(r"\s+", " ", normalize(text).lower())
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# --------------------------------------------------------------------------- #
# Sentence segmentation
# --------------------------------------------------------------------------- #

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "inc", "ltd", "llc", "co", "corp",
    "vs", "etc", "eg", "ie", "approx", "est", "fig", "no", "vol", "dept", "univ", "gov", "al",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "u.s", "u.k", "e.g", "i.e", "p.m", "a.m", "min", "max", "sec", "hr",
}
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences without any NLP dependency.

    Handles the failure modes that actually break naive ``text.split('.')``:
    decimals (``3.5%``), known abbreviations (``Dr.``, ``e.g.``), initials
    (``J. R. Smith``), version strings (``v1.2.3``) and ellipses.
    """
    text = normalize(text)
    if not text:
        return []
    sentences: List[str] = []
    start = 0
    for match in _SENT_BOUNDARY_RE.finditer(text):
        end = match.start()
        candidate = text[start:end].strip()
        if not candidate:
            continue
        if _is_false_boundary(text, end):
            continue
        sentences.append(candidate)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def _is_false_boundary(text: str, end: int) -> bool:
    """``end`` is the index just past the terminating punctuation."""
    idx = end - 1
    while idx >= 0 and text[idx] in "\"')]":
        idx -= 1
    if idx < 0:
        return True
    if text[idx] != ".":
        return False  # '!' and '?' are reliable boundaries
    # Ellipsis: "..." is usually mid-sentence in prose.
    if idx >= 2 and text[idx - 2 : idx] == "..":
        return True
    # Digit before and after the period -> decimal / version number.
    if idx >= 1 and text[idx - 1].isdigit():
        nxt = text[end : end + 1]
        if nxt.isdigit():
            return True
    # Preceding word is a known abbreviation or a single-letter initial.
    j = idx - 1
    while j >= 0 and (text[j].isalnum() or text[j] == "."):
        j -= 1
    word = text[j + 1 : idx].lower().strip(".")
    if not word:
        return False
    if len(word) == 1 and word.isalpha():
        return True
    return word in _ABBREVIATIONS


# --------------------------------------------------------------------------- #
# Numeric extraction
# --------------------------------------------------------------------------- #

_MAGNITUDES = {
    "hundred": 1e2, "thousand": 1e3, "k": 1e3,
    "million": 1e6, "m": 1e6, "mn": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12,
}
_NUM_RE = re.compile(
    r"""
    (?P<sign>[-+]?)
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
    (?P<suffix>%|percent|per\ cent|hundred|thousand|million|billion|trillion|bn|mn|k(?![a-z]))?
    """,
    re.VERBOSE | re.IGNORECASE,
)
_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7,
    "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9,
    "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
# "May" and "March" are also ordinary English words, so they only count as a
# month when a day or year sits next to them.
_AMBIGUOUS_MONTHS = {"May", "March", "Mar", "Sept"}


def extract_numbers(text: str) -> List[Tuple[float, str]]:
    """Return ``(value, kind)`` pairs for every numeric literal in ``text``.

    ``kind`` is one of ``"percent"``, ``"year"``, ``"month"`` or ``"number"``.
    Magnitude words are folded into the value so ``1.2 billion`` and
    ``1,200,000,000`` compare equal, and percentages live in their own
    namespace so ``50%`` never silently matches the bare number ``50``.
    """
    if not text:
        return []
    text = normalize(text)
    out: List[Tuple[float, str]] = []
    for match in _NUM_RE.finditer(text):
        raw = match.group("num").replace(",", "")
        try:
            value = float(raw)
        except ValueError:  # pragma: no cover - regex guarantees parseability
            continue
        if match.group("sign") == "-":
            value = -value
        suffix = (match.group("suffix") or "").strip().lower()
        kind = "number"
        if suffix in ("%", "percent", "per cent"):
            kind = "percent"
        elif suffix in _MAGNITUDES:
            value *= _MAGNITUDES[suffix]
        elif len(raw) == 4 and "." not in raw and 1500 <= value <= 2200:
            kind = "year"
        out.append((round(value, 6), kind))

    for word, idx in _MONTHS.items():
        pattern = rf"\b{word}\b\.?"
        if word in _AMBIGUOUS_MONTHS:
            pattern = rf"(?:\d{{1,2}}\s+{word}\b|\b{word}\s+\d)"
        if re.search(pattern, text):
            key = (float(idx), "month")
            if key not in out:
                out.append(key)
    return out


def extract_number_units(text: str) -> Set[Tuple[float, str, str]]:
    """Return ``(value, kind, unit)`` triples - a figure plus what it measures.

    The unit is the first *content* word after the figure (stopwords and the
    magnitude suffix are skipped), stemmed for comparability:
    ``"13 hours"`` -> ``(13.0, "number", "hour")`` and
    ``"48 per megawatt hour"`` -> ``(48.0, "number", "megawatt")``.

    Percentages carry the literal unit ``"percent"``, and years and months carry
    no unit at all, because "in 2024," is followed by an arbitrary word and a
    spurious unit there would create false mismatches.

    Why this matters: unit-aware comparison is what separates *"the claim says
    14 hours where the source says 13 hours"* (a contradiction) from *"the claim
    mentions 36,000 feet, which appears in a different sentence of the same
    passage"* (perfectly fine). Value-only matching cannot tell those apart.
    """
    if not text:
        return set()
    normalised = normalize(text)
    lowered = re.sub(r"(?<=\d),(?=\d{3}\b)", "", normalised.lower())
    out: Set[Tuple[float, str, str]] = set()
    for match in _NUM_RE.finditer(normalised):
        raw = match.group("num").replace(",", "")
        try:
            value = float(raw)
        except ValueError:  # pragma: no cover
            continue
        if match.group("sign") == "-":
            value = -value
        suffix = (match.group("suffix") or "").strip().lower()
        kind = "number"
        unit = ""
        if suffix in ("%", "percent", "per cent"):
            kind, unit = "percent", "percent"
        elif suffix in _MAGNITUDES:
            value *= _MAGNITUDES[suffix]
        elif len(raw) == 4 and "." not in raw and 1500 <= value <= 2200:
            kind = "year"
        if kind == "number":
            unit = _next_content_token(normalised, match.end())
        out.add((round(value, 6), kind, unit))
    return out


def _next_content_token(text: str, start: int) -> str:
    """First non-stopword token at or after ``start``, stemmed. ``""`` if none."""
    tail = text[start : start + 80].lower()
    tail = re.sub(r"(?<=\d),(?=\d{3}\b)", "", tail)
    for match in _TOKEN_RE.finditer(tail):
        token = match.group(0)
        if token in STOPWORDS:
            continue
        return stem(token)
    return ""


def number_keys(text: str) -> Set[str]:
    """Canonical string keys for the numbers in ``text`` (order-insensitive)."""
    return {f"{k}:{v:.6g}" for v, k in extract_numbers(text)}


# --------------------------------------------------------------------------- #
# Entity extraction
# --------------------------------------------------------------------------- #

_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9-]+)(?:\s+(?:of|the|de|van|and)?\s*[A-Z][a-zA-Z0-9-]+)*\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}[A-Za-z0-9]*(?:-[A-Z0-9]+)?\b")


def extract_entities(text: str, *, min_len: int = 3) -> List[str]:
    """Heuristically pull proper-noun-ish spans and acronyms out of ``text``.

    Two rules keep the false-positive rate usable without any POS tagger:

    * a **single** capitalised word at the start of a sentence is ignored - it
      is capitalised by orthography, not because it names anything;
    * stopwords are never entities, and all-caps tokens are handled by the
      acronym pass instead.

    Multi-word spans (``Flight NR-482``, ``Halden Regional``) and acronyms
    (``AD-2024-07``, ``ACR20``) survive both rules, which is exactly the set of
    strings a generator is most likely to fabricate.
    """
    if not text:
        return []
    text = normalize(text)
    entities: List[str] = []
    seen: Set[str] = set()

    def add(span: str) -> None:
        key = span.lower()
        if key not in seen:
            seen.add(key)
            entities.append(span)

    for match in _ACRONYM_RE.finditer(text):
        token = match.group(0)
        if len(token) >= 2 and token.lower() not in STOPWORDS:
            add(token)

    for sentence in split_sentences(text) or [text]:
        for match in _ENTITY_RE.finditer(sentence):
            span = match.group(0).strip()
            if len(span) < min_len:
                continue
            words = span.split()
            if len(words) == 1:
                word = words[0]
                if word.lower() in STOPWORDS or word.isupper():
                    continue
                if match.start() == 0:
                    continue  # sentence-initial capitalisation, not a name
            add(span)
    return entities


# Opposite groups depend on ``stem``, so they are built once the module is fully
# defined rather than at the point of declaration.
_build_opposites()
