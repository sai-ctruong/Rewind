"""Query representation for Vietnamese and English retrieval.

The rule this module exists to enforce: **the original query is never replaced**. CLIP was
trained on natural text and should see exactly what the user typed; only the lexical
channels (BM25, objects, metadata) consume folded and expanded forms. A pipeline that
globally accent-strips its input silently degrades its own dense retrieval.

`QueryRepresentation` therefore carries several views side by side rather than one
"cleaned" string, and every expanded term records where it came from.

Three things are handled carefully because getting them wrong changes meaning:

* **Accent folding** is deterministic Unicode work (`NFD`, drop combining marks) plus the
  one Vietnamese letter that does not decompose: `đ`/`Đ`.
* **Negation** (`không`, `not`, `without`, ...) is preserved and marked. A naive object
  expansion would turn "không có xe" into a positive query for cars.
* **Temporal markers** (`trước`, `sau đó`, `then`, ...) are preserved for TRAKE consumers
  rather than filtered out as stopwords.

The bilingual vocabulary is a small, explicit, versioned retrieval aid for detector
labels — not a translation system, and no external API is ever called.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# Bump when the vocabulary or folding rules change in a way that affects retrieval.
QUERY_VOCABULARY_VERSION = 1

# Vietnamese `đ` carries no combining mark, so NFD leaves it intact and it must be
# mapped explicitly. Everything else folds through Unicode decomposition.
_STANDALONE_FOLDS = {"đ": "d", "Đ": "D", "ð": "d", "Ð": "D"}

NEGATION_TERMS = frozenset(
    {
        "khong", "chua", "chẳng", "chang", "không", "chưa",
        "no", "not", "without", "never", "none", "neither", "nor",
    }
)
TEMPORAL_TERMS = frozenset(
    {
        "truoc", "trước", "sau", "roi", "rồi", "tiep", "tiếp", "theo",
        "dau", "đầu", "cuoi", "cuối", "khi", "luc", "lúc", "trong",
        "before", "after", "then", "next", "first", "last", "while",
        "during", "finally", "afterwards", "later", "earlier",
    }
)

# A deliberately small, explicit retrieval aid: Vietnamese phrases to the English words
# an object detector actually emits. It is not a translation database and is expected to
# be extended by hand, with tests, when a real gap shows up.
VIETNAMESE_ENGLISH_TERMS: dict[str, tuple[str, ...]] = {
    "nguoi": ("person", "people", "man", "woman"),
    "dan ong": ("man",),
    "phu nu": ("woman",),
    "tre em": ("child", "boy", "girl"),
    "xe hoi": ("car",),
    "o to": ("car",),
    "xe": ("vehicle", "car"),
    "xe may": ("motorcycle",),
    "xe gan may": ("motorcycle",),
    "xe dap": ("bicycle",),
    "xe buyt": ("bus",),
    "xe tai": ("truck",),
    "xe cuu thuong": ("ambulance",),
    "tau": ("train", "boat"),
    "thuyen": ("boat",),
    "may bay": ("airplane",),
    "duong": ("road", "street"),
    "pho": ("street",),
    "nga tu": ("intersection",),
    "cua": ("door",),
    "cua so": ("window",),
    "ghe": ("chair",),
    "ban": ("table", "desk"),
    "giuong": ("bed",),
    "nha": ("house", "building"),
    "toa nha": ("building",),
    "phong": ("room",),
    "cho": ("dog",),
    "meo": ("cat",),
    "chim": ("bird",),
    "cay": ("tree", "plant"),
    "hoa": ("flower",),
    "den": ("lamp", "light"),
    "den giao thong": ("traffic light",),
    "bien bao": ("traffic sign",),
    "dien thoai": ("mobile phone",),
    "may tinh": ("computer", "laptop"),
    "sach": ("book",),
    "tui": ("bag", "handbag"),
    "mu": ("hat", "helmet"),
    "mu bao hiem": ("helmet",),
    "ao": ("shirt", "clothing"),
    "giay": ("shoe", "footwear"),
    "banh": ("cake",),
    "nen": ("candle",),
    "qua": ("gift",),
    "thuc an": ("food",),
    "nuoc": ("water",),
}

# The reverse direction, so an English query still reaches Vietnamese lexical fields.
ENGLISH_VIETNAMESE_TERMS: dict[str, tuple[str, ...]] = {}
for _vi, _english in VIETNAMESE_ENGLISH_TERMS.items():
    for _en in _english:
        ENGLISH_VIETNAMESE_TERMS.setdefault(_en, ())
        if _vi not in ENGLISH_VIETNAMESE_TERMS[_en]:
            ENGLISH_VIETNAMESE_TERMS[_en] = ENGLISH_VIETNAMESE_TERMS[_en] + (_vi,)


def fold_accents(text: str) -> str:
    """Deterministic Vietnamese/Latin accent folding.

    `NFD` splits a letter from its combining marks, which are then dropped; `đ` and `Đ`
    have no decomposition and are mapped explicitly. Input in either NFC or NFD gives the
    same result, so upstream normalization differences cannot change retrieval.
    """
    if not text:
        return ""
    replaced = "".join(_STANDALONE_FOLDS.get(character, character) for character in str(text))
    decomposed = unicodedata.normalize("NFD", replaced)
    stripped = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return unicodedata.normalize("NFC", stripped)


def tokenize_query(text: str) -> tuple[str, ...]:
    """Split into word tokens, keeping negation and temporal words intact."""
    if not text:
        return ()
    cleaned = re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", str(text)), flags=re.UNICODE)
    return tuple(token for token in cleaned.casefold().split() if token)


@dataclass(frozen=True)
class ExpandedTerm:
    """One lexical term plus where it came from, so provenance survives fusion."""

    term: str
    source: str
    origin: str = ""
    negated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "source": self.source,
            "origin": self.origin,
            "negated": self.negated,
        }


@dataclass(frozen=True)
class QueryRepresentation:
    """Several views of one query. The original is always available, never replaced."""

    original: str
    normalized_whitespace: str = ""
    lowercase: str = ""
    accent_folded: str = ""
    tokens_original: tuple[str, ...] = ()
    tokens_folded: tuple[str, ...] = ()
    negated_tokens: tuple[str, ...] = ()
    temporal_markers: tuple[str, ...] = ()
    number_terms: tuple[str, ...] = ()
    expanded_terms: tuple[ExpandedTerm, ...] = ()
    vocabulary_version: int = QUERY_VOCABULARY_VERSION

    @property
    def dense_query(self) -> str:
        """What CLIP sees: the user's own words, unfolded and unexpanded."""
        return self.normalized_whitespace or self.original

    @property
    def object_terms(self) -> tuple[str, ...]:
        """Positive lexical terms usable for object matching; negated ones excluded."""
        return tuple(
            dict.fromkeys(item.term for item in self.expanded_terms if not item.negated)
        )

    @property
    def negated_terms(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(item.term for item in self.expanded_terms if item.negated)
        )

    @property
    def lexical_terms(self) -> tuple[str, ...]:
        """Everything a sparse channel may match on, positive terms only."""
        return tuple(dict.fromkeys(self.tokens_folded + self.object_terms))

    def to_dict(self, *, include_terms: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "original": self.original,
            "accent_folded": self.accent_folded,
            "tokens_folded": list(self.tokens_folded),
            "negated_tokens": list(self.negated_tokens),
            "temporal_markers": list(self.temporal_markers),
            "number_terms": list(self.number_terms),
            "vocabulary_version": self.vocabulary_version,
        }
        if include_terms:
            payload["expanded_terms"] = [item.to_dict() for item in self.expanded_terms]
            payload["object_terms"] = list(self.object_terms)
        return payload


def _phrase_expansions(
    folded_tokens: Sequence[str], negated_positions: set[int]
) -> list[ExpandedTerm]:
    """Match multi-word vocabulary phrases first, then single words.

    Longest-phrase-first matters: `xe may` must beat the bare `xe`, or a motorcycle query
    would expand to a generic vehicle one.
    """
    out: list[ExpandedTerm] = []
    used: set[int] = set()
    max_words = max((len(key.split()) for key in VIETNAMESE_ENGLISH_TERMS), default=1)
    for width in range(max_words, 0, -1):
        for start in range(0, len(folded_tokens) - width + 1):
            if any(position in used for position in range(start, start + width)):
                continue
            phrase = " ".join(folded_tokens[start:start + width])
            english = VIETNAMESE_ENGLISH_TERMS.get(phrase)
            if not english:
                continue
            negated = any(position in negated_positions for position in range(start, start + width))
            used.update(range(start, start + width))
            for term in english:
                out.append(
                    ExpandedTerm(term=term, source="vi_en_vocabulary", origin=phrase, negated=negated)
                )
    # English input reaching Vietnamese lexical fields.
    for position, token in enumerate(folded_tokens):
        for term in ENGLISH_VIETNAMESE_TERMS.get(token, ()):  # noqa: B007 - small map
            out.append(
                ExpandedTerm(
                    term=term,
                    source="en_vi_vocabulary",
                    origin=token,
                    negated=position in negated_positions,
                )
            )
    return out


# A negation word scopes over the next few tokens; beyond that the connection is too weak
# to assume. Deliberately short and explicit rather than a parser.
NEGATION_SCOPE_TOKENS = 3


def normalize_query(text: str, *, expand: bool = True) -> QueryRepresentation:
    """Build every view of a query without discarding the user's own words."""
    original = str(text or "")
    whitespace = " ".join(unicodedata.normalize("NFC", original).split())
    lowercase = whitespace.casefold()
    folded = fold_accents(lowercase)
    tokens_original = tokenize_query(whitespace)
    tokens_folded = tuple(fold_accents(token) for token in tokens_original)

    negated_positions: set[int] = set()
    negated_tokens: list[str] = []
    for position, token in enumerate(tokens_folded):
        if token in NEGATION_TERMS or tokens_original[position] in NEGATION_TERMS:
            negated_tokens.append(tokens_original[position])
            # The negation itself, plus the short window it scopes over.
            for offset in range(1, NEGATION_SCOPE_TOKENS + 1):
                if position + offset < len(tokens_folded):
                    negated_positions.add(position + offset)
    temporal = tuple(
        dict.fromkeys(
            tokens_original[position]
            for position, token in enumerate(tokens_folded)
            if token in TEMPORAL_TERMS or tokens_original[position] in TEMPORAL_TERMS
        )
    )
    numbers = tuple(token for token in tokens_folded if token.isdigit())

    expanded: list[ExpandedTerm] = []
    if expand:
        for position, token in enumerate(tokens_folded):
            if token in NEGATION_TERMS:
                continue
            expanded.append(
                ExpandedTerm(
                    term=token,
                    source="query_token",
                    origin=tokens_original[position],
                    negated=position in negated_positions,
                )
            )
        expanded.extend(_phrase_expansions(tokens_folded, negated_positions))

    # Deduplicate on (term, negated) keeping the first provenance, so a term that is both
    # positive and negated in one query keeps both records.
    seen: set[tuple[str, bool]] = set()
    unique: list[ExpandedTerm] = []
    for item in expanded:
        key = (item.term, item.negated)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return QueryRepresentation(
        original=original,
        normalized_whitespace=whitespace,
        lowercase=lowercase,
        accent_folded=folded,
        tokens_original=tokens_original,
        tokens_folded=tokens_folded,
        negated_tokens=tuple(dict.fromkeys(negated_tokens)),
        temporal_markers=temporal,
        number_terms=tuple(dict.fromkeys(numbers)),
        expanded_terms=tuple(unique),
    )


def normalize_label(value: str) -> str:
    """Canonical form for an indexed label or lexical field: folded, singularized."""
    folded = fold_accents(unicodedata.normalize("NFKC", str(value)).casefold().replace("_", " "))
    cleaned = re.sub(r"[^\w\s-]", " ", folded, flags=re.UNICODE)
    words = []
    for word in cleaned.split():
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


def label_tokens(value: str) -> tuple[str, ...]:
    return tuple(normalize_label(value).split())


__all__ = [
    "ENGLISH_VIETNAMESE_TERMS",
    "NEGATION_SCOPE_TOKENS",
    "NEGATION_TERMS",
    "QUERY_VOCABULARY_VERSION",
    "TEMPORAL_TERMS",
    "VIETNAMESE_ENGLISH_TERMS",
    "ExpandedTerm",
    "QueryRepresentation",
    "fold_accents",
    "label_tokens",
    "normalize_label",
    "normalize_query",
    "tokenize_query",
]
