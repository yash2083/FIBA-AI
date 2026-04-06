"""Query parser for FIBA AI action retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

try:
    import spacy
except ImportError:  # pragma: no cover - optional dependency
    spacy = None


@dataclass
class QueryResult:
    raw_query: str
    action_verb: str
    action_category: str
    object_noun: str
    tool_noun: Optional[str]


VERB_CATEGORY_MAP = {
    # CUT family
    "cut": "CUT",
    "cutting": "CUT",
    "chop": "CUT",
    "chopping": "CUT",
    "slice": "CUT",
    "slicing": "CUT",
    "dice": "CUT",
    "dicing": "CUT",
    # OPEN family
    "open": "OPEN",
    "opening": "OPEN",
    "unscrew": "OPEN",
    "unscrewing": "OPEN",
    "unlock": "OPEN",
    "peel": "OPEN",
    "peeling": "OPEN",
    "unwrap": "OPEN",
    "unwrapping": "OPEN",
    # POUR family (includes dip, insert, submerge — liquid/container interactions)
    "pour": "POUR",
    "pouring": "POUR",
    "fill": "POUR",
    "filling": "POUR",
    "drain": "POUR",
    "draining": "POUR",
    "dip": "DIP",
    "dipping": "DIP",
    "dunk": "DIP",
    "dunking": "DIP",
    "submerge": "DIP",
    "submerging": "DIP",
    "soak": "DIP",
    "soaking": "DIP",
    "steep": "DIP",
    "steeping": "DIP",
    "insert": "DIP",
    "inserting": "DIP",
    # PICK family
    "pick": "PICK",
    "picking": "PICK",
    "grab": "PICK",
    "grabbing": "PICK",
    "take": "PICK",
    "taking": "PICK",
    "lift": "PICK",
    "lifting": "PICK",
    "hold": "PICK",
    "holding": "PICK",
    # PLACE family
    "place": "PLACE",
    "placing": "PLACE",
    "put": "PLACE",
    "putting": "PLACE",
    "set": "PLACE",
    "drop": "PLACE",
    "dropping": "PLACE",
    # MIX family
    "mix": "MIX",
    "mixing": "MIX",
    "stir": "MIX",
    "stirring": "MIX",
    "shake": "MIX",
    "shaking": "MIX",
    "blend": "MIX",
    "whisk": "MIX",
    "whisking": "MIX",
    # CLOSE family
    "close": "CLOSE",
    "closing": "CLOSE",
    "shut": "CLOSE",
    "cap": "CLOSE",
    "cover": "CLOSE",
    "covering": "CLOSE",
    "seal": "CLOSE",
    # PUSH / PULL
    "push": "PUSH",
    "pushing": "PUSH",
    "press": "PUSH",
    "pressing": "PUSH",
    "pull": "PULL",
    "pulling": "PULL",
    # SQUEEZE
    "squeeze": "SQUEEZE",
    "squeezing": "SQUEEZE",
    "wring": "SQUEEZE",
    # SPREAD
    "spread": "SPREAD",
    "spreading": "SPREAD",
    "smear": "SPREAD",
    # SCOOP
    "scoop": "SCOOP",
    "scooping": "SCOOP",
    "ladle": "SCOOP",
    # WASH / WIPE
    "wash": "WASH",
    "washing": "WASH",
    "rinse": "WASH",
    "rinsing": "WASH",
    "wipe": "WASH",
    "wiping": "WASH",
    "clean": "WASH",
    "cleaning": "WASH",
    # FOLD / TEAR
    "fold": "FOLD",
    "folding": "FOLD",
    "tear": "TEAR",
    "tearing": "TEAR",
    "rip": "TEAR",
}

CATEGORY_TOOL_MAP = {
    "CUT": "knife",
    "OPEN": None,
    "POUR": None,
    "DIP": None,
    "PICK": None,
    "PLACE": None,
    "MIX": "spoon",
    "CLOSE": None,
    "PUSH": None,
    "PULL": None,
    "SQUEEZE": None,
    "SPREAD": "knife",
    "SCOOP": "spoon",
    "WASH": None,
    "FOLD": None,
    "TEAR": None,
}

TOOL_WORDS = {"knife", "spoon", "fork", "scissors", "hand", "finger"}
STOP_WORDS = {
    "a",
    "an",
    "the",
    "some",
    "my",
    "with",
    "using",
    "from",
    "into",
    "onto",
    "off",
    "up",
    "down",
    "is",
    "are",
    "be",
    "i",
    "want",
    "to",
    "know",
    "where",
    "when",
    "how",
    "what",
    "who",
    "find",
    "show",
    "me",
    "person",
    "someone",
    "man",
    "woman",
    "they",
    "he",
    "she",
    "it",
    "this",
    "that",
    "of",
    "in",
    "on",
    "at",
    "and",
    "or",
    "but",
    "for",
    "not",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "was",
    "were",
    "been",
    "being",
    "there",
    "here",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "may",
    "might",
}

_SPACY_MODEL = "en_core_web_sm"
_NLP = None
_SPACY_LOAD_ATTEMPTED = False


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower().strip())
    return [tok for tok in tokens if tok not in STOP_WORDS]


def _load_spacy_model():
    global _NLP
    global _SPACY_LOAD_ATTEMPTED
    if _SPACY_LOAD_ATTEMPTED:
        return _NLP
    _SPACY_LOAD_ATTEMPTED = True

    if spacy is None:
        return None

    try:
        _NLP = spacy.load(_SPACY_MODEL)
    except Exception:
        _NLP = None
    return _NLP


def _parse_with_spacy(raw_query: str, tokens: list[str]) -> Optional[QueryResult]:
    nlp = _load_spacy_model()
    if nlp is None:
        return None

    doc = nlp(raw_query)
    action_verb = None
    action_category = None
    verb_index = None

    for token in doc:
        txt = token.text.lower()
        lemma = token.lemma_.lower()
        if token.pos_ in {"VERB", "AUX"} or txt in VERB_CATEGORY_MAP or lemma in VERB_CATEGORY_MAP:
            if txt in VERB_CATEGORY_MAP:
                action_verb = txt
            elif lemma in VERB_CATEGORY_MAP:
                action_verb = lemma
            else:
                action_verb = txt
            action_category = VERB_CATEGORY_MAP.get(action_verb, "UNKNOWN")
            verb_index = token.i
            break

    if action_verb is None:
        if tokens:
            action_verb = tokens[0]
            action_category = VERB_CATEGORY_MAP.get(action_verb, "UNKNOWN")
        else:
            action_verb = "unknown"
            action_category = "UNKNOWN"

    object_noun = None
    if verb_index is not None:
        for token in doc:
            if token.i <= verb_index:
                continue
            if token.pos_ in {"NOUN", "PROPN"}:
                candidate = token.lemma_.lower() if token.lemma_ else token.text.lower()
                if candidate and candidate not in STOP_WORDS:
                    object_noun = candidate
                    break

    if object_noun is None:
        fallback_tokens = tokens[1:] if tokens else []
        object_noun = fallback_tokens[0] if fallback_tokens else "object"

    tool_noun = CATEGORY_TOOL_MAP.get(action_category)
    for token in tokens:
        if token in TOOL_WORDS:
            tool_noun = token
            break

    return QueryResult(
        raw_query=raw_query,
        action_verb=action_verb,
        action_category=action_category,
        object_noun=object_noun,
        tool_noun=tool_noun,
    )


# Compound noun map — maps multi-word objects to single tokens for YOLO grounding
COMPOUND_NOUNS = {
    "tea bag": "cup",
    "teabag": "cup",
    "tea cup": "cup",
    "water bottle": "bottle",
    "wine glass": "cup",
    "coffee cup": "cup",
    "frying pan": "pan",
    "cutting board": "knife",
    "paper towel": "towel",
}


def _resolve_object_noun(remaining_tokens: list[str], raw_query: str) -> str:
    """Resolve object noun, handling compound nouns and YOLO-friendly mapping."""
    # First check for compound nouns in the raw query
    raw_lower = raw_query.lower()
    for compound, coco_label in COMPOUND_NOUNS.items():
        if compound in raw_lower:
            return coco_label

    # Join remaining to check for two-word compounds
    if len(remaining_tokens) >= 2:
        pair = f"{remaining_tokens[0]} {remaining_tokens[1]}"
        if pair in COMPOUND_NOUNS:
            return COMPOUND_NOUNS[pair]

    return remaining_tokens[0] if remaining_tokens else "object"


def _parse_with_regex(raw_query: str, tokens: list[str]) -> QueryResult:
    action_verb = "unknown"
    action_category = "UNKNOWN"
    verb_idx = -1

    for idx, token in enumerate(tokens):
        if token in VERB_CATEGORY_MAP:
            action_verb = token
            action_category = VERB_CATEGORY_MAP[token]
            verb_idx = idx
            break

    if action_verb == "unknown" and tokens:
        action_verb = tokens[0]
        action_category = VERB_CATEGORY_MAP.get(action_verb, "UNKNOWN")
        verb_idx = 0

    remaining = tokens[verb_idx + 1 :] if verb_idx >= 0 else tokens
    object_noun = _resolve_object_noun(remaining, raw_query)

    tool_noun = CATEGORY_TOOL_MAP.get(action_category)
    for token in remaining[1:]:
        if token in TOOL_WORDS:
            tool_noun = token
            break

    return QueryResult(
        raw_query=raw_query,
        action_verb=action_verb,
        action_category=action_category,
        object_noun=object_noun,
        tool_noun=tool_noun,
    )


def parse_query(query_text: str) -> QueryResult:
    """Parse natural language query into action/object/tool components."""
    if not isinstance(query_text, str):
        raise TypeError("query_text must be a string")

    cleaned = query_text.strip()
    if not cleaned:
        return QueryResult(
            raw_query=query_text,
            action_verb="unknown",
            action_category="UNKNOWN",
            object_noun="object",
            tool_noun=None,
        )

    tokens = _tokenize(cleaned)
    spacy_result = _parse_with_spacy(cleaned, tokens)
    if spacy_result is not None:
        return spacy_result
    return _parse_with_regex(cleaned, tokens)


if __name__ == "__main__":
    sample_queries = [
        "cutting onion",
        "opening a box",
        "pouring water into cup",
        "picking up the bottle",
        "mixing ingredients with spoon",
    ]

    for query in sample_queries:
        result = parse_query(query)
        print(f"Query: {query}")
        print(
            "  -> verb=%s (%s), object=%s, tool=%s"
            % (
                result.action_verb,
                result.action_category,
                result.object_noun,
                result.tool_noun,
            )
        )
        print()
