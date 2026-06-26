"""Question-aware keyword filtering for SST records.

Keeps only the scene content that is relevant to the question terms (plus the
objects connected to them via relations), enabling the Keyword-Aware variant
to drop roughly half the scene tokens with negligible accuracy loss.
"""

from __future__ import annotations

import re

STOPWORDS = {
    "what", "which", "where", "who", "whom", "whose", "why", "how",
    "is", "are", "was", "were", "be", "being", "been",
    "a", "an", "the", "this", "that", "these", "those",
    "there", "in", "on", "at", "to", "from", "of", "for", "with",
    "and", "or", "but", "by", "near", "next", "beside",
    "left", "right", "top", "bottom", "front", "behind",
    "color", "kind", "type", "called", "name", "doing",
    "does", "do", "did", "can", "could", "would", "should",
    "visible", "shown", "picture", "image", "scene",
}
SYNONYMS = {
    "people": "person", "persons": "person", "men": "man", "women": "woman",
    "children": "child", "kids": "child", "boys": "boy", "girls": "girl",
    "cars": "car", "buses": "bus", "trucks": "truck", "bikes": "bike",
    "bicycles": "bicycle", "birds": "bird", "animals": "animal",
    "dogs": "dog", "cats": "cat", "plates": "plate", "tables": "table",
    "chairs": "chair", "signs": "sign", "trees": "tree",
    "windows": "window", "doors": "door",
}


def normalize_term(term: str) -> str:
    term = re.sub(r"[^a-z0-9 ]+", " ", str(term).lower().replace("_", " ")).strip()
    term = re.sub(r"\s+", " ", term).strip()
    term = SYNONYMS.get(term, term)
    if term.endswith("ies") and len(term) > 4:
        term = term[:-3] + "y"
    elif term.endswith("es") and len(term) > 3:
        term = term[:-2]
    elif term.endswith("s") and len(term) > 3:
        term = term[:-1]
    return SYNONYMS.get(term, term)


def extract_terms(question: str) -> set:
    toks = re.findall(r"[a-zA-Z0-9_]+", str(question).lower())
    return {normalize_term(t) for t in toks if t not in STOPWORDS and len(normalize_term(t)) > 1}


def item_matches(item: str, terms: set) -> bool:
    inorm = normalize_term(item)
    itoks = set(inorm.split())
    for t in terms:
        tnorm = normalize_term(t)
        ttoks = set(tnorm.split())
        if tnorm == inorm or tnorm in itoks or inorm in ttoks or (itoks & ttoks):
            return True
    return False


def obj_name_from_fact(fact: str) -> str:
    return str(fact).split(":")[0].strip() if ":" in str(fact) else str(fact).strip()


def keyword_aware_sst_filter(sample: dict, max_obj=25, max_attr=25, max_rel=25) -> dict:
    q, sst = sample["question"], sample["sst"]
    objs  = list(map(str, sst.get("objects", [])))
    cnts  = list(map(str, sst.get("counts", [])))
    attrs = list(map(str, sst.get("attributes", [])))
    rels  = list(map(str, sst.get("relations", [])))
    txts  = list(map(str, sst.get("text", [])))
    terms = extract_terms(q)
    matched = [o for o in objs if item_matches(o, terms)]
    if not matched:
        return {"objects": objs[:max_obj], "counts": cnts,
                "attributes": attrs[:max_attr], "relations": rels[:max_rel], "text": txts}
    connected = set(matched)
    kept_rels = []
    for r in rels:
        if any(o in r for o in matched):
            kept_rels.append(r)
            for o in objs:
                if o in r:
                    connected.add(o)
    kept_cnts  = [c for c in cnts  if obj_name_from_fact(c) in connected or item_matches(obj_name_from_fact(c), terms)]
    kept_attrs = [a for a in attrs if obj_name_from_fact(a) in connected or item_matches(obj_name_from_fact(a), terms)]
    return {
        "objects":    list(connected)[:max_obj],
        "counts":     kept_cnts,
        "attributes": kept_attrs[:max_attr],
        "relations":  kept_rels[:max_rel],
        "text":       txts,
    }
