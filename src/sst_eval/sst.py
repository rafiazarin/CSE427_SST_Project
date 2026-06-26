"""SST construction, serialisation, and variant builders.

An SST ("Scene-to-Structured-Text") record is a dict with five fields:
objects, counts, attributes, relations, text.
"""

from __future__ import annotations

from collections import Counter

SST_FIELDS = ["objects", "counts", "attributes", "relations", "text"]

# Body parts and other fine-grained nodes that add scene-graph noise without
# helping most questions. Filtered out of the oracle SST.
LOW_VALUE_OBJECTS = {
    "nose", "eye", "eyes", "ear", "ears", "mouth", "hair", "head",
    "hand", "hands", "arm", "arms", "leg", "legs", "foot", "feet",
    "face", "neck", "finger", "fingers", "tail",
}
WEAK_RELATIONS = {"of"}


# ── Serialisers ────────────────────────────────────────────────────────────
def sst_to_prompt_structured(sst: dict) -> str:
    lines = [
        "Objects: "    + ", ".join(map(str, sst.get("objects", []))),
        "Counts: "     + ", ".join(map(str, sst.get("counts", []))),
        "Attributes: " + ", ".join(map(str, sst.get("attributes", []))),
        "Relations: "  + ", ".join(map(str, sst.get("relations", []))),
        "Text: "       + ", ".join(map(str, sst.get("text", []))),
    ]
    return "\n".join(lines)


def sst_to_prompt_compact(sst: dict) -> str:
    parts = []
    if sst.get("objects"):    parts.append("obj: "  + ", ".join(map(str, sst["objects"])))
    if sst.get("counts"):     parts.append("cnt: "  + ", ".join(map(str, sst["counts"])))
    if sst.get("attributes"): parts.append("attr: " + ", ".join(map(str, sst["attributes"])))
    if sst.get("relations"):  parts.append("rel: "  + ", ".join(map(str, sst["relations"])))
    if sst.get("text"):       parts.append("text: " + ", ".join(map(str, sst["text"])))
    return " | ".join(parts) if parts else "no scene information"


def sst_to_prompt_caption(sst: dict) -> str:
    s = []
    if sst.get("objects"):    s.append("The scene contains " + ", ".join(map(str, sst["objects"])) + ".")
    if sst.get("counts"):     s.append("Object counts: " + ", ".join(map(str, sst["counts"])) + ".")
    if sst.get("attributes"): s.append("Attributes: " + ", ".join(map(str, sst["attributes"])) + ".")
    if sst.get("relations"):  s.append("Relations: " + ", ".join(map(str, sst["relations"])) + ".")
    if sst.get("text"):       s.append("Visible text: " + ", ".join(map(str, sst["text"])) + ".")
    return " ".join(s) if s else "No scene information."


# ── Variant builders (field ablations) ─────────────────────────────────────
def make_full_sst(sst: dict) -> dict:
    return {k: list(sst.get(k, [])) for k in SST_FIELDS}


def make_objects_only_sst(sst: dict) -> dict:
    return {"objects": list(sst.get("objects", [])),
            "counts": [], "attributes": [], "relations": [], "text": []}


def make_no_relations_sst(sst: dict) -> dict:
    return {k: list(sst.get(k, [])) for k in ["objects", "counts", "attributes", "text"]} | {"relations": []}


def make_no_attributes_sst(sst: dict) -> dict:
    return {k: list(sst.get(k, [])) for k in ["objects", "counts", "relations", "text"]} | {"attributes": []}


# ── Oracle scene-graph -> SST ──────────────────────────────────────────────
def gqa_scene_to_clean_sst(scene: dict) -> dict:
    """Convert a GQA ground-truth scene graph into a clean oracle SST record.

    Note: the GQA scene graph carries no OCR, so the ``text`` field is always
    empty here. Visible-text extraction only happens in the detected-SST
    pipeline (YOLO + EasyOCR).
    """
    objects_dict = scene.get("objects", {})
    names, attrs, rels, id2name = [], [], [], {}
    for oid, obj in objects_dict.items():
        name = str(obj.get("name", "")).strip().lower()
        if not name or name in LOW_VALUE_OBJECTS:
            continue
        id2name[oid] = name
        names.append(name)
        for a in obj.get("attributes", []):
            a = str(a).strip().lower()
            if a:
                attrs.append(f"{name}: {a}")
    cnts = [f"{n}: {c}" for n, c in Counter(names).items() if c > 1]
    for oid, obj in objects_dict.items():
        subj = str(obj.get("name", "")).strip().lower()
        if not subj or subj in LOW_VALUE_OBJECTS:
            continue
        for rel in obj.get("relations", []):
            rname = str(rel.get("name", "")).strip().lower()
            tobj = id2name.get(str(rel.get("object", "")), "")
            if rname and rname not in WEAK_RELATIONS and tobj:
                rels.append(f"{subj} {rname} {tobj}")

    def dedup(lst):
        seen = set()
        return [x for x in lst if not (x in seen or seen.add(x))]

    return {
        "objects":    dedup(names)[:15],
        "counts":     dedup(cnts)[:15],
        "attributes": dedup(attrs)[:20],
        "relations":  dedup(rels)[:25],
        "text":       [],
    }


# ── Sample helpers ─────────────────────────────────────────────────────────
def get_semantic_type(sample: dict) -> str:
    types = sample.get("types")
    return types.get("semantic", "unknown") if isinstance(types, dict) else "unknown"


def validate_sample(sample: dict, idx: int) -> dict:
    from .normalize import normalize_answer

    s = dict(sample)
    s["question_id"] = s.get("question_id", f"sample_{idx}")
    s["question"]    = str(s.get("question", "")).strip()
    s["answer"]      = normalize_answer(s.get("answer", ""))
    if not isinstance(s.get("types"), dict):
        s["types"] = {"semantic": "unknown"}
    if "semantic" not in s["types"]:
        s["types"]["semantic"] = "unknown"
    if not isinstance(s.get("sst"), dict):
        s["sst"] = {}
    for f in SST_FIELDS:
        if f not in s["sst"]:
            s["sst"][f] = []
    return s
