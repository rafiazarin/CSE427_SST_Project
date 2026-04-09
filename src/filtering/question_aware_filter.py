from typing import Dict, Any


def filter_sst_by_question_type(sst: Dict[str, Any], semantic_type: str) -> Dict[str, Any]:
    """
    Filter SST fields based on question semantic type.

    semantic_type can be:
    - attribute
    - count
    - relation
    """

    filtered = {
        "objects": sst.get("objects", []),
        "counts": [],
        "attributes": [],
        "relations": [],
        "text": []
    }

    if semantic_type == "attribute":
        filtered["attributes"] = sst.get("attributes", [])
        filtered["text"] = sst.get("text", [])

    elif semantic_type == "count":
        filtered["counts"] = sst.get("counts", [])

    elif semantic_type == "relation":
        filtered["relations"] = sst.get("relations", [])

    else:
        # fallback → keep everything
        return sst

    return filtered