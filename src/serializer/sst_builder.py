from typing import List, Dict, Any


def build_sst(
    objects: List[str],
    counts: Dict[str, int],
    attributes: List[str],
    relations: List[str],
    text: List[str]
) -> Dict[str, Any]:
    """
    Build a Scene-to-Structured-Text (SST) dictionary.
    """

    sst = {
        "objects": objects,
        "counts": [f"{key}: {value}" for key, value in counts.items()],
        "attributes": attributes,
        "relations": relations,
        "text": text
    }

    return sst


def sst_to_prompt(sst: Dict[str, Any]) -> str:
    """
    Convert SST dictionary into a structured text prompt.
    """

    prompt = []
    prompt.append("Scene-to-Structured-Text Representation:")

    prompt.append(f"Objects: {', '.join(sst.get('objects', [])) if sst.get('objects') else 'None'}")
    prompt.append(f"Counts: {', '.join(sst.get('counts', [])) if sst.get('counts') else 'None'}")
    prompt.append(f"Attributes: {', '.join(sst.get('attributes', [])) if sst.get('attributes') else 'None'}")
    prompt.append(f"Relations: {', '.join(sst.get('relations', [])) if sst.get('relations') else 'None'}")
    prompt.append(f"Text: {', '.join(sst.get('text', [])) if sst.get('text') else 'None'}")

    return "\n".join(prompt)


if __name__ == "__main__":
    sample_sst = build_sst(
        objects=["bus", "person", "stop_sign"],
        counts={"person": 3},
        attributes=["bus: red"],
        relations=["person near bus", "stop_sign right of bus"],
        text=["STOP"]
    )

    print(sample_sst)
    print()
    print(sst_to_prompt(sample_sst))