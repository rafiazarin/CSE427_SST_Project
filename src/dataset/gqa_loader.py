import json
import os
from typing import Dict, List, Any


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def gqa_questions_to_list(questions_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert GQA question dictionary format into a list of records.

    Expected GQA-style structure:
    {
        "question_id_1": {
            "imageId": "...",
            "question": "...",
            "answer": "...",
            ...
        },
        ...
    }
    """
    records = []

    for question_id, item in questions_dict.items():
        record = {
            "question_id": question_id,
            "image_id": item.get("imageId"),
            "question": item.get("question"),
            "answer": item.get("answer"),
            "types": item.get("types", {}),
            "full_record": item
        }
        if "sst" in item:
          record["sst"] = item["sst"]
        records.append(record)

    return records


def load_gqa_questions(file_path: str) -> List[Dict[str, Any]]:
    """
    Load GQA questions JSON file and convert to a list format.
    """
    raw_data = load_json(file_path)
    return gqa_questions_to_list(raw_data)


if __name__ == "__main__":
    sample_path = "data/raw/sample_questions.json"

    if os.path.exists(sample_path):
        records = load_gqa_questions(sample_path)
        print(f"Loaded {len(records)} question records.")
        if len(records) > 0:
            print(records[0])
    else:
        print(f"Sample file not found at: {sample_path}")