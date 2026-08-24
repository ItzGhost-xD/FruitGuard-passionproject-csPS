from __future__ import annotations

import json
from functools import lru_cache

from ml.config import TAXONOMY_PATH


@lru_cache(maxsize=1)
def load_taxonomy() -> dict:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def class_ids() -> list[str]:
    return [item["id"] for item in load_taxonomy()["classes"]]


def class_index() -> dict[str, int]:
    return {cid: i for i, cid in enumerate(class_ids())}


def by_id(class_id: str) -> dict:
    for item in load_taxonomy()["classes"]:
        if item["id"] == class_id:
            return item
    raise KeyError(class_id)


def folder_to_id() -> dict[str, str]:
    return {item["plantvillage_folder"]: item["id"] for item in load_taxonomy()["classes"]}
