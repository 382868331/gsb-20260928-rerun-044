from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


EMPTY_DATABASE: dict[str, list[dict[str, Any]]] = {
    "datasets": [],
    "batches": [],
    "tests": [],
    "permutation_samples": [],
    "batch_assignments": [],
    "batch_states": [],
    "run_states": [],
    "test_results": [],
}


def dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def loads(value: str) -> object:
    return json.loads(value)


class JsonStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        if not self.path.exists():
            return {key: list(value) for key, value in EMPTY_DATABASE.items()}
        with self.path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return {key: list(loaded.get(key, [])) for key in EMPTY_DATABASE}

    def commit(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def next_id(self, table: str) -> int:
        rows = self.data[table]
        return 1 if not rows else max(int(row["id"]) for row in rows) + 1

    def insert(self, table: str, row: dict[str, Any]) -> int:
        if "id" not in row:
            row = {"id": self.next_id(table), **row}
        self.data[table].append(row)
        return int(row["id"])

    def find_one(self, table: str, **predicate: Any) -> dict[str, Any] | None:
        for row in self.data[table]:
            if all(row.get(key) == value for key, value in predicate.items()):
                return row
        return None

    def find_many(self, table: str, **predicate: Any) -> list[dict[str, Any]]:
        return [
            row
            for row in self.data[table]
            if all(row.get(key) == value for key, value in predicate.items())
        ]

    def close(self) -> None:
        self.commit()


def connect(path: str | Path) -> JsonStore:
    return JsonStore(path)
