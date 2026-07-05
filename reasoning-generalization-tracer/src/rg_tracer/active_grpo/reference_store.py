"""Append-only JSONL reference store for Active-GRPO records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from ..transactions.transition_log import _file_lock, _path_lock
from .types import ReferenceRecord


class JSONLReferenceStore:
    """Lightweight append-only persistence for reference history."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: ReferenceRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = _path_lock(self.path)
        with lock, _file_lock(self.path):
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
                handle.flush()

    def load_all(self) -> List[ReferenceRecord]:
        records: List[ReferenceRecord] = []
        lock = _path_lock(self.path)
        with lock, _file_lock(self.path):
            try:
                handle = self.path.open("r", encoding="utf-8")
            except FileNotFoundError:
                return []
            with handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        records.append(ReferenceRecord.from_mapping(json.loads(stripped)))
        return records

    def latest_by_task(self) -> Dict[str, ReferenceRecord]:
        latest: Dict[str, ReferenceRecord] = {}
        for record in self.load_all():
            latest[record.task_id] = record
        return latest

    def iter_task(self, task_id: str) -> Iterable[ReferenceRecord]:
        for record in self.load_all():
            if record.task_id == task_id:
                yield record


__all__ = ["JSONLReferenceStore"]
