"""Append-only transition log and effective-state projection."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Mapping

from .proposal import AdmissionDecision, Proposal

logger = logging.getLogger(__name__)

if os.name == "nt":  # pragma: no cover - platform-specific locking
    import msvcrt
else:  # pragma: no cover - platform-specific locking
    import fcntl


_PATH_LOCKS: Dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


class TransitionLog:
    """JSONL transition log for accepted and rejected proposals."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, proposal: Proposal, decision: AdmissionDecision) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"proposal": proposal.to_dict(), "decision": decision.to_dict()}
        lock = _path_lock(self.path)
        with lock, _file_lock(self.path):
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def load(self) -> List[Mapping[str, object]]:
        records: List[Mapping[str, object]] = []
        lock = _path_lock(self.path)
        with lock, _file_lock(self.path):
            try:
                handle = self.path.open("r", encoding="utf-8")
            except FileNotFoundError:
                return []
            with handle:
                self._load_records(handle, records)
        return records

    def _load_records(self, handle: object, records: List[Mapping[str, object]]) -> None:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.error(
                        "Corrupted transition log line %d in %s",
                        line_number,
                        self.path,
                    )
                    continue

    def effective_state(self) -> Dict[str, Dict[str, object]]:
        state: Dict[str, Dict[str, object]] = {}
        for record in self.load():
            if not isinstance(record, Mapping):
                continue
            proposal = record.get("proposal")
            decision = record.get("decision")
            if isinstance(proposal, dict) and isinstance(decision, dict):
                task_id = proposal.get("task_id")
                if decision.get("accepted") is True and task_id:
                    state[str(task_id)] = dict(proposal)
        return state


__all__ = ["TransitionLog"]


def _path_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        if key not in _PATH_LOCKS:
            _PATH_LOCKS[key] = threading.Lock()
        return _PATH_LOCKS[key]


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    canonical_path = path.resolve()
    lock_path = Path(f"{canonical_path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            handle.seek(0)
            _lock_windows_file(handle.fileno())
        else:
            _lock_posix_file(handle.fileno())
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lock_windows_file(file_descriptor: int, attempts: int = 10) -> None:
    for attempt in range(attempts):
        try:
            msvcrt.locking(file_descriptor, msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if attempt == attempts - 1:
                raise TimeoutError("timed out acquiring transition log lock") from exc
            time.sleep(0.05)


def _lock_posix_file(file_descriptor: int, attempts: int = 10, wait: float = 0.05) -> None:
    for attempt in range(attempts):
        try:
            fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as exc:
            if attempt == attempts - 1:
                raise TimeoutError("timed out acquiring transition log lock") from exc
            time.sleep(wait)
