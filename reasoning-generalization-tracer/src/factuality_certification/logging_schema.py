from __future__ import annotations

from datetime import datetime, timezone


def make_log_bundle(**kwargs) -> dict:
    bundle = dict(kwargs)
    bundle["timestamp"] = datetime.now(timezone.utc).isoformat()
    return bundle
