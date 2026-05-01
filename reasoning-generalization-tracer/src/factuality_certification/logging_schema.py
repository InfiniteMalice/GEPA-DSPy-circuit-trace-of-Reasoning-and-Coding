from __future__ import annotations

from datetime import datetime, timezone


def make_log_bundle(**kwargs) -> dict:
    bundle = {"timestamp": datetime.now(timezone.utc).isoformat()}
    bundle.update(kwargs)
    return bundle
