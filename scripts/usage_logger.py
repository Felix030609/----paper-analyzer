from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOCAL_USAGE_LOG = OUTPUT_DIR / "usage_events.jsonl"
ERROR_MESSAGE_LIMIT = 200


def _read_streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        return None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _read_config(name: str) -> str | None:
    return _read_streamlit_secret(name) or os.getenv(name)


def get_session_id() -> str:
    try:
        import streamlit as st

        session_id = st.session_state.get("session_id")
        if session_id:
            return str(session_id)
    except Exception:
        pass
    return "unknown"


def sanitize_event(event_type: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = dict(metadata or {})
    allowed_fields = {
        "file_type",
        "file_size_mb",
        "text_length",
        "model_name",
        "analysis_mode",
        "top_k",
        "duration_seconds",
        "success",
        "error_type",
        "error_message_short",
    }
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": get_session_id(),
        "event_type": str(event_type),
    }
    for field in allowed_fields:
        value = metadata.get(field)
        if value is not None:
            event[field] = value

    if "error_message_short" in event:
        event["error_message_short"] = str(event["error_message_short"])[:ERROR_MESSAGE_LIMIT]
    return event


def supabase_configured() -> bool:
    return all(
        _read_config(name)
        for name in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_USAGE_TABLE")
    )


def _get_supabase_client():
    try:
        from supabase import create_client
    except Exception as exc:
        raise RuntimeError(f"Supabase 依赖不可用：{exc}") from exc

    url = _read_config("SUPABASE_URL")
    key = _read_config("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Supabase 未配置。")
    return create_client(url, key)


def _write_supabase(event: dict[str, Any]) -> None:
    table = _read_config("SUPABASE_USAGE_TABLE")
    if not table:
        raise RuntimeError("SUPABASE_USAGE_TABLE 未配置。")
    client = _get_supabase_client()
    client.table(table).insert(event).execute()


def _write_local(event: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOCAL_USAGE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def log_event(event_type: str, metadata: dict[str, Any] | None = None) -> None:
    event = sanitize_event(event_type, metadata)
    try:
        if supabase_configured():
            try:
                _write_supabase(event)
                return
            except Exception as exc:
                print(f"Supabase usage log failed, fallback to local jsonl: {exc}", flush=True)
        _write_local(event)
    except Exception as exc:
        print(f"Usage log failed: {exc}", flush=True)


def read_usage_events(limit: int | None = None) -> list[dict[str, Any]]:
    try:
        if supabase_configured():
            table = _read_config("SUPABASE_USAGE_TABLE")
            client = _get_supabase_client()
            query = client.table(table).select("*").order("timestamp", desc=True)
            if limit:
                query = query.limit(limit)
            response = query.execute()
            data = response.data or []
            return list(reversed(data)) if limit else data
    except Exception as exc:
        print(f"Supabase usage read failed, fallback to local jsonl: {exc}", flush=True)

    if not LOCAL_USAGE_LOG.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        for line in LOCAL_USAGE_LOG.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                events.append(item)
    except Exception as exc:
        print(f"Local usage read failed: {exc}", flush=True)
        return []
    return events[-limit:] if limit else events


def usage_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(event.get("event_type", "")) for event in events)
    completed = [
        float(event.get("duration_seconds"))
        for event in events
        if event.get("event_type") == "analysis_completed"
        and isinstance(event.get("duration_seconds"), (int, float))
    ]
    model_counts = Counter(
        str(event.get("model_name", ""))
        for event in events
        if event.get("event_type") == "analysis_started"
    )
    avg_duration = round(sum(completed) / len(completed), 1) if completed else 0
    return {
        "counts": counts,
        "flash_count": model_counts.get("deepseek-v4-flash", 0),
        "pro_count": model_counts.get("deepseek-v4-pro", 0),
        "avg_duration": avg_duration,
    }
