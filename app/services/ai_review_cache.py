from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from json import dumps, loads
from pathlib import Path
import sqlite3
from typing import Any

from app.core.config import settings


def canonical_json(value: Any) -> str:
    return dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cache_key(
    *, request_body: dict[str, Any], model_version: str, knowledge_base_version: str
) -> str:
    body_hash = sha256(canonical_json(request_body).encode("utf-8")).hexdigest()
    return f"ai-review-opinions:{model_version}:{knowledge_base_version}:{body_hash}"


class AiReviewCache:
    def __init__(self, path: str | None = None, ttl_seconds: int | None = None) -> None:
        self.path = Path(path or settings.ai_review_cache_path)
        self.ttl_seconds = ttl_seconds or settings.ai_review_cache_ttl_seconds
        self._ensure_table()

    def get(self, key: str) -> dict[str, Any] | None:
        now = datetime.now(UTC).isoformat()
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                select response_json
                from ai_review_cache
                where cache_key = ? and expires_at > ?
                """,
                (key, now),
            ).fetchone()
        if row is None:
            return None
        return loads(row["response_json"])

    def set(self, key: str, response: dict[str, Any]) -> None:
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=self.ttl_seconds)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                insert into ai_review_cache(cache_key, response_json, created_at, expires_at)
                values (?, ?, ?, ?)
                on conflict(cache_key) do update set
                    response_json = excluded.response_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    key,
                    dumps(response, ensure_ascii=False),
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )
            connection.commit()

    def _ensure_table(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                create table if not exists ai_review_cache (
                    cache_key text primary key,
                    response_json text not null,
                    created_at text not null,
                    expires_at text not null
                )
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection
