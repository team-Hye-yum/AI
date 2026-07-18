from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from json import dumps, loads
from pathlib import Path
from re import finditer, search, split
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid5

from app.core.config import settings


DocumentProperties = dict[str, Any]

BATCH_SIZE = 100
MAX_CHUNK_CHARS = 6000
CHUNK_OVERLAP_CHARS = 400


@dataclass(frozen=True)
class WeaviateIngestResult:
    enabled: bool
    collection: str
    object_count: int


class WeaviateIngestError(RuntimeError):
    pass


def ingest_training_file(
    *,
    dataset_type: str,
    path: Path,
    original_filename: str,
    sha256: str,
    uploaded_at: datetime,
) -> WeaviateIngestResult:
    if not settings.weaviate_ingest_enabled:
        return WeaviateIngestResult(
            enabled=False,
            collection=settings.weaviate_collection,
            object_count=0,
        )

    text = path.read_text(encoding="utf-8-sig")
    documents = _parse_documents(
        dataset_type=dataset_type,
        text=text,
        original_filename=original_filename,
        stored_filename=path.name,
        stored_path=path.as_posix(),
        sha256=sha256,
        uploaded_at=uploaded_at,
    )
    if not documents:
        raise WeaviateIngestError("no ingestible document chunks were found")

    _ensure_schema()
    _batch_insert(documents)

    return WeaviateIngestResult(
        enabled=True,
        collection=settings.weaviate_collection,
        object_count=len(documents),
    )


def count_ingestible_documents(*, dataset_type: str, path: Path) -> int:
    text = path.read_text(encoding="utf-8-sig")
    count = 0
    for chunk in _split_by_dataset_type(dataset_type, text):
        count += len([piece for piece in _split_large_chunk(chunk) if piece.strip()])
    return count


def count_training_documents(*, dataset_type: str, sha256: str) -> int:
    query = f"""
    {{
      Aggregate {{
        {settings.weaviate_collection}(
          where: {{
            operator: And
            operands: [
              {{ path: ["datasetType"], operator: Equal, valueText: {dumps(dataset_type)} }}
              {{ path: ["sha256"], operator: Equal, valueText: {dumps(sha256)} }}
            ]
          }}
        ) {{
          meta {{
            count
          }}
        }}
      }}
    }}
    """
    try:
        response = _request("POST", "/v1/graphql", {"query": query})
    except WeaviateIngestError as exc:
        if _is_missing_schema_error(exc):
            return 0
        raise
    aggregate = response.get("data", {}).get("Aggregate", {})
    collection = aggregate.get(settings.weaviate_collection, [])
    if not collection:
        return 0
    return int(collection[0].get("meta", {}).get("count", 0))


def delete_training_documents(*, dataset_type: str, sha256: str) -> None:
    payload = {
        "match": {
            "class": settings.weaviate_collection,
            "where": {
                "operator": "And",
                "operands": [
                    {
                        "path": ["datasetType"],
                        "operator": "Equal",
                        "valueText": dataset_type,
                    },
                    {
                        "path": ["sha256"],
                        "operator": "Equal",
                        "valueText": sha256,
                    },
                ],
            },
        },
        "output": "minimal",
    }
    _request("DELETE", "/v1/batch/objects", payload)


def _is_missing_schema_error(exc: WeaviateIngestError) -> bool:
    message = str(exc)
    return (
        "no graphql provider present" in message
        or f'Cannot query field "{settings.weaviate_collection}"' in message
        or "schema" in message and "Import a schema first" in message
    )


def _parse_documents(
    *,
    dataset_type: str,
    text: str,
    original_filename: str,
    stored_filename: str,
    stored_path: str,
    sha256: str,
    uploaded_at: datetime,
) -> list[DocumentProperties]:
    documents = []

    for source_chunk in _split_by_dataset_type(dataset_type, text):
        metadata = _extract_metadata(dataset_type, source_chunk)
        for chunk in _split_large_chunk(source_chunk):
            chunk = chunk.strip()
            if not chunk:
                continue

            documents.append(
                {
                    "datasetType": dataset_type,
                    "sourceFilename": original_filename,
                    "storedFilename": stored_filename,
                    "storedPath": stored_path,
                    "sha256": sha256,
                    "chunkIndex": len(documents),
                    "title": metadata.get("title", ""),
                    "code": metadata.get("code", ""),
                    "name": metadata.get("name", ""),
                    "section": metadata.get("section", ""),
                    "text": chunk,
                    "uploadedAt": uploaded_at.isoformat(),
                }
            )

    return documents


def _split_large_chunk(chunk: str) -> list[str]:
    chunk = chunk.strip()
    if len(chunk) <= MAX_CHUNK_CHARS:
        return [chunk]

    pieces: list[str] = []
    start = 0
    while start < len(chunk):
        hard_end = min(start + MAX_CHUNK_CHARS, len(chunk))
        end = hard_end
        if hard_end < len(chunk):
            paragraph_end = chunk.rfind("\n\n", start, hard_end)
            line_end = chunk.rfind("\n", start, hard_end)
            end = max(paragraph_end, line_end)
            if end <= start:
                end = hard_end

        piece = chunk[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= len(chunk):
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)

    return pieces


def _split_by_dataset_type(dataset_type: str, text: str) -> list[str]:
    if dataset_type == "ksic":
        return _split_ksic(text)
    if dataset_type == "notice":
        return [
            chunk
            for chunk in split(r"\n={20,}\n(?=FILE: )", text)
            if chunk.strip()
        ]
    if dataset_type == "equipment":
        return [
            chunk
            for chunk in split(r"\n={20,}\n(?=eq_seq: )", text)
            if chunk.strip()
        ]
    return [text]


def _split_ksic(text: str) -> list[str]:
    matches = list(finditer(r"(?m)^\[[A-Z]\d{5}\]\s*$", text))
    if not matches:
        return [text]

    header = text[: matches[0].start()].strip()
    chunks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunk = text[match.start() : end].strip()
        if header:
            chunk = f"{header}\n\n{chunk}"
        chunks.append(chunk)

    return chunks


def _extract_metadata(dataset_type: str, chunk: str) -> dict[str, str]:
    if dataset_type == "ksic":
        return {
            "title": _first_match(r"(?m)^Name:\s*(.+)$", chunk),
            "code": _first_match(r"(?m)^KSIC code:\s*(.+)$", chunk),
            "name": _first_match(r"(?m)^Name:\s*(.+)$", chunk),
            "section": _first_match(r"(?m)^Section:\s*(.+)$", chunk),
        }
    if dataset_type == "notice":
        return {"title": _first_match(r"(?m)^FILE:\s*(.+)$", chunk)}
    if dataset_type == "equipment":
        return {"code": _first_match(r"(?m)^eq_seq:\s*(.+)$", chunk)}
    return {}


def _first_match(pattern: str, text: str) -> str:
    matched = search(pattern, text)
    return matched.group(1).strip() if matched else ""


def _ensure_schema() -> None:
    try:
        _request("GET", f"/v1/schema/{settings.weaviate_collection}")
        return
    except WeaviateIngestError as exc:
        if "404" not in str(exc):
            raise

    vectorizer = "text2vec-openai" if settings.openai_api_key else "none"
    payload: dict[str, Any] = {
        "class": settings.weaviate_collection,
        "description": "Uploaded Hyeyum training documents",
        "vectorizer": vectorizer,
        "properties": [
            {"name": "datasetType", "dataType": ["text"]},
            {"name": "sourceFilename", "dataType": ["text"]},
            {"name": "storedFilename", "dataType": ["text"]},
            {"name": "storedPath", "dataType": ["text"]},
            {"name": "sha256", "dataType": ["text"]},
            {"name": "chunkIndex", "dataType": ["int"]},
            {"name": "title", "dataType": ["text"]},
            {"name": "code", "dataType": ["text"]},
            {"name": "name", "dataType": ["text"]},
            {"name": "section", "dataType": ["text"]},
            {"name": "text", "dataType": ["text"]},
            {"name": "uploadedAt", "dataType": ["date"]},
        ],
    }
    if vectorizer == "text2vec-openai":
        payload["moduleConfig"] = {
            "text2vec-openai": {
                "model": "text-embedding-3-small",
                "type": "text",
                "vectorizeClassName": False,
            }
        }

    _request("POST", "/v1/schema", payload)


def _batch_insert(documents: list[DocumentProperties]) -> None:
    for start in range(0, len(documents), BATCH_SIZE):
        objects = [
            {
                "class": settings.weaviate_collection,
                "id": _document_id(document),
                "properties": document,
            }
            for document in documents[start : start + BATCH_SIZE]
        ]
        response = _request("POST", "/v1/batch/objects", {"objects": objects})
        failed = [
            item
            for item in response
            if item.get("result", {}).get("errors")
        ]
        if failed:
            raise WeaviateIngestError(f"weaviate batch insert failed: {failed[:1]}")


def _document_id(document: DocumentProperties) -> str:
    source_key = ":".join(
        [
            settings.weaviate_collection,
            str(document["datasetType"]),
            str(document["sha256"]),
            str(document["chunkIndex"]),
        ]
    )
    return str(uuid5(NAMESPACE_URL, source_key))


def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    url = urljoin(settings.weaviate_url.rstrip("/") + "/", path.lstrip("/"))
    body = dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if settings.openai_api_key:
        headers["X-OpenAI-Api-Key"] = settings.openai_api_key

    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(
            request,
            timeout=settings.weaviate_request_timeout_seconds,
        ) as response:
            response_body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise WeaviateIngestError(
            f"weaviate request failed with {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise WeaviateIngestError(f"weaviate is unreachable: {exc.reason}") from exc

    if not response_body:
        return None
    return loads(response_body.decode("utf-8"))
