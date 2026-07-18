from __future__ import annotations

import logging
from datetime import UTC, datetime
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from pathlib import Path
from time import sleep
from typing import Any

from app.core.config import settings
from app.services.weaviate_ingest import (
    WeaviateIngestError,
    count_ingestible_documents,
    count_training_documents,
    delete_training_documents,
    ingest_training_file,
)


logger = logging.getLogger(__name__)

DATASET_TYPES = ("equipment", "notice", "ksic")


def get_training_data_seed_status() -> dict[str, Any]:
    upload_root = Path(settings.upload_dir)
    seed_root = Path(settings.seed_training_data_dir)
    state_path = upload_root / settings.seed_training_data_state_file
    state = _load_state(state_path)

    files = []
    for dataset_type, path, digest, expected_count in _find_seed_files(seed_root):
        state_entry = state.get("files", {}).get(_state_key(dataset_type, path), {})
        weaviate_count: int | None = None
        weaviate_error = ""
        try:
            weaviate_count = count_training_documents(
                dataset_type=dataset_type,
                sha256=digest,
            )
        except WeaviateIngestError as exc:
            weaviate_error = str(exc)

        files.append(
            {
                "dataset_type": dataset_type,
                "path": path.relative_to(seed_root).as_posix(),
                "sha256": digest,
                "expected_object_count": expected_count,
                "weaviate_object_count": weaviate_count,
                "weaviate_error": weaviate_error,
                "seeded": weaviate_count == expected_count,
                "state_seeded": (
                    isinstance(state_entry, dict)
                    and state_entry.get("sha256") == digest
                    and state_entry.get("object_count") == expected_count
                ),
                "seeded_at": state_entry.get("seeded_at", "")
                if isinstance(state_entry, dict)
                else "",
            }
        )

    return {
        "seed_enabled": settings.seed_training_data_on_startup,
        "weaviate_ingest_enabled": settings.weaviate_ingest_enabled,
        "weaviate_collection": settings.weaviate_collection,
        "upload_dir": upload_root.as_posix(),
        "seed_dir": seed_root.as_posix(),
        "seed_dir_exists": seed_root.exists(),
        "state_file": state_path.as_posix(),
        "state_file_exists": state_path.exists(),
        "files": files,
    }


def seed_training_data_once() -> None:
    if not settings.seed_training_data_on_startup:
        logger.info("Training data seed is disabled.")
        return

    if not settings.weaviate_ingest_enabled:
        logger.info("Training data seed skipped because Weaviate ingest is disabled.")
        return

    upload_root = Path(settings.upload_dir)
    seed_root = Path(settings.seed_training_data_dir)
    files = _find_seed_files(seed_root)
    if not files:
        logger.info("No training data seed files found in %s.", seed_root)
        return

    state_path = upload_root / settings.seed_training_data_state_file
    state = _load_state(state_path)
    last_error: WeaviateIngestError | None = None
    for attempt in range(1, settings.seed_training_data_max_attempts + 1):
        try:
            pending = _find_pending_files(files, state, state_path, seed_root)
            if not pending:
                logger.info("All training data seed files were already ingested.")
                return
            _seed_pending_files(pending, state, state_path, seed_root)
            return
        except WeaviateIngestError as exc:
            last_error = exc
            if attempt == settings.seed_training_data_max_attempts:
                break
            logger.warning(
                "Training data seed failed on attempt %s/%s: %s",
                attempt,
                settings.seed_training_data_max_attempts,
                exc,
            )
            sleep(settings.seed_training_data_retry_delay_seconds)

    logger.error("Training data seed failed after retries: %s", last_error)


def _find_pending_files(
    files: list[tuple[str, Path, str, int]],
    state: dict[str, Any],
    state_path: Path,
    seed_root: Path,
) -> list[tuple[str, Path, str]]:
    pending = []
    for dataset_type, path, digest, expected_count in files:
        existing_count = count_training_documents(dataset_type=dataset_type, sha256=digest)
        if existing_count == expected_count:
            _mark_seeded(state, seed_root, dataset_type, path, digest, existing_count)
            _save_state(state_path, state)
            logger.info(
                "Training data seed file %s already exists in Weaviate (%s objects).",
                path,
                existing_count,
            )
            continue

        if existing_count > 0:
            delete_training_documents(dataset_type=dataset_type, sha256=digest)
            logger.warning(
                "Deleted incomplete training data seed file %s from Weaviate "
                "(existing %s objects, expected %s).",
                path,
                existing_count,
                expected_count,
            )

        pending.append((dataset_type, path, digest))
    return pending


def _find_seed_files(upload_root: Path) -> list[tuple[str, Path, str, int]]:
    files: list[tuple[str, Path, str, int]] = []
    for dataset_type in DATASET_TYPES:
        dataset_root = upload_root / dataset_type
        if not dataset_root.exists():
            continue
        for path in sorted(dataset_root.glob("*.txt")):
            files.append(
                (
                    dataset_type,
                    path,
                    _file_sha256(path),
                    count_ingestible_documents(dataset_type=dataset_type, path=path),
                )
            )
    return files


def _seed_pending_files(
    pending: list[tuple[str, Path, str]],
    state: dict[str, Any],
    state_path: Path,
    seed_root: Path,
) -> None:
    for dataset_type, path, digest in pending:
        result = ingest_training_file(
            dataset_type=dataset_type,
            path=path,
            original_filename=path.name,
            sha256=digest,
            uploaded_at=datetime.now(UTC),
        )
        _mark_seeded(state, seed_root, dataset_type, path, digest, result.object_count)
        _save_state(state_path, state)
        logger.info(
            "Seeded training data file %s into %s (%s objects).",
            path,
            result.collection,
            result.object_count,
        )


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "files": {}}

    try:
        loaded = loads(path.read_text(encoding="utf-8"))
    except (JSONDecodeError, OSError):
        logger.warning("Ignoring unreadable training data seed state file: %s", path)
        return {"version": 1, "files": {}}

    if not isinstance(loaded, dict):
        return {"version": 1, "files": {}}
    loaded.setdefault("version", 1)
    loaded.setdefault("files", {})
    return loaded


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_already_seeded(
    state: dict[str, Any],
    dataset_type: str,
    path: Path,
    digest: str,
    expected_count: int,
) -> bool:
    files = state.get("files", {})
    if not isinstance(files, dict):
        return False

    entry = files.get(_state_key(dataset_type, path))
    return (
        isinstance(entry, dict)
        and entry.get("sha256") == digest
        and entry.get("object_count") == expected_count
    )


def _mark_seeded(
    state: dict[str, Any],
    seed_root: Path,
    dataset_type: str,
    path: Path,
    digest: str,
    object_count: int,
) -> None:
    files = state.setdefault("files", {})
    files[_state_key(dataset_type, path)] = {
        "dataset_type": dataset_type,
        "path": path.relative_to(seed_root).as_posix(),
        "sha256": digest,
        "object_count": object_count,
        "seeded_at": datetime.now(UTC).isoformat(),
    }


def _state_key(dataset_type: str, path: Path) -> str:
    return f"{dataset_type}/{path.name}"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
