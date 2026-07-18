from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from re import sub
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from app.core.config import settings
from app.services.training_data_seed import get_training_data_seed_status
from app.services.weaviate_ingest import WeaviateIngestError, ingest_training_file

router = APIRouter()

DatasetType = Literal["equipment", "notice", "ksic"]

CHUNK_SIZE = 1024 * 1024
ALLOWED_SUFFIXES = {".txt"}


class TrainingDataUploadResponse(BaseModel):
    dataset_type: DatasetType
    original_filename: str
    stored_filename: str
    stored_path: str
    content_type: str | None
    size_bytes: int
    sha256: str
    uploaded_at: datetime
    weaviate_enabled: bool
    weaviate_collection: str
    weaviate_object_count: int


class UploadedTrainingDataItem(BaseModel):
    dataset_type: str
    stored_filename: str
    stored_path: str
    size_bytes: int
    modified_at: datetime


class UploadedTrainingDataListResponse(BaseModel):
    total_count: int
    items: list[UploadedTrainingDataItem]


class TrainingDataSeedFileStatus(BaseModel):
    dataset_type: str
    path: str
    sha256: str
    expected_object_count: int
    weaviate_object_count: int | None
    weaviate_error: str
    seeded: bool
    state_seeded: bool
    seeded_at: str


class TrainingDataSeedStatusResponse(BaseModel):
    seed_enabled: bool
    weaviate_ingest_enabled: bool
    weaviate_collection: str
    upload_dir: str
    seed_dir: str
    seed_dir_exists: bool
    state_file: str
    state_file_exists: bool
    files: list[TrainingDataSeedFileStatus]


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="filename is required",
        )

    safe_name = sub(r"[^0-9A-Za-z\uac00-\ud7a3._ -]+", "_", name).strip(" .")
    return safe_name or f"upload-{uuid4().hex}.txt"


def _validate_upload(file: UploadFile) -> str:
    safe_name = _safe_filename(file.filename or "")
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"only {allowed} files are allowed",
        )
    return safe_name


def _iter_uploaded_files(dataset_type: DatasetType | None) -> list[Path]:
    upload_root = Path(settings.upload_dir)
    if not upload_root.exists():
        return []

    if dataset_type is not None:
        search_roots = [upload_root / dataset_type]
    else:
        search_roots = [path for path in upload_root.iterdir() if path.is_dir()]

    files: list[Path] = []
    for search_root in search_roots:
        if search_root.exists():
            files.extend(path for path in search_root.iterdir() if path.is_file())

    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


@router.get("/uploads/training-data", response_model=UploadedTrainingDataListResponse)
def list_uploaded_training_data(
    dataset_type: DatasetType | None = Query(default=None),
) -> UploadedTrainingDataListResponse:
    files = _iter_uploaded_files(dataset_type)
    items = []

    for path in files:
        stat = path.stat()
        items.append(
            UploadedTrainingDataItem(
                dataset_type=path.parent.name,
                stored_filename=path.name,
                stored_path=path.as_posix(),
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, UTC),
            )
        )

    return UploadedTrainingDataListResponse(total_count=len(items), items=items)


@router.get(
    "/training-data/seed-status",
    response_model=TrainingDataSeedStatusResponse,
)
def get_training_data_seed_status_response() -> TrainingDataSeedStatusResponse:
    return TrainingDataSeedStatusResponse(**get_training_data_seed_status())


@router.post(
    "/uploads/training-data",
    response_model=TrainingDataUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_training_data(
    dataset_type: DatasetType = Form(...),
    file: UploadFile = File(...),
) -> TrainingDataUploadResponse:
    safe_name = _validate_upload(file)
    uploaded_at = datetime.now(UTC)
    upload_root = Path(settings.upload_dir)
    target_dir = upload_root / dataset_type
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = uploaded_at.strftime("%Y%m%dT%H%M%S")
    stored_filename = f"{timestamp}-{uuid4().hex[:12]}-{safe_name}"
    target_path = target_dir / stored_filename
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")

    max_size = settings.max_upload_file_size_mb * 1024 * 1024
    digest = sha256()
    size_bytes = 0

    try:
        with temp_path.open("wb") as output:
            while chunk := await file.read(CHUNK_SIZE):
                size_bytes += len(chunk)
                if size_bytes > max_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"file exceeds {settings.max_upload_file_size_mb}MB limit",
                    )
                digest.update(chunk)
                output.write(chunk)

        if size_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="empty files are not allowed",
            )

        temp_path.replace(target_path)
    except HTTPException:
        if temp_path.exists():
            temp_path.unlink()
        raise
    except OSError as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to store upload: {exc}",
        ) from exc
    finally:
        await file.close()

    try:
        weaviate_result = ingest_training_file(
            dataset_type=dataset_type,
            path=target_path,
            original_filename=file.filename or safe_name,
            sha256=digest.hexdigest(),
            uploaded_at=uploaded_at,
        )
    except WeaviateIngestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"failed to ingest into weaviate: {exc}",
        ) from exc

    return TrainingDataUploadResponse(
        dataset_type=dataset_type,
        original_filename=file.filename or safe_name,
        stored_filename=stored_filename,
        stored_path=target_path.as_posix(),
        content_type=file.content_type,
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
        uploaded_at=uploaded_at,
        weaviate_enabled=weaviate_result.enabled,
        weaviate_collection=weaviate_result.collection,
        weaviate_object_count=weaviate_result.object_count,
    )
