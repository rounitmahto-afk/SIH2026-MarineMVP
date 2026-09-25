import hashlib
import json
import shutil
from pathlib import Path

import cv2
from fastapi import UploadFile

from services.api.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_ROOT = PROJECT_ROOT / "storage" / "uploads"
STAGING_ROOT = UPLOAD_ROOT / ".staging"

ALLOWED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


class IngestionValidationError(ValueError):
    pass


def canonical_manifest_bytes(manifest_model) -> bytes:
    return json.dumps(
        manifest_model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def calculate_bundle_checksum(manifest_bytes: bytes, file_hashes: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    digest.update(manifest_bytes)

    for filename, file_hash in file_hashes:
        digest.update(filename.encode("utf-8"))
        digest.update(b"\n")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")

    return digest.hexdigest()


async def save_and_validate_image(
    upload: UploadFile,
    destination: Path,
    total_bytes: int,
) -> tuple[int, int, str, int]:
    filename = Path(upload.filename or "").name

    if filename != upload.filename:
        raise IngestionValidationError(
            f"Invalid filename: {upload.filename!r}"
        )

    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise IngestionValidationError(
            f"Unsupported image extension for SSS input: {suffix or '<none>'}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    file_size = 0

    with destination.open("wb") as output:
        while True:
            chunk = await upload.read(1024 * 1024)

            if not chunk:
                break

            file_size += len(chunk)
            total_bytes += len(chunk)

            if file_size > settings.max_upload_bytes:
                raise IngestionValidationError(
                    f"File exceeds MAX_UPLOAD_BYTES ({settings.max_upload_bytes})."
                )

            if total_bytes > settings.max_upload_bytes:
                raise IngestionValidationError(
                    f"Job exceeds MAX_UPLOAD_BYTES ({settings.max_upload_bytes})."
                )

            digest.update(chunk)
            output.write(chunk)

    if file_size == 0:
        raise IngestionValidationError(
            f"Empty image file: {filename}"
        )

    image = cv2.imread(
        str(destination),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise IngestionValidationError(
            f"Image could not be decoded: {filename}"
        )

    height, width = image.shape[:2]

    if width > settings.max_image_width:
        raise IngestionValidationError(
            f"Image width {width} exceeds MAX_IMAGE_WIDTH "
            f"({settings.max_image_width}): {filename}"
        )

    if height > settings.max_image_height:
        raise IngestionValidationError(
            f"Image height {height} exceeds MAX_IMAGE_HEIGHT "
            f"({settings.max_image_height}): {filename}"
        )

    return width, height, digest.hexdigest(), total_bytes


def remove_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
