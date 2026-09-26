import asyncio
import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np

from services.api.services.ingestion import (
    IngestionValidationError,
    calculate_bundle_checksum,
    canonical_manifest_bytes,
    save_and_validate_image,
)
from services.api.schemas.ingestion import SSSIngestionManifest


class FakeUpload:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._content):
            return b""

        if size < 0:
            chunk = self._content[self._offset :]
            self._offset = len(self._content)
            return chunk

        end = min(self._offset + size, len(self._content))
        chunk = self._content[self._offset : end]
        self._offset = end
        return chunk


def make_jpeg(width: int = 32, height: int = 24) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)

    assert ok, "OpenCV failed to encode test image"
    return encoded.tobytes()


def test_checksum_is_deterministic() -> None:
    manifest = b"{\"modality\":\"side_scan_sonar\",\"source_id\":\"source\",\"sequence_id\":\"sequence\"}"
    file_hashes = [
        ("b.jpg", "bb" * 32),
        ("a.jpg", "aa" * 32),
    ]

    first = calculate_bundle_checksum(manifest, file_hashes)
    second = calculate_bundle_checksum(manifest, file_hashes)

    assert first == second
    assert len(first) == 64
    assert first == hashlib.sha256(
        manifest
        + b"b.jpg\n"
        + ("bb" * 32).encode("ascii")
        + b"\n"
        + b"a.jpg\n"
        + ("aa" * 32).encode("ascii")
        + b"\n"
    ).hexdigest()


def test_canonical_manifest_bytes_are_stable() -> None:
    manifest_a = SSSIngestionManifest.model_validate(
        {
            "schema_version": "1",
            "modality": "side_scan_sonar",
            "source_id": "source",
            "sequence_id": "sequence",
            "frames": [
                {
                    "filename": "sample.jpg",
                    "frame_index": 0,
                }
            ],
        }
    )

    manifest_b = SSSIngestionManifest.model_validate(
        {
            "frames": [
                {
                    "frame_index": 0,
                    "filename": "sample.jpg",
                }
            ],
            "sequence_id": "sequence",
            "source_id": "source",
            "modality": "side_scan_sonar",
            "schema_version": "1",
        }
    )

    assert canonical_manifest_bytes(manifest_a) == canonical_manifest_bytes(manifest_b)


def test_valid_image_is_saved_and_hashed() -> None:
    content = make_jpeg()
    upload = FakeUpload("sample.jpg", content)

    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp) / "sample.jpg"
        width, height, sha256, total_bytes = asyncio.run(
            save_and_validate_image(upload, destination, 0)
        )

        assert destination.exists()
        assert destination.read_bytes() == content
        assert (width, height) == (32, 24)
        assert sha256 == hashlib.sha256(content).hexdigest()
        assert total_bytes == len(content)


def assert_image_rejected(filename: str, content: bytes) -> None:
    upload = FakeUpload(filename, content)

    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp) / Path(filename).name

        try:
            asyncio.run(save_and_validate_image(upload, destination, 0))
        except IngestionValidationError:
            return

    raise AssertionError(f"Expected image validation to reject {filename!r}")


def test_unsupported_extension_is_rejected() -> None:
    assert_image_rejected("sample.txt", b"not an image")


def test_directory_path_is_rejected() -> None:
    upload = FakeUpload("frames/sample.jpg", make_jpeg())

    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp) / "sample.jpg"

        try:
            asyncio.run(save_and_validate_image(upload, destination, 0))
        except IngestionValidationError:
            return

    raise AssertionError("Directory paths in uploaded filenames must be rejected")


def test_empty_image_is_rejected() -> None:
    assert_image_rejected("empty.jpg", b"")


def test_corrupt_image_is_rejected() -> None:
    assert_image_rejected("corrupt.jpg", b"this is not a valid image")


def run() -> None:
    test_checksum_is_deterministic()
    test_canonical_manifest_bytes_are_stable()
    test_valid_image_is_saved_and_hashed()
    test_unsupported_extension_is_rejected()
    test_directory_path_is_rejected()
    test_empty_image_is_rejected()
    test_corrupt_image_is_rejected()

    print("Bundle checksum determinism: PASS")
    print("Canonical manifest stability: PASS")
    print("Valid image save/decode/hash: PASS")
    print("Unsupported extension rejection: PASS")
    print("Directory path rejection: PASS")
    print("Empty image rejection: PASS")
    print("Corrupt image rejection: PASS")
    print("Ingestion service unit tests: PASS")


if __name__ == "__main__":
    run()
