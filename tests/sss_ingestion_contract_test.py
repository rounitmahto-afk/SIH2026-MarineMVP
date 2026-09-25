from pathlib import Path
from datetime import datetime, timezone

from services.api.schemas.ingestion import SSSIngestionManifest


def run() -> None:
    valid = SSSIngestionManifest.model_validate(
        {
            "schema_version": "1",
            "modality": "side_scan_sonar",
            "source_id": "ghostvision-smoke-source",
            "sequence_id": "ghostvision-smoke-sequence",
            "frames": [
                {
                    "filename": "sample.jpg",
                    "frame_index": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        }
    )

    assert valid.modality == "side_scan_sonar"

    rejected = False

    try:
        SSSIngestionManifest.model_validate(
            {
                "schema_version": "1",
                "modality": "rgb",
                "source_id": "invalid",
                "sequence_id": "invalid",
                "frames": [
                    {
                        "filename": "rgb.jpg",
                        "frame_index": 0,
                    }
                ],
            }
        )
    except Exception:
        rejected = True

    assert rejected, "Non-SSS modality must be rejected"

    duplicate_rejected = False

    try:
        SSSIngestionManifest.model_validate(
            {
                "schema_version": "1",
                "modality": "side_scan_sonar",
                "source_id": "duplicate-test",
                "sequence_id": "duplicate-test",
                "frames": [
                    {
                        "filename": "a.jpg",
                        "frame_index": 0,
                    },
                    {
                        "filename": "a.jpg",
                        "frame_index": 1,
                    },
                ],
            }
        )
    except Exception:
        duplicate_rejected = True

    assert duplicate_rejected, "Duplicate SSS filenames must be rejected"

    print("Valid SSS manifest: PASS")
    print("Non-SSS modality rejection: PASS")
    print("Duplicate filename rejection: PASS")
    print("SSS ingestion contract test: PASS")


if __name__ == "__main__":
    run()
