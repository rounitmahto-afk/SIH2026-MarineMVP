from __future__ import annotations

import json
from pathlib import Path

from ml.tracking.frame_manifest import (
    SurveyFrame,
    validate_sequence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "sequence_contract_example.json"
)


def main() -> None:
    """
    Contract-only validation.

    This example intentionally contains no invented
    survey sequence. It validates the schema mechanics
    using an empty example contract definition.
    """

    contract = {
        "contract": "verified_survey_frame_sequence",
        "version": 1,
        "requires_verified_frame_index": True,
        "requires_sequence_id": True,
        "requires_source_id": True,
        "geolocation_optional": True,
        "heading_optional": True,
        "frames": [],
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            contract,
            indent=2,
        ),
        encoding="utf-8",
    )

    if contract["frames"]:
        frames = [
            SurveyFrame(**frame)
            for frame in contract["frames"]
        ]

        validate_sequence(frames)
    else:
        print(
            "No sequence supplied: contract definition "
            "validated without inventing frame order."
        )

    print("")
    print("============================================================")
    print("FRAME MANIFEST CONTRACT VALIDATED")
    print("============================================================")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
