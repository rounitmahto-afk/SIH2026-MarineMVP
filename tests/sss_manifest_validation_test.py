from services.api.schemas.ingestion import SSSIngestionManifest


def valid_manifest(**frame_overrides):
    frame = {
        "filename": "sample.jpg",
        "frame_index": 0,
    }
    frame.update(frame_overrides)

    return {
        "schema_version": "1",
        "modality": "side_scan_sonar",
        "source_id": "validation-test-source",
        "sequence_id": "validation-test-sequence",
        "frames": [frame],
    }


def assert_rejected(payload, message):
    rejected = False

    try:
        SSSIngestionManifest.model_validate(payload)
    except Exception:
        rejected = True

    assert rejected, message


def run() -> None:
    assert_rejected(
        valid_manifest(frame_index=-1),
        "Negative frame_index must be rejected",
    )

    assert_rejected(
        valid_manifest(latitude=90.1),
        "Latitude above 90 must be rejected",
    )

    assert_rejected(
        valid_manifest(longitude=-180.1),
        "Longitude below -180 must be rejected",
    )

    assert_rejected(
        valid_manifest(heading_deg=360.0),
        "Heading of 360 degrees must be rejected",
    )

    assert_rejected(
        valid_manifest(filename="frames/sample.jpg"),
        "Filenames containing directory paths must be rejected",
    )

    assert_rejected(
        {
            "schema_version": "1",
            "modality": "side_scan_sonar",
            "source_id": "validation-test-source",
            "sequence_id": "validation-test-sequence",
            "frames": [
                {"filename": "a.jpg", "frame_index": 0},
                {"filename": "b.jpg", "frame_index": 0},
            ],
        },
        "Duplicate frame_index values must be rejected",
    )

    assert_rejected(
        valid_manifest(filename=""),
        "Empty filenames must be rejected",
    )

    assert_rejected(
        valid_manifest(filename="a" * 256),
        "Filenames longer than 255 characters must be rejected",
    )

    print("Negative frame_index rejection: PASS")
    print("Latitude range validation: PASS")
    print("Longitude range validation: PASS")
    print("Heading range validation: PASS")
    print("Directory path rejection: PASS")
    print("Duplicate frame_index rejection: PASS")
    print("Empty filename rejection: PASS")
    print("Filename length validation: PASS")
    print("SSS manifest validation tests: PASS")


if __name__ == "__main__":
    run()
